"""
Data cleaning pipeline for urban_traffic_dataset.csv

Steps
  1. Load + parse timestamps, sort per intersection
  2. Remove exact / key duplicates
  3. Validate the 15-minute time grid (report gaps)
  4. Fix calendar columns that disagree with the timestamp
  5. Fix impossible values (negatives, count above physical capacity, bad flags/labels)
  6. Detect sensor outliers (upward speed spikes) and mark them missing
  7. Impute missing sensor values CAUSALLY (past-only: forward-fill, then typical profile)
  8. Save:
       urban_traffic_cleaned.csv      -> all rows, cleaned
       urban_traffic_model_ready.csv  -> rows with complete lags + targets (no NaNs)
       cleaning_report.json           -> everything that was changed

Usage
  python data_cleaning.py
  python data_cleaning.py --input urban_traffic_dataset.csv --outdir .
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SENSOR_COLS = ["avg_speed_kmh", "vehicle_density_per_km",
               "avg_waiting_time_sec", "queue_length_veh"]
BINARY_COLS = ["is_weekend", "is_holiday", "event_nearby", "incident_flag"]
LAG_COLS = ["prev_count_15min", "prev_count_1h", "count_same_time_yesterday",
            "count_same_time_last_week", "rolling_mean_1h"]
TARGET_COLS = ["target_next_count", "target_congestion"]
VALID_LEVELS = ["Low", "Moderate", "High", "Severe"]
FREQ = "15min"
MAX_FFILL_STEPS = 4          # carry a reading forward at most 1 hour
HAMPEL_WINDOW = 9            # 9 x 15 min = ~2 h centred window
HAMPEL_K = 12.0              # robust z-score threshold (5 flags ordinary sensor noise; 12 isolates real spikes)
SPEED_CAP_FACTOR = 1.10      # speed above 1.10 x the intersection's 99.5th percentile is physically implausible


# --------------------------------------------------------------------------- #
# 1. load
# --------------------------------------------------------------------------- #
def load(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)
    return df, {"rows_loaded": int(len(df) + bad_ts), "unparseable_timestamps_dropped": bad_ts}


# --------------------------------------------------------------------------- #
# 2. duplicates
# --------------------------------------------------------------------------- #
def drop_duplicates(df):
    n0 = len(df)
    df = df.drop_duplicates()
    exact = n0 - len(df)
    n1 = len(df)
    df = df.drop_duplicates(subset=["intersection_id", "timestamp"], keep="first")
    key = n1 - len(df)
    return df.reset_index(drop=True), {"exact_duplicates_removed": int(exact),
                                       "duplicate_key_rows_removed": int(key)}


# --------------------------------------------------------------------------- #
# 3. time grid
# --------------------------------------------------------------------------- #
def check_time_grid(df):
    gaps = {}
    for name, g in df.groupby("intersection_id"):
        full = pd.date_range(g["timestamp"].min(), g["timestamp"].max(), freq=FREQ)
        gaps[name] = int(len(full) - g["timestamp"].nunique())
    return {"missing_timestamps_per_intersection": gaps}


# --------------------------------------------------------------------------- #
# 4. calendar consistency
# --------------------------------------------------------------------------- #
def fix_calendar(df):
    ts = df["timestamp"]
    expected = {"hour": ts.dt.hour, "minute": ts.dt.minute,
                "day_of_week": ts.dt.dayofweek, "is_weekend": (ts.dt.dayofweek >= 5).astype(int)}
    fixed = {}
    for col, exp in expected.items():
        mismatch = int((df[col] != exp).sum())
        fixed[col] = mismatch
        df[col] = exp.astype(int)
    return df, {"calendar_values_corrected": fixed}


# --------------------------------------------------------------------------- #
# 5. impossible values
# --------------------------------------------------------------------------- #
def fix_impossible(df):
    rep = {}
    # negative sensor readings are impossible -> missing
    for col in SENSOR_COLS + ["vehicle_count"]:
        neg = df[col] < 0
        rep[f"negative_{col}"] = int(neg.sum())
        df.loc[neg, col] = np.nan
    # count above ~physical capacity (allow 5 % tolerance used by the simulator)
    over = df["vehicle_count"] > df["road_capacity_veh_15min"] * 1.05
    rep["count_over_capacity_clipped"] = int(over.sum())
    df.loc[over, "vehicle_count"] = (df.loc[over, "road_capacity_veh_15min"] * 1.05).round()
    # binary flags must be 0/1
    for col in BINARY_COLS:
        bad = ~df[col].isin([0, 1])
        rep[f"invalid_{col}"] = int(bad.sum())
        df.loc[bad, col] = df[col].where(~bad).mode().iloc[0] if (~bad).any() else 0
        df[col] = df[col].astype(int)
    # labels must be one of the 4 classes
    bad_lbl = df["target_congestion"].notna() & ~df["target_congestion"].isin(VALID_LEVELS)
    rep["invalid_target_labels_set_nan"] = int(bad_lbl.sum())
    df.loc[bad_lbl, "target_congestion"] = np.nan
    return df, rep


# --------------------------------------------------------------------------- #
# 6. sensor outliers (Hampel filter, upward spikes only)
# --------------------------------------------------------------------------- #
def mark_speed_outliers(df):
    """Upward speed spikes (e.g. sensor reading doubled) are physically impossible.
    Two rules, either one flags a reading:
      a) Hampel: jump above the local rolling median by > HAMPEL_K robust sigmas
      b) Physical cap: above SPEED_CAP_FACTOR x the intersection's 99.5th percentile
    Downward drops are kept: real incidents cause sudden slowdowns."""
    flagged = pd.Series(False, index=df.index)
    for name, g in df.groupby("intersection_id"):
        s = g["avg_speed_kmh"]
        med = s.rolling(HAMPEL_WINDOW, center=True, min_periods=3).median()
        dev = s - med
        mad = 1.4826 * dev.abs().median()
        cap = s.quantile(0.995) * SPEED_CAP_FACTOR
        flagged.loc[g.index] = (dev > HAMPEL_K * mad) | (s > cap)
    n = int(flagged.sum())
    df.loc[flagged, "avg_speed_kmh"] = np.nan
    return df, {"speed_outliers_set_missing": n}


# --------------------------------------------------------------------------- #
# 7. causal imputation
# --------------------------------------------------------------------------- #
def impute_sensors(df):
    """Past-only imputation (no future leakage):
         a) forward-fill up to MAX_FFILL_STEPS within each intersection
         b) fallback: median of the same intersection / hour / weekend profile"""
    rep = {"missing_before": {c: int(df[c].isna().sum()) for c in SENSOR_COLS}}
    imputed_count = pd.Series(0, index=df.index)
    for col in SENSOR_COLS:
        was_na = df[col].isna()
        df[col] = df.groupby("intersection_id")[col].ffill(limit=MAX_FFILL_STEPS)
        profile = df.groupby(["intersection_id", "hour", "is_weekend"])[col].transform("median")
        df[col] = df[col].fillna(profile)
        imputed_count += was_na.astype(int)
    df["n_imputed_sensor_values"] = imputed_count
    rep["missing_after"] = {c: int(df[c].isna().sum()) for c in SENSOR_COLS}
    rep["rows_with_any_imputed_value"] = int((imputed_count > 0).sum())
    return df, rep


# --------------------------------------------------------------------------- #
# 8. final typing + model-ready subset
# --------------------------------------------------------------------------- #
def finalise(df):
    df["intersection_id"] = df["intersection_id"].astype("category")
    for col in SENSOR_COLS:
        df[col] = df[col].round(1)
    int_cols = ["hour", "minute", "day_of_week", "vehicle_count", "road_capacity_veh_15min"] + BINARY_COLS
    for col in int_cols:
        df[col] = df[col].astype("int32")
    return df


def model_ready(df):
    need = LAG_COLS + TARGET_COLS
    out = df.dropna(subset=need).copy()
    return out.reset_index(drop=True), {"rows_dropped_missing_lags_or_targets": int(len(df) - len(out))}


# --------------------------------------------------------------------------- #
def main(input_path, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    report = {}

    df, r = load(input_path);              report["load"] = r
    report["shape_before"] = list(df.shape)
    report["missing_before_total"] = {c: int(v) for c, v in df.isna().sum().items() if v}
    df, r = drop_duplicates(df);           report["duplicates"] = r
    report["time_grid"] = check_time_grid(df)
    df, r = fix_calendar(df);              report["calendar"] = r
    df, r = fix_impossible(df);            report["impossible_values"] = r
    df, r = mark_speed_outliers(df);       report["outliers"] = r
    df, r = impute_sensors(df);            report["imputation"] = r
    df = finalise(df)
    ready, r = model_ready(df);            report["model_ready"] = r
    report["shape_cleaned"] = list(df.shape)
    report["shape_model_ready"] = list(ready.shape)

    df.to_csv(outdir / "urban_traffic_cleaned.csv", index=False)
    ready.to_csv(outdir / "urban_traffic_model_ready.csv", index=False)
    with open(outdir / "cleaning_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {outdir.resolve()}:")
    print("  urban_traffic_cleaned.csv | urban_traffic_model_ready.csv | cleaning_report.json")
    return df, ready, report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean the urban traffic dataset")
    ap.add_argument("--input", default="urban_traffic_dataset.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()
    main(a.input, a.outdir)
