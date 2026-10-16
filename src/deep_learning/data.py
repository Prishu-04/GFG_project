"""
Data preparation for the sequence models (no torch needed here, so it can be tested anywhere).

Pipeline
  model-ready CSV -> features -> chronological split -> train-only scaling -> sliding windows
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C


@dataclass
class SplitData:
    X: np.ndarray            # (n_windows, seq_len, n_features) float32
    y: np.ndarray            # (n_windows,) utilisation of the NEXT 15 min
    capacity: np.ndarray     # (n_windows,) road capacity, to convert back to vehicles
    timestamp: np.ndarray    # (n_windows,) timestamp of the last step in each window
    site: np.ndarray         # (n_windows,) intersection id
    y_class: np.ndarray      # (n_windows,) true congestion class index (0..3)


def util_to_class(util: np.ndarray) -> np.ndarray:
    """Convert (predicted or true) utilisation to class index with the data-derived cut-offs."""
    return np.digitize(util, C.UTIL_BINS).astype(int)


def load_frame(path=C.DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values(["intersection_id", "timestamp"]).reset_index(drop=True)
    needed = ["vehicle_count", "target_next_count", "count_same_time_yesterday", "count_same_time_last_week",
              "rolling_mean_1h", "avg_speed_kmh", "vehicle_density_per_km", "avg_waiting_time_sec",
              "queue_length_veh", "temperature_c", "rain_mm_15min"]
    if df[needed].isna().any().any():
        raise ValueError("NaNs found - use data/processed/urban_traffic_model_ready.csv")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cap = df["road_capacity_veh_15min"].astype(float)
    df["util_now"] = df["vehicle_count"] / cap
    df["util_same_time_yesterday"] = df["count_same_time_yesterday"] / cap
    df["util_same_time_last_week"] = df["count_same_time_last_week"] / cap
    df["util_rolling_1h"] = df["rolling_mean_1h"] / cap

    tod = (df["hour"] * 60 + df["minute"]) / 1440.0
    df["tod_sin"], df["tod_cos"] = np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    for s in C.INTERSECTIONS:
        df[f"site_{s}"] = (df["intersection_id"] == s).astype(float)

    df["y_util"] = df["target_next_count"] / cap
    return df


def check_contiguous(df: pd.DataFrame) -> None:
    """Windows only make sense if every intersection has an unbroken 15-minute series."""
    for site, g in df.groupby("intersection_id"):
        steps = g["timestamp"].diff().dropna().unique()
        if len(steps) != 1 or steps[0] != pd.Timedelta("15min"):
            raise ValueError(f"{site}: time series has gaps / irregular steps: {steps[:5]}")


def split_boundaries(df: pd.DataFrame, train_frac=C.TRAIN_FRAC, val_frac=C.VAL_FRAC):
    ts = np.sort(df["timestamp"].unique())
    t_train_end = ts[int(len(ts) * train_frac) - 1]
    t_val_end = ts[int(len(ts) * (train_frac + val_frac)) - 1]
    return t_train_end, t_val_end


def make_windows(df: pd.DataFrame, scaler: dict, seq_len: int):
    """Sliding windows per intersection; a window is labelled by its LAST row."""
    Xs, ys, caps, tss, sites = [], [], [], [], []
    mean, scale = scaler["mean"], scaler["scale"]
    for site, g in df.groupby("intersection_id"):
        feats = g[C.FEATURES].to_numpy(dtype=np.float64).copy()
        n_cont = len(C.CONTINUOUS)
        feats[:, :n_cont] = (feats[:, :n_cont] - mean) / scale
        win = np.lib.stride_tricks.sliding_window_view(feats, seq_len, axis=0)   # (N-L+1, F, L)
        Xs.append(np.ascontiguousarray(win.transpose(0, 2, 1), dtype=np.float32))
        sl = slice(seq_len - 1, None)
        ys.append(g["y_util"].to_numpy()[sl])
        caps.append(g["road_capacity_veh_15min"].to_numpy()[sl])
        tss.append(g["timestamp"].to_numpy()[sl])
        sites.append(g["intersection_id"].to_numpy()[sl])
    return (np.concatenate(Xs), np.concatenate(ys).astype(np.float32),
            np.concatenate(caps).astype(np.float32), np.concatenate(tss), np.concatenate(sites))


def prepare(seq_len: int = C.SEQ_LEN, path=C.DATA_PATH):
    """Returns dict(train=SplitData, val=SplitData, test=SplitData), scaler dict, feature list."""
    df = add_features(load_frame(path))
    check_contiguous(df)
    t_tr, t_va = split_boundaries(df)

    # scaler uses ONLY rows from the training period
    train_rows = df[df["timestamp"] <= t_tr]
    cont = train_rows[C.CONTINUOUS].to_numpy(dtype=np.float64)
    scaler = {"mean": cont.mean(0), "scale": cont.std(0) + 1e-8}

    X, y, cap, ts, site = make_windows(df, scaler, seq_len)
    ts = pd.to_datetime(ts)
    masks = {
        "train": ts <= t_tr,
        "val": (ts > t_tr) & (ts <= t_va),
        "test": ts > t_va,
    }
    out = {}
    for name, m in masks.items():
        out[name] = SplitData(X[m], y[m], cap[m], ts[m].to_numpy(), site[m], util_to_class(y[m]))
    return out, scaler, list(C.FEATURES)
