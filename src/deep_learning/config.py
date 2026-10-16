"""
Shared settings for the deep-learning / transformer experiments.

Task (matches the project README): predict the NEXT 15-minute vehicle count for an
intersection from its recent history, then convert the predicted volume to a
congestion class (Low / Moderate / High / Severe).

We predict *utilisation* (= vehicle count / road capacity) instead of the raw count so that
the four intersections (capacities 260-520) share one target scale. The count is recovered
by multiplying back by the intersection's capacity.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "urban_traffic_model_ready.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "deep_learning"
REPORT_DIR = PROJECT_ROOT / "reports" / "deep_learning"
PRED_DIR = REPORT_DIR / "predictions"

# --- congestion classes --------------------------------------------------------------
# These cut-offs are not invented: in this dataset `target_congestion` is exactly
#   Low < 0.35 <= Moderate < 0.60 <= High < 0.85 <= Severe   for  target_next_count / capacity
# (verified on all 66,428 model-ready rows). The README asks for thresholds justified by the data.
CLASS_NAMES = ["Low", "Moderate", "High", "Severe"]
UTIL_BINS = [0.35, 0.60, 0.85]

# --- features -------------------------------------------------------------------------
INTERSECTIONS = ["INT_01_CBD", "INT_02_Arterial", "INT_03_Residential", "INT_04_Stadium"]

# continuous columns -> standardised with TRAIN statistics only
CONTINUOUS = [
    "util_now", "util_same_time_yesterday", "util_same_time_last_week", "util_rolling_1h",
    "avg_speed_kmh", "vehicle_density_per_km", "avg_waiting_time_sec", "queue_length_veh",
    "temperature_c", "rain_mm_15min", "road_capacity_veh_15min",
]
# already bounded / binary -> left as they are
UNSCALED = [
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "is_weekend", "is_holiday", "event_nearby", "incident_flag",
] + [f"site_{i}" for i in INTERSECTIONS]

FEATURES = CONTINUOUS + UNSCALED

# Deliberately NOT used as inputs (would leak the answer): target_next_count, target_congestion.

# --- split & windows ------------------------------------------------------------------
SEQ_LEN = 16            # 16 steps x 15 min = 4 hours of history
TRAIN_FRAC = 0.70       # chronological: first 70% of the timeline
VAL_FRAC = 0.15         # next 15%; the last 15% is the untouched test period
SEED = 42
