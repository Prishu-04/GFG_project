import os
import sys

import numpy as np
import pandas as pd

# --- make src/logger.py and src/exception.py importable no matter where
# this script is launched from (this file lives in "src/Data cleaning/") ---
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from logger import logging          # noqa: E402
from exception import CustomException  # noqa: E402

# --- relative, machine-independent paths -----------------------------------
PROJECT_ROOT = os.path.dirname(SRC_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "traffic_dataset.csv")
CLEAN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "traffic_dataset_clean.csv")

# Columns whose raw values are messy free-text / list-like and are dropped
# in the basic version of the project (see README "Recommended Scope").
COLUMNS_TO_DROP = ["IR Presence (Lane 1-4)", "Vehicle Types Detected"]

# Columns that must be numeric for modelling; coerced + median-imputed.
NUMERIC_COLUMNS_TO_FIX = ["Vehicle Count", "Avg Speed (km/h)", "Vehicle Density (%)"]


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw traffic CSV."""
    try:
        logging.info(f"Loading raw dataset from {path}")
        df = pd.read_csv(path, encoding="latin1")
        logging.info(f"Loaded raw dataset with shape {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray newlines / whitespace from column headers."""
    df = df.copy()
    df.columns = (
        df.columns.str.replace("\r\n", " ", regex=False)
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", " ", regex=False)
        .str.strip()
    )
    return df


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_dupes = df.duplicated().sum()
    logging.info(f"Dropping {n_dupes} duplicate rows")
    return df.drop_duplicates().reset_index(drop=True)


def extract_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = df["Timestamp"].astype(str).str.split(":").str[0].astype(int)
    df["Hour"] = df["Hour"] % 24
    return df


def fix_numeric_columns(df: pd.DataFrame, columns=NUMERIC_COLUMNS_TO_FIX) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        n_missing = df[col].isna().sum()
        if n_missing:
            median_val = df[col].median()
            logging.info(f"Imputing {n_missing} missing values in '{col}' with median {median_val}")
            df[col] = df[col].fillna(median_val)
    return df


def add_previous_traffic_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Previous_Traffic"] = df["Vehicle Count"].shift(1)
    df["Previous_Traffic"] = df["Previous_Traffic"].bfill()
    return df


def drop_unusable_columns(df: pd.DataFrame, columns=COLUMNS_TO_DROP) -> pd.DataFrame:
    existing = [c for c in columns if c in df.columns]
    logging.info(f"Dropping unusable columns: {existing}")
    return df.drop(columns=existing)


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = clean_column_names(df)
        df = drop_duplicate_rows(df)
        df = extract_hour_feature(df)
        df = fix_numeric_columns(df)
        df = add_previous_traffic_feature(df)
        df = drop_unusable_columns(df)

        remaining_nulls = df.isnull().sum().sum()
        logging.info(f"Cleaning complete. Final shape {df.shape}, remaining nulls: {remaining_nulls}")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def save_clean_data(df: pd.DataFrame, path: str = CLEAN_DATA_PATH) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"Saved cleaned dataset to {path}")
    except Exception as e:
        raise CustomException(e, sys)


def run_pipeline(raw_path: str = RAW_DATA_PATH, clean_path: str = CLEAN_DATA_PATH) -> pd.DataFrame:
    raw_df = load_raw_data(raw_path)
    clean_df = clean_traffic_data(raw_df)
    save_clean_data(clean_df, clean_path)
    return clean_df


if __name__ == "__main__":
    clean_df = run_pipeline()
    print("Final shape:", clean_df.shape)
    print(clean_df.isnull().sum())
    print(clean_df.head())
