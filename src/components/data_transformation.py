import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

TARGET = "target_congestion"
# Ordered labels -> numbers (keeps the final numpy array numeric)
LABEL_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}

# target_next_count is the other target (congestion is derived from it) -> would leak
DROP_COLS = ["timestamp", "target_next_count"]

CATEGORICAL_COLUMNS = ["intersection_id"]
NUMERICAL_COLUMNS = [
    "hour", "minute", "day_of_week", "is_weekend", "is_holiday", "event_nearby",
    "temperature_c", "rain_mm_15min", "road_capacity_veh_15min", "vehicle_count",
    "avg_speed_kmh", "vehicle_density_per_km", "avg_waiting_time_sec",
    "queue_length_veh", "incident_flag", "prev_count_15min", "prev_count_1h",
    "count_same_time_yesterday", "count_same_time_last_week", "rolling_mean_1h",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning + cyclical time features (reuse this at prediction time too)."""
    df = df.copy()
    if "avg_speed_kmh" in df.columns:
        df["avg_speed_kmh"] = df["avg_speed_kmh"].clip(upper=120)  # sensor glitches
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
            ])
            return ColumnTransformer([
                ("num_pipeline", num_pipeline, NUMERICAL_COLUMNS),
                ("cat_pipeline", cat_pipeline, CATEGORICAL_COLUMNS),
            ])
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path).dropna(subset=[TARGET])
            test_df = pd.read_csv(test_path).dropna(subset=[TARGET])
            logging.info("Read train and test data")

            train_df = add_features(train_df)
            test_df = add_features(test_df)

            y_train = train_df[TARGET].map(LABEL_MAP)
            y_test = test_df[TARGET].map(LABEL_MAP)

            X_train = train_df.drop(columns=[TARGET] + DROP_COLS)
            X_test = test_df.drop(columns=[TARGET] + DROP_COLS)

            preprocessing_obj = self.get_data_transformer_object()
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr = np.c_[X_test_arr, y_test.to_numpy()]

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_obj,
            )
            logging.info("Preprocessor saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)