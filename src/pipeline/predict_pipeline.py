import os
import sys

import pandas as pd

from src.components.data_transformation import (
    CATEGORICAL_COLUMNS,
    LABEL_MAP,
    NUMERICAL_COLUMNS,
    add_features,
)
from src.exception import CustomException
from src.utils import load_object

# Every intersection has a fixed road capacity in the dataset
ROAD_CAPACITY = {
    "INT_01_CBD": 420,
    "INT_02_Arterial": 520,
    "INT_03_Residential": 260,
    "INT_04_Stadium": 380,
}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features: pd.DataFrame):
        """Returns (label, {label: probability} or None)."""
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            features = add_features(features)
            data = preprocessor.transform(features[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS])

            pred = int(model.predict(data)[0])
            label = INV_LABEL_MAP[pred]

            probs = None
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(data)[0]
                probs = {INV_LABEL_MAP[int(c)]: float(v) for c, v in zip(model.classes_, p)}

            return label, probs
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        intersection_id: str,
        hour: int,
        minute: int,
        day_of_week: int,
        is_holiday: int,
        event_nearby: int,
        temperature_c: float,
        rain_mm_15min: float,
        vehicle_count: int,
        avg_speed_kmh: float,
        vehicle_density_per_km: float,
        avg_waiting_time_sec: float,
        queue_length_veh: float,
        incident_flag: int,
        prev_count_15min: float,
        prev_count_1h: float,
        count_same_time_yesterday: float,
        count_same_time_last_week: float,
        rolling_mean_1h: float,
    ):
        self.intersection_id = intersection_id
        self.hour = hour
        self.minute = minute
        self.day_of_week = day_of_week  # 0 = Monday ... 6 = Sunday
        self.is_holiday = is_holiday
        self.event_nearby = event_nearby
        self.temperature_c = temperature_c
        self.rain_mm_15min = rain_mm_15min
        self.vehicle_count = vehicle_count
        self.avg_speed_kmh = avg_speed_kmh
        self.vehicle_density_per_km = vehicle_density_per_km
        self.avg_waiting_time_sec = avg_waiting_time_sec
        self.queue_length_veh = queue_length_veh
        self.incident_flag = incident_flag
        self.prev_count_15min = prev_count_15min
        self.prev_count_1h = prev_count_1h
        self.count_same_time_yesterday = count_same_time_yesterday
        self.count_same_time_last_week = count_same_time_last_week
        self.rolling_mean_1h = rolling_mean_1h

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            data = dict(self.__dict__)
            # Derived fields (not asked from the user)
            data["is_weekend"] = int(self.day_of_week >= 5)
            data["road_capacity_veh_15min"] = ROAD_CAPACITY[self.intersection_id]
            return pd.DataFrame([data])
        except Exception as e:
            raise CustomException(e, sys)