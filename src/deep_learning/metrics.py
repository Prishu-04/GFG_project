"""Regression (vehicle count) + congestion-class metrics. Pure numpy/sklearn."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)

from . import config as C
from .data import util_to_class


def evaluate(y_util_true, y_util_pred, capacity) -> dict:
    """All numbers are on the VEHICLE-COUNT scale (utilisation x capacity) unless stated."""
    y_util_pred = np.clip(y_util_pred, 0.0, 1.25)
    true_cnt, pred_cnt = y_util_true * capacity, y_util_pred * capacity
    cls_true, cls_pred = util_to_class(y_util_true), util_to_class(y_util_pred)
    labels = list(range(len(C.CLASS_NAMES)))
    return {
        "MAE_vehicles": float(mean_absolute_error(true_cnt, pred_cnt)),
        "RMSE_vehicles": float(np.sqrt(mean_squared_error(true_cnt, pred_cnt))),
        "R2": float(r2_score(true_cnt, pred_cnt)),
        "MAE_utilisation": float(mean_absolute_error(y_util_true, y_util_pred)),
        "class_accuracy": float(accuracy_score(cls_true, cls_pred)),
        "class_macro_F1": float(f1_score(cls_true, cls_pred, average="macro", labels=labels, zero_division=0)),
        "class_within_one_level": float(np.mean(np.abs(cls_true - cls_pred) <= 1)),
        "confusion_matrix": confusion_matrix(cls_true, cls_pred, labels=labels).tolist(),
        "n_test_windows": int(len(y_util_true)),
    }
