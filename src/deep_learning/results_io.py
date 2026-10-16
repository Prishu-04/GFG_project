"""Save/load experiment outputs in one common format (used by baselines AND the neural nets)."""
import json

import numpy as np
import pandas as pd

from . import config as C
from .data import SplitData
from .metrics import evaluate


def save_results(name: str, test: SplitData, y_pred_util: np.ndarray, extra: dict | None = None,
                 history: list | None = None) -> dict:
    C.PRED_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred_util = np.clip(np.asarray(y_pred_util, dtype=np.float64), 0.0, 1.25)

    metrics = evaluate(test.y, y_pred_util, test.capacity)
    payload = {"model": name, "metrics": metrics, **(extra or {})}
    (C.REPORT_DIR / f"{name}_metrics.json").write_text(json.dumps(payload, indent=2))

    pd.DataFrame({
        "timestamp": test.timestamp, "intersection_id": test.site, "capacity": test.capacity,
        "true_util": test.y, "pred_util": y_pred_util,
        "true_count": test.y * test.capacity, "pred_count": y_pred_util * test.capacity,
    }).to_csv(C.PRED_DIR / f"{name}.csv", index=False)

    if history:
        pd.DataFrame(history).to_csv(C.REPORT_DIR / f"{name}_history.csv", index=False)
    return payload
