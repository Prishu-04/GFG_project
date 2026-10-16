"""
Reference models to beat. Run:  python -m src.deep_learning.baselines

  persistence      : "next 15 min = right now"                      (the naive forecast)
  hist_gb_tabular  : sklearn HistGradientBoosting on the flattened window (strong classical model)
"""
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .data import prepare
from .results_io import save_results


def run(seq_len: int = C.SEQ_LEN, verbose: bool = True):
    splits, scaler, _ = prepare(seq_len)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    out = {}

    # 1) persistence: util_now of the last step, un-scaled back to utilisation
    pers = te.X[:, -1, 0] * scaler["scale"][0] + scaler["mean"][0]
    out["persistence"] = save_results("persistence", te, pers)

    # 2) gradient boosting on the flattened window (train on train, early-stop on val)
    t0 = time.time()
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                                       early_stopping=False, random_state=C.SEED)
    gb.fit(tr.X.reshape(len(tr.X), -1), tr.y)
    pred = gb.predict(te.X.reshape(len(te.X), -1))
    out["hist_gb_tabular"] = save_results("hist_gb_tabular", te, pred,
                                          extra={"train_seconds": round(time.time() - t0, 1)})
    if verbose:
        for k, v in out.items():
            m = v["metrics"]
            print(f"{k:18s} MAE={m['MAE_vehicles']:.2f} RMSE={m['RMSE_vehicles']:.2f} "
                  f"R2={m['R2']:.4f} class_acc={m['class_accuracy']:.4f} macroF1={m['class_macro_F1']:.4f}")
    return out


if __name__ == "__main__":
    run()
