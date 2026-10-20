"""
Classical machine-learning models on the SAME windows, split and metrics as the neural networks, so the
comparison table is fair. Run:  python -m src.deep_learning.classical

  linear_ridge   : ridge regression on the flattened window (simple linear baseline)
  knn            : k-nearest neighbours (k=15, distance weighted)
  decision_tree  : single regression tree
  random_forest  : bagged trees
(persistence and hist_gb_tabular come from baselines.py)

Like the neural nets, each model predicts the next-15-minute utilisation; the congestion class is then
derived from it with the data-derived cut-offs. The label is never an input, so there is no leakage.
"""
import argparse
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from . import config as C
from .data import prepare
from .results_io import save_results


def make_models(fast: bool = False):
    return {
        "linear_ridge": Ridge(alpha=1.0),
        "knn": KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1),
        "decision_tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=10, random_state=C.SEED),
        "random_forest": RandomForestRegressor(n_estimators=40 if fast else 150, max_features=0.3,
                                               min_samples_leaf=3, n_jobs=-1, random_state=C.SEED),
    }


def run(seq_len: int = C.SEQ_LEN, fast: bool = False, verbose: bool = True):
    splits, _, _ = prepare(seq_len)
    tr, te = splits["train"], splits["test"]
    Xtr, Xte = tr.X.reshape(len(tr.X), -1), te.X.reshape(len(te.X), -1)
    out = {}
    for name, model in make_models(fast).items():
        t0 = time.time()
        model.fit(Xtr, tr.y)
        pred = model.predict(Xte)
        out[name] = save_results(name, te, pred, extra={"train_seconds": round(time.time() - t0, 1)})
        if verbose:
            m = out[name]["metrics"]
            print(f"{name:15s} MAE={m['MAE_vehicles']:.2f} RMSE={m['RMSE_vehicles']:.2f} R2={m['R2']:.4f} "
                  f"class_acc={m['class_accuracy']:.4f} macroF1={m['class_macro_F1']:.4f}  ({time.time() - t0:.0f}s)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="fewer trees in the random forest")
    run(fast=ap.parse_args().fast)
