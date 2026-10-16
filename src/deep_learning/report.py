"""
Build the model-comparison table and charts from whatever has been run so far.

  python -m src.deep_learning.report

Writes to reports/deep_learning/ : model_comparison.csv / .md and PNG charts (in reports/figures/deep_learning/).
Works with any subset of models (baselines only, or baselines + mlp/lstm/gru/transformer).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

ORDER = ["persistence", "hist_gb_tabular", "mlp", "lstm", "gru", "transformer"]
FIG_DIR = C.PROJECT_ROOT / "reports" / "figures" / "deep_learning"


def load_all() -> pd.DataFrame:
    rows = []
    for path in C.REPORT_DIR.glob("*_metrics.json"):
        d = json.loads(path.read_text())
        if d["model"].endswith("_smoke"):
            continue
        m = d["metrics"]
        rows.append({"model": d["model"], "MAE (vehicles)": m["MAE_vehicles"], "RMSE (vehicles)": m["RMSE_vehicles"],
                     "R2": m["R2"], "class accuracy": m["class_accuracy"], "class macro-F1": m["class_macro_F1"],
                     "within 1 level": m["class_within_one_level"], "parameters": d.get("n_parameters"),
                     "train seconds": d.get("train_seconds")})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No results yet - run baselines.py and/or train.py first.")
    df["_o"] = df["model"].map({n: i for i, n in enumerate(ORDER)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def plot_comparison(df):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#9e9e9e" if m in ("persistence", "hist_gb_tabular") else "#1f77b4" for m in df["model"]]
    ax[0].bar(df["model"], df["MAE (vehicles)"], color=colors)
    ax[0].set_title("Test MAE (vehicles per 15 min) - lower is better")
    ax[1].bar(df["model"], df["class accuracy"] * 100, color=colors)
    ax[1].set_title("Congestion class accuracy (%) - higher is better")
    ax[1].set_ylim(max(0, df["class accuracy"].min() * 100 - 10), 100)
    for a in ax:
        a.tick_params(axis="x", rotation=25)
        a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=130)
    plt.close(fig)


def plot_forecast(df, site="INT_01_CBD", days=3):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    first = True
    for model in df["model"]:
        p = pd.read_csv(C.PRED_DIR / f"{model}.csv", parse_dates=["timestamp"])
        p = p[p["intersection_id"] == site].sort_values("timestamp")
        p = p[p["timestamp"] < p["timestamp"].min() + pd.Timedelta(days=days)]
        if first:
            ax.plot(p["timestamp"], p["true_count"], color="black", lw=2, label="actual")
            first = False
        ax.plot(p["timestamp"], p["pred_count"], lw=1.2, alpha=0.85, label=model)
    ax.set_title(f"Next-15-min vehicle count - first {days} test days at {site}")
    ax.set_ylabel("vehicles / 15 min")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "forecast_vs_actual.png", dpi=130)
    plt.close(fig)


def plot_confusion(name):
    d = json.loads((C.REPORT_DIR / f"{name}_metrics.json").read_text())
    cm = np.array(d["metrics"]["confusion_matrix"], dtype=float)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xticks(range(4), C.CLASS_NAMES)
    ax.set_yticks(range(4), C.CLASS_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Congestion confusion matrix - {name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"confusion_{name}.png", dpi=130)
    plt.close(fig)


def plot_training_curves():
    hist = {p.stem.replace("_history", ""): pd.read_csv(p) for p in C.REPORT_DIR.glob("*_history.csv")
            if "_smoke" not in p.stem}
    if not hist:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hist.items():
        ax[0].plot(h["epoch"], h["train_loss"], label=name)
        ax[1].plot(h["epoch"], h["val_MAE_vehicles"], label=name)
    ax[0].set_title("Training loss (Huber, utilisation scale)")
    ax[1].set_title("Validation MAE (vehicles)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_all()
    df.to_csv(C.REPORT_DIR / "model_comparison.csv", index=False)
    try:                                              # to_markdown needs the optional `tabulate` package
        table = df.round(4).to_markdown(index=False)
    except ImportError:
        table = df.round(4).to_string(index=False)
    (C.REPORT_DIR / "model_comparison.md").write_text(table)
    plot_comparison(df)
    plot_forecast(df)
    best = df.loc[df["MAE (vehicles)"].idxmin(), "model"]
    plot_confusion(best)
    plot_training_curves()
    print(df.round(4).to_string(index=False))
    print(f"\nBest by test MAE: {best}\nCharts saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
