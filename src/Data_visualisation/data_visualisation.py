"""
data_visualisation.py
----------------------
Complete, script form of visualization.ipynb: feature-exploration plots for
the Dynamic Urban Traffic Prediction dataset, aimed at answering one
question -- which features actually matter for predicting `Vehicle Count`?

Run directly:

    python "src/Data_visualisation/data_visualisation.py"

Each function saves its chart to `src/Data_visualisation/plots/` (created
automatically) as a PNG, and also returns any computed tables (e.g. the
correlation / mutual-information series) so this module can be imported
and reused, e.g. from a report-generation script.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe default for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# --- make src/logger.py and src/exception.py importable ---------------------
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from logger import logging              # noqa: E402
from exception import CustomException    # noqa: E402

PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "traffic_dataset_clean.csv")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved plot: {path}")
    return path


def load_clean_data(path: str = DATA_PATH) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df["Time"] = pd.to_timedelta(df["Timestamp"].astype(str))
        logging.info(f"Loaded clean dataset with shape {df.shape}")
        return df
    except Exception as e:
        raise CustomException(e, sys)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """Numeric feature columns, excluding the helper 'Time' timedelta column."""
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in cols if c != "Time"]


def plot_target_overview(df: pd.DataFrame):
    """Section 1: Vehicle Count over time + its distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(df["Time"].dt.total_seconds() / 60, df["Vehicle Count"], linewidth=0.8)
    axes[0].set_title("Vehicle Count Over Time (minute-level)")
    axes[0].set_xlabel("Minutes since start of recording")
    axes[0].set_ylabel("Vehicle Count")

    sns.histplot(df["Vehicle Count"], bins=30, kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title("Distribution of Vehicle Count")
    axes[1].set_xlabel("Vehicle Count")

    fig.tight_layout()
    return _save(fig, "01_target_overview.png")


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    """Section 2: full correlation heatmap + sorted correlation-with-target bar chart."""
    corr = df[numeric_cols].corr()

    fig1 = plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Heatmap — Numeric Features")
    plt.tight_layout()
    path1 = _save(fig1, "02_correlation_heatmap.png")

    target_corr = corr["Vehicle Count"].drop("Vehicle Count").sort_values(key=np.abs, ascending=False)

    fig2 = plt.figure(figsize=(8, 6))
    sns.barplot(x=target_corr.values, y=target_corr.index,
                hue=target_corr.index, palette="vlag", legend=False)
    plt.title("Feature Correlation with Vehicle Count (sorted by strength)")
    plt.xlabel("Correlation coefficient")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    path2 = _save(fig2, "03_correlation_with_target.png")

    logging.info(f"Top correlated features:\n{target_corr}")
    return corr, target_corr, (path1, path2)


def plot_feature_distributions(df: pd.DataFrame):
    """Section 3: grouped distribution plots by feature family."""
    paths = []

    speed_flow_cols = ["Avg Speed (km/h)", "FreeFlowSpeed (km/h)", "Speed Factor"]
    fig, axes = plt.subplots(1, len(speed_flow_cols), figsize=(16, 4))
    for ax, col in zip(axes, speed_flow_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="seagreen")
        ax.set_title(col)
    fig.suptitle("Speed-Related Features", y=1.03)
    fig.tight_layout()
    paths.append(_save(fig, "04_speed_features.png"))

    density_flow_cols = ["Vehicle Density (%)", "Saturation Flow Rate(veh/hr/lane)",
                          "Volume to  Saturation Lane Traffic ratio(%)"]
    fig, axes = plt.subplots(1, len(density_flow_cols), figsize=(16, 4))
    for ax, col in zip(axes, density_flow_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="darkorange")
        ax.set_title(col)
    fig.suptitle("Density / Flow-Ratio Features", y=1.03)
    fig.tight_layout()
    paths.append(_save(fig, "05_density_flow_features.png"))

    index_cols = ["TSR", "VLSR", "CI"]
    fig, axes = plt.subplots(1, len(index_cols), figsize=(16, 4))
    for ax, col in zip(axes, index_cols):
        sns.histplot(df[col], bins=20, kde=True, ax=ax, color="mediumpurple")
        ax.set_title(col)
    fig.suptitle("Composite Index Features (TSR / VLSR / CI)", y=1.03)
    fig.tight_layout()
    paths.append(_save(fig, "06_index_features.png"))

    fig = plt.figure(figsize=(8, 5))
    sns.histplot(df["Previous_Traffic"], bins=20, kde=True, color="crimson")
    plt.title("Distribution of Previous_Traffic (lag feature)")
    plt.xlabel("Previous Traffic")
    plt.tight_layout()
    paths.append(_save(fig, "07_previous_traffic.png"))

    return paths


def plot_hourly_patterns(df: pd.DataFrame):
    """Section 4: Vehicle Count by hour (rush-hour pattern) + row coverage per hour."""
    fig = plt.figure(figsize=(12, 5))
    sns.boxplot(data=df, x="Hour", y="Vehicle Count", hue="Hour", palette="crest", legend=False)
    plt.title("Vehicle Count by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Vehicle Count")
    plt.tight_layout()
    path1 = _save(fig, "08_vehicle_count_by_hour.png")

    hourly_counts = df["Hour"].value_counts().sort_index()
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, color="slateblue")
    plt.title("Number of Readings per Hour (row coverage, not traffic volume)")
    plt.xlabel("Hour")
    plt.ylabel("Row Count")
    plt.tight_layout()
    path2 = _save(fig, "09_row_coverage_by_hour.png")

    return path1, path2


def plot_congestion_level_analysis(df: pd.DataFrame):
    """Section 5: Vehicle Count vs Congestion Level + class balance."""
    order = df.groupby("Congestion Level")["Vehicle Count"].median().sort_values().index

    fig = plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Congestion Level", y="Vehicle Count", order=order,
                hue="Congestion Level", palette="flare", legend=False)
    plt.title("Vehicle Count by Congestion Level")
    plt.tight_layout()
    path1 = _save(fig, "10_vehicle_count_by_congestion.png")

    counts = df["Congestion Level"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="flare", legend=False, ax=axes[0])
    axes[0].set_title("Congestion Level — Row Counts")
    axes[0].set_ylabel("Frequency")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")

    axes[1].pie(counts.values, labels=[f"{idx} ({p}%)" for idx, p in zip(counts.index, pct)],
                colors=sns.color_palette("flare", len(counts)))
    axes[1].set_title("Congestion Level — Class Balance")
    fig.tight_layout()
    path2 = _save(fig, "11_congestion_class_balance.png")

    logging.info(f"Congestion Level class balance (%):\n{pct}")
    return counts, pct, (path1, path2)


def plot_mutual_information(df: pd.DataFrame, numeric_cols: list):
    """Section 6: non-linear feature-importance proxy via mutual information."""
    feature_cols = [c for c in numeric_cols if c != "Vehicle Count"]
    X = df[feature_cols].fillna(0)
    y = df["Vehicle Count"]

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

    fig = plt.figure(figsize=(8, 6))
    sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, palette="mako", legend=False)
    plt.title("Mutual Information with Vehicle Count")
    plt.xlabel("Mutual Information Score")
    plt.tight_layout()
    path = _save(fig, "12_mutual_information.png")

    logging.info(f"Mutual information scores:\n{mi_series}")
    return mi_series, path


def plot_top_feature_pairplot(df: pd.DataFrame, target_corr: pd.Series, top_n: int = 4):
    """Section 7: pairwise scatter of the top-|correlation| features vs. target."""
    top_features = target_corr.abs().sort_values(ascending=False).head(top_n).index.tolist()
    plot_cols = top_features + ["Vehicle Count"]

    grid = sns.pairplot(df[plot_cols], corner=True, plot_kws={"alpha": 0.4, "s": 12})
    grid.fig.suptitle("Pairwise Relationships — Top Correlated Features vs. Vehicle Count", y=1.02)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, "13_top_feature_pairplot.png")
    grid.fig.savefig(path, bbox_inches="tight")
    plt.close(grid.fig)
    logging.info(f"Saved plot: {path}")
    return top_features, path


def run_all():
    """Run every plotting step end-to-end and print a short feature-selection summary."""
    try:
        df = load_clean_data()
        numeric_cols = get_numeric_columns(df)

        plot_target_overview(df)
        corr, target_corr, _ = plot_correlation_heatmap(df, numeric_cols)
        plot_feature_distributions(df)
        plot_hourly_patterns(df)
        counts, pct, _ = plot_congestion_level_analysis(df)
        mi_series, _ = plot_mutual_information(df, numeric_cols)
        top_features, _ = plot_top_feature_pairplot(df, target_corr)

        print(f"All plots saved to: {PLOTS_DIR}")
        print("\nTop features by |correlation| with Vehicle Count:")
        print(target_corr.abs().sort_values(ascending=False).head(6))
        print("\nTop features by mutual information with Vehicle Count:")
        print(mi_series.head(6))
        print("\nCongestion Level class balance (%):")
        print(pct)

        return {
            "correlation": corr,
            "target_correlation": target_corr,
            "mutual_information": mi_series,
            "congestion_class_balance": pct,
            "top_features": top_features,
        }
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    run_all()
