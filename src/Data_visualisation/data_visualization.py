"""
Data visualisation for the urban traffic dataset (uses the CLEANED data).

Produces 12 charts and saves them as PNG files in ./visualizations/

  01 Congestion class balance (overall + per intersection)
  02 Hourly traffic profile: weekday vs weekend, per intersection
  03 Utilisation heatmap: hour of day x day of week
  04 One-week time series per intersection
  05 Speed vs road utilisation (the speed-flow relationship)
  06 Rain effect on speed and waiting time
  07 Incident impact on speed, waiting time, queue
  08 Stadium events: evening traffic with vs without an event
  09 Correlation heatmap of key numeric features
  10 Autocorrelation of vehicle count (persistence + daily cycle)
  11 Share of congestion levels by hour of day
  12 Daily totals over time with 7-day rolling mean

Usage
  python data_visualization.py
  python data_visualization.py --input urban_traffic_cleaned.csv --outdir visualizations
  python data_visualization.py --show          # also open the plots on screen
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LEVELS = ["Low", "Moderate", "High", "Severe"]
LEVEL_COLORS = {"Low": "#4CAF50", "Moderate": "#FFC107", "High": "#FF7043", "Severe": "#C62828"}
SITE_COLORS = {"CBD": "#1f77b4", "Arterial": "#d62728", "Residential": "#2ca02c", "Stadium": "#9467bd"}
SITES = list(SITE_COLORS)

plt.rcParams.update({
    "figure.dpi": 100, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})


# --------------------------------------------------------------------------- #
def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df["site"] = df["intersection_id"].str[7:]                     # CBD, Arterial, ...
    df["utilisation"] = df["vehicle_count"] / df["road_capacity_veh_15min"]
    df["date"] = df["timestamp"].dt.date
    return df


# --------------------------------------------------------------------------- #
# 01
def plot_congestion_distribution(df):
    d = df.dropna(subset=["target_congestion"])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    share = d["target_congestion"].value_counts(normalize=True).reindex(LEVELS) * 100
    bars = ax[0].bar(LEVELS, share.values, color=[LEVEL_COLORS[l] for l in LEVELS])
    for b, v in zip(bars, share.values):
        ax[0].text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontsize=10)
    ax[0].set_title("Congestion level (next 15 min): overall")
    ax[0].set_ylabel("% of intervals"); ax[0].set_ylim(0, share.max() + 10)

    ct = pd.crosstab(d["site"], d["target_congestion"], normalize="index").reindex(SITES)[LEVELS] * 100
    ct.plot(kind="bar", stacked=True, ax=ax[1], color=[LEVEL_COLORS[l] for l in LEVELS], width=0.7)
    ax[1].set_title("Congestion mix per intersection"); ax[1].set_ylabel("% of intervals")
    ax[1].set_xlabel(""); ax[1].tick_params(axis="x", rotation=0); ax[1].legend(title="", ncol=4, loc="upper center",
                                                                              bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    return fig


# 02
def plot_hourly_profile(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, site in zip(axes.ravel(), SITES):
        d = df[df["site"] == site]
        prof = d.groupby(["hour", "is_weekend"])["vehicle_count"].mean().unstack()
        ax.plot(prof.index, prof[0], lw=2.4, color=SITE_COLORS[site], label="Weekday")
        ax.plot(prof.index, prof[1], lw=2.4, color="gray", ls="--", label="Weekend")
        ax.fill_between(prof.index, prof[0], alpha=0.12, color=SITE_COLORS[site])
        ax.set_title(site); ax.set_ylabel("Vehicles / 15 min"); ax.legend(frameon=False)
        ax.set_xticks(range(0, 24, 3))
    for ax in axes[1]:
        ax.set_xlabel("Hour of day")
    fig.suptitle("Average hourly traffic: weekday vs weekend", fontweight="bold", y=1.0)
    fig.tight_layout()
    return fig


# 03
def plot_hour_day_heatmap(df):
    piv = df.pivot_table(index="day_of_week", columns="hour", values="utilisation", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 3.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7)); ax.set_yticklabels(DAY_NAMES)
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24))
    ax.set_xlabel("Hour of day"); ax.grid(False)
    ax.set_title("Road utilisation (vehicles / capacity) by day and hour")
    fig.colorbar(im, ax=ax, label="Mean utilisation", pad=0.02)
    fig.tight_layout()
    return fig


# 04
def plot_week_timeseries(df, week_start="2025-01-13"):
    start = pd.Timestamp(week_start)
    w = df[(df["timestamp"] >= start) & (df["timestamp"] < start + pd.Timedelta(days=7))]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, site in zip(axes, SITES):
        d = w[w["site"] == site]
        ax.plot(d["timestamp"], d["vehicle_count"], color=SITE_COLORS[site], lw=1.3)
        ax.set_ylabel(site); ax.margins(x=0)
        for day in range(7):
            if (start + pd.Timedelta(days=day)).dayofweek >= 5:
                ax.axvspan(start + pd.Timedelta(days=day), start + pd.Timedelta(days=day + 1),
                           color="gray", alpha=0.12)
    axes[0].set_title(f"One week of traffic (shaded = weekend), week of {week_start}")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


# 05
def plot_speed_vs_load(df, sample=12000):
    d = df.sample(min(sample, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for site in SITES:
        s = d[d["site"] == site]
        ax.scatter(s["utilisation"], s["avg_speed_kmh"], s=6, alpha=0.25, color=SITE_COLORS[site], label=site)
    bins = np.linspace(0, 1.05, 22)
    mid = (bins[:-1] + bins[1:]) / 2
    mean = df.groupby(pd.cut(df["utilisation"], bins), observed=True)["avg_speed_kmh"].mean()
    ax.plot(mid[:len(mean)], mean.values, color="black", lw=2.5, label="Overall mean")
    ax.set_xlabel("Utilisation (vehicles / capacity)"); ax.set_ylabel("Average speed (km/h)")
    ax.set_title("Speed falls as the road fills up")
    ax.legend(markerscale=3, frameon=False)
    fig.tight_layout()
    return fig


# 06
def plot_weather_effect(df):
    bins = [-0.01, 0.0, 2.0, 4.0, np.inf]
    labels = ["Dry", "Light (<2 mm)", "Moderate (2-4 mm)", "Heavy (>4 mm)"]
    d = df.assign(rain=pd.cut(df["rain_mm_15min"], bins, labels=labels))
    g = d.groupby("rain", observed=True)[["avg_speed_kmh", "avg_waiting_time_sec"]].mean()
    n = d["rain"].value_counts().reindex(g.index)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#90caf9", "#64b5f6", "#2196f3", "#0d47a1"][:len(g)]
    for a, col, title, unit in zip(ax, g.columns, ["Average speed", "Average waiting time"], ["km/h", "seconds"]):
        bars = a.bar(range(len(g)), g[col].values, color=colors)
        a.set_xticks(range(len(g))); a.set_xticklabels([f"{i}\n(n={n[i]:,})" for i in g.index], fontsize=8)
        a.set_title(f"{title} by rainfall"); a.set_ylabel(unit)
        for b, v in zip(bars, g[col].values):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
        a.set_ylim(0, g[col].max() * 1.15)
    fig.tight_layout()
    return fig


# 07
def plot_incident_impact(df):
    cols = [("avg_speed_kmh", "Average speed (km/h)"), ("avg_waiting_time_sec", "Waiting time (sec)"),
            ("queue_length_veh", "Queue length (vehicles)")]
    g = df.groupby("incident_flag")[[c for c, _ in cols]].mean()
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, (c, title) in zip(ax, cols):
        bars = a.bar(["No incident", "Incident"], g[c].values, color=["#78909c", "#e53935"], width=0.55)
        for b, v in zip(bars, g[c].values):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom")
        a.set_title(title); a.set_ylim(0, g[c].max() * 1.2)
    fig.suptitle("Incident impact (all intersections)", fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# 08
def plot_stadium_event(df):
    s = df[df["site"] == "Stadium"]
    hours = range(15, 24)
    prof = s[s["hour"].isin(hours)].groupby(["hour", "event_nearby"])["vehicle_count"].mean().unstack()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(prof.index, prof[0], marker="o", lw=2.4, color="gray", label="No event")
    ax.plot(prof.index, prof[1], marker="o", lw=2.4, color="#9467bd", label="Event nearby")
    ax.fill_between(prof.index, prof[0], prof[1], where=prof[1] > prof[0], color="#9467bd", alpha=0.15)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Vehicles / 15 min")
    ax.set_title("Stadium intersection: evening traffic with and without an event")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


# 09
def plot_correlation(df):
    cols = ["vehicle_count", "avg_speed_kmh", "vehicle_density_per_km", "avg_waiting_time_sec",
            "queue_length_veh", "rain_mm_15min", "temperature_c", "incident_flag", "prev_count_15min",
            "count_same_time_yesterday", "target_next_count"]
    c = df[cols].corr()
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols); ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(c.values[i, j]) > 0.6 else "black")
    ax.set_title("Correlation between key features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return fig


# 10
def plot_autocorrelation(df, max_lag=2 * 96 + 8):
    s = df[df["site"] == "CBD"].set_index("timestamp")["vehicle_count"]
    lags = np.arange(1, max_lag + 1)
    acf = np.array([s.autocorr(int(l)) for l in lags])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.vlines(lags, 0, acf, color="#1f77b4", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    for day in (1, 2):
        ax.axvline(day * 96, color="red", ls="--", lw=1)
        ax.text(day * 96 + 2, 0.92, f"{day} day", color="red", fontsize=9)
    ax.set_xlabel("Lag (15-minute steps)"); ax.set_ylabel("Autocorrelation")
    ax.set_title("Vehicle count is highly persistent and repeats daily (CBD)")
    fig.tight_layout()
    return fig


# 11
def plot_congestion_by_hour(df):
    d = df.dropna(subset=["target_congestion"])
    ct = pd.crosstab(d["hour"], d["target_congestion"], normalize="index").reindex(columns=LEVELS) * 100
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.stackplot(ct.index, [ct[l] for l in LEVELS], labels=LEVELS, colors=[LEVEL_COLORS[l] for l in LEVELS])
    ax.set_xlim(0, 23); ax.set_ylim(0, 100); ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day"); ax.set_ylabel("% of intervals")
    ax.set_title("When does congestion happen? Share of each level by hour")
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.tight_layout()
    return fig


# 12
def plot_daily_totals(df):
    daily = df.groupby(["date", "site"])["vehicle_count"].sum().unstack()[SITES]
    daily.index = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for site in SITES:
        ax.plot(daily.index, daily[site], color=SITE_COLORS[site], alpha=0.25, lw=1)
        ax.plot(daily.index, daily[site].rolling(7, center=True).mean(), color=SITE_COLORS[site], lw=2.4, label=site)
    ax.set_ylabel("Vehicles per day"); ax.set_title("Daily traffic volume (thin = daily, thick = 7-day average)")
    ax.legend(ncol=4, frameon=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
CHARTS = [
    ("01_congestion_distribution", plot_congestion_distribution),
    ("02_hourly_profile", plot_hourly_profile),
    ("03_hour_day_heatmap", plot_hour_day_heatmap),
    ("04_week_timeseries", plot_week_timeseries),
    ("05_speed_vs_load", plot_speed_vs_load),
    ("06_weather_effect", plot_weather_effect),
    ("07_incident_impact", plot_incident_impact),
    ("08_stadium_event", plot_stadium_event),
    ("09_correlation", plot_correlation),
    ("10_autocorrelation", plot_autocorrelation),
    ("11_congestion_by_hour", plot_congestion_by_hour),
    ("12_daily_totals", plot_daily_totals),
]


def main(input_path, outdir, show=False):
    if not show:
        plt.switch_backend("Agg")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = load_data(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    for name, fn in CHARTS:
        fig = fn(df)
        fig.savefig(outdir / f"{name}.png", dpi=130, bbox_inches="tight")
        print(f"  saved {name}.png")
        if not show:
            plt.close(fig)
    print(f"\nAll charts saved in {outdir.resolve()}")
    if show:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Visualise the urban traffic dataset")
    ap.add_argument("--input", default="urban_traffic_cleaned.csv")
    ap.add_argument("--outdir", default="visualizations")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    main(a.input, a.outdir, a.show)
