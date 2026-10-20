"""Builds the data shown on the Results page.

Everything is read from files the project already produces (reports/ and data/processed/),
so the page updates itself when the team re-runs the training or visualisation scripts.
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_ROOT = os.path.join(ROOT, "reports", "figures")
REPORT_DIR = os.path.join(ROOT, "reports", "deep_learning")
DATA_CSV = os.path.join(ROOT, "data", "processed", "urban_traffic_model_ready.csv")

LEVELS = ["Low", "Moderate", "High", "Severe"]

MODEL_BLURB = {
    "persistence": ("Persistence baseline", "Baseline", "Predicts that the next 15 minutes will look exactly like the last 15 minutes. No learning involved."),
    "linear_ridge": ("Ridge regression", "Classical ML", "Linear model on the flattened 4-hour window."),
    "knn": ("k-nearest neighbours", "Classical ML", "Averages the 15 most similar past windows."),
    "decision_tree": ("Decision tree", "Classical ML", "A single regression tree of depth 12."),
    "random_forest": ("Random forest", "Classical ML", "150 bagged regression trees."),
    "hist_gb_tabular": ("Gradient boosting", "Classical ML", "HistGradientBoosting on the flattened window (lags, calendar, weather, sensors)."),
    "mlp": ("MLP", "Deep learning", "Small feed-forward neural network on the flattened window."),
    "lstm": ("LSTM", "Deep learning", "Recurrent network that reads the 4-hour window step by step."),
    "gru": ("GRU", "Deep learning", "Lighter recurrent network on the 4-hour window."),
    "transformer": ("Transformer", "Transformer", "Self-attention encoder over the 16 time steps."),
}
ORDER = ["persistence", "linear_ridge", "knn", "decision_tree", "random_forest", "hist_gb_tabular",
         "mlp", "lstm", "gru", "transformer"]
FAMILY_ORDER = ["Baseline", "Classical ML", "Deep learning", "Transformer"]

# (file, group, title, what it shows)
FIGURES = [
    ("eda/01_congestion_distribution.png", "eda", "Congestion balance", "How often each congestion level occurs overall and at each of the four intersections."),
    ("eda/02_hourly_profile.png", "eda", "Hourly profile", "Average vehicles per 15 minutes by hour, weekdays against weekends, one panel per intersection."),
    ("eda/03_hour_day_heatmap.png", "eda", "Hour by day heatmap", "Road utilisation (vehicles divided by capacity) for every hour of every weekday."),
    ("eda/04_week_timeseries.png", "eda", "One week of traffic", "A single week of raw counts with weekends shaded."),
    ("eda/05_speed_vs_load.png", "eda", "Speed versus load", "Average speed against road utilisation. Speed drops as the road fills."),
    ("eda/06_weather_effect.png", "eda", "Weather effect", "Speed and volume grouped by rainfall."),
    ("eda/07_incident_impact.png", "eda", "Incident impact", "Average speed, waiting time and queue length with and without an incident."),
    ("eda/08_stadium_event.png", "eda", "Stadium event", "Evening traffic at the stadium intersection with and without an event nearby."),
    ("eda/09_correlation.png", "eda", "Correlations", "How strongly the key features move together."),
    ("eda/10_autocorrelation.png", "eda", "Autocorrelation", "How much today's count resembles the count 15 minutes, a day and a week earlier."),
    ("eda/11_congestion_by_hour.png", "eda", "Congestion by hour", "Share of each congestion level in every hour of the day."),
    ("eda/12_daily_totals.png", "eda", "Daily totals", "Vehicles per day, with a 7-day average."),
    ("deep_learning/model_comparison.png", "model", "Model comparison", "Test error, class accuracy and macro-F1 for every model that has been run."),
    ("deep_learning/forecast_vs_actual.png", "model", "Forecast versus actual", "Predicted and observed counts over the first days of the test period, for the best model of each family."),
    ("deep_learning/training_curves.png", "model", "Training curves", "Training loss and validation error per epoch for the neural networks."),
]

_cache = {}


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def figures():
    """Only figures that exist on disk are listed. Confusion matrices are picked up for whichever models have one."""
    items = list(FIGURES)
    for p in sorted(glob.glob(os.path.join(FIG_ROOT, "deep_learning", "confusion_*.png"))):
        name = os.path.basename(p)[len("confusion_"):-len(".png")]
        title = MODEL_BLURB.get(name, (name,))[0]
        items.append((f"deep_learning/{os.path.basename(p)}", "model", f"Confusion matrix: {title}",
                      f"Where {title} gets the congestion level right or wrong."))
    out = []
    for rel, group, title, text in items:
        if os.path.isfile(os.path.join(FIG_ROOT, rel)):
            out.append({"id": os.path.splitext(os.path.basename(rel))[0], "path": rel,
                        "group": group, "title": title, "text": text})
    return out


def _per_class_recall(cm):
    rec = []
    for i, row in enumerate(cm):
        s = sum(row)
        rec.append(row[i] / s if s else 0.0)
    return rec


def models():
    """Metrics for every model that has a *_metrics.json report."""
    rows = []
    for p in sorted(glob.glob(os.path.join(REPORT_DIR, "*_metrics.json"))):
        d = _load_json(p)
        if not d or "metrics" not in d:
            continue
        m = d["metrics"]
        name = d.get("model", os.path.basename(p).replace("_metrics.json", ""))
        if name.endswith("_smoke"):
            continue
        title, family, blurb = MODEL_BLURB.get(name, (name, "Other", ""))
        rows.append({
            "key": name, "title": title, "blurb": blurb, "family": family,
            "params": d.get("n_parameters"),
            "mae": m["MAE_vehicles"], "rmse": m["RMSE_vehicles"], "r2": m["R2"],
            "acc": m["class_accuracy"], "f1": m["class_macro_F1"],
            "within1": m["class_within_one_level"],
            "recall": _per_class_recall(m["confusion_matrix"]) if "confusion_matrix" in m else None,
            "n": m.get("n_test_windows"), "secs": d.get("train_seconds"),
        })
    rows.sort(key=lambda r: r["mae"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def ordered(rows):
    """Same models in the project's natural order: baseline, classical, deep learning, transformer."""
    pos = {n: i for i, n in enumerate(ORDER)}
    return sorted(rows, key=lambda r: pos.get(r["key"], 99))


def model_differences(rows):
    """Plain-language differences: best model against the persistence baseline, and between model families."""
    if len(rows) < 2:
        return None
    best = rows[0]
    base = next((r for r in rows if r["key"] == "persistence"), rows[-1])
    if best is base:
        return None
    pct = lambda a, b: (b - a) / b * 100
    notes = [
        f"{best['title']} is wrong by {best['mae']:.1f} vehicles per 15 minutes on average, against "
        f"{base['mae']:.1f} for the {base['title'].lower()}. That is {pct(best['mae'], base['mae']):.0f}% less error.",
        f"RMSE falls from {base['rmse']:.1f} to {best['rmse']:.1f} "
        f"({pct(best['rmse'], base['rmse']):.0f}% lower). RMSE punishes big misses harder than MAE, "
        f"so the gap shows the better model also avoids the large misses.",
        f"Congestion-class accuracy moves from {base['acc']*100:.1f}% to {best['acc']*100:.1f}% "
        f"({(best['acc']-base['acc'])*100:+.1f} points) and macro-F1 from {base['f1']:.3f} to {best['f1']:.3f}.",
    ]

    # family against family
    fam_best = {}
    for r in rows:
        if r["family"] not in fam_best:
            fam_best[r["family"]] = r
    cl, dl, tf = fam_best.get("Classical ML"), fam_best.get("Deep learning"), fam_best.get("Transformer")
    if cl and dl:
        gap = abs(cl["mae"] - dl["mae"])
        lead = cl if cl["mae"] < dl["mae"] else dl
        if gap < 0.5:
            verdict = (f"They are effectively tied: the gap is {gap:.2f} vehicles per 15 minutes, "
                       f"far smaller than the {base['mae'] - best['mae']:.1f} gained over persistence.")
        else:
            verdict = f"{lead['title']} is ahead by {gap:.2f} vehicles per 15 minutes."
        notes.append(f"Best classical model: {cl['title']} (MAE {cl['mae']:.2f}). Best deep-learning model: "
                     f"{dl['title']} (MAE {dl['mae']:.2f}). {verdict}")
    if tf:
        ref = min([r for r in (cl, dl) if r], key=lambda r: r["mae"]) if (cl or dl) else None
        if ref:
            d = tf["mae"] - ref["mae"]
            notes.append(f"The Transformer scores MAE {tf['mae']:.2f} and accuracy {tf['acc']*100:.1f}%, "
                         f"{'ahead of' if d < 0 else 'behind'} the best other model ({ref['title']}) by {abs(d):.2f} vehicles. "
                         f"With only 16 time steps of history and four sites, attention may have little to exploit beyond what simpler models already use.")
    ridge = next((r for r in rows if r["key"] == "linear_ridge"), None)
    if ridge and base["mae"] > best["mae"]:
        share = (base["mae"] - ridge["mae"]) / (base["mae"] - best["mae"]) * 100
        notes.append(f"A plain linear model (ridge) already captures {share:.0f}% of the improvement from persistence to the best model. "
                     f"This suggests the extra inputs (recent history, calendar, weather) matter more than model complexity.")
    knn = next((r for r in rows if r["key"] == "knn"), None)
    if knn and knn["mae"] > base["mae"]:
        notes.append(f"k-nearest neighbours is worse than persistence here (MAE {knn['mae']:.2f} against {base['mae']:.2f}), so it is not a good fit for this data.")
    timed = [r for r in rows if r["secs"]]
    if timed:
        fast, slow = min(timed, key=lambda r: r["secs"]), max(timed, key=lambda r: r["secs"])
        notes.append(f"Training cost differs a lot: {fast['title']} took {('under 1' if fast['secs'] < 1 else f"{fast['secs']:.0f}")} s and {slow['title']} took {slow['secs']:.0f} s on this machine. "
                     f"Accuracy differences between the top models are small compared with that.")
    notes.append(f"Both {base['title'].lower()} and {best['title']} are usually at most one congestion level off "
                 f"({base['within1']*100:.2f}% and {best['within1']*100:.2f}%), so the mistakes are mostly neighbouring classes.")
    if best["recall"] and base["recall"]:
        gains = [(LEVELS[i], best["recall"][i] - base["recall"][i]) for i in range(4)]
        lvl, g = max(gains, key=lambda x: x[1])
        notes.append(f"The biggest class-level gain over persistence is on {lvl}, where recall improves by {g*100:.1f} points. "
                     f"High and Severe are the levels a traffic operator acts on.")
    return {"best": best["title"], "base": base["title"], "base_row": base, "notes": notes}


def _stats():
    """A few numbers from the data used in the written comparisons. Empty dict if data is missing."""
    if "stats" in _cache:
        return _cache["stats"]
    stats = {}
    try:
        import pandas as pd
        df = pd.read_csv(DATA_CSV, usecols=["intersection_id", "hour", "event_nearby", "vehicle_count",
                                           "avg_speed_kmh", "avg_waiting_time_sec", "queue_length_veh",
                                           "incident_flag", "rain_mm_15min", "is_weekend", "target_congestion"])
        g = df.groupby("incident_flag")[["avg_speed_kmh", "avg_waiting_time_sec", "queue_length_veh"]].mean()
        stats["inc"] = {c: (float(g.loc[0, c]), float(g.loc[1, c])) for c in g.columns}
        s = df[(df["intersection_id"] == "INT_04_Stadium") & df["hour"].between(16, 19)]
        e = s.groupby("event_nearby")["vehicle_count"].mean()
        stats["event"] = (float(e.loc[0]), float(e.loc[1]))
        r = df.assign(wet=df["rain_mm_15min"] > 0).groupby("wet")[["avg_speed_kmh", "vehicle_count"]].mean()
        stats["rain"] = {c: (float(r.loc[False, c]), float(r.loc[True, c])) for c in r.columns}
        h = df.groupby(["is_weekend", "hour"])["vehicle_count"].mean().unstack(0)
        hs = df[df["target_congestion"].isin(["High", "Severe"])]["hour"]
        stats["mix"] = (float((df["target_congestion"] == "Low").mean()),
                        float(hs.between(7, 9).add(hs.between(15, 19)).mean()))
        stats["peak"] = {"weekday": int(h[0].idxmax()), "weekend": int(h[1].idxmax())}
    except Exception:
        stats = {}
    _cache["stats"] = stats
    return stats


def presets():
    """Curated side-by-side pairs, each with a written explanation of how they differ."""
    st = _stats()
    avail = {f["id"] for f in figures()}
    out = []

    def add(a, b, title, intro, diffs):
        if a in avail and b in avail:
            out.append({"a": a, "b": b, "title": title, "intro": intro, "diffs": diffs})

    inc, ev = st.get("inc"), st.get("event")
    d1 = ["The incident chart compares conditions on the road (speed, waiting, queues). The stadium chart compares volume by hour.",
          "An incident makes a road slower without adding cars. A stadium event adds cars at a particular time."]
    if inc:
        sp, wt, q = inc["avg_speed_kmh"], inc["avg_waiting_time_sec"], inc["queue_length_veh"]
        d1.append(f"With an incident, average speed drops from {sp[0]:.1f} to {sp[1]:.1f} km/h, waiting time rises from "
                  f"{wt[0]:.1f} to {wt[1]:.1f} s ({(wt[1]/wt[0]-1)*100:.0f}% longer) and queues grow from {q[0]:.1f} to {q[1]:.1f} vehicles.")
    if ev:
        d1.append(f"At the stadium between 16:00 and 19:59, an event raises the average count from {ev[0]:.0f} to {ev[1]:.0f} "
                  f"vehicles per 15 minutes ({(ev[1]/ev[0]-1)*100:.0f}% more).")
    d1.append("For the model this means the incident flag mostly changes how the road behaves, while the event flag mostly changes how many vehicles arrive.")
    add("07_incident_impact", "08_stadium_event", "Incident versus stadium event",
        "Two things that disturb normal traffic, shown in different ways.", d1)

    pk = st.get("peak")
    d2 = ["Both charts show the same weekday and weekend rhythm. The line chart gives vehicle counts for each intersection. The heatmap gives utilisation, which is count divided by road capacity.",
          "The heatmap makes the weekday/weekend split obvious in one glance. The line chart is better for reading actual numbers."]
    if pk:
        d2.append(f"Counts peak at {pk['weekday']}:00 on weekdays and at {pk['weekend']}:00 on weekends. "
                  f"Weekday rush hours show as dark bands, and weekends have a flatter midday bulge.")
    add("02_hourly_profile", "03_hour_day_heatmap", "Hourly profile versus heatmap",
        "The same daily pattern as counts and as utilisation.", d2)

    rn = st.get("rain")
    d3 = ["The weather chart splits records by rainfall. The incident chart splits them by whether an incident was flagged.",
          "Rain slows traffic a little. An incident is a sharper effect on speed and waiting time."]
    if rn and inc:
        d3.append(f"Average speed is {rn['avg_speed_kmh'][0]:.1f} km/h when dry and {rn['avg_speed_kmh'][1]:.1f} km/h when raining, "
                  f"against {inc['avg_speed_kmh'][0]:.1f} and {inc['avg_speed_kmh'][1]:.1f} km/h without and with an incident.")
    add("06_weather_effect", "07_incident_impact", "Weather versus incidents",
        "Two outside factors and how strongly each one slows traffic.", d3)

    conf = next((f["id"] for f in figures() if f["id"].startswith("confusion_")), "confusion_hist_gb_tabular")
    add("model_comparison", conf, "Model scores versus confusion matrix",
        "A summary of how good the models are, and the detail behind one of them.",
        ["The bar chart gives one number per model for error and accuracy. The confusion matrix opens up a single model and shows which levels it mixes up.",
         "Read the bars to choose a model. Read the matrix to see how it fails. The off-diagonal cells sit next to the diagonal, so mistakes are usually one level away."])

    d5 = ["The overall chart answers how often each level occurs, and the by-hour chart answers when."]
    if st.get("mix"):
        d5.append(f"Low is {st['mix'][0]*100:.0f}% of all intervals, yet {st['mix'][1]*100:.0f}% of High and Severe intervals fall between "
                  f"07:00 and 09:59 or 15:00 and 19:59. That is why hour and day of week matter so much to the model.")
    add("11_congestion_by_hour", "01_congestion_distribution", "Congestion by hour versus overall mix",
        "When congestion happens, and how common each level is in total.", d5)
    return out


def summary():
    ms = models()
    best = ms[0] if ms else None
    figs = figures()
    return {
        "n_figs": len(figs),
        "n_models": len(ms),
        "best_acc": f"{best['acc']*100:.1f}%" if best else "n/a",
        "best_mae": f"{best['mae']:.1f}" if best else "n/a",
        "best_name": best["title"] if best else "n/a",
        "families": sorted({m["family"] for m in ms}),
    }
