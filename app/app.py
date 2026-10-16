import os
import sys

# Reorganisation: make the project root importable so `src.*` resolves when running `python app/app.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

INT_FIELDS = ["hour", "minute", "day_of_week", "is_holiday", "event_nearby", "incident_flag"]
FLOAT_FIELDS = [
    "temperature_c", "rain_mm_15min", "vehicle_count", "avg_speed_kmh",
    "vehicle_density_per_km", "avg_waiting_time_sec", "queue_length_veh",
    "prev_count_15min", "prev_count_1h", "count_same_time_yesterday",
    "count_same_time_last_week", "rolling_mean_1h",
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html", form={})

    form = request.form.to_dict()
    try:
        values = {"intersection_id": form["intersection_id"]}
        values.update({k: int(form[k]) for k in INT_FIELDS})
        values.update({k: float(form[k]) for k in FLOAT_FIELDS})

        df = CustomData(**values).get_data_as_data_frame()
        label, probs = PredictPipeline().predict(df)
        return render_template("home.html", form=form, result=label, probs=probs)
    except Exception as e:
        return render_template("home.html", form=form, error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)