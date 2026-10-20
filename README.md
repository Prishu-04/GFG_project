# Dynamic Urban Traffic Prediction

Predict the next-15-minute traffic volume at four city intersections and classify the expected congestion (Low / Moderate / High / Severe).
Group project — **who did what is in [`CONTRIBUTION_LOG.md`](CONTRIBUTION_LOG.md)**. The original plan README is kept in `docs/original_project_plan_README.md`.

## Project layout
```
data/
  raw/                  urban_traffic_dataset.csv (69,120 rows, 4 intersections, 15-min, Jan-Jul 2025)
                        traffic_dataset.csv (first single-intersection, minute-level dataset - legacy)
  processed/            urban_traffic_cleaned.csv, urban_traffic_model_ready.csv (66,428 rows, use this for models)
    team_alt_cleaning/  other members' cleaned versions (dropna-based, not for sequence models)
notebooks/
  01_data_cleaning/     prishu_ / kaushal_ / khushi_ notebooks
  02_data_visualisation/
  03_classical_ml/      scikit-learn model comparison (read the review note at the top)
  04_deep_learning/     deep_learning_training.ipynb  <- MLP, LSTM, GRU, Transformer
  legacy_single_intersection/
src/
  data_processing/      data_cleaning.py
  visualization/        data_visualization.py
  components/ pipeline/ utils.py   modular scikit-learn pipeline (ingestion -> transformation -> trainer -> predict)
  deep_learning/        data.py models.py train.py baselines.py report.py ...
app/                    Flask web app (python app/app.py): templates/, static/, results_data.py
artifacts/              saved scikit-learn model + preprocessor
models/deep_learning/   trained PyTorch checkpoints (created by train.py)
reports/                figures and metrics
docs/                   project-plan PDFs, original READMEs
logs/                   runtime logs
CONTRIBUTION_LOG.md
```

## Quick start (run everything from the project root)
```bash
pip install -r requirements.txt

python src/data_processing/data_cleaning.py        # raw -> data/processed
python src/visualization/data_visualization.py     # 12 EDA charts -> reports/figures/eda
python -m src.pipeline.train_pipeline              # scikit-learn pipeline -> artifacts/
python app/app.py                                  # web app at http://localhost:5000
```

The web app has three pages: Home (`/`), Predict (`/predictdata`) and Results (`/results`). Results shows every generated chart from
`reports/figures/`, a model comparison built from `reports/deep_learning/*_metrics.json`, and a side-by-side chart comparison with written
explanations. Re-run the visualisation or training scripts and the page picks up the new files.

## Deep learning and transformer experiments
```bash
python -m src.deep_learning.baselines                    # persistence + gradient-boosting reference
python -m src.deep_learning.classical                    # ridge, k-NN, decision tree, random forest
python -m src.deep_learning.train --model transformer --smoke   # 1-minute wiring check
python -m src.deep_learning.train --model all            # mlp, lstm, gru, transformer
python -m src.deep_learning.report                       # comparison table + charts
```
Setup: last 4 hours (16 steps) -> next 15 minutes; chronological 70/15/15 split; the congestion label is **never** an input
(it is computed from the target count, so using it would leak the answer).

### Results on the test period (9 Jun - 4 Jul 2025, 9,968 windows)
All ten models use the same windows, chronological split and metrics. Regenerate with the commands above (`classical`, `baselines`, `train --model all`, `report`).

| Model | Family | MAE (vehicles) | RMSE | R2 | Class accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| persistence | Baseline | 16.51 | 25.92 | 0.9640 | 87.6 % | 0.796 |
| linear_ridge | Classical ML | 13.49 | 19.28 | 0.9801 | 91.0 % | 0.858 |
| knn | Classical ML | 15.37 | 24.54 | 0.9677 | 89.0 % | 0.823 |
| decision_tree | Classical ML | 12.90 | 19.81 | 0.9790 | 90.3 % | 0.839 |
| random_forest | Classical ML | 11.02 | 16.57 | 0.9853 | 92.0 % | 0.872 |
| hist_gb_tabular | Classical ML | 11.22 | 16.84 | 0.9848 | 91.9 % | 0.869 |
| mlp | Deep learning | 11.27 | 16.75 | 0.9850 | 91.9 % | 0.869 |
| lstm | Deep learning | 10.98 | 16.69 | 0.9851 | 91.9 % | 0.871 |
| gru | Deep learning | 10.90 | 16.35 | 0.9857 | 92.1 % | 0.872 |
| transformer | Transformer | 10.94 | 16.47 | 0.9855 | 91.9 % | 0.870 |

One training run per model with seed 42, so gaps of a few hundredths of a vehicle between the top models are within noise.
The model behind the web app's Predict page is the separate classifier from `src/pipeline` (about 91.6 % accuracy on its own split), so it is not a row here.
