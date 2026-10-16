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
app/                    Flask web app (python app/app.py)
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
python app/app.py                                  # web app at http://localhost:5000/predictdata
```

## Deep learning and transformer experiments
```bash
python -m src.deep_learning.baselines                    # persistence + gradient-boosting reference
python -m src.deep_learning.train --model transformer --smoke   # 1-minute wiring check
python -m src.deep_learning.train --model all            # mlp, lstm, gru, transformer
python -m src.deep_learning.report                       # comparison table + charts
```
Setup: last 4 hours (16 steps) -> next 15 minutes; chronological 70/15/15 split; the congestion label is **never** an input
(it is computed from the target count, so using it would leak the answer).

### Reference results on the test period (9 Jun - 4 Jul 2025, 9,968 windows)
| Model | MAE (vehicles) | RMSE | R2 | Class accuracy | Macro-F1 |
|---|---|---|---|---|---|
| persistence | 16.51 | 25.92 | 0.9640 | 87.6 % | 0.796 |
| hist_gb_tabular | 11.22 | 16.84 | 0.9848 | 91.9 % | 0.869 |
| mlp / lstm / gru / transformer | *run `train.py` and `report.py` to fill in* | | | | |

The neural-network rows are intentionally empty: they have to be produced by running the training on your machine.
