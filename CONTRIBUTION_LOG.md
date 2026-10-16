# Contribution Log — Dynamic Urban Traffic Prediction

How this log was built: from the repository's **git history** (authors, dates, commit messages, files touched) and from reading the files themselves. Git only knows what was committed under each account, so where a folder was never committed, or a name is inferred, it is marked **(to confirm)**. Please correct anything that is wrong before submitting.

## 1. Summary

| Team member (git identity) | Commits | Active | Main contribution |
|---|---|---|---|
| **Prishu-04** (pratyakshsrivastava23ks@gmail.com) | 10 | 25 Sep – 6 Oct 2026 | Repo setup, README, datasets, committed the project-plan PDFs, **data-cleaning pipeline** (script + notebook), **data-visualisation** (script + notebook) |
| **Kaushal Kumar** (kr.kaushal779@gmail.com) | 1 | 8 Oct 2026 | Own cleaning and visualisation notebooks (matplotlib + seaborn), project-overview README, alternative cleaned datasets |
| **Khushi Kiran** — git `Kkiran217` (mailtokhushi.kiran217@gmail.com); name taken from the `khushikiran/` folder **(to confirm)** | 1 | 8 Oct 2026 | Cleaning + seaborn visualisation notebook (hotspot heatmap, weekday/weekend, waiting time, speed) |
| **`setup/` folder owner — not in git (to confirm)**; `setup.py` lists author e-mail gananya226@gmail.com, so probably that team member | 0 (untracked) | 29 Sep – 8 Oct 2026 (from log timestamps) | **Modular ML pipeline** (ingestion → transformation → trainer → predict), **Flask web app**, classical-ML comparison notebook, saved model |

## 2. Detail per member

### Prishu-04 — foundation, cleaning, visualisation (10 commits)
| Date | Commit | What |
|---|---|---|
| 25 Sep | `519e28d`, `8fd952f` | README: project overview, objectives, features, development plan |
| 25 Sep | `33c7aa5` | Virtual environment created *(committed to git — should not be tracked; excluded in this reorganised copy)* |
| 25 Sep | `196fbe8`, `16a3d33`, `4efbf25` | First dataset added; `setup.py` + `req.txt`; `src/` directory |
| 27 Sep | `09cd453` | First EDA + visualisation notebooks and `traffic_dataset_clean.csv` (first, minute-level single-intersection dataset) |
| 28 Sep | `3e40566` | `data_cleaning.py`, `data_visualisation.py`, project-plan and roadmap PDFs, first runtime log |
| 6 Oct | `b134216` | **Switched to the 4-intersection dataset** (`urban_traffic_dataset.csv`, 69,120 rows). Rewrote `data_cleaning.py` (+ notebook); produced `urban_traffic_cleaned.csv` and `urban_traffic_model_ready.csv` (66,428 rows) |
| 6 Oct | `7f8d1b0` | New `data_visualization.py` (12 charts) and `data_visualization.ipynb` |

Cleaning approach (the most thorough in the repo): duplicate and 15-minute-grid checks, calendar-column consistency, impossible-value handling, Hampel-filter detection of speed spikes, and **causal imputation** (forward-fill, then typical hourly profile) so no future information leaks; writes a `cleaning_report.json`. Now in `src/data_processing/data_cleaning.py` and `notebooks/01_data_cleaning/prishu_data_cleaning.ipynb`.
Visualisation: congestion balance, hourly profile (weekday vs weekend), hour × day heatmap, speed-vs-load, rain / incident / stadium-event effects, correlation, autocorrelation, congestion share by hour, daily totals. Now in `src/visualization/` and `notebooks/02_data_visualisation/prishu_data_visualization.ipynb`.

### Kaushal Kumar — notebooks and overview (1 commit, `241feea`, 8 Oct)
- `notebooks/legacy_single_intersection/kaushal_legacy_data_cleaning.ipynb` — cleaning of the first dataset (checks for `24:00:00` timestamps, converts to time deltas, derives `Hour`).
- `notebooks/01_data_cleaning/kaushal_data_cleaning.ipynb` — urban dataset: `dropna`, timestamp parsing, numeric congestion label.
- `notebooks/02_data_visualisation/kaushal_visualisation_matplotlib.ipynb` — 11 matplotlib charts (distribution, hour, day-of-week, intersection, congestion levels, vehicle count vs speed / waiting time / queue / rain / temperature, current vs next traffic).
- `notebooks/02_data_visualisation/kaushal_visualisation_seaborn.ipynb` — seaborn versions of the key charts.
- `docs/kaushal_project_overview_README.md` — long project overview and problem statement; `requirements.txt`.
- Alternative cleaned datasets in `data/processed/team_alt_cleaning/` (`kaushal_*`).

### Khushi Kiran (`Kkiran217`) — 1 commit, `cb13444`, 8 Oct
- `notebooks/01_data_cleaning/khushi_cleaning_and_visualisation.ipynb` — loads the urban dataset, `dropna`, duplicate and negative-value checks, numeric congestion mapping, saves a cleaned file; then five seaborn charts: volume vs speed by congestion, **hour × day-of-week hotspot heatmap**, weekday vs weekend hourly profile, waiting time by congestion level, average speed by congestion level.
- Output: `data/processed/team_alt_cleaning/khushi_dropna_cleaned.csv`.

### `setup/` folder (untracked) — ML pipeline and web app (owner to confirm)
- `src/components/data_ingestion.py`, `data_transformation.py`, `model_trainer.py`, `src/utils.py`, `src/pipeline/train_pipeline.py`, `predict_pipeline.py` — modular pipeline: time-ordered 80/20 split, cyclical hour/day features, scaling + one-hot, grid-searched Logistic Regression / Decision Tree / Random Forest / HistGradientBoosting. Intentionally **drops `target_next_count`** to avoid leakage.
- `app/app.py` + `app/templates/` — Flask app: input form → predicted congestion class with probabilities.
- `artifacts/` — saved `model.pkl`, `preprocessor.pkl`, train/test split.
- `notebooks/03_classical_ml/setup_team_classical_ml_models.ipynb` — comparison of Logistic Regression, SVC, KNN, Random Forest.
- `logs/runtime/setup_pipeline/` — 16 run logs. They show the pipeline first running on the old dataset (29–30 Sep), then moving to the urban dataset on 6 Oct, where the best model was **HistGradientBoosting, accuracy 0.9155**; the Flask app was run on 6 Oct.
- `EDA.ipynb` in that folder is empty (0 bytes), so it was not carried over.

## 3. Reorganisation and additions made after the hand-over (AI-assisted, with Claude)
These are not individual members' work and should be reviewed by the team.

**Reorganised** (nothing deleted from the originals; files were copied into a stage-based layout): `data/raw`, `data/processed`, `notebooks/01…04`, `src/{data_processing,visualization,components,pipeline,deep_learning}`, `app/`, `artifacts/`, `docs/`, `reports/`, `logs/`. The committed `venv/` and `__pycache__/` were left out; `.gitignore` added.

**Fixes (no logic changed):**
- `prishu_data_cleaning.ipynb`: `pd.read_csv()` had **no file path** and would crash; now reads `data/raw/…` and writes to `data/processed/`.
- Hard-coded `C:\Users\KIIT\…` paths in Kaushal's notebooks replaced by relative paths; Khushi's and the setup notebooks pointed at the working directory, now relative too.
- `src/logger.py` writes to `logs/runtime/` of the project (was: whichever folder you ran from); `data_ingestion.py` points to `data/raw/…`; `app/app.py` can find `src` when started from anywhere; script defaults in Prishu's scripts use project paths.
- Checked after moving: `data_cleaning.py` reproduces `urban_traffic_model_ready.csv` byte-for-byte; the visualisation script writes all 12 charts; ingestion + transformation run; the saved model still predicts through `PredictPipeline`.

**New:** `src/deep_learning/` (data windows, MLP / LSTM / GRU / Transformer in PyTorch, training, baselines, report), `notebooks/04_deep_learning/deep_learning_training.ipynb`, `requirements.txt`, this log.

## 4. Review findings for the team
1. **Target leakage in the classical-ML notebook (99%+ accuracy).** `target_congestion` is computed exactly from `target_next_count ÷ road capacity` (Low < 0.35 ≤ Moderate < 0.60 ≤ High < 0.85 ≤ Severe, verified on all 66,428 rows). The notebook leaves `target_next_count` in `X`. A note was added at the top of the notebook. Honest numbers: the `src/` pipeline gets 0.9155 accuracy; see `reports/deep_learning/model_comparison.md`.
2. **Random split** in that notebook; use the chronological split instead.
3. **`dropna` cleaning** (Kaushal, Khushi) removes 5,290 of 69,120 rows (7.7 %), including the whole first week (needs last-week lag) and scattered sensor rows. That breaks the 15-minute time series, so those files are unsuitable for sequence models. Use `data/processed/urban_traffic_model_ready.csv`.
4. Khushi's `urban_traffic_dataset.csv` in the repo is identical to her *cleaned* file, so the original raw data is not preserved there. The canonical raw file is `data/raw/urban_traffic_dataset.csv`.
5. The README still describes 3 classes and XGBoost; the data has **4 classes** (incl. *Severe*) and the team's models so far are scikit-learn ones. Decide and update the README.
6. 4 rows have no label; `LabelEncoder` in the classical notebook turned them into a fifth class.

## 5. How to keep this log up to date
Add a row under the member's section for each task you finish (date, files, one-line description). Run `git shortlog -sne --all` before submitting and make sure every member has committed their own work under their own account — the untracked `setup/` work currently has no author in git.
