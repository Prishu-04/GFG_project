# Attribution Guide — Dynamic Urban Traffic Prediction

Prepared 16 Oct 2026 for the group project archive. **No original file was edited or removed.** This guide is the only file added.

## 1. Method

**Sources used** (only these):
1. `CONTRIBUTION_LOG.md`: the team's log. It says it was built from the repository's git history plus a reading of the files.
2. File and folder names (the `prishu_`, `kaushal_` and `khushi_` prefixes).
3. Details inside the files: `setup.py` author fields, PDF metadata, notebook contents, run-log dates and the READMEs.

**Limit:** the archive has no `.git` folder, so commit IDs, dates and authors cannot be checked here. They are the log's claims. Each one is cross-checked against filenames and file contents wherever a file allows it.

**Status labels**
| Label | Meaning |
|---|---|
| **Documented** | The log names the person *and* a filename, folder or file content agrees. |
| **Documented (caveat)** | The log supports the attribution, but something is unclear. The caveat is stated. Examples: the person may only have *committed* the file, or the file was renamed. |
| **Unverified** | The documentation does not establish who contributed. The reason is given. |
| **Not a member contribution** | Added during the post-hand-over reorganisation (log §3, AI-assisted), or an empty folder. |

Nothing is attributed on a guess. Where the documents conflict or are silent, the item is marked Unverified.

## 2. Team members and how each identity is supported

| Member | Evidence in the project | Notes |
|---|---|---|
| **Prishu** | Git handle `Prishu-04` (10 commits per log); `prishu_*` notebook names | The link between the handle and the name rests on the log and the filename prefix. The git e-mail given in the log does not contain the name, so it adds no support. |
| **Kaushal Kumar** | Git name "Kaushal Kumar", e-mail `kr.kaushal779@gmail.com` (1 commit); `kaushal_*` files | Name, e-mail and filenames agree. |
| **Khushi Kiran** | Git account `Kkiran217`, e-mail `mailtokhushi.kiran217@gmail.com` (1 commit); `khushi_*` files | The log marks the name "(to confirm)". The e-mail contains "khushi.kiran", which supports it. |
| **Ananya** | **Not named anywhere in the archive** | See Section 4. Unverified. |

## 3. Folder-level summary

| Folder / item | Attribution | Status |
|---|---|---|
| `data/raw/` | Committed to the repo by Prishu; who produced the data is not documented | Documented (caveat) |
| `data/processed/` (main files and `cleaning_report.json`) | Prishu | Documented |
| `data/processed/team_alt_cleaning/` | `kaushal_*` → Kaushal Kumar; `khushi_*` → Khushi Kiran; `setup_team_*` → Unverified | Mixed |
| `src/data_processing/`, `src/visualization/` | Prishu | Documented |
| `src/components/`, `src/pipeline/`, `src/utils.py` | Unnamed owner of the original `setup/` folder | Unverified |
| `src/logger.py`, `src/exception.py` | Not stated anywhere | Unverified |
| `src/deep_learning/` | Reorganisation stage (AI-assisted) | Not a member contribution |
| `app/` | Unnamed owner of the original `setup/` folder | Unverified |
| `artifacts/` | Output of the `setup/` pipeline; runner not recorded | Unverified |
| `notebooks/01_data_cleaning/` | Prishu, Kaushal Kumar and Khushi Kiran, one notebook each | Documented |
| `notebooks/02_data_visualisation/` | Prishu (1), Kaushal Kumar (2) | Documented |
| `notebooks/legacy_single_intersection/` | Kaushal Kumar | Documented |
| `notebooks/03_classical_ml/` | Unnamed owner of the original `setup/` folder | Unverified |
| `notebooks/04_deep_learning/` | Reorganisation stage (AI-assisted) | Not a member contribution |
| `docs/` | READMEs: Prishu (as committer) and Kaushal Kumar. PDFs: authorship not established | Mixed |
| `logs/` | Program output; first log committed by Prishu; the `setup_pipeline/` runs are Unverified | Mixed |
| `reports/figures/eda/` | Produced by Prishu's script; who generated these images is not recorded | Documented (caveat) |
| `reports/deep_learning/`, `reports/figures/deep_learning/` | Reorganisation stage (AI-assisted) | Not a member contribution |
| `models/` | Empty placeholder | n/a |
| Root files | `setup.py` (caveat), `requirements.txt` (Unverified), `README.md` (Unverified), `.gitignore` and `CONTRIBUTION_LOG.md` (reorganisation stage) | Mixed |

## 4. The `setup.py` contributor ("anonymous") and Ananya

What the files actually say:
- `setup.py` has `author='Traffic Detection'` and `author_email='gananya226@gmail.com'`. It does not contain the word "anonymous". The author name is a project label, not a person.
- The word "anonymous" appears in the metadata of both PDFs in `docs/`: `Author: (anonymous)`.
- The name "Ananya" does not appear in any project document, including `CONTRIBUTION_LOG.md`. The log says only that the `setup/` folder owner is "probably" the person behind the `setup.py` e-mail, and it names no one.

**Decision: Unverified. This contribution is not attributed to Ananya.**
- The e-mail local part is `gananya226`, with a leading "g" that nothing in the project explains. It could be a first initial, part of a different name, or an arbitrary handle. A near-match with "Ananya" does not identify a person.
- Nothing links the e-mail to the team roster. The only link between "Ananya" and this project is the request itself.
- The e-mail sits in `setup.py`, which the log says Prishu-04 committed. An e-mail in the metadata shows a contact address, not who wrote the `setup/` pipeline, app or notebook.

What would settle it: Ananya confirming that `gananya226@gmail.com` is hers and that she wrote the `setup/` work, or git author lines for those files (the log says they were never committed). Then the "Owner of the original `setup/` folder" entries below can be updated.

## 5. Items whose ownership remains Unverified

1. **The `setup/` work** (`src/components/`, `src/pipeline/`, `src/utils.py`, `app/`, `notebooks/03_classical_ml/`, `artifacts/`, `logs/runtime/setup_pipeline/`, `data/processed/team_alt_cleaning/setup_team_imputed_cleaned.csv`): the log says this was never committed, so there is no author. The person is unnamed (Section 4).
2. **`src/logger.py`, `src/exception.py`**: they existed by 28 Sep (roadmap PDF), but no document says who wrote them.
3. **`src/*/__init__.py` markers** other than `src/__init__.py`: not mentioned anywhere.
4. **Both PDFs in `docs/`**: committed by Prishu, but the PDF metadata says "(anonymous)" and the log does not say who wrote them.
5. **`requirements.txt`**: the log credits it to Kaushal Kumar (§2) and also lists it as a reorganisation addition (§3).
6. **`README.md`** (root): rewritten for the reorganised layout and not listed in the log.
7. **Data creator**: nobody documents who collected or produced the two raw datasets. The log only shows who added them to the repo.
8. **Who ran generated outputs** (run logs, `artifacts/`, EDA figures, `reports/deep_learning/`): outputs record what ran, not who ran it.
9. **Log discrepancy:** the log says 16 `setup/` run logs. The archive has 17 entries in `logs/runtime/setup_pipeline/`. Ten of them are folders named `*.log` that contain a same-named file. This is left as received.
10. **Structure note:** the original upload contained `CONTRIBUTION_LOG.md` twice (outside the inner ZIP and inside the project). The two copies are byte-identical.

## 6. File-by-file attribution

Files with identical attribution in the same folder are grouped.

| Path | Contributor | Status | Basis / what is unclear |
|---|---|---|---|
| `.gitignore` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Log §3: ".gitignore added" during reorganisation. |
| `CONTRIBUTION_LOG.md` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Log §3 lists "this log" under New additions made after hand-over. |
| `README.md` | Unverified | Unverified | Rewritten to describe the reorganised layout and says the original plan README was moved to docs/. Log §3 does not list it, so no author is stated. The original (25 Sep) README is attributed to Prishu-04 under docs/. |
| `app/app.py` | Owner of the original "setup/" folder (not identified) | Unverified | Log: Flask app belongs to the untracked setup/ folder. app.py was edited in reorganisation to find src (log §3). |
| `app/templates/home.html`, `app/templates/index.html` | Owner of the original "setup/" folder (not identified) | Unverified | Log: Flask app belongs to the untracked setup/ folder. app.py was edited in reorganisation to find src (log §3). |
| `artifacts/` — 4 files: `model.pkl`, `preprocessor.pkl`, `test.csv`, `train.csv` | Generated by the setup/ pipeline run (runner not identified) | Unverified | Log: model.pkl, preprocessor.pkl and the train/test split are outputs of the setup/ pipeline. Outputs of running code, so who ran it is not recorded. |
| `data/processed/cleaning_report.json` | Prishu | Documented | Log: outputs of Prishu-04's data_cleaning.py (b134216); reproducibility re-checked after reorganisation (log §3). |
| `data/processed/legacy_traffic_dataset_clean.csv` | Prishu (probable) | Documented (caveat) | Log credits Prishu-04 with "traffic_dataset_clean.csv" (09cd453, 27 Sep). This file has a different name, so the match is by description and was probably renamed in reorganisation; the rename is not stated in the log. |
| `data/processed/team_alt_cleaning/` — 4 files: `kaushal_legacy_cleaned_traffic.csv`, `kaushal_legacy_cleaned_traffic2.csv`, `kaushal_mat_visualized_data.csv`, `kaushal_re_cleaned.csv` | Kaushal Kumar | Documented | Filename prefix and log: Kaushal Kumar's alternative cleaned datasets. |
| `data/processed/team_alt_cleaning/khushi_dropna_cleaned.csv` | Khushi Kiran | Documented | Filename prefix and log: output of her cleaning notebook. |
| `data/processed/team_alt_cleaning/setup_team_imputed_cleaned.csv` | Unverified | Unverified | Prefix "setup_team" points to the setup/ folder, but this file is not mentioned in the log (it is read by the classical-ML notebook). "setup_team" names a folder, not a person. |
| `data/processed/urban_traffic_cleaned.csv`, `data/processed/urban_traffic_model_ready.csv` | Prishu | Documented | Log: outputs of Prishu-04's data_cleaning.py (b134216); reproducibility re-checked after reorganisation (log §3). |
| `data/raw/traffic_dataset.csv`, `data/raw/urban_traffic_dataset.csv` | Prishu (added to repo, per log); data creator Unverified | Documented (caveat) | Log: Prishu-04 added the first dataset (25 Sep) and switched to the 4-intersection dataset (6 Oct). That shows who committed the data. Nothing in the project says who collected or produced it. |
| `docs/Dynamic_Urban_Traffic_Prediction_Roadmap.pdf`, `docs/Traffic_Prediction_Project_Plan.pdf` | Prishu (as committer only, per log); authorship Unverified | Unverified | Log: Prishu-04 committed both PDFs (3e40566, 28 Sep). PDF metadata for both says Author "(anonymous)". The roadmap text addresses a reader ("your GFG_project repository"), so it reads like advice written for the team. The log does not say who wrote either PDF. |
| `docs/kaushal_project_overview_README.md` | Kaushal Kumar | Documented | Filename prefix and log: Kaushal Kumar project-overview README. |
| `docs/original_project_plan_README.md` | Prishu (as committer, per log) | Documented (caveat) | Log: Prishu-04 README commits 519e28d and 8fd952f (25 Sep). The file was moved here during reorganisation. The log does not say whether Prishu wrote the text or only committed it. |
| `logs/runtime/09_28_2026_00_32_11.log/09_28_2026_00_32_11.log` | Prishu (as committer, per log) | Documented (caveat) | Log: first runtime log was committed in 3e40566 (Prishu-04, 28 Sep); the date matches. A log file is program output; who ran the code is not recorded. |
| `logs/runtime/setup_pipeline/` — 17 files: `09_29_2026_19_39_05.log/09_29_2026_19_39_05.log`, `09_29_2026_19_40_49.log/09_29_2026_19_40_49.log`, `09_29_2026_19_46_22.log/09_29_2026_19_46_22.log`, `09_29_2026_19_49_10.log/09_29_2026_19_49_10.log`, `09_29_2026_19_52_41.log/09_29_2026_19_52_41.log`, `09_29_2026_19_56_21.log/09_29_2026_19_56_21.log`, `09_29_2026_20_11_12.log/09_29_2026_20_11_12.log`, `09_30_2026_16_33_09.log/09_30_2026_16_33_09.log`, `10_06_2026_14_04_28.log/10_06_2026_14_04_28.log`, `10_06_2026_14_18_44.log/10_06_2026_14_18_44.log`, `10_06_2026_14_42_56.log`, `10_06_2026_15_07_49.log`, `10_06_2026_15_08_35.log`, `10_06_2026_15_09_32.log`, `10_06_2026_15_09_35.log`, `10_06_2026_15_13_00.log`, `10_06_2026_15_13_03.log` | Run of the setup/ pipeline (runner not identified) | Unverified | Log: setup/ pipeline run logs, dated 29 Sep–6 Oct. Program output; runner not recorded. Note: this folder holds 17 log entries, while the log says 16. |
| `notebooks/01_data_cleaning/kaushal_data_cleaning.ipynb` | Kaushal Kumar | Documented | Filename prefix and log: Kaushal Kumar commit 241feea (8 Oct). Hard-coded paths replaced in reorganisation. |
| `notebooks/01_data_cleaning/khushi_cleaning_and_visualisation.ipynb` | Khushi Kiran | Documented | Filename prefix and log: commit cb13444 under git account Kkiran217 (8 Oct). Her email (mailtokhushi.kiran217) contains "khushi.kiran", which supports the name. |
| `notebooks/01_data_cleaning/prishu_data_cleaning.ipynb` | Prishu | Documented | Filename prefix and log: Prishu-04 cleaning notebook. Log §3: missing read_csv path fixed during reorganisation. |
| `notebooks/02_data_visualisation/kaushal_visualisation_matplotlib.ipynb`, `notebooks/02_data_visualisation/kaushal_visualisation_seaborn.ipynb` | Kaushal Kumar | Documented | Filename prefix and log: Kaushal Kumar commit 241feea (matplotlib / seaborn notebooks). |
| `notebooks/02_data_visualisation/prishu_data_visualization.ipynb` | Prishu | Documented | Filename prefix and log: Prishu-04 commit 7f8d1b0 (data_visualization.ipynb). |
| `notebooks/03_classical_ml/setup_team_classical_ml_models.ipynb` | Owner of the original "setup/" folder (not identified) | Unverified | Log assigns this notebook to the setup/ folder. The review note at the top of the notebook was added during reorganisation, not by the author. |
| `notebooks/04_deep_learning/deep_learning_training.ipynb` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Log §3: New, added after hand-over with AI assistance. |
| `notebooks/legacy_single_intersection/kaushal_legacy_data_cleaning.ipynb` | Kaushal Kumar | Documented | Filename prefix and log: Kaushal Kumar legacy-dataset cleaning notebook. |
| `reports/deep_learning/` — 4 files: `hist_gb_tabular_metrics.json`, `model_comparison.csv`, `model_comparison.md`, `persistence_metrics.json` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Outputs of the src/deep_learning code, which the log lists as New (AI-assisted). Who ran it is not recorded. |
| `reports/deep_learning/predictions/hist_gb_tabular.csv`, `reports/deep_learning/predictions/persistence.csv` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Outputs of the src/deep_learning code, which the log lists as New (AI-assisted). Who ran it is not recorded. |
| `reports/figures/deep_learning/confusion_hist_gb_tabular.png`, `reports/figures/deep_learning/forecast_vs_actual.png`, `reports/figures/deep_learning/model_comparison.png` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Outputs of the src/deep_learning code, which the log lists as New (AI-assisted). Who ran it is not recorded. |
| `reports/figures/eda/` — 12 files: `01_congestion_distribution.png`, `02_hourly_profile.png`, `03_hour_day_heatmap.png`, `04_week_timeseries.png`, `05_speed_vs_load.png`, `06_weather_effect.png`, `07_incident_impact.png`, `08_stadium_event.png`, `09_correlation.png`, `10_autocorrelation.png`, `11_congestion_by_hour.png`, `12_daily_totals.png` | Output of Prishu's data_visualization.py | Documented (caveat) | Log: that script writes all 12 charts, and this was checked after reorganisation. The charts may have been produced during that check, so who generated these exact images is not recorded. |
| `requirements.txt` | Unverified | Unverified | Log gives conflicting statements: §2 credits "requirements.txt" to Kaushal Kumar, while §3 lists "requirements.txt" as New in the reorganisation. The file also contains a torch/deep-learning line that fits the reorganisation work. Unclear which version this is. |
| `setup.py` | Prishu (as committer, per log) | Documented (caveat) | Log credits Prishu-04 with committing "setup.py + req.txt" (196fbe8/16a3d33, 25 Sep). The file's own author fields read author='Traffic Detection' and author_email='gananya226@gmail.com', which names no team member. Who wrote its content is not established; see Section 4. |
| `src/__init__.py` | Prishu (directory creation, per log) | Documented (caveat) | Log: Prishu-04 commit 4efbf25 "src/ directory". Whether this exact file was in that commit is not stated. |
| `src/components/__init__.py` | Unverified | Unverified | Package marker; the log credits the pipeline modules to the setup/ folder owner, who is not identified. |
| `src/components/data_ingestion.py`, `src/components/data_transformation.py`, `src/components/model_trainer.py` | Owner of the original "setup/" folder (not identified) | Unverified | Log §1/§2 assigns the modular pipeline to the untracked setup/ folder, which has no git author. Person is unnamed; see Section 4. data_ingestion.py path edited in reorganisation (log §3). |
| `src/data_processing/__init__.py` | Unverified | Unverified | Package marker; not mentioned in the log. |
| `src/data_processing/data_cleaning.py` | Prishu | Documented | Log: Prishu-04 wrote data_cleaning.py (28 Sep; rewritten 6 Oct, b134216). Script defaults adjusted during reorganisation (log §3). |
| `src/deep_learning/` — 9 files: `__init__.py`, `baselines.py`, `config.py`, `data.py`, `metrics.py`, `models.py`, `report.py`, `results_io.py`, `train.py` | Reorganisation stage (AI-assisted, not a member) | Not a member contribution | Log §3: src/deep_learning/ listed as New, added after hand-over with AI assistance. |
| `src/exception.py`, `src/logger.py` | Unverified | Unverified | Roadmap PDF says src/logger.py and src/exception.py already existed by 28 Sep, but no document says who wrote them. Not listed under any member in the log. logger.py was later edited in the reorganisation (log paths, per log §3). |
| `src/pipeline/__init__.py` | Unverified | Unverified | Package marker; the log credits the pipeline modules to the setup/ folder owner, who is not identified. |
| `src/pipeline/predict_pipeline.py`, `src/pipeline/train_pipeline.py` | Owner of the original "setup/" folder (not identified) | Unverified | Log §1/§2 assigns the modular pipeline to the untracked setup/ folder, which has no git author. Person is unnamed; see Section 4. data_ingestion.py path edited in reorganisation (log §3). |
| `src/utils.py` | Owner of the original "setup/" folder (not identified) | Unverified | Log §1/§2 assigns the modular pipeline to the untracked setup/ folder, which has no git author. Person is unnamed; see Section 4. data_ingestion.py path edited in reorganisation (log §3). |
| `src/visualization/__init__.py` | Unverified | Unverified | Package marker; not mentioned in the log. |
| `src/visualization/data_visualization.py` | Prishu | Documented | Log: Prishu-04 commit 7f8d1b0 (6 Oct) added the 12-chart data_visualization.py. Script defaults adjusted during reorganisation. |

## 7. Keeping this current
When someone confirms an Unverified item, change its row and cite where the confirmation came from (a git author line, a signed-off note, the person's own statement). Do not rely on filenames alone.
