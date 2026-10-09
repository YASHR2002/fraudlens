# Hyperparameter tuning on Kaggle (Phase 5)

Tuning runs about 100 model fits on 1.15 million rows, which is too heavy for an 8 GB laptop.
A free Kaggle notebook (4 CPU cores, about 30 GB RAM, 12-hour sessions) runs it instead.

## 1. Build the upload bundle (laptop)

```powershell
uv run fraudlens kaggle-bundle --kaggle-user <your-kaggle-username>
```

This creates `data\kaggle_upload\` (git-ignored) containing:

| File | What it is |
|---|---|
| `features_dev.parquet` | 18 features + label + split, **train and validation rows only** (about 50 MB). The test set is never uploaded. |
| `fraudlens-0.1.0-py3-none-any.whl` | this project as a package, so Kaggle runs the same pipelines and metric code |
| `manifest.json` | split dates, row and fraud counts, the data file's SHA-256, local package versions |
| `dataset-metadata.json` | only needed for the Kaggle CLI route |

## 2. Upload it as a private Kaggle dataset

**Website (simplest):**

1. Sign in at [kaggle.com](https://www.kaggle.com), click **Create** (left menu) > **New Dataset**.
2. Drag in `features_dev.parquet`, the `.whl` file and `manifest.json` from `data\kaggle_upload\`.
3. Title: `fraudlens-features`. Leave visibility **Private**. Click **Create**.

**Or with the Kaggle CLI** (after `uv run kaggle auth login`, see `setup_windows.md`):

```powershell
uv run kaggle datasets create -p data\kaggle_upload     # private by default
# after rebuilding the bundle later:
uv run kaggle datasets version -p data\kaggle_upload -m "rebuilt bundle"
```

## 3. Create and configure the notebook

1. **Create** > **New Notebook**. Then **File** > **Import Notebook** and upload
   `notebooks\kaggle_tuning.ipynb` from the project.
2. Right-hand panel > **Add Input** > **Your Work / Datasets** > `fraudlens-features`.
3. **Settings** (right panel or the gear menu):
   - **Internet: On.** Needed to `pip install` the pinned library versions. Kaggle asks for a
     one-time phone verification before Internet can be enabled.
   - **Accelerator: None** (CPU). Gradient boosting here does not need a GPU, and CPU sessions
     have the larger time budget.

## 4. Run it in the background

Click **Save Version** > **Save & Run All (Commit)** > **Save**. The notebook then runs on
Kaggle's servers even if you close the browser. Interactive sessions stop after a period of
inactivity, so a committed run is safer for a long job.

- **How long:** each model is capped at 150 minutes (`TIMEOUT_PER_MODEL_MIN`), so the whole run
  takes at most about 5 hours; pruning usually makes it shorter. Watch progress in the version's
  **Logs** tab.
- The first cells check, and stop with a clear error if anything is wrong: the dataset is
  attached, the library versions match the laptop (Internet must be on), the split dates and data
  checksum match the manifest, and only train/validation rows are present.

## 5. Bring the result back

1. Open the finished version > **Output** tab > download `best_params.json`
   (`trials_xgboost.csv` and `trials_lightgbm.csv` are optional logs of every trial).
2. Save it in the project as **`configs\best_params.json`** (this file is committed).
3. Tell Claude Code; Phase 6 retrains locally with these parameters
   (`uv run fraudlens train --use-best-params`).

## Reproducibility notes

- **Same data:** the notebook refuses to run if the parquet file's SHA-256 differs from the
  manifest, and `best_params.json` records that hash. Rebuilding the bundle from the same
  `features.parquet` produces the same hash.
- **Same code:** the notebook installs the project wheel and calls
  `fraudlens.models.tuning.tune`, which builds the exact pipelines used locally.
- **Same libraries:** scikit-learn, XGBoost, LightGBM and Optuna are pinned to the laptop's
  versions; the versions actually used are written into `best_params.json`.
- **Same search:** Optuna's TPE sampler is seeded (42). Trial *timings* on shared hardware vary,
  so a re-run with a time cap may complete a different number of trials.
- **Local check:** the notebook has a smoke-test mode (2 tiny trials per model) that runs on the
  laptop against a copy of the bundle; it was run before upload.
