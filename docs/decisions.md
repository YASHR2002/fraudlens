# Design Decisions Log

Each entry records the decision, the alternatives considered, and why. Newest phase last.

---

## Phase 0: Project setup

### D0.1 `uv` + `pyproject.toml` with dependency groups

- **Decision:** One `pyproject.toml`. Shared libraries (pandas, scikit-learn, XGBoost,
  LightGBM, SHAP, google-genai) are core dependencies; component-specific libraries are in
  PEP 735 dependency groups `train`, `api`, `dashboard`, `dev`. `uv sync` installs everything
  locally; Docker images install only core + their group. Exact versions are pinned in
  `uv.lock`; `pyproject.toml` holds major-version bounds.
- **Alternatives:** `requirements*.txt` files; Poetry; conda; optional extras instead of groups.
- **Why:** `uv` is fast, manages the Python version on Windows, and produces a cross-platform
  lock file. Groups keep the API image small (no Optuna, Evidently, Jupyter, Streamlit) while
  keeping one source of truth. Model libraries must be core because the API has to load and
  explain exactly the same model types that training produces.

### D0.2 Python 3.11

- **Decision:** `requires-python = ">=3.11,<3.12"`, `.python-version` = 3.11.
- **Why:** Fixed by the plan, and widely supported by the ML stack and by the slim Docker
  base images. Pinning one minor version avoids "works on my machine" differences between
  the laptop, CI, Kaggle and the Docker images.

### D0.3 Typer CLI as the single entry point

- **Decision:** `python -m fraudlens <command>` (also installed as `fraudlens`).
- **Alternatives:** Makefile; PowerShell or bash scripts.
- **Why:** Works the same in PowerShell, CI (Linux) and containers. Every command logs what it
  did and how long it took (`log_duration`), which makes long data steps observable.

### D0.4 Configuration: YAML for settings, `.env` for secrets, both validated

- **Decision:** `configs/config.yaml` is parsed into strict Pydantic models (unknown keys are
  rejected, so typos fail loudly). Secrets and deployment settings come from environment
  variables / `.env` via `pydantic-settings`, with secrets typed as `SecretStr` so they are
  masked when printed or logged.
- **Why:** One place for every tunable value (split dates, costs, promotion rules), no
  hard-coded settings, and no secrets in Git, logs or MLflow.

### D0.5 Docker Compose profiles and memory limits for an 8 GB laptop

- **Decision:** WSL2 capped at 3 GB. Each service has a `mem_limit`, and profiles (`data`,
  `train`, `serve`, `monitoring`) start only what a phase needs. All ports bind to
  `127.0.0.1` only.
- **Alternatives:** One always-on stack; running services natively on Windows.
- **Why:** The full stack plus model training would not fit in 8 GB. Profiles make the
  memory budget explicit and keep each phase reproducible with one command.

### D0.6 MLflow server: SQLite backend, served artifacts, explicit host allow-list

- **Decision:** `ghcr.io/mlflow/mlflow` pinned to the same version as the Python client in
  `uv.lock`. SQLite backend and artifacts in a named volume, served through the tracking
  server so the host and other containers log artifacts over HTTP. `--workers=1` and
  `--allowed-hosts=localhost,...,mlflow,...`.
- **Why / deviation from the plan:** MLflow 3.x added Host-header validation (DNS-rebinding
  protection). The default allows `localhost` and private IPs but not the compose service name
  `mlflow`, which the API container uses, so it is added explicitly rather than disabling the
  security middleware. The default of 4 workers does not fit a 512 MB limit; one worker is
  enough for a single user. PostgreSQL as MLflow backend was not used, to keep the `serve`
  profile from needing the database.

### D0.7 Data never enters Git

- **Decision:** `data/`, `reports/`, `models/`, MLflow files, `*.parquet`, `*.csv` (except
  test fixtures), model binaries and `.env` are git-ignored; the directory skeleton is kept with
  `.gitkeep` files. A pre-commit hook blocks files over 1 MB and private keys.
- **Why:** The dataset has (synthetic) personal data and is large; models and reports are
  reproducible build outputs.

### D0.8 Gemini model choice

- **Decision:** Default `GEMINI_MODEL=gemini-3.8-flash` (newest stable Flash model with a free
  tier, checked against the Gemini API pricing page in October 2026). The model name is
  configuration, not code.
- **Why:** Flash models are fast and on the free tier; a 3-5 sentence analyst note doesn't
  need a Pro model. Keeping the name in `.env` means a deprecation is a config change.

---

## Phase 1: Data ingestion and validation

### D1.1 Explicit raw schema, streamed with Arrow

- **Decision:** One schema module (`data/schema.py`) declares every raw column and its type.
  Files are streamed in 32 MB blocks with `pyarrow.csv.open_csv` using those types; a changed
  header or an unparseable value raises `SchemaError` immediately.
- **Alternatives:** `pandas.read_csv(chunksize=...)` with type inference.
- **Why:** Inference silently changes types when a file changes (e.g. `zip` read as an
  integer loses leading zeros). Arrow's streaming reader is several times faster than pandas
  chunks and keeps peak memory under 600 MB for the 335 MB training file.

### D1.2 `trans_date_trans_time` is the event time; `unix_time` is not used

- **Finding:** `unix_time` is 7 years earlier than `trans_date_trans_time` (the generator ran on
  a 2012 calendar). 2012 is a leap year, so the 29 February 2012 transactions were stamped
  `2019-02-28`: that date holds two days of activity, and it is the single place the training
  file goes backwards in time. The offset shifts by one day between 2019-02-28 and 2020-02-28.
- **Decision:** Keep `trans_date_trans_time` (renamed `trans_ts`) as the event time. It is the
  documented column and what a live transaction would carry. About 0.14% of rows (one date
  early in training) get slightly inflated velocity features.
- **Alternative:** Rebuild time from `unix_time` + 7 years: strictly monotonic, but moves
  ~924,000 rows by a day relative to their labelled date. Not worth it for one date.

### D1.3 Processed types: compact where lossless, float64 for money and coordinates

- **Decision:** Text columns are dictionary-encoded (pandas `category`), `city_pop` int32,
  `is_fraud` int8, `zip` left-padded to 5 digits (81,782 train and 34,607 test zips had lost a
  leading zero). `amt` and coordinates stay float64.
- **Why:** Halves in-memory size (427 MB to 208 MB for train) and Parquet is 4.3x smaller than
  CSV. float32 would change amounts (107.23 becomes 107.2300034) and break the exact
  SQL-vs-Python feature parity test planned for Phase 2.

### D1.4 One `transactions` table, loaded with COPY

- **Decision:** Both files go into one table with a `source` column (`train` / `test`),
  loaded from the Parquet files with PostgreSQL `COPY ... FROM STDIN` and indexed on
  `(cc_num, trans_ts)`. The load fails if PostgreSQL row counts differ from the Parquet files.
- **Why:** Features must be computed over train and test together in time order, so test
  transactions see their card's earlier history, exactly as a live system would. `COPY` loads
  1.85M rows in about 90 s, far faster than row inserts.

### D1.5 Time-based split

- **Decision:** train = fraudTrain up to 2020-04-21 (1,145,597 rows, 6,558 fraud);
  validation = fraudTrain 2020-04-22 to 2020-06-21 (151,078 rows, 948 fraud);
  test = all of fraudTest (555,719 rows, 2,145 fraud), used once in Phase 6.
- **Alternatives:** A random stratified split.
- **Why:** Fraud models are used on future transactions; a random split leaks future
  patterns into training and overstates performance. Two months of validation gives 948 fraud
  cases, enough for a stable PR-AUC and cost-based threshold. Test is later still and has a
  lower fraud rate (0.386% vs 0.579%, and 0.185% in December), which is realistic drift.

### D1.6 Connect to Docker services on `127.0.0.1`, not `localhost`

- **Finding:** Each PostgreSQL connection to `localhost` took about 2 minutes: Windows tries
  IPv6 `::1` first, and Docker Desktop's port proxy leaves it hanging. `127.0.0.1` connects in
  0.02 s.
- **Decision:** Defaults and `.env.example` use `127.0.0.1` for PostgreSQL and MLflow, and the
  connection has a 10 s timeout so a stopped database fails fast with a clear message.

### D1.7 Synthetic test fixture

- **Decision:** `fraudlens make-fixture` generates `tests/fixtures/sample_transactions.csv`
  (200 rows, 8 cards, 8 fraud) from invented values in the raw CSV format, including the
  quirks the pipeline must handle (empty index header, 4-digit zips). A test checks the file
  is exactly what the generator produces.
- **Why:** Tests and CI must not need the real data; generating rather than sampling avoids
  committing any real records.

### D1.8 Stopping the stack: `docker compose --profile "*" down`

- **Finding:** Every service belongs to a profile, and a plain `docker compose down` ignores
  profiled services, so it stopped nothing and `down -v` deleted no volumes. PostgreSQL kept its
  old password because its data volume survived.
- **Decision:** All docs use `docker compose --profile "*" down` (add `-v` to delete data).

---

## Phase 2: SQL feature engineering

### D2.1 Features in SQL with window functions, over train and test together

- **Decision:** 18 features (`features/feature_names.py`) are computed in PostgreSQL by
  `sql/02_features.sql` into a `features` table, then exported to
  `data/processed/features.parquet`. History features use window functions partitioned by
  card and ordered by time, over both files at once.
- **Alternatives:** pandas `groupby().rolling()`; computing train and test separately.
- **Why:** SQL window functions are the natural, reviewable way to express "per card, in the
  previous hour", and it is where a bank's transaction data lives. Computing test together with
  train means a test transaction sees its card's real earlier history (legitimate past data),
  exactly as a live system would; computing test alone would wrongly reset every card's history
  at the split date.

### D2.2 "Prior" means strictly earlier in time (deviation from the plan)

- **Finding:** 44 pairs of transactions share a card and the same second, 87 consecutive gaps
  are exactly 1 hour, and 4 exactly 24 hours.
- **Decision:** Every history frame is `RANGE BETWEEN <window> PRECEDING AND CURRENT ROW
  EXCLUDE GROUP`: it drops the current row *and* any same-second peers, so windows are
  `[t - window, t)`. The plan's `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING` for the
  prior average was replaced.
- **Why:** With `ROWS`, which of two same-second transactions counts as "before" depends on an
  arbitrary sort order, so the SQL result would not be reproducible and could not be matched by
  the online code. "Strictly earlier" is deterministic and order-independent, and the Python
  implementation reproduces it exactly whatever order ties arrive in (tested).

### D2.3 Missing history stays missing

- **Decision:** For a card's first transaction, `card_avg_amt_prior`, `amt_to_card_avg_ratio`
  and `secs_since_last_txn` are NULL (NaN), not filled with 0 or 1; `card_txn_number = 0` also
  marks it.
- **Why:** Any fill value is a made-up number (a gap of 0 seconds would look like a burst).
  XGBoost and LightGBM handle missing values natively; the logistic regression baseline will
  impute inside its own pipeline in Phase 4.

### D2.4 No leakage by construction

- Nothing reads `is_fraud` of any row; no target encoding. Windows end before the current
  transaction. `gender`, names, street and `trans_num` are never inputs; `gender` is carried in
  the table only for the fairness audit, and `cc_num` only as the partition key.
- A manual spot-check of real cards (first transaction, a same-second tie, a fraud burst,
  1-hour window edges) confirmed each value by hand; see the Phase 2 journal entry.

### D2.5 Training-serving parity test

- **Decision:** `features/online.py` reimplements every feature in pure Python with a small
  per-card state (last 7 days of timestamps and amounts, running count and sum, first time
  each category was seen). `tests/test_feature_parity.py` checks it against the SQL:
  1. on the synthetic fixture, against the SQL output committed in
     `tests/fixtures/sample_features_sql.csv` (runs in CI with no database);
  2. on the full two-year history of 40 random real cards, against `features.parquet`
     (runs locally when the data exists).
- **Why:** The model trains on SQL features but the API computes them in Python. Any
  difference (an off-by-one window, a different age rule, tie handling) is training-serving
  skew: the model silently receives different inputs in production than it was trained on.
  The test fails loudly instead.
- **Result (2026-10-07):** both levels pass. The real-data check compares 79,057 transactions
  (40 cards, full history, ties included) and every feature matches within 1e-9.
  Building the table takes about 5.5 minutes in PostgreSQL (1 GB container); the export to
  Parquet 17 s (124 MB, 258 MB in pandas).

---

## Phase 3: EDA

### D3.1 EDA on the train split only

- **Decision:** `notebooks/01_eda.ipynb` analyses only the train split (1,145,597 rows). The
  fraud-rate-over-time chart alone adds validation months; test labels are never loaded
  (`load_features` filters test rows out at read time unless asked).
- **Why:** Patterns spotted during EDA shape modelling choices. Looking at validation or test
  outcomes would let those sets influence decisions and make their scores optimistic.

### D3.2 Keep the feature set as built, despite two findings

- **Findings:** `distance_km` carries no signal (fraud rate 0.55-0.61% in every decile; the
  simulator places merchants randomly near home), and `is_night` (22:00-05:59) is slightly too
  wide (fraud is concentrated 22:00-03:59).
- **Decision:** No feature changes. Trees can use `hour` directly, and distance is a realistic
  feature that real data would likely reward; the model should learn to ignore it here, which
  SHAP will show in Phase 7.
- **Alternative:** Drop distance and redefine `is_night`. Rejected: tuning features to quirks of
  a simulator would not transfer, and every SQL change has to be mirrored online.

### D3.3 Known synthetic artefacts to state in the model card

- No fraud above $1,500 (fraud amounts are capped at ~$1,372) and fraud amounts are bimodal.
  **Measured on the Phase 4 LightGBM (validation):** the cap does *not* let flagged frauds escape:
  raising the amount of the 920 caught frauds to $1,600, $2,000 or $5,000 leaves 97.3% flagged,
  because night-time, category and card-history signals still fire. But it does teach "very large
  amounts are normal": of 160 legitimate purchases above $1,500, only 3 are flagged, although
  their median is 34.8x the card's average (frauds: 5.5x). Moved to 23:00 as online purchases,
  41% of them are flagged. On real data, a daytime in-store purchase 35x a card's usual spend
  deserves review; a production system would add a business rule for that (to be stated in the
  model card, and visible in SHAP explanations in Phase 7).
- Fraud arrives as per-card bursts (median 10 fraud transactions per affected card), which
  inflates velocity features' usefulness and is why the split is by time, not random.

---

## Phase 4: Baseline models with MLflow

### D4.1 Accuracy is meaningless here; PR-AUC is the primary metric, business cost decides

- **Accuracy:** with 0.57% fraud, flagging nothing is 99.43% accurate while catching zero fraud
  and losing $508,489 on the validation months. Any model looks "accurate".
- **ROC-AUC (reported, not used to choose):** it averages over the false-positive rate, which
  the ~150,000 easy legitimate transactions keep tiny. Logistic regression scores 0.988 ROC-AUC
  yet only 0.42 PR-AUC: at most recall levels over half its alerts would be false.
- **PR-AUC (primary, threshold-free):** summarises precision (how clean the analysts' alert
  queue is) across all recall levels, and its no-skill level is the fraud rate, not 0.5.
- **Total business cost (decides the operating point):** each missed fraud costs its amount,
  each false alarm $5 of analyst time (`config.yaml`). This is the quantity the business pays.

### D4.2 Threshold minimises cost on validation, not F1

- **Decision:** `models/threshold.py` evaluates every distinct score as a threshold in one sorted
  pass (O(n log n), ties flagged together) and picks the minimum total cost; among equal costs the
  highest threshold (fewest alerts). The F1-optimal threshold is reported only for comparison.
- **Why:** F1 weighs a false alarm and a missed fraud equally; here a missed fraud averages ~$530
  versus $5 for a review, about 100 to 1. For LightGBM the F1 threshold (0.93) misses 66 frauds
  ($12,894 total cost); the cost threshold (0.36) misses 28 ($3,015): F1 would cost 4.3x more.
- **Cost model choice:** following the plan, correctly flagged fraud costs nothing. In practice
  it also takes a review; adding $5 per true positive would add $4,600 for LightGBM's 920
  caught frauds but barely move the threshold, because missed fraud dominates.

### D4.3 Class weighting, not resampling; no early stopping

- **Decision:** logistic regression and LightGBM use `class_weight="balanced"`, XGBoost
  `scale_pos_weight` = legit/fraud in train (173.7). No SMOTE or undersampling. Fixed numbers of
  trees, no early stopping.
- **Why:** weighting keeps every real row and needs no synthetic data; the probabilities become
  uncalibrated, but the threshold is chosen on validation scores so it absorbs that. Early
  stopping would use validation to choose the number of trees and then report validation scores
  that were partly fit to it.

### D4.4 One pipeline interface, saved with skops (MLflow 3.16 default)

- **Decision:** every model is a scikit-learn `Pipeline` taking the same 18 raw features
  (`prepare_features`): trees get ordinal-encoded categories (unseen -> missing) and raw
  numerics with NaN; logistic regression gets one-hot, median imputation with missing
  indicators, and scaling. Models are logged with skops plus an explicit list of trusted types
  per model (all library classes), `pyfunc_predict_fn="predict_proba"`, and pinned pip
  requirements for just the six packages a model needs.
- **Why:** MLflow 3.16 saves scikit-learn models with skops by default and refuses to load
  pickles unless explicitly allowed. Skops only reconstructs trusted types instead of executing
  arbitrary code, so avoiding custom transformers keeps the trust list short and auditable. A test
  fails if a library upgrade adds a type the list does not cover.

### D4.5 MLflow server: job runner disabled (plan's 512 MB limit was too small)

- **Finding:** MLflow 3.16's server starts a job runner and 7 worker processes for its GenAI
  features, about 225 MB each. The container was killed for exceeding 512 MB six times in a row.
- **Decision:** `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` (a documented setting; the features are
  unused here). The server then idles at ~400 MB; the limit is 768 MB for artifact uploads.
- **Also:** MLflow prints emoji when a run ends, which crashed on the Windows cp1252 console; the
  CLI switches stdout/stderr to UTF-8 at startup.

### D4.6 Baseline results (validation, 2026-10-08)

| Model | PR-AUC | ROC-AUC | Cost-optimal threshold | Precision | Recall | Cost | F1-threshold cost |
|---|---|---|---|---|---|---|---|
| LightGBM | 0.984 | 0.9996 | 0.364 | 0.91 | 0.97 | $3,015 | $12,894 |
| XGBoost | 0.982 | 0.9996 | 0.489 | 0.86 | 0.97 | $3,216 | $9,866 |
| Logistic regression | 0.421 | 0.988 | 0.594 | 0.28 | 0.92 | $19,904 | $68,074 |

Flag-nothing cost on validation: $508,489. Re-running `fraudlens train` reproduces every metric
exactly (fixed seeds). LightGBM's train PR-AUC is 1.000 (memorising the training set at default
settings), which Phase 5 tuning should address.

---

## Phase 5: Hyperparameter tuning on Kaggle

### D5.1 Tune off-laptop, on a free Kaggle notebook

- **Decision:** Optuna studies for XGBoost and LightGBM (50 trials each, 150-minute cap per
  model) run on Kaggle's free CPU notebooks (4 cores, ~30 GB RAM). The laptop only builds the
  upload bundle and later retrains once with the chosen parameters.
- **Alternatives:** tune locally (about 100 fits of up to a few minutes each, while Docker holds
  3 GB of the 8 GB, for several hours); a paid cloud VM; a much smaller local search.
- **Why:** free, enough memory, and it keeps the laptop usable. Only the *search* happens off the
  laptop; the final model is trained locally and tracked in MLflow like the baselines.

### D5.2 How reproducibility is preserved across machines

- **Same data, provably:** `fraudlens kaggle-bundle` exports train + validation features with a
  SHA-256 in `manifest.json`; the notebook refuses to run on a different file, and the hash is
  written into `best_params.json`. Rebuilding the bundle gave the identical hash.
- **Same code:** the bundle carries the project wheel; the notebook calls
  `fraudlens.models.tuning.tune`, which uses the same `build_pipeline` and `prepare_features` as
  local training. Nothing is re-implemented in the notebook.
- **Same libraries:** scikit-learn 1.9.1, XGBoost 3.2.0, LightGBM 4.7.0 and Optuna 5.0.0 are
  pinned and checked; actual versions are recorded in the output.
- **Same split:** dates are copied into the notebook's first cell (as planned) and asserted equal
  to the manifest's.
- **No test leakage:** test rows are never exported; the notebook also asserts that only `train`
  and `validation` splits are present.
- **Observed on Kaggle (2026-10-08):** the image runs Python 3.13.15 (laptop: 3.11), pandas 2.3.3
  and numpy 2.1.3 (laptop: 3.0.6 / 2.4.6). The wheel installed only thanks to
  `--ignore-requires-python`. The model libraries were pinned to the laptop's exact versions,
  and pandas/numpy only load the data, whose checksum matched. Any remaining numeric differences
  do not matter for the final model: Phase 6 retrains and re-evaluates on the laptop.
- **Tested locally:** a smoke-test mode (2 tiny trials per model) ran the whole notebook on the
  laptop against a copy of the bundle; `tests/test_tuning.py` covers the objective, the payload
  and both pruning callbacks.

### D5.3 Search design

- **Objective:** validation PR-AUC (the primary metric). Because validation is used to choose
  hyperparameters, its scores become slightly optimistic; the untouched test set in Phase 6 gives
  the unbiased estimate.
- **Sampler:** seeded TPE. **Pruner:** median pruner on validation average precision reported
  every 25 boosting rounds (after 5 complete trials and 100 rounds), so weak configurations stop
  early.
- **Spaces:** trees 200-1,500, learning rate 0.01-0.3 (log), depth 3-10 / leaves 15-255 (log),
  minimum leaf size, row and column subsampling, L1/L2 penalties, minimum split gain. These
  include strongly regularised settings, since the LightGBM baseline memorised the training set.
- **Fixed:** class weighting (legit/fraud ratio) and the feature set, so the comparison with the
  baselines isolates the effect of the hyperparameters.

### D5.4 Tuning results (Kaggle, 2026-10-08)

| | XGBoost | LightGBM |
|---|---|---|
| Trials (complete / pruned) | 16 / 34 | 20 / 30 |
| Best validation PR-AUC | 0.9865 (baseline 0.9820) | 0.9853 (baseline 0.9840) |
| Key changes vs baseline | 700 trees, depth 8, learning rate 0.22, min_child_weight 15, gamma 1.2, L1/L2 1.2/6.6 | 450 trees, **26 leaves (was 63)**, **min 217 rows per leaf (was 50)**, L1/L2 3.0/2.3, min_split_gain 0.46 |

- The whole run took about 70 minutes; pruning stopped 64 of 100 trials early.
- Checks on return: the file's data SHA-256 and split dates match the local bundle and config,
  and the four model libraries match the laptop's versions exactly.
- LightGBM's winning configuration is much more regularised (smaller trees, larger leaves,
  L1/L2 penalties), which addresses the baseline's memorisation of the training set.
- These validation scores were used to choose the parameters, so they are slightly optimistic;
  Phase 6 retrains locally, re-evaluates, and reports the untouched test set once.

---

## Phase 6: Final training, promotion gate, and test report

### D6.1 One registered model, aliases not stages

- **Decision:** every candidate is a version of `fraudlens-fraud-detector`; the production model is
  whichever version holds the alias `@champion`. Each version carries its cost-optimal threshold
  and validation metrics as tags, plus `promotion_status` / `promotion_reason` after the gate.
- **Why:** MLflow deprecated stages (Staging/Production) in favour of aliases. Tags make the
  version self-describing: the API can load `models:/fraudlens-fraud-detector@champion` and read
  its threshold without recomputing anything.

### D6.2 Promotion gate: rules on validation, winner by business cost

- **Rules** (config): recall >= 0.80, precision >= 0.50, PR-AUC >= the current champion's.
- **Winner:** the passing version with the **lowest validation cost** (ties: higher PR-AUC, then
  newer). PR-AUC is a guard against regressions; cost is what the business optimises.
- **Recorded:** every candidate gets a status and the full list of PASS/FAIL reasons on both its
  model version and its training run; a beaten champion is tagged `retired`. Nothing is deleted.
- **Tested:** pure-function tests of every rule and tie-break, plus an end-to-end registry test on
  a throwaway SQLite MLflow (first promotion, a rejection, a blocked weaker challenger, a takeover).

### D6.3 The test set is used once, with everything fixed beforehand

- The model, its threshold, and the "flag everything over $X" rule's X were all chosen on
  validation. `fraudlens evaluate --final` is the only code that loads test rows; it tags the
  champion `test_evaluated_utc` and refuses to run again without `--force` (a forced re-run is
  labelled as such in the report and in MLflow).
- The best threshold *in hindsight* on test is reported for reference only, to measure drift.

### D6.4 Results (2026-10-09)

| | Validation | Test |
|---|---|---|
| Champion | LightGBM v2 (tuned) | |
| PR-AUC | 0.9853 | 0.9745 |
| Precision / recall at threshold 0.4326 | 0.901 / 0.976 | 0.861 / 0.952 |
| Total cost | $2,975 (flag nothing $508,489) | $25,150 (flag nothing $1,133,325) |
| Best rule "flag >= $257.51" (X from validation) | | $91,335, 14,820 alerts, recall 0.731 |

- Tuned vs baseline on validation: LightGBM $2,975 vs $3,015; XGBoost $3,037 vs $3,216. Both
  passed the gate; LightGBM won on cost. XGBoost's laptop PR-AUC (0.9852) was 0.0013 below its
  Kaggle score; LightGBM matched exactly. Multi-threaded histogram sums in XGBoost depend on the
  core count (8 vs 4), so tiny differences compound over 700 trees.
- **Threshold drift:** on test, the hindsight-optimal threshold is 0.0385 ($13,363), far below the
  validation choice. The validation-chosen threshold is too strict for late 2020 (most of the
  remaining cost is $23,495 of missed fraud). Not exploited here; Phase 9 monitors it.

---

## Phase 7: Explainability, LLM explanations, and fairness

### D7.1 SHAP: exact TreeSHAP, grouped into plain-English factors

- **Decision:** `shap.TreeExplainer` on the champion's booster (verified identical to LightGBM's
  native contributions, max difference 0.0; additivity checked in tests for LightGBM and
  XGBoost). Related features are grouped for explanations (amount + log amount; hour + night
  flag); SHAP values add, so a group's contribution is the sum. Each factor carries a sentence
  with the real values ("amount $330.52 is 6.0x this card's average of $54.94").
- **Global plots:** beeswarm and bar on a seeded 10,000-row *validation* sample, logged to the
  champion's training run. In the beeswarm, colour for "merchant category" shows its integer code,
  which has no meaningful order.

### D7.2 LLM: facts in, validated JSON out, never a hard dependency

- **Model:** `gemini-3.5-flash-lite` (configurable). Measured: ~1.7-2.5 s per note, against
  6-13 s for `gemini-3.8-flash`, whose free tier also allowed only 5 requests per minute.
  `thinking_level=low`; `minimal` is not supported by these models.
- **Grounding:** the prompt holds only system-computed facts: score, threshold, decision, top 5
  SHAP factors with values, and context (category, hour, distance) explicitly marked as
  background that may not be presented as a reason. This rule was added after a live note listed
  distance (not a top factor) as a reason.
- **Validation:** Pydantic schema (summary, 1-5 key reasons, approve/review/block); the action
  must agree with the model's decision; no protected terms (word-boundary match, so "average" is
  fine) and no card-number-like digits. An invalid reply counts as a failure.
- **Robustness:** 15 s timeout; one retry for timeouts, 429 and 5xx, none for 400/401/403/404;
  then a deterministic template note from the same factors (it never recommends "block": without
  the LLM, a human decides). LLM notes are cached by `trans_num` (LRU); fallbacks are not cached,
  so a transient outage does not stick. Every result says `explanation_source: llm | fallback`.
- **Observed live:** 504 deadline, 503 overload and 429 quota errors all ended in a correct
  fallback note; the schema initially failed because Pydantic's `extra="forbid"` becomes
  `additionalProperties`, which the Gemini API rejects (removed).
- **Tests:** the Gemini client is always mocked (15 tests: cache, retry, permanent errors,
  invalid replies, fallback wording).

### D7.3 Protected attributes and the LLM

- Gender is not a model feature. Age is a feature, so it appears in SHAP charts for analysts and
  auditors, but age factors are removed before the LLM call and from fallback wording, and any
  note mentioning age or gender is rejected. Explanations to analysts should not suggest a
  customer's age or gender is a reason to block them.

### D7.4 Fairness audit: measure with uncertainty, do not auto-correct

- **Decision:** recall, precision and false positive rate by gender and age band (under 30,
  30-49, 50-69, 70+) on the test set at the production threshold, each with a 95% Wilson
  interval; a gap is called real only if the best and worst groups' intervals do not overlap.
- **Findings:** real gaps in recall by gender (97.5% male vs 93.3% female), recall and precision
  by age (30-49 lowest), and false positive rate by age (under 30: 0.073%, 70+: 0.031%).
  Discussed in `docs/model_card.md`; no automatic mitigation, as planned.

---

## Phase 8: Scoring API and dashboard

### D8.1 Online state: snapshot at the end of fraudTrain, updated by every scored transaction

- **Decision:** `fraudlens build-state` writes each card's state after its last fraudTrain
  transaction (2020-06-21 12:13) to `card_state.json.gz` (983 cards, 259 KB, gzipped JSON, no
  pickle). It is built with vectorised aggregates in 2 s and tested equal to a full replay. The
  API loads it into memory and records every scored transaction (`update_state=true`).
- **Production note:** in production this would be Redis or a feature store shared by replicas;
  an in-memory store fits one process and this data size.
- **Concurrency and retries:** one lock makes compute-then-record atomic per request; recorded
  `trans_num`s are remembered, so a client retry is scored again but never double-counted (same
  features: the earlier copy is a same-second peer and excluded). A transaction older than the
  card's latest recorded one returns **409**, since its history can no longer be reconstructed.

### D8.2 Demo transactions: only those with exact history

- **Finding:** scoring an arbitrary December test transaction against the June snapshot would
  silently miss its card's intervening history and produce wrong features.
- **Decision:** the demo set is each card's *first* transaction after the snapshot (924, of
  which 20 fraud), whose true history is exactly the snapshot. `build-state` verifies the API's
  features equal the SQL features for all 924 (0 mismatches). The API serves them at
  `GET /demo/transactions`, so the dashboard needs no data files.

### D8.3 Fast and explained paths; latency

- `/predict` computes features online and scores; `/predict_explained` adds SHAP factors (top 8)
  and the LLM note. `/predict/batch` takes up to 1,000, scores them in time order and reports
  per-item errors; `/model-info` reports version, threshold (env override or model tag), test and
  validation metrics, global SHAP importance and fairness from MLflow; `/metrics` is Prometheus.
- **Measured** (1,000 sequential requests, warm, laptop): first version p50 **19.0 ms** server
  time. Profiling showed the model was not the cost: building a one-row pandas frame took 5.2 ms
  and LightGBM spent more on starting threads than predicting. Fixes: a direct one-row frame
  builder (scores bit-identical on all 924 demo transactions, tested) and `n_jobs=1` at serving
  time. Result: **p50 5.5 ms server-side; round trip p50 8.6 ms, p95 12.0 ms in Docker.**
  `/predict_explained` takes about 2 s, almost all of it the LLM call.

### D8.4 Images

- **API:** two-stage `python:3.11-slim` build with uv from the lock file, only the core + `api`
  groups, non-root user, `libgomp1` for LightGBM/XGBoost, health check on `/ready`. Model from
  MLflow (or a mounted export later); state and demo data mounted read-only; secrets only via
  `.env` at run time (`.dockerignore` excludes `.env`, data, models, notebooks, tests).
- **Size:** 2.66 GB at first; 469 MB was NVIDIA NCCL, which XGBoost pulls in on Linux for
  multi-GPU training. A uv `override-dependencies` entry drops it (CPU only; XGBoost verified to
  train and predict in the image): **1.82 GB**. XGBoost's own GPU kernels (~230 MB) remain.
- **Import hygiene:** the API image does not ship matplotlib; an isolated `api`-group environment
  check caught that the SHAP module imported it at module level (now lazy) and that the API
  pulled in training code via a helper (moved).
- **Dashboard:** Streamlit + Plotly + httpx only (no project package, no ML libraries), talking to
  the API over HTTP (`API_URL`): 1.03 GB, ~70 MB RAM.

### D8.5 Running footprint (serve profile)

MLflow ~330 MB, API ~530 MB (1 GB limit), dashboard ~70 MB: about 0.93 GB of Docker's 3 GB.

---

## Phase 9: Monitoring and drift

### D9.1 What the API exposes to Prometheus

- HTTP metrics from `prometheus-fastapi-instrumentator`, with latency buckets from 1 ms to 10 s
  (dense between 5 and 25 ms); the library's default buckets start at 100 ms, too coarse for a
  5 ms endpoint.
- Model metrics: transactions scored by endpoint and decision (flag rate), a fraud-score
  histogram (an early drift signal), analyst notes by source (LLM vs fallback) with LLM latency,
  duplicates, out-of-order rejections, and gauges for the serving model, threshold and cards.
- Each app owns its Prometheus registry, so tests can build many apps in one process.

### D9.2 Monitoring profile without MLflow

- `fraudlens export-model` copies `@champion` (skops model + metadata: threshold, metrics,
  importance, fairness) to `models/champion/` (1.4 MB) and verifies it loads before replacing
  the previous export. A second Compose service, `api-monitoring`, shares the API definition
  through a YAML anchor but sets `MODEL_SOURCE=local` and has no MLflow dependency; a network
  alias keeps `api:8000` as the single scrape target. Footprint: API ~370 MB, Prometheus ~120 MB,
  Grafana ~130 MB.

### D9.3 Grafana is provisioned, not clicked together

- Datasource and a 12-panel dashboard (`docker/grafana/provisioning/`) load at startup: serving
  model and threshold, transactions scored, flag rate, p50/p95/p99 latency, requests/s, error
  rate, score distribution, LLM notes by source and LLM latency. Anonymous read-only viewing is
  enabled for the local stack only (ports bind to 127.0.0.1).
- Ratios use `or vector(0)` so "no fallbacks" and "no errors" show 0% instead of "No data".

### D9.4 Replay: live traffic, exact features

- `fraudlens replay --speed <tx per second>` sends test transactions in time order with
  `update_state=true`, continuing the snapshot's history, so features stay exact; every
  response is saved with its true label for live recall/precision. Every Nth can go to the
  explained endpoint to exercise the LLM without exhausting the free quota.
- **First run (6,000 transactions, 21 Jun 12:14 to 22 Jun 23:28):** 241 s at 24.9/s, 0 errors,
  0 out-of-order, recall 1.000 (21 of 21 frauds), precision 0.913, 12 of 12 LLM notes.

### D9.5 Latency under sparse traffic (finding)

- During the 25/s replay, handler p95 was ~20 ms against ~8 ms in the back-to-back benchmark.
  A controlled test on unseen transactions (all updating state) isolated the cause:

  | Traffic | p50 | p95 | p99 |
  |---|---|---|---|
  | back-to-back | 5.6 ms | 7.9 ms | 9.8 ms |
  | 100 requests/s | 5.7 ms | 8.4 ms | 10.5 ms |
  | 25 requests/s | 6.6 ms | 19.9 ms | 23.2 ms |

- The model and state work are unchanged; with ~40 ms idle between requests the laptop CPU and
  the WSL2 VM drop into power saving, and waking costs ~10-15 ms on a share of requests. This is
  a property of sparse traffic on a laptop, not of the code; under steady load the p95 is under
  10 ms. Both numbers are reported; the 20 ms target holds at p95 in both regimes.

### D9.6 Drift reports (Evidently) and what they found

- **Setup:** reference = 20,000 seeded train rows (features + champion score); current = each
  test month (up to 20,000 rows). Evidently picks Wasserstein distance (numeric) and
  Jensen-Shannon (categorical), drift at >= 0.1. Output: one HTML report per month,
  `reports/drift/summary.md`, and a per-column CSV. Runs in ~22 s.
- **Findings (4 to 8 of 19 columns drift each month; dataset-level drift never triggered):**
  1. `card_txn_number` drifts most (distance up to 2.9) **by construction**: it is a running
     count, so later months always exceed the training range. The model is extrapolating on it;
     a production model would cap it or use a recency-bounded count instead.
  2. Velocity features (`card_txn_count_7d/24h`, `card_amt_sum_24h`, `secs_since_last_txn`)
     drift most in December, when transaction volume doubles for the holidays.
  3. `day_of_week` drifts every month: a **simulator calendar artefact**. Weekday shares are
     shifted by exactly one day between Mar 2019 and Feb 2020 (busiest Sun/Mon/Sat) versus the
     other periods (busiest Mon/Tue/Sun), because the generator's 2012 calendar was relabelled
     onto 2019-2020 across a leap day (same root cause as D1.2). Most of the reference falls in
     the shifted year.
  4. The **score** distribution barely drifts (distance <= 0.05; mean score 0.0029 in December
     vs 0.0064 in the reference), so the model's outputs are stable even where inputs move; the
     larger issue remains the threshold drift found in D6.4 as the fraud rate changes.
- Harmless `invalid value encountered in divide` warnings come from Evidently on columns with
  zero variance in a month's sample.

---

## Phase 10: Testing and CI

### D10.1 What the tests cover (180 tests, 92% line + branch coverage)

| Area | Tests |
|---|---|
| Features | hand-computed window edges, ties, no-history and leap-day ages; haversine; parity of Python vs SQL on the synthetic fixture (CI) and on 40 real cards (locally) |
| Thresholds and gate | cost/F1 thresholds on worked examples; every promotion rule and tie-break; a registry scenario on a throwaway SQLite MLflow |
| API | every endpoint with FastAPI's TestClient and a tiny model: validation (422), out-of-order (409), duplicates, batch limits, metrics, readiness when loading fails |
| LLM | Gemini always mocked: cache, retry policy, permanent errors, invalid replies, fallback wording |
| Dashboard | the real Streamlit script run headlessly against the in-process API (Streamlit's AppTest) |
| Workflow | one end-to-end test runs the real CLI in a throwaway project: download (detect) -> validate -> convert -> train -> register -> promote -> evaluate (once, refused twice) -> export -> explain -> SHAP -> fairness -> drift -> build-state (0 parity mismatches) -> Kaggle bundle -> production entry point |

- **Not unit-tested:** `load_db` and `features/build`, which need a live PostgreSQL; they are
  exercised by the real pipeline runs (row counts and parity checked there).
- **Bugs the new tests found:** a split without fraud crashed evaluation with a bare
  `ZeroDivisionError`; training now stops with a clear message pointing at the split dates.
- **Offline by design:** a clean copy of the repository (no data, no `.env`, no services) gives
  179 passed and 1 skipped (real-data parity).

### D10.2 Coverage measured by path

- `[tool.coverage.run] source = ["src/fraudlens"]` instead of the package name: Streamlit runs
  the dashboard by file path, which package-based coverage did not trace (0% -> 94%).

### D10.3 CI pipeline (GitHub Actions)

- On push to `main`, pull requests and manual runs: `uv sync --frozen` (Python 3.11, cached),
  `ruff check`, `ruff format --check`, `pytest` with coverage (summary written to the run page,
  `coverage.xml` uploaded), then both Docker images built with BuildKit layer caching (never
  pushed). Least-privilege token (`contents: read`); superseded runs are cancelled.

## Phase 11: free public demo

### D11.1 Where each piece runs

| Piece | Host | Why |
|---|---|---|
| Model, threshold, card state, demo set | Hugging Face **model repo** | free, versioned (each upload is a Git commit), public model card; no MLflow server needed in the cloud |
| Scoring API | **Render** free web service (Docker) | runs the exact API image from this repo; deploys only after CI passes (`autoDeployTrigger: checksPass` in `render.yaml`) |
| Dashboard | **Streamlit Community Cloud** | free, deploys from this GitHub repo on every push; installs only `src/fraudlens/dashboard/requirements.txt` (Streamlit, Plotly, httpx, pandas, pinned to the lock) because the dashboard only talks to the API over HTTP |

- The API downloads the Hub repo at startup (`MODEL_SOURCE=huggingface`), so a new model is
  published with `fraudlens export-model; fraudlens push-model` and picked up by the next restart,
  with no rebuild. Hugging Face replaced the local MLflow registry for the demo, not for training.
- The model is loaded straight from its **skops** file with the same trusted-type list as
  training (scores identical to the MLflow loader on 5,000 rows), so the cloud API never imports
  MLflow.
- Infrastructure as code: `render.yaml` (Render Blueprint) defines the service; secrets
  (`GOOGLE_API_KEY`, `HF_TOKEN`) are `sync: false`, entered once in Render, never in Git.
  The dashboard's `API_URL` is a Streamlit Cloud secret (top-level secrets are exposed as
  environment variables, which the dashboard already reads).
- **Plan change:** the plan put the dashboard on a Hugging Face Space, but Hugging Face now
  requires a paid PRO subscription to run Docker or Gradio Spaces on its free CPU (the upload was
  refused with HTTP 402; only static Spaces are free). Streamlit Community Cloud is free for
  public apps and builds straight from GitHub, so no image or upload command is needed.

### D11.2 Fitting the free tier (measured locally with `--cpus 0.1 --memory 512m`)

- **Memory:** ~410 MB peak while loading, ~350 MB serving (limit 512 MB). Possible because the
  API image has no MLflow server, matplotlib is imported lazily, and SHAP's explainer is built
  on first use.
- **Cold start:** ~150 s from container start to ready at 0.1 CPU (importing NumPy, pandas,
  LightGBM and SHAP, and rebuilding 983 card states) plus the Hub download. Render free
  instances sleep after 15 minutes idle, so the README warns visitors, and the dashboard waits
  up to 3 minutes with a "waking up" message instead of failing after 60 s.
- **First explanation:** ~38 s once per start in the local 0.1-CPU simulation (SHAP
  TreeExplainer setup), but 6 s on Render, whose CPU share bursts above 0.1. Not pre-warmed at
  startup, which would slow every wake-up for visitors who only score.
- **Measured on Render (live):** `/predict` ~0.1 s and `/predict_explained` ~1.7 s end to end
  (Gemini note included, `explanation_source: llm`), from India to the Singapore region.
- The container listens on `$PORT` (Render sets it), falling back to 8000 locally and in Compose.

### D11.3 What is public

- Uploaded: the model, its metadata (metrics, threshold, global feature importance, fairness summary),
  the feature list, card state and 924 demo transactions. All of it derives from the
  synthetic, CC0 Sparkov data; no real cardholder data exists anywhere in the project.
- Not uploaded: the raw dataset, `.env`, MLflow runs. The Hugging Face token is read from
  `.env` and only used to authenticate.

