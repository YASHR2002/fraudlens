# FraudLens: Explainable Real-Time Fraud Detection Platform

## Project Plan and Implementation Guide (for Claude Code)

---

## 0. Instructions for Claude Code (read this first)

You are implementing an end-to-end, portfolio-grade fraud detection project for a data scientist with 2 years of experience who is targeting Data Science / AI roles in banking and fintech. The project must look and behave like a real production system, and every decision should be explainable in a job interview.

**How to work:**

1. Implement this plan **one phase at a time, in order**. Each phase ends with a **CHECKPOINT**.
2. At every checkpoint, stop, summarize what you built, show how to verify it (commands to run, what output to expect), and **wait for my approval** before starting the next phase.
3. If something in this plan is wrong, outdated, or impossible (for example a library API has changed, a dataset column name differs, a model name no longer exists), **tell me and propose a fix** instead of silently working around it.
4. Never commit the dataset, `.env`, API keys, model binaries, or MLflow artifacts to Git.
5. Before running any command that creates paid cloud resources (Phase 12, AWS), **show me the command and wait for my confirmation**. Prefer giving me the commands to run myself.
6. Keep code clean and professional: type hints, docstrings on public functions, logging instead of print in library code, a single config file for settings, fixed random seeds.
7. Notebooks are for exploration only. Anything reused must live in the `src/` package and be tested.

**Hardware and OS constraints (important):**

* Windows 11 laptop with **8 GB RAM**. Docker Desktop runs on WSL2.
* Docker is capped at **3 GB RAM** via `.wslconfig` (see Phase 0).
* VS Code, Git, and Docker Desktop are already installed.
* Never design a step that needs all containers running while also training a model. Use Docker Compose **profiles** so each phase starts only what it needs.
* Commands in docs must work in **PowerShell**. Avoid Makefiles and bash-only scripts for the local workflow; use a cross-platform Python CLI instead (see Phase 0). Shell scripts are fine only for things that run inside Linux containers or on the AWS server.

---

## 1. Project Summary

**Business problem:** A card issuer needs to flag fraudulent credit card transactions in real time, without overwhelming its fraud analysts with false alarms, and must be able to explain every decision (regulators and customers ask "why was my card blocked?").

**What the system does:**

1. Ingests transaction data into PostgreSQL and builds behavioral features in SQL (spending velocity, amount vs. customer's normal spend, distance from home, time of day, and so on).
2. Trains and compares models, tracks every experiment in MLflow, and promotes a model to production only if it passes fixed business rules (a promotion gate).
3. Chooses the decision threshold by **minimizing total business cost** (missed fraud vs. analyst review cost), not by F1.
4. Serves predictions through a FastAPI service, with a fast endpoint and a slower "explained" endpoint.
5. Explains each flagged transaction with SHAP, then turns the SHAP output into a plain-English analyst note using Google Gemini.
6. Monitors the live service (Prometheus + Grafana) and data drift (Evidently AI).
7. Runs tests and linting in GitHub Actions, deploys a free public demo (Render + Hugging Face), and is deployed once to AWS (EC2 + ECR + S3) as a documented production blueprint.
8. Includes a fairness check of model performance across gender and age groups.

---

## 2. Tech Stack

| Area | Tool |
|---|---|
| Language | Python 3.11 |
| Package / env management | `uv` with `pyproject.toml` (fallback: `venv` + `pip`) |
| Database | PostgreSQL 16 (Docker) |
| Data processing | pandas, pyarrow (Parquet), SQLAlchemy, psycopg |
| Modelling | scikit-learn, XGBoost, LightGBM |
| Tuning | Optuna (run on Kaggle Notebooks) |
| Experiment tracking / registry | MLflow (Docker, SQLite backend, served artifacts) |
| Explainability | SHAP |
| LLM | Google Gemini via the `google-genai` SDK, free API key from Google AI Studio |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit, Plotly |
| Monitoring | Prometheus, Grafana, `prometheus-fastapi-instrumentator`, Evidently AI |
| CLI | Typer |
| Testing / quality | pytest, ruff, pre-commit |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose (with profiles) |
| Model hosting | Hugging Face Hub |
| Public demo | Render (API), Hugging Face Spaces or Streamlit Community Cloud (dashboard) |
| Cloud blueprint | AWS EC2 (t3.micro), ECR, S3, IAM |

Pin versions in `pyproject.toml`. Check current stable versions when you start rather than relying on memory.

---

## 3. Dataset

**Name:** Sparkov synthetic credit card transactions ("Credit Card Transactions Fraud Detection Dataset" by Kartik Shenoy on Kaggle; the slug is believed to be `kartik2112/fraud-detection`, so verify it).

**Files:** `fraudTrain.csv` (~1.3M rows) and `fraudTest.csv` (~555K rows). Roughly 0.5% fraud. US transactions spanning about 2019 to 2020; the test file continues in time after the train file. Verify all of this after download and report actual numbers.

**Expected columns** (verify and adapt):
`trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud` (plus an unnamed index column to drop).

**Download:** Support both options in docs:
* Kaggle API (`kaggle datasets download ...`), which needs `kaggle.json` in `%USERPROFILE%\.kaggle\`.
* Manual download into `data/raw/`.

**Data rules:**
* `data/` is git-ignored entirely, except a tiny synthetic fixture used by tests (`tests/fixtures/sample_transactions.csv`, about 200 rows, generated by a script, containing a few fraud rows).
* **Personally identifying columns are never used as model features:** `first, last, street, cc_num, trans_num`. `cc_num` is used only as a grouping key (card ID) for features.
* **Protected attributes:** `gender` is **excluded from model features** and used only for the fairness audit (Phase 7). Age is used as a feature (it is a meaningful fraud signal) but is also audited. Document this decision in the README.

**Honest limitation to document:** the data is synthetic, so fraud patterns are cleaner than reality and scores will be high. The project's value is in engineering, evaluation discipline, explainability, and MLOps, not the headline score.

---

## 4. Repository Structure

```
fraudlens/
├── .github/workflows/ci.yml
├── configs/
│   ├── config.yaml              # all settings: paths, split dates, costs, promotion rules
│   └── best_params.json         # output of Kaggle tuning (Phase 5)
├── data/                        # git-ignored
│   ├── raw/
│   ├── processed/               # parquet files
│   └── reference/               # drift reference sample
├── docker/
│   ├── api.Dockerfile
│   ├── dashboard.Dockerfile
│   ├── postgres/init.sql
│   ├── prometheus/prometheus.yml
│   └── grafana/provisioning/    # datasource + dashboard JSON
├── docs/
│   ├── setup_windows.md
│   ├── architecture.md
│   ├── model_card.md
│   ├── decisions.md             # design decisions and trade-offs log
│   ├── aws_deployment.md
│   └── screenshots/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── kaggle_tuning.ipynb      # runs on Kaggle, not locally
├── reports/                     # git-ignored generated reports (drift, evaluation)
├── sql/
│   ├── 01_create_tables.sql
│   └── 02_features.sql
├── src/fraudlens/
│   ├── __init__.py
│   ├── cli.py                   # Typer CLI: `python -m fraudlens <command>`
│   ├── config.py
│   ├── data/                    # download, validate, convert, load_to_postgres
│   ├── features/                # sql runner, online (Python) feature computation, feature names
│   ├── models/                  # train, evaluate, threshold, promote, registry helpers
│   ├── explain/                 # shap_explainer, llm_explainer, prompts, fallback
│   ├── fairness/                # group metrics
│   ├── monitoring/              # drift reports, traffic replay
│   ├── api/                     # FastAPI app, schemas, model loader, online state
│   └── dashboard/               # Streamlit app
├── tests/
│   ├── fixtures/
│   ├── test_features.py
│   ├── test_feature_parity.py
│   ├── test_threshold.py
│   ├── test_promotion.py
│   ├── test_api.py
│   └── test_llm_explainer.py
├── .env.example
├── .gitattributes               # force LF endings for *.sh, Dockerfiles, yml
├── .gitignore
├── .pre-commit-config.yaml
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 5. Docker Compose Design (8 GB RAM)

Use **profiles** so each phase starts only what it needs. Put a memory limit on every service.

| Service | Image | Profiles | Memory limit |
|---|---|---|---|
| `postgres` | `postgres:16` | `data`, `train` | 1 GB (tune `shared_buffers=256MB`, `work_mem=32MB`) |
| `mlflow` | official MLflow image | `train`, `serve` | 512 MB |
| `api` | built from `docker/api.Dockerfile` | `serve`, `monitoring` | 1 GB |
| `dashboard` | built from `docker/dashboard.Dockerfile` | `serve` | 512 MB |
| `prometheus` | `prom/prometheus` | `monitoring` | 256 MB |
| `grafana` | `grafana/grafana` | `monitoring` | 256 MB |

MLflow server command (adapt flags to the current MLflow version):
`mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db --artifacts-destination /mlflow/artifacts --serve-artifacts`
with a named volume for `/mlflow`, so notebooks and scripts on the host can log artifacts through `http://localhost:5000`.

Usage examples to document in the README:
```powershell
docker compose --profile data up -d        # Phase 1-2
docker compose --profile train up -d       # Phase 4, 6
docker compose --profile serve up -d       # Phase 8
docker compose --profile monitoring up -d  # Phase 9
docker compose down                        # stop everything
```

---

## 6. Configuration

`configs/config.yaml` holds everything tunable. Minimum contents:

```yaml
seed: 42
paths: {raw: data/raw, processed: data/processed, reference: data/reference}
split:
  # time-based; Claude Code: set actual dates after inspecting data
  train_end: "YYYY-MM-DD"          # train = fraudTrain up to this date
  validation_end: "YYYY-MM-DD"     # validation = rest of fraudTrain (last ~2 months)
  # test = all of fraudTest, used ONCE for final reporting
costs:
  false_positive_cost: 5.0         # analyst review cost per false alarm (USD)
  false_negative_cost: "amount"    # a missed fraud costs the transaction amount
promotion:
  min_recall: 0.80
  min_precision: 0.50
  require_pr_auc_at_least_champion: true
shap:
  global_sample_size: 10000
llm:
  provider: gemini
  model_env_var: GEMINI_MODEL
  timeout_seconds: 15
```

`.env.example`:
```
POSTGRES_USER=fraudlens
POSTGRES_PASSWORD=change_me
POSTGRES_DB=fraudlens
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
MLFLOW_TRACKING_URI=http://localhost:5000
GOOGLE_API_KEY=
GEMINI_MODEL=            # Claude Code: set to a current free-tier Flash model name and tell me which one
FRAUD_THRESHOLD=          # filled in after Phase 6
MODEL_SOURCE=mlflow      # mlflow | huggingface
HF_REPO_ID=
HF_TOKEN=
```

---

## 7. Phases

### Phase 0: Project Setup

**Tasks:**
1. Create the repository structure above, `pyproject.toml` with dependency groups (`core`, `train`, `api`, `dashboard`, `dev`), `.gitignore`, `.gitattributes`, `.pre-commit-config.yaml` (ruff lint + format).
2. Create `src/fraudlens/cli.py` (Typer) as the single entry point. Commands get added phase by phase.
3. Write `docs/setup_windows.md` explaining:
   * Creating `C:\Users\<name>\.wslconfig` with `memory=3GB`, `processors=2`, `swap=4GB`, then `wsl --shutdown` and restarting Docker Desktop.
   * Installing `uv` on Windows and creating the environment.
   * Getting a free Gemini API key from Google AI Studio and putting it in `.env`.
   * Kaggle API setup.
4. Create `docker-compose.yml` with all services and profiles (later phases fill in the Dockerfiles).
5. Create a README skeleton (title, one-paragraph problem statement, "status: in progress").
6. Initialize Git, make the first commit.

**Acceptance criteria:** `uv sync` works; `python -m fraudlens --help` runs; `docker compose --profile data up -d` starts PostgreSQL; `pre-commit run --all-files` passes.

**CHECKPOINT 0.**

---

### Phase 1: Data Ingestion and Validation

**Tasks:**
1. `fraudlens download-data`: download via Kaggle API or detect manually placed files.
2. `fraudlens validate-data`: check schema (columns, types), null counts, duplicates on `trans_num`, value ranges (amount > 0, valid lat/long), fraud rate, date range of each file. Write a short report to `reports/data_quality.md` and fail loudly on schema errors.
3. `fraudlens convert-data`: CSV to Parquet with memory-efficient types (categories for text columns, float32/int32 where safe, parsed datetime). Read CSVs in chunks. Log memory before/after.
4. `fraudlens load-db`: create tables (`sql/01_create_tables.sql`) and load both files into one `transactions` table with a `source` column (`train`/`test`), using PostgreSQL `COPY` for speed. Add indexes on `(cc_num, trans_ts)`.
5. Generate the small synthetic test fixture with a script (not copied from the real data).

**Acceptance criteria:** row counts in PostgreSQL match the files; data quality report generated; Parquet files exist and are much smaller than the CSVs.

**CHECKPOINT 1.** Report actual row counts, fraud rates, and date ranges.

---

### Phase 2: SQL Feature Engineering

All features must use **only information available before the current transaction** (no leakage). Compute features over train and test together, ordered by time, so test transactions correctly see earlier history (which is legitimate past information).

**Features to build in `sql/02_features.sql` (materialize into a `features` table):**

| Feature | Description |
|---|---|
| `amt` | transaction amount |
| `log_amt` | log(1 + amount) |
| `category` | merchant category |
| `hour`, `day_of_week`, `is_night` | time of day; night = 22:00–05:59 |
| `age_at_txn` | age in years from `dob` |
| `log_city_pop` | log of city population |
| `distance_km` | haversine distance between customer and merchant |
| `card_txn_count_1h`, `_24h`, `_7d` | prior transactions on the card in the time window (use `RANGE BETWEEN INTERVAL ... PRECEDING AND ... PRECEDING`, excluding the current row) |
| `card_amt_sum_24h` | prior spend on the card in the last 24h |
| `card_avg_amt_prior` | average of all prior amounts on the card (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`) |
| `amt_to_card_avg_ratio` | current amount / prior average (handle no history) |
| `secs_since_last_txn` | time since the card's previous transaction |
| `is_new_category_for_card` | first time this card uses this category |
| `card_txn_number` | how many transactions the card has made before (history depth) |

**Do NOT build:** target-encoded features or any feature derived from `is_fraud` of other rows (leakage risk), and nothing using `gender`, names, street, or `trans_num`.

**Tasks:**
1. `fraudlens build-features`: run the SQL, then export the features table to `data/processed/features.parquet`.
2. Write `src/fraudlens/features/feature_names.py` with a mapping from technical names to human-readable descriptions (used later by SHAP plots and the LLM prompt).
3. Write `src/fraudlens/features/online.py`: a **pure-Python** implementation of the same features for a single incoming transaction, given a per-card state object (recent timestamps and amounts, running average, categories seen, last timestamp). This is used by the API.
4. Write `tests/test_feature_parity.py`: for a sample of cards, replay their transactions through the Python online path and assert results match the SQL output (within float tolerance). This guards against **training-serving skew**. Explain this in `docs/decisions.md`.

**Acceptance criteria:** features table built; parity test passes; leakage check documented (spot-check a few cards manually and show it).

**CHECKPOINT 2.**

---

### Phase 3: EDA

`notebooks/01_eda.ipynb` covering: class imbalance, fraud rate by category, hour, age band, amount distribution (fraud vs. legit), distance and velocity features vs. fraud, fraud rate over time (for later drift discussion). Keep each chart with a 1–2 sentence business takeaway. Load only needed columns from Parquet.

**CHECKPOINT 3.** Summarize the 5 most important findings.

---

### Phase 4: Baseline Models with MLflow

**Split:** time-based, from `config.yaml`: train / validation from `fraudTrain`, test = `fraudTest`. The **test set is not touched until Phase 6's final report.**

**Tasks:**
1. `src/fraudlens/models/train.py` with a reusable training function that logs to MLflow: params, metrics, feature list, data split dates, dataset row counts and fraud rates, the model, and evaluation plots (PR curve, confusion matrix at the chosen threshold, feature importance).
2. Models to compare (logged as separate MLflow runs under one experiment):
   * Logistic regression (scaled, `class_weight="balanced"`) as the baseline.
   * XGBoost with `scale_pos_weight`, `tree_method="hist"`.
   * LightGBM with class weighting.
3. Metrics on validation: **PR-AUC (primary)**, ROC-AUC (secondary, for reference), precision, recall, and F1 at the threshold, plus **total business cost** at the threshold.
4. `src/fraudlens/models/threshold.py`: choose the threshold that **minimizes total cost** on validation, where each false negative costs its transaction amount and each false positive costs `false_positive_cost`. Also report the threshold that maximizes F1, for comparison. Unit-test this module.
5. Explicitly explain in `docs/decisions.md` why accuracy is meaningless here and why PR-AUC and cost are used.
6. Set up `notebooks/02_baselines.ipynb` to call the `src` functions (not duplicate them).

**Acceptance criteria:** three runs visible in the MLflow UI at `localhost:5000` with all artifacts; comparison table printed.

**CHECKPOINT 4.**

---

### Phase 5: Hyperparameter Tuning on Kaggle

My laptop has 8 GB RAM, so heavy tuning runs on a free Kaggle Notebook.

**Tasks:**
1. Create `notebooks/kaggle_tuning.ipynb` that:
   * Uploads/uses `features.parquet` as a private Kaggle dataset (document the steps for me).
   * Uses the **same split dates** from config (copy them into the notebook's first cell).
   * Runs Optuna (around 50 trials, with pruning) for XGBoost and LightGBM, optimizing validation PR-AUC.
   * Saves the best params of each model as JSON.
2. I will download the JSON into `configs/best_params.json`.
3. Document in `docs/decisions.md` why tuning was done off-laptop and how reproducibility is preserved.

**Acceptance criteria:** notebook runs end to end on Kaggle; `best_params.json` committed.

**CHECKPOINT 5.**

---

### Phase 6: Final Training, Promotion Gate, and Test Report

**Tasks:**
1. `fraudlens train --use-best-params`: retrain tuned XGBoost and LightGBM locally on the train split, log to MLflow, and register both in the MLflow Model Registry.
2. `fraudlens promote`: apply the promotion gate on **validation** metrics at the cost-optimal threshold:
   * recall ≥ `min_recall`, precision ≥ `min_precision`,
   * PR-AUC ≥ current champion's PR-AUC (if a champion exists).

   The winner gets the registry alias **`champion`** (use aliases, not the deprecated stages). Log the decision and reasons as a run tag and print a clear summary. Rejected models stay registered with the reason recorded. Unit-test the gate logic.
3. Save the chosen threshold with the model (as a registered model version tag) and also write it to `.env` guidance.
4. `fraudlens evaluate --final`: evaluate the champion **once** on the test set. Produce `reports/final_evaluation.md` with PR-AUC, precision/recall at threshold, confusion matrix, fraud caught, false alarms, total cost vs. two baselines ("flag nothing" and "flag everything over $X"). This feeds the README results table.

**Acceptance criteria:** a model holds the `champion` alias; final evaluation report generated; nothing in Phases 4–6 used the test set for decisions.

**CHECKPOINT 6.**

---

### Phase 7: Explainability, LLM Explanations, and Fairness Audit

**SHAP (`src/fraudlens/explain/shap_explainer.py`):**
1. `TreeExplainer` on the champion.
2. Global: SHAP summary (beeswarm) and bar plot on a 10,000-row sample; log as MLflow artifacts.
3. Local: function returning the top N contributing features for one transaction, with feature values, SHAP values, direction (toward / away from fraud), and human-readable names.

**LLM explainer (`src/fraudlens/explain/llm_explainer.py`):**
1. Use the `google-genai` SDK; model name from `GEMINI_MODEL`; API key from `GOOGLE_API_KEY`.
2. Prompt (in `prompts.py`) provides: the fraud score, threshold, decision, top SHAP factors in plain words with actual values (for example "amount $1,250 is 11.4x this card's average of $109"), and transaction context (category, hour, distance). Exclude names and card number.
3. Instruct the model to: use **only** the facts provided, not invent information, not mention protected attributes, write for a fraud analyst in 3–5 sentences, and return **JSON** with `summary`, `key_reasons` (list), `recommended_action` (`approve` / `review` / `block`), validated by a Pydantic model.
4. Robustness: timeout, one retry, then a **deterministic template fallback** built from the SHAP factors (so the API never fails because the LLM is down or rate-limited). Cache explanations by `trans_num`. Response includes `explanation_source: "llm" | "fallback"`.
5. Tests mock the Gemini client (no real API calls in tests or CI).

**Fairness audit (`src/fraudlens/fairness/`):**
1. On the test set, compute recall, precision, and false positive rate by `gender` and by age band (for example under 30, 30–50, 50–70, 70+).
2. Report gaps in `reports/fairness.md` and in the model card, with a short discussion. No automatic "fixing", just measurement and honest commentary.

**Acceptance criteria:** global SHAP plots in MLflow; `fraudlens explain --trans-num <id>` prints SHAP factors and an LLM explanation; fallback works when the API key is removed; fairness report generated.

**CHECKPOINT 7.**

---

### Phase 8: Scoring API and Dashboard

**Online feature state:** Sparkov has a small number of cards (around 1,000), so the API keeps a per-card state store in memory, initialized from a snapshot built from history up to the end of the training period (`fraudlens build-state` writes `data/processed/card_state.parquet` or similar). Each scored transaction updates the state. Document that in production this would be Redis or a feature store.

**FastAPI app (`src/fraudlens/api/`):**

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness |
| `GET /ready` | model and state loaded |
| `POST /predict` | raw transaction in → features computed online → score, decision, threshold, model version, latency |
| `POST /predict/batch` | up to 1,000 transactions |
| `POST /predict_explained` | same as `/predict` plus SHAP factors and LLM explanation |
| `GET /model-info` | model name, version, alias, threshold, training date, metrics |
| `GET /metrics` | Prometheus metrics |

* Pydantic request/response schemas with validation and example payloads (visible in `/docs`).
* Model loading controlled by `MODEL_SOURCE`: `mlflow` (load alias `champion`) or `huggingface` (download from `HF_REPO_ID`). Threshold from env var, falling back to the value stored with the model.
* Log each request's latency. Target: `/predict` under 20 ms on the laptop. Measure and report.
* `docker/api.Dockerfile`: slim Python base, multi-stage if helpful, non-root user, only API dependencies.

**Streamlit dashboard (`src/fraudlens/dashboard/`):**
1. **Score a transaction:** pick a test-set transaction (filter to show fraud and legit examples) or edit fields manually → call the API → show score gauge, decision, SHAP waterfall, and the LLM analyst note.
2. **Model overview:** metrics from `/model-info`, final evaluation numbers, global SHAP plot, fairness summary.
3. The dashboard talks to the API only through HTTP (`API_URL` env var).

**Acceptance criteria:** `docker compose --profile serve up` runs API + dashboard + MLflow; API tests pass with FastAPI `TestClient`; latency measured.

**CHECKPOINT 8.**

---

### Phase 9: Monitoring and Drift

**Tasks:**
1. Add `prometheus-fastapi-instrumentator` plus custom metrics: total predictions, fraud flags, a histogram of fraud scores, LLM call count, LLM fallback count, LLM latency.
2. Prometheus config scraping the API; Grafana with **provisioned** datasource and a dashboard JSON (requests/sec, p50/p95 latency, error rate, flag rate, score distribution, LLM fallback rate) so it appears automatically on startup.
3. `fraudlens replay --speed <n>`: replays test-set transactions in time order against the API to simulate live traffic (use a modest speed so the laptop copes).
4. `fraudlens drift-report`: Evidently AI comparing a reference sample (from training data, saved in `data/reference/`) with each month of replayed/test data; write HTML reports to `reports/drift/` and a short summary of which features drifted.
5. Screenshots of Grafana and a drift report into `docs/screenshots/`.

**Note for 8 GB RAM:** the `monitoring` profile runs API + Prometheus + Grafana only (PostgreSQL and MLflow stopped; the API loads the model from a local exported copy or Hugging Face). Add a `fraudlens export-model` command that copies the champion out of MLflow into `models/champion/` for this.

**Acceptance criteria:** Grafana dashboard shows live data during a replay; drift reports generated.

**CHECKPOINT 9.**

---

### Phase 10: Testing and CI

**Tasks:**
1. Make sure tests cover: feature calculations (including haversine and no-history edge cases), feature parity, threshold selection, promotion gate, API endpoints (with a tiny dummy model fixture), LLM explainer (mocked, including fallback path), schema validation.
2. Tests must run **without** the real dataset, Docker, PostgreSQL, MLflow, or internet. Use fixtures.
3. `.github/workflows/ci.yml`: on push and PR → set up Python with uv → ruff lint and format check → pytest with coverage → build the API Docker image (no push).
4. Add a CI status badge and coverage number to the README.

**Acceptance criteria:** CI green on GitHub.

**CHECKPOINT 10.**

---

### Phase 11: Free Public Demo

**Tasks:**
1. `fraudlens push-model`: upload the exported champion model, threshold, feature list, and card state snapshot to a Hugging Face Hub model repo (I will provide `HF_TOKEN`).
2. Deploy the API Docker image to **Render** (free web service) with `MODEL_SOURCE=huggingface`; set secrets in the Render dashboard (`GOOGLE_API_KEY`, `HF_REPO_ID`, `HF_TOKEN` if the repo is private). Write `render.yaml` if useful.
3. Deploy the Streamlit dashboard to **Hugging Face Spaces** (or Streamlit Community Cloud) pointing at the Render API URL.
4. Note in the README that Render's free tier sleeps when idle, so the first request may take some time.
5. Optionally, a GitHub Actions job that triggers a Render deploy hook after CI passes on `main`.

**Acceptance criteria:** public URLs for the API docs and the dashboard work end to end, including an LLM explanation.

**CHECKPOINT 11.**

---

### Phase 12: AWS Production Blueprint (deploy once, document, tear down)

**Rules:** I am on the AWS Free plan using credits. **Do not run any AWS command that creates resources without showing it to me first.** Prefer writing the commands into `docs/aws_deployment.md` for me to run. I need to install the AWS CLI on Windows and configure it with an IAM user that is not root.

**Architecture:**
* **ECR:** private repository for the API image.
* **S3:** bucket for the exported model, card state, and reference data.
* **IAM role + instance profile:** EC2 can read S3 and pull from ECR (least privilege, no stored keys on the server).
* **EC2:** t3.micro, Amazon Linux 2023, 20 GB gp3 disk, **2 GB swap file**, Docker + Docker Compose installed via a user-data script.
* **Security group:** API port open publicly; Grafana and Prometheus ports restricted to my IP only; SSH restricted to my IP.
* Runs a `docker-compose.aws.yml`: API + Prometheus + Grafana, with the API loading the model from S3 (add `MODEL_SOURCE=s3` support).

**`docs/aws_deployment.md` must include:**
1. Pre-flight: confirm Free plan, set a **billing alarm/budget**, pick region (ap-south-1, Mumbai).
2. Step-by-step commands: create the ECR repo, build and push the image from Windows, create the S3 bucket and upload artifacts, create the IAM role, launch EC2 with user data, verify.
3. Things to avoid because they cause unexpected charges: NAT gateways, unattached Elastic IPs, larger instance types, leaving resources running.
4. Verification checklist + screenshots to capture (EC2 console, ECR image, S3 bucket, API `/docs` on the EC2 public IP, Grafana).
5. **Teardown checklist** with commands: terminate EC2, delete EBS volumes and snapshots, delete ECR repo, empty and delete S3 bucket, delete IAM role, release any Elastic IPs, then confirm in Billing that nothing is running.
6. An architecture diagram (Mermaid) of the AWS setup.

**Acceptance criteria:** deployment verified with screenshots, everything torn down, doc complete.

**CHECKPOINT 12.**

---

### Phase 13: Documentation and Polish

**README.md structure:**
1. Title, one-line pitch, badges (CI, coverage, Python version), links to live demo and API docs.
2. **Business problem** (short, in plain language).
3. **Architecture diagram** (Mermaid): data → PostgreSQL → features → training → MLflow registry → API → dashboard / monitoring; plus the deployment targets.
4. **Results table:** test-set PR-AUC, recall, precision, fraud caught, false alarms, total cost vs. baselines; plus API latency.
5. **Key design decisions** (link to `docs/decisions.md`): time-based split, leakage-free SQL features, training-serving parity test, cost-based threshold, promotion gate, separate fast/explained endpoints, LLM with grounded prompt and fallback, gender excluded from features, phased Docker profiles for an 8 GB machine, tuning on Kaggle.
6. **Explainability example:** a real flagged transaction with its SHAP chart and LLM note.
7. **Fairness findings** (short).
8. **Monitoring** screenshots.
9. **How to run locally** (PowerShell commands, phase by phase).
10. **Limitations and future work:** synthetic data, in-memory state instead of Redis, no real-time streaming (Kafka), retraining not automated, possible next steps.

**`docs/model_card.md`:** intended use, data, features, metrics, threshold and cost assumptions, fairness results, limitations, ethical considerations.

**Acceptance criteria:** a recruiter can understand the project from the README in under 2 minutes; a developer can run it from the docs.

**FINAL CHECKPOINT.** Produce a short list of 3–4 resume bullet points based on the actual measured results, and a list of likely interview questions about this project with brief answer notes.

---

## 8. Global Quality Rules

* Fixed seeds everywhere; log package versions to MLflow.
* No test-set usage for any decision before Phase 6's final evaluation.
* No secrets in code, logs, Git history, Docker images, or MLflow.
* All settings in `configs/config.yaml` or `.env`, never hard-coded.
* Every CLI command logs what it did and how long it took.
* Keep memory in mind: load only needed columns, use Parquet, sample for SHAP, stream CSVs in chunks.
* Keep `docs/decisions.md` updated as you go: each entry states the decision, alternatives considered, and why.
