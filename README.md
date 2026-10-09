# FraudLens

[![CI](https://github.com/YASHR2002/fraudlens/actions/workflows/ci.yml/badge.svg)](https://github.com/YASHR2002/fraudlens/actions/workflows/ci.yml)
![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)

**Explainable real-time credit card fraud detection: from SQL features to a monitored, explained scoring API.**

> **Status: in progress.** Built phase by phase; see [PROJECT_PLAN.md](PROJECT_PLAN.md).

## Live demo

- **Dashboard:** https://huggingface.co/spaces/YASHR2002/fraudlens (Hugging Face Spaces)
- **Scoring API:** https://fraudlens-api.onrender.com/docs (Render free tier)
- **Model:** https://huggingface.co/YASHR2002/fraudlens-model

> The API runs on Render's free tier (0.1 CPU, 512 MB), which **sleeps after 15 minutes without
> traffic**. The first request after that wakes it up and loads the model, which takes about
> **2-3 minutes**; the dashboard waits and shows a message meanwhile. The first explained
> prediction then takes ~40 s more (building the SHAP explainer once), after which scoring is
> ~50 ms and explanations ~1 s plus the LLM call.

## Business problem

A card issuer must flag fraudulent transactions in real time without burying its fraud
analysts in false alarms, and it must be able to explain every decision, because regulators
and customers ask "why was my card blocked?". FraudLens builds leakage-free behavioural
features in PostgreSQL, picks the decision threshold that minimises total business cost
(missed fraud vs. analyst review time) instead of F1, promotes models only through a
fixed promotion gate, and serves scores through a FastAPI service that explains each flagged
transaction with SHAP and a plain-English analyst note written by an LLM, with a
deterministic fallback for when the LLM is unavailable.

## Getting started

Windows setup (WSL2 memory cap, `uv`, API keys, Kaggle): [docs/setup_windows.md](docs/setup_windows.md).

```powershell
uv sync                                   # create .venv and install everything
uv run python -m fraudlens --help         # CLI entry point
docker compose --profile data up -d       # start PostgreSQL
```

Docker Compose profiles start only what each phase needs, so the stack fits on an 8 GB laptop:

```powershell
docker compose --profile data up -d        # Phase 1-2: PostgreSQL
docker compose --profile train up -d       # Phase 4, 6: PostgreSQL + MLflow
docker compose --profile serve up -d       # Phase 8: MLflow + API + dashboard
docker compose --profile monitoring up -d  # Phase 9: API + Prometheus + Grafana
docker compose --profile "*" down           # stop everything (a plain `down` skips profiled services)
```

### Pipeline (PowerShell, from the project folder)

```powershell
uv run fraudlens validate-data; uv run fraudlens convert-data      # data quality + Parquet
docker compose --profile data up -d; uv run fraudlens load-db       # PostgreSQL
uv run fraudlens build-features                                     # SQL features -> Parquet
docker compose --profile train up -d mlflow                         # MLflow at http://127.0.0.1:5000
uv run fraudlens train --use-best-params; uv run fraudlens promote  # tuned models -> @champion
uv run fraudlens evaluate --final                                   # test set, once
uv run fraudlens build-state                                        # card state for the API
docker compose --profile serve up -d                                # MLflow + API + dashboard
```

- API docs: http://127.0.0.1:8000/docs (`/predict`, `/predict_explained`, `/model-info`, ...)
- Dashboard: http://127.0.0.1:8501
- Explain one transaction from the terminal: `uv run fraudlens explain --trans-num <id>`

Monitoring (Prometheus + Grafana, no MLflow needed):

```powershell
uv run fraudlens export-model                        # champion -> models\champion\ (needs MLflow up)
docker compose --profile "*" down; docker compose --profile monitoring up -d
uv run fraudlens replay --speed 25 --limit 6000      # simulated live traffic
$env:MODEL_SOURCE="local"; uv run fraudlens drift-report   # reports\drift```

- Grafana: http://127.0.0.1:3000 (dashboard "FraudLens: scoring API"), Prometheus: http://127.0.0.1:9090

Design decisions and trade-offs are logged in [docs/decisions.md](docs/decisions.md).
