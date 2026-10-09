# FraudLens scoring API: python:3.11-slim, two stages, non-root, API dependencies only.
# Build:  docker compose build api      Run: docker compose --profile serve up -d

# ---- build stage: resolve and install dependencies with uv from the lock file -----------------
FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
# Dependencies first (cached unless pyproject/uv.lock change), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --group api --no-install-project
COPY src ./src
RUN uv sync --frozen --no-default-groups --group api --no-editable

# ---- runtime stage --------------------------------------------------------------------------
FROM python:3.11-slim
# libgomp: OpenMP runtime needed by LightGBM and XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs/config.yaml ./configs/config.yaml
ENV PATH="/app/.venv/bin:$PATH" \
    FRAUDLENS_CONFIG=/app/configs/config.yaml \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
USER app
EXPOSE 8000
# Model state and demo data are mounted at /app/data/processed; the model comes from MLflow
# (MODEL_SOURCE=mlflow) or a mounted export in /app/models (MODEL_SOURCE=local).
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"
CMD ["uvicorn", "fraudlens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
