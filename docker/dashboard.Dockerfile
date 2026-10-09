# FraudLens dashboard: Streamlit only. It talks to the API over HTTP (API_URL), so the image
# needs no model libraries, data or project package: just the dashboard group and one file.

FROM python:3.11-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --only-group dashboard --no-hashes -o requirements.txt \
    && uv venv /app/.venv \
    && uv pip install --python /app/.venv/bin/python -r requirements.txt

FROM python:3.11-slim
RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/fraudlens/dashboard/app.py ./dashboard.py
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER app
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
