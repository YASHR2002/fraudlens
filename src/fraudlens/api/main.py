"""Production entry point: ``uvicorn fraudlens.api.main:app``.

Loads the model (``MODEL_SOURCE``), the card-state snapshot and the LLM settings at startup.
"""

from __future__ import annotations

import os

import pandas as pd

from fraudlens.api.app import Services, create_app
from fraudlens.api.model_loader import load_model_bundle
from fraudlens.api.state import LiveState
from fraudlens.config import get_config, get_env
from fraudlens.explain.llm_explainer import llm_from_settings
from fraudlens.features.state_io import load_state
from fraudlens.logging_utils import setup_logging
from fraudlens.models.estimators import set_serving_threads

STATE_FILE = "card_state.json.gz"
DEMO_FILE = "demo_transactions.parquet"


def build_services() -> Services:
    """Load everything the API needs from config.yaml and the environment."""
    config, env = get_config(), get_env()
    bundle = load_model_bundle(config, env)
    set_serving_threads(bundle.pipeline, n_jobs=1)  # single-row scoring: no thread start-up
    processed = config.resolve(config.paths.processed)
    store, meta = load_state(processed / STATE_FILE)
    demo_path = processed / DEMO_FILE
    demo = pd.read_parquet(demo_path) if demo_path.is_file() else None
    return Services(
        bundle=bundle,
        state=LiveState(store=store, meta=meta),
        llm=llm_from_settings(config, env),
        top_factors=config.llm.top_factors,
        demo=demo,
    )


os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
setup_logging(os.environ.get("LOG_LEVEL", "INFO"))
app = create_app(loader=build_services)
