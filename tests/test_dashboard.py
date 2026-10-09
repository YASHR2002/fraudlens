"""Dashboard UI test: the real Streamlit script, driven headlessly, against the in-process API.

``httpx.Client`` is swapped for FastAPI's TestClient (an httpx.Client subclass) wired to an app
with a tiny model, so the dashboard's HTTP calls reach real endpoints without a network.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from fraudlens.api.app import Services, create_app
from fraudlens.api.model_loader import ModelBundle
from fraudlens.api.state import LiveState
from fraudlens.features.state_io import build_state

from .test_api import (  # noqa: F401
    payload,  # noqa: F401  (shared fixtures below reuse test_api's)
    pipeline,
    raw,
)

DASHBOARD = Path(__file__).parents[1] / "src" / "fraudlens" / "dashboard" / "app.py"


@pytest.fixture(autouse=True)
def fresh_streamlit_caches():
    """st.cache_* persist across AppTest runs in one process; start each test clean."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


@pytest.fixture
def api_client(raw, pipeline):  # noqa: F811
    demo = raw.iloc[150:170][
        [
            "trans_num",
            "cc_num",
            "trans_ts",
            "amt",
            "category",
            "dob",
            "city_pop",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
            "is_fraud",
        ]
    ]
    bundle = ModelBundle(pipeline=pipeline, model_threshold=0.5, name="demo-model", version="3",
                         family="lightgbm", source="test",
                         metrics={"test_pr_auc": 0.97, "test_recall": 0.95, "test_precision": 0.86,
                                  "test_cost": 25150.0, "val_pr_auc": 0.98, "val_recall": 0.97,
                                  "val_precision": 0.9, "val_cost": 2975.0},
                         global_importance={"amt": 0.9, "category": 1.3},
                         fairness={"gender.female": {"recall": 0.93, "precision": 0.84,
                                                     "fpr": 0.0007}})  # fmt: skip
    services = Services(bundle=bundle, state=LiveState(store=build_state(raw.iloc[:150])),
                        demo=demo.assign(is_fraud=[1, 0] * 10))  # fmt: skip
    with TestClient(create_app(services=services)) as client:
        yield client


def test_dashboard_scores_and_explains(api_client, monkeypatch) -> None:
    monkeypatch.setattr(httpx, "Client", lambda *a, **k: api_client)
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert [t.label for t in at.tabs] == ["Score a transaction", "Model overview"]
    assert at.selectbox[0].options  # demo transactions loaded through /demo/transactions
    at.button[0].click().run()
    assert not at.exception, at.exception
    metrics = [(m.label, m.value) for m in at.metric]
    assert ("Threshold", "0.5000") in metrics
    # Overview tab, from /model-info: test and validation PR-AUC.
    assert [v for label, v in metrics if label == "PR-AUC"] == ["0.970", "0.980"]
    notes = [m.value for m in at.markdown if "Analyst note" in m.value]
    assert notes and "template fallback" in notes[0]  # no LLM configured in this app
    assert len(at.get("plotly_chart")) >= 3  # gauge, waterfall, importance


def test_dashboard_reports_unreachable_api(monkeypatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda *a, **k: real_client(transport=httpx.MockTransport(refuse))
    )
    at = AppTest.from_file(str(DASHBOARD), default_timeout=60)
    at.run()
    assert not at.exception
    assert any("Cannot reach the scoring API" in e.value for e in at.error)
