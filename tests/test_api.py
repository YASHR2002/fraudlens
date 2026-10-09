"""API tests with FastAPI's TestClient, a tiny fixture-trained model and a fake LLM.

No MLflow, Docker, database or network: everything is injected through ``create_app``.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraudlens.api.app import Services, create_app
from fraudlens.api.model_loader import ModelBundle
from fraudlens.api.state import LiveState
from fraudlens.explain.llm_explainer import LLMExplainer
from fraudlens.features.online import CardStateStore, Transaction
from fraudlens.features.state_io import build_state, load_state, save_state
from fraudlens.models.estimators import (
    build_pipeline,
    fit_kwargs,
    prepare_features,
    single_row_frame,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "sample_transactions.csv", dtype={"zip": str})
    df["trans_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"]).dt.date
    return df.sort_values("trans_ts", kind="stable").reset_index(drop=True)


@pytest.fixture(scope="module")
def pipeline():
    feats = pd.read_csv(FIXTURES / "sample_features_sql.csv")
    params = {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 2}
    pipe = build_pipeline("lightgbm", params, seed=0, scale_pos_weight=20.0)
    pipe.fit(prepare_features(feats), feats["is_fraud"].to_numpy(), **fit_kwargs("lightgbm"))
    return pipe


def payload(row: dict, update_state: bool = True) -> dict:
    return {
        "trans_num": row["trans_num"], "cc_num": int(row["cc_num"]),
        "trans_ts": row["trans_ts"].isoformat(), "amt": float(row["amt"]),
        "category": row["category"], "dob": row["dob"].isoformat(),
        "city_pop": int(row["city_pop"]), "lat": row["lat"], "long": row["long"],
        "merch_lat": row["merch_lat"], "merch_long": row["merch_long"],
        "update_state": update_state,
    }  # fmt: skip


class FakeModels:
    def __init__(self) -> None:
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        flagged = "Decision: FLAGGED" in contents
        return SimpleNamespace(parsed=None, text=__import__("json").dumps({
            "summary": "A short, factual analyst note written from the provided facts only.",
            "key_reasons": ["reason one", "reason two"],
            "recommended_action": "review" if flagged else "approve",
        }))  # fmt: skip


@pytest.fixture
def client(raw, pipeline):
    history = raw.iloc[:150]  # state "as of" the 150th transaction, like the real snapshot
    bundle = ModelBundle(pipeline=pipeline, model_threshold=0.5, name="test-model",
                         version="7", family="lightgbm", source="test",
                         metrics={"val_pr_auc": 0.9})  # fmt: skip
    fake = FakeModels()
    services = Services(
        bundle=bundle,
        state=LiveState(store=build_state(history), meta={"cutoff": str(history.trans_ts.max())}),
        llm=LLMExplainer(None, "fake-llm", client=SimpleNamespace(models=fake),
                         retry_delay_seconds=0),
    )  # fmt: skip
    with TestClient(create_app(services=services)) as c:
        c.fake_llm = fake
        yield c


# --- ops ---------------------------------------------------------------------------------


def test_health_and_ready(client) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model_version"] == "7"


def test_not_ready_without_model() -> None:
    with TestClient(create_app(services=Services(load_error="boom"))) as c:
        assert c.get("/health").status_code == 200
        r = c.get("/ready")
        assert r.status_code == 503 and r.json()["reason"] == "boom"
        assert c.post("/predict", json={}).status_code == 422  # validation runs first
        assert c.get("/model-info").status_code == 503


def test_loader_failure_is_reported_not_raised() -> None:
    def broken():
        raise RuntimeError("mlflow unreachable")

    with TestClient(create_app(loader=broken)) as c:
        assert c.get("/ready").status_code == 503
        assert "mlflow unreachable" in c.get("/ready").json()["reason"]


def test_metrics_endpoint(client, raw) -> None:
    client.post("/predict", json=payload(raw.iloc[150].to_dict(), update_state=False))
    text = client.get("/metrics").text
    assert 'handler="/predict"' in text and "http_request_duration_seconds" in text


# --- /predict ----------------------------------------------------------------------------


def test_predict_matches_offline_pipeline(client, raw, pipeline) -> None:
    """Features computed online + the served model = the same score as offline scoring."""
    row = raw.iloc[150].to_dict()
    r = client.post("/predict", json=payload(row, update_state=False))
    assert r.status_code == 200
    body = r.json()
    store = build_state(raw.iloc[:150])
    feats = store.features(Transaction.from_mapping(row))
    expected = pipeline.predict_proba(prepare_features(pd.DataFrame([feats])))[0, 1]
    assert body["score"] == pytest.approx(expected, rel=1e-12)
    assert body["decision"] == ("flagged" if expected >= 0.5 else "approved")
    assert body["model_version"] == "7" and body["state_updated"] is False
    assert body["latency_ms"] > 0
    assert "X-Process-Time-Ms" in r.headers


def test_state_updates_and_duplicates_are_idempotent(client, raw) -> None:
    row = raw.iloc[150].to_dict()
    first = client.post("/predict", json=payload(row)).json()
    assert first["state_updated"] is True and first["duplicate"] is False
    again = client.post("/predict", json=payload(row)).json()  # e.g. a client retry
    assert again["state_updated"] is False and again["duplicate"] is True
    assert again["score"] == pytest.approx(first["score"])  # same features, not double-counted
    info = client.get("/model-info").json()
    assert info["state"]["transactions_recorded_since_start"] == 1


def test_out_of_order_transaction_is_rejected(client, raw) -> None:
    client.post("/predict", json=payload(raw.iloc[160].to_dict()))
    old = raw.iloc[151].to_dict()
    r = client.post("/predict", json=payload(old))
    assert r.status_code == 409 and "older" in r.json()["detail"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("amt", -5), ("lat", 123.0), ("trans_num", ""), ("trans_ts", "not a date"), ("cc_num", 0)],
)
def test_invalid_input_is_422(client, raw, field, value) -> None:
    body = payload(raw.iloc[150].to_dict()) | {field: value}
    assert client.post("/predict", json=body).status_code == 422


def test_unknown_category_is_scored(client, raw) -> None:
    body = payload(raw.iloc[150].to_dict(), update_state=False) | {"category": "brand_new"}
    r = client.post("/predict", json=body)
    assert r.status_code == 200 and 0 <= r.json()["score"] <= 1


# --- /predict/batch ----------------------------------------------------------------------


def test_batch_scores_in_time_order_and_keeps_input_order(client, raw) -> None:
    rows = [payload(raw.iloc[i].to_dict()) for i in (155, 152, 153)]  # deliberately unsorted
    r = client.post("/predict/batch", json={"transactions": rows})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["trans_num"] for x in results] == [p["trans_num"] for p in rows]
    assert all(x["result"] and x["error"] is None for x in results)


def test_batch_reports_per_item_errors(client, raw) -> None:
    client.post("/predict", json=payload(raw.iloc[170].to_dict()))
    rows = [payload(raw.iloc[160].to_dict()), payload(raw.iloc[175].to_dict())]
    results = client.post("/predict/batch", json={"transactions": rows}).json()["results"]
    assert results[0]["error"] and results[0]["result"] is None  # older than the card's state
    assert results[1]["result"] is not None


def test_batch_size_limits(client, raw) -> None:
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
    many = [payload(raw.iloc[150].to_dict(), update_state=False)] * 1001
    assert client.post("/predict/batch", json={"transactions": many}).status_code == 422


# --- /predict_explained and /model-info ---------------------------------------------------


def test_predict_explained(client, raw) -> None:
    r = client.post("/predict_explained", json=payload(raw.iloc[150].to_dict(), False))
    assert r.status_code == 200
    body = r.json()
    assert len(body["shap_factors"]) == 8
    assert set(body["features"]) >= {"amt", "card_txn_count_24h"}
    note = body["explanation"]
    assert note["explanation_source"] == "llm"
    assert note["recommended_action"] == ("review" if body["flagged"] else "approve")
    # Cached by trans_num: explaining it again does not call the LLM.
    client.post("/predict_explained", json=payload(raw.iloc[150].to_dict(), False))
    assert client.fake_llm.calls == 1


def test_model_info(client) -> None:
    info = client.get("/model-info").json()
    assert info["model_name"] == "test-model" and info["threshold"] == 0.5
    assert info["threshold_source"] == "model" and info["metrics"]["val_pr_auc"] == 0.9
    assert info["llm"] == {"enabled": True, "model": "fake-llm"}


def test_threshold_override_from_env() -> None:
    bundle = ModelBundle(pipeline=None, model_threshold=0.4, name="m", version="1",
                         family="x", source="t", threshold_override=0.7)  # fmt: skip
    assert bundle.threshold == 0.7 and bundle.threshold_source == "env"


# --- state snapshot + fast path ------------------------------------------------------------


def test_state_snapshot_equals_replay_and_round_trips(raw, tmp_path) -> None:
    replayed = CardStateStore()
    for r in raw.to_dict("records"):
        replayed.record(Transaction.from_mapping(r))
    built = build_state(raw)
    assert set(built.cards) == set(replayed.cards)
    for cc, s in replayed.cards.items():
        b = built.cards[cc]
        assert list(b.recent) == list(s.recent)
        assert (b.n_total, b.last_ts, b.prev_distinct_ts) == (s.n_total, s.last_ts,
                                                              s.prev_distinct_ts)  # fmt: skip
        assert b.sum_total == pytest.approx(s.sum_total)
        assert b.category_first_seen == s.category_first_seen
    path = tmp_path / "state.json.gz"
    save_state(built, path, {"cutoff": "x"})
    loaded, meta = load_state(path)
    assert meta == {"cutoff": "x"}
    later = raw.trans_ts.max() + dt.timedelta(hours=1)
    probe = Transaction.from_mapping({**raw.iloc[-1].to_dict(), "trans_ts": later})
    assert loaded.features(probe) == built.features(probe)


def test_single_row_frame_matches_prepare_features(raw, pipeline) -> None:
    store = build_state(raw.iloc[:100])
    for r in raw.iloc[100:130].to_dict("records"):
        feats = store.features(Transaction.from_mapping(r))
        fast, slow = single_row_frame(feats), prepare_features(pd.DataFrame([feats]))
        assert list(fast.columns) == list(slow.columns)
        np.testing.assert_array_equal(pipeline.predict_proba(fast), pipeline.predict_proba(slow))


def test_demo_transactions_endpoint(raw, pipeline) -> None:
    demo = raw.iloc[150:160][
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
    demo = demo.assign(is_fraud=[1, 0] * 5)
    bundle = ModelBundle(pipeline=pipeline, model_threshold=0.5, name="m", version="1",
                         family="lightgbm", source="test")  # fmt: skip
    svc = Services(bundle=bundle, state=LiveState(store=build_state(raw.iloc[:150])), demo=demo)
    with TestClient(create_app(services=svc)) as c:
        fraud = c.get("/demo/transactions", params={"label": "fraud"}).json()
        assert len(fraud) == 5 and all(r["is_fraud"] == 1 for r in fraud)
        assert len(c.get("/demo/transactions", params={"limit": 3}).json()) == 3
        # A demo row is directly usable as a request body.
        body = {k: v for k, v in fraud[0].items() if k != "is_fraud"} | {"update_state": False}
        assert c.post("/predict", json=body).status_code == 200
    with TestClient(create_app(services=Services(bundle=bundle))) as c:
        assert c.get("/demo/transactions").status_code == 404
