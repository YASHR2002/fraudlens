"""Tests for traffic replay (mocked HTTP) and drift detection (Evidently on fixture data)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from fraudlens.models.estimators import prepare_features
from fraudlens.monitoring.drift import (
    SCORE,
    ColumnDrift,
    column_drift,
    drift_report,
    render_summary,
)
from fraudlens.monitoring.replay import paced, replay, to_payload

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "sample_transactions.csv", dtype={"zip": str})
    df["trans_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"]).dt.date
    return df.sort_values("trans_ts").head(12).reset_index(drop=True)


# --- replay --------------------------------------------------------------------------------


def test_payload_is_live_traffic_json(rows) -> None:
    body = to_payload(rows.iloc[0].to_dict())
    assert body["update_state"] is True
    assert isinstance(body["cc_num"], int) and isinstance(body["amt"], float)
    assert body["trans_ts"].startswith("2019-01-01T") and len(body["dob"]) == 10
    json.dumps(body)  # serialisable


def test_paced_releases_items_at_the_requested_rate() -> None:
    start = time.perf_counter()
    assert list(paced(21, speed=200)) == list(range(21))
    assert time.perf_counter() - start >= 20 / 200 * 0.9  # ~0.1 s for 20 intervals


def mock_api(flag_fraud: bool = True) -> tuple[httpx.Client, list[str]]:
    """A fake API: flags the 3rd and 6th transactions, 409s the 5th, 500s the 8th."""
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        n = len(paths)
        if n == 5:
            return httpx.Response(409, json={"detail": "older"})
        if n == 8:
            return httpx.Response(500, text="boom")
        flagged = flag_fraud and n in (3, 6)
        return httpx.Response(200, json={"score": 0.9 if flagged else 0.01, "flagged": flagged,
                                         "latency_ms": 5.0})  # fmt: skip

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api"), paths


def test_replay_counts_outcomes_and_live_metrics(rows) -> None:
    labelled = rows.assign(is_fraud=[0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0])
    client, paths = mock_api()
    summary, results = replay(labelled, "http://api", speed=1000, explain_every=4, client=client)
    assert summary.sent == 12 and summary.scored == 10
    assert summary.rejected_out_of_order == 1 and summary.errors == 1
    # Fraud rows 3 and 6 flagged (TP 2); fraud row 7 approved (FN 1); nothing else flagged.
    assert (summary.tp, summary.fp, summary.fn) == (2, 0, 1)
    assert summary.recall == pytest.approx(2 / 3) and summary.precision == 1.0
    assert paths.count("/predict_explained") == 3  # every 4th of 12
    assert summary.explained == 2  # the 8th (an explained call) hit the mocked 500
    assert len(results) == 12 and set(results["status"]) == {200, 409, 500}


def test_replay_rejects_non_positive_speed(rows) -> None:
    with pytest.raises(ValueError, match="speed"):
        replay(rows.assign(is_fraud=0), "http://api", speed=0)


# --- drift ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "value", "drifted"),
    [
        ("Wasserstein distance (normed)", 0.15, True),
        ("Wasserstein distance (normed)", 0.05, False),
        ("Jensen-Shannon distance", 0.1, True),
        ("K-S p_value", 0.01, True),  # p-value: drift when small
        ("K-S p_value", 0.2, False),
    ],
)
def test_column_drift_direction(method: str, value: float, drifted: bool) -> None:
    assert ColumnDrift("x", method, value, 0.1 if "distance" in method else 0.05).drifted is drifted


def test_drift_report_detects_a_shifted_feature(tmp_path) -> None:
    feats = pd.read_csv(FIXTURES / "sample_features_sql.csv")
    rng = np.random.default_rng(0)
    base = prepare_features(pd.concat([feats] * 5, ignore_index=True))
    base[SCORE] = rng.uniform(0, 0.05, len(base))
    # Identical data except one shifted feature, so nothing else can drift by chance.
    reference = base.copy()
    current = base.copy()
    current["amt"] = current["amt"] * 4
    current["log_amt"] = np.log1p(current["amt"])
    html = tmp_path / "drift.html"
    results = drift_report(reference, current, html)
    by_col = {r.column: r for r in results}
    assert html.is_file() and html.stat().st_size > 10_000
    assert len(by_col) == 19  # 18 features + score
    assert by_col["amt"].drifted and by_col["log_amt"].drifted
    assert not by_col["hour"].drifted and not by_col["category"].drifted


def test_column_drift_parses_snapshot_dict() -> None:
    snap = {"metrics": [
        {"config": {"type": "evidently:metric_v2:DriftedColumnsCount"}, "value": {"count": 1}},
        {"config": {"type": "evidently:metric_v2:ValueDrift", "column": "amt",
                    "method": "Wasserstein distance (normed)", "threshold": 0.1}, "value": 0.3},
    ]}  # fmt: skip
    assert column_drift(snap) == [ColumnDrift("amt", "Wasserstein distance (normed)", 0.3, 0.1)]


def test_render_summary() -> None:
    table = pd.DataFrame([
        {"month": "2020-07", "column": "amt", "method": "W", "value": 0.2, "threshold": 0.1,
         "drifted": True, "rows": 100, "fraud_rate": 0.004, "mean_score": 0.01},
        {"month": "2020-07", "column": SCORE, "method": "W", "value": 0.05, "threshold": 0.1,
         "drifted": False, "rows": 100, "fraud_rate": 0.004, "mean_score": 0.01},
    ])  # fmt: skip
    text = render_summary(table, pd.DataFrame({SCORE: [0.01, 0.02]}), 0.0057)
    assert "| 2020-07 | 100 | 0.400% | 0.0100 | 0.050 | 1 of 2 | amt |" in text
