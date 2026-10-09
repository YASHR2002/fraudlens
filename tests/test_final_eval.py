"""Tests for the final test-set evaluation logic (pure parts; no MLflow or real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraudlens.models.final_eval import (
    amount_rule_threshold,
    compute_final_result,
    monthly_breakdown,
    render_report,
)
from fraudlens.models.registry import PENDING, VersionInfo

FP = 5.0
VERSION = VersionInfo(
    version="2", family="lightgbm", run_id="r", threshold=0.5, val_pr_auc=0.98,
    val_precision=0.9, val_recall=0.95, val_cost=3000.0, status="champion",
)  # fmt: skip


def test_amount_rule_threshold_minimises_validation_cost() -> None:
    # Frauds of $900 and $600: a $600 cut-off catches both with no false alarm (cost $0);
    # $500 would add a $5 false alarm, and $900 would miss $600 of fraud.
    y = np.array([1, 1, 0, 0, 0])
    amt = np.array([900.0, 600.0, 500.0, 40.0, 10.0])
    assert amount_rule_threshold(y, amt, FP) == 600.0
    # A cheap fraud is not worth chasing: catching the $12 fraud means flagging everything
    # >= $12, i.e. 3 false alarms ($15) to save $12, so the rule stays at $600.
    y2 = np.array([1, 1, 0, 0, 0, 1])
    amt2 = np.array([900.0, 600.0, 500.0, 40.0, 30.0, 12.0])
    assert amount_rule_threshold(y2, amt2, FP) == 600.0


def test_threshold_is_not_reoptimised_on_test() -> None:
    y = np.array([1, 0, 1, 0, 0, 0])
    scores = np.array([0.9, 0.6, 0.45, 0.3, 0.2, 0.1])
    amt = np.array([100.0, 1, 80, 1, 1, 1])
    months = pd.Series(["2020-07"] * 3 + ["2020-08"] * 3)
    r = compute_final_result(VERSION, y, scores, amt, months, amount_rule_x=50.0,
                             false_positive_cost=FP)  # fmt: skip
    assert r.model.threshold == 0.5  # the validation threshold, as given
    assert (r.model.tp, r.model.fp, r.model.fn) == (1, 1, 1)
    assert r.hindsight.threshold == 0.45  # best on test: reported, not used
    assert r.hindsight.total_cost < r.model.total_cost
    assert r.flag_nothing_cost == 180.0
    assert r.amount_rule.tp == 2 and r.amount_rule.fp == 0  # amounts 100 and 80 >= 50
    assert list(r.monthly.index) == ["2020-07", "2020-08"]
    assert r.monthly["fraud"].sum() == 2


def test_monthly_breakdown_matches_overall_counts() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.1).astype(int)
    scores, amt = rng.random(300), rng.uniform(1, 500, 300)
    months = pd.Series(np.repeat(["2020-07", "2020-08", "2020-09"], 100))
    table = monthly_breakdown(months, y, scores, amt, 0.5, FP)
    assert table["transactions"].sum() == 300
    assert table["fraud"].sum() == y.sum()
    assert table["alerts"].sum() == int((scores >= 0.5).sum())


def test_report_contains_headline_numbers_and_baselines() -> None:
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.7, 0.2, 0.1])
    amt = np.array([300.0, 5.0, 200.0, 5.0])
    r = compute_final_result(VERSION, y, scores, amt, pd.Series(["2020-07"] * 4), 100.0, FP)
    text = render_report(r, "fraudlens-fraud-detector", FP, rerun=False)
    for expected in ("# Final evaluation on the test set", "fraudlens-fraud-detector v2",
                     "Flag nothing", "Rule: flag every transaction >= $100.00",
                     "Threshold drift", "By month"):  # fmt: skip
        assert expected in text
    assert "forced re-run" not in text
    assert "forced re-run" in render_report(r, "m", FP, rerun=True)


def test_version_info_requires_metric_tags() -> None:
    class FakeMV:
        version, run_id, tags = "1", "r", {"threshold": "0.5", "promotion_status": PENDING}

    with pytest.raises(ValueError, match="lacks required tags"):
        VersionInfo.from_model_version(FakeMV())
