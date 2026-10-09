"""Tests for cost-based and F1-based threshold selection (hand-computed examples)."""

from __future__ import annotations

import numpy as np
import pytest

from fraudlens.models.threshold import (
    best_cost_threshold,
    best_f1_threshold,
    cost_curve,
    operating_point,
)

FP_COST = 5.0


def test_worked_example() -> None:
    # Flag score >= t. Fraud amounts 100 and 50; a false alarm costs 5.
    y = [1, 0, 1, 0, 0]
    s = [0.9, 0.8, 0.6, 0.3, 0.1]
    amt = [100, 10, 50, 20, 5]
    curve = cost_curve(y, s, amt, FP_COST)
    # thresholds: inf (flag nothing), 0.9, 0.8, 0.6, 0.3, 0.1
    assert curve["threshold"].tolist() == [np.inf, 0.9, 0.8, 0.6, 0.3, 0.1]
    assert curve["cost"].tolist() == [150, 50, 55, 5, 10, 15]
    best = best_cost_threshold(y, s, amt, FP_COST)
    assert best.threshold == 0.6
    assert (best.tp, best.fp, best.fn, best.tn) == (2, 1, 0, 2)
    assert best.total_cost == 5
    assert best.precision == pytest.approx(2 / 3)
    assert best.recall == 1.0


def test_cost_and_f1_can_disagree() -> None:
    # Catching the second fraud saves $2 but needs a $5 false alarm: cost says stop at 0.9,
    # F1 says go on to 0.7. This is why the operating threshold is chosen by cost.
    y, s, amt = [1, 0, 1], [0.9, 0.8, 0.7], [100, 30, 2]
    cost_pt = best_cost_threshold(y, s, amt, FP_COST)
    f1_pt = best_f1_threshold(y, s, amt, FP_COST)
    assert cost_pt.threshold == 0.9 and cost_pt.total_cost == 2
    assert f1_pt.threshold == 0.7 and f1_pt.total_cost == 5
    assert f1_pt.f1 > cost_pt.f1


def test_tied_scores_are_flagged_together() -> None:
    curve = cost_curve([1, 0, 0], [0.5, 0.5, 0.2], [100, 1, 1], FP_COST)
    assert curve["threshold"].tolist() == [np.inf, 0.5, 0.2]
    assert curve["tp"].tolist() == [0, 1, 1]
    assert curve["fp"].tolist() == [0, 1, 2]


def test_flag_nothing_wins_when_alerts_cost_more_than_fraud() -> None:
    best = best_cost_threshold([1, 0, 0], [0.9, 0.95, 0.99], [3, 1, 1], FP_COST)
    assert best.threshold == np.inf
    assert best.alerts == 0
    assert best.total_cost == 3


def test_equal_cost_prefers_higher_threshold() -> None:
    # Flag nothing costs 5 (missed fraud); flagging the top score also costs 5 (one false alarm)
    # but catches no fraud. Equal cost -> fewer alerts.
    best = best_cost_threshold([0, 1], [0.9, 0.1], [7, 5], FP_COST)
    assert best.threshold == np.inf


def test_curve_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    s = np.round(rng.random(400), 2)  # rounding creates many ties
    amt = rng.uniform(1, 500, 400)
    curve = cost_curve(y, s, amt, FP_COST)
    for t, cost, tp, fp in zip(curve["threshold"], curve["cost"], curve["tp"], curve["fp"],
                               strict=True):  # fmt: skip
        pt = operating_point(y, s, amt, t, FP_COST)
        assert (pt.tp, pt.fp) == (tp, fp)
        assert pt.total_cost == pytest.approx(cost)
    brute = min(operating_point(y, s, amt, t, FP_COST).total_cost for t in [*set(s), np.inf])
    assert best_cost_threshold(y, s, amt, FP_COST).total_cost == pytest.approx(brute)


def test_metrics_are_safe_with_no_alerts() -> None:
    pt = operating_point([1, 0], [0.1, 0.2], [10, 1], threshold=0.5, false_positive_cost=FP_COST)
    assert pt.precision == 0.0 and pt.recall == 0.0 and pt.f1 == 0.0
    assert pt.as_dict()["total_cost"] == 10


@pytest.mark.parametrize(
    ("y", "s", "amt", "match"),
    [
        ([1, 0], [0.1], [1, 1], "same length"),
        ([1, 2], [0.1, 0.2], [1, 1], "only 0 and 1"),
        ([1, 0], [0.1, np.nan], [1, 1], "NaN"),
        ([1, 0], [0.1, 0.2], [1], "same length as y"),
        ([1, 0], [0.1, 0.2], [-1, 1], "non-negative"),
    ],
)
def test_invalid_inputs(y, s, amt, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        cost_curve(y, s, amt, FP_COST)
