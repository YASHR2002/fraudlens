"""Tests for SHAP explanations, plain-English factors and the fairness audit (fixture data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from fraudlens.explain.factors import (
    GROUPS,
    PROTECTED_GROUPS,
    build_factors,
    describe,
    top_factors,
)
from fraudlens.explain.shap_explainer import ShapExplainer, global_plots, mean_abs_shap
from fraudlens.fairness.audit import (
    age_band,
    audit,
    group_metrics,
    render_tables,
    wilson_interval,
)
from fraudlens.features.feature_names import FEATURE_NAMES
from fraudlens.models.estimators import build_pipeline, fit_kwargs, prepare_features

FIXTURE = Path(__file__).parent / "fixtures" / "sample_features_sql.csv"


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="module", params=["lightgbm", "xgboost"])
def explainer(request, data) -> ShapExplainer:
    params = {"lightgbm": {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 2},
              "xgboost": {"n_estimators": 20, "max_depth": 3}}[request.param]  # fmt: skip
    pipe = build_pipeline(request.param, params, seed=0, scale_pos_weight=20.0)
    pipe.fit(prepare_features(data), data["is_fraud"].to_numpy(), **fit_kwargs(request.param))
    return ShapExplainer(pipe)


# --- SHAP ----------------------------------------------------------------------------------


def test_shap_values_are_additive_and_in_feature_order(explainer, data) -> None:
    rows = data.head(20)
    values = explainer.shap_values(rows)
    assert values.shape == (20, len(FEATURE_NAMES))
    margin = explainer.base_value + values.sum(axis=1)
    scores = explainer.pipeline.predict_proba(prepare_features(rows))[:, 1]
    np.testing.assert_allclose(expit(margin), scores, rtol=1e-5, atol=1e-7)


def test_local_explanation_groups_and_sorts_factors(explainer, data) -> None:
    exp = explainer.explain(data.iloc[[60]])
    assert {f.group for f in exp.factors} == set(GROUPS)
    sizes = [abs(f.shap) for f in exp.factors]
    assert sizes == sorted(sizes, reverse=True)
    amount = next(f for f in exp.factors if f.group == "amount")
    assert amount.shap == pytest.approx(exp.shap_by_feature["amt"] + exp.shap_by_feature["log_amt"])
    with pytest.raises(ValueError, match="exactly one row"):
        explainer.explain(data.head(2))


def test_global_plots_and_importance(explainer, data) -> None:
    figs = global_plots(explainer, data, "test")
    assert set(figs) == {"shap_beeswarm.png", "shap_bar.png"}
    imp = mean_abs_shap(explainer, data)
    assert list(imp.index) != [] and set(imp.index) == set(FEATURE_NAMES)


# --- plain-English factors -----------------------------------------------------------------

ROW = {
    "amt": 330.52, "log_amt": 5.8, "category": "shopping_net", "hour": 22, "day_of_week": 2,
    "is_night": 1, "age_at_txn": 48, "log_city_pop": np.log1p(640), "distance_km": 98.4,
    "card_txn_count_1h": 1, "card_txn_count_24h": 3, "card_txn_count_7d": 28,
    "card_amt_sum_24h": 3037.01, "card_avg_amt_prior": 54.94,
    "amt_to_card_avg_ratio": 6.016, "secs_since_last_txn": 68_760.0,
    "is_new_category_for_card": 0, "card_txn_number": 412,
}  # fmt: skip


@pytest.mark.parametrize(
    ("group", "expected"),
    [
        ("amount", "transaction amount is $330.52"),
        ("time_of_day", "made between 22:00 and 22:59 (at night)"),
        ("day_of_week", "made on a Tuesday"),
        ("category", "merchant category is online shopping"),
        ("amount_vs_card_average", "amount $330.52 is 6.0x this card's average of $54.94"),
        ("card_spend_24h", "card spent $3,037.01 in the previous 24 hours"),
        ("card_txns_1h", "1 earlier transaction on this card in the previous hour"),
        ("time_since_last", "made 19.1 hours after the card's previous transaction"),
        ("new_category", "card has been used in this merchant category before"),
        ("distance", "merchant is 98 km from the cardholder's home"),
        ("city_size", "cardholder's city has about 640 residents"),
    ],
)
def test_describe(group: str, expected: str) -> None:
    assert describe(group, ROW) == expected


def test_describe_first_transaction_on_card() -> None:
    first = {**ROW, "amt_to_card_avg_ratio": float("nan"), "card_avg_amt_prior": float("nan"),
             "secs_since_last_txn": float("nan")}  # fmt: skip
    assert "first transaction on this card" in describe("amount_vs_card_average", first)
    assert describe("time_since_last", first) == "no previous transaction on this card"


def test_protected_factors_can_be_excluded() -> None:
    shap_values = dict.fromkeys(FEATURE_NAMES, 0.0) | {"age_at_txn": 5.0, "amt": 1.0}
    factors = build_factors(ROW, shap_values)
    assert factors[0].group == "age"  # largest effect
    kept = top_factors(factors, n=3, exclude_protected=True)
    assert all(f.group not in PROTECTED_GROUPS for f in kept)
    assert kept[0].group == "amount"


# --- fairness ------------------------------------------------------------------------------


def test_wilson_interval() -> None:
    lo, hi = wilson_interval(50, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.19, abs=0.01)
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == 1.0
    assert all(np.isnan(wilson_interval(0, 0)))


def test_age_bands() -> None:
    bands = age_band(pd.Series([18, 29, 30, 49, 50, 69, 70, 95]))
    assert bands.astype(str).tolist() == [
        "under 30", "under 30", "30-49", "30-49", "50-69", "50-69", "70+", "70+",
    ]  # fmt: skip


def test_group_metrics_and_gaps() -> None:
    # Group A: 2 fraud, both caught, 1 false alarm in 6 legit. Group B: 2 fraud, 1 caught.
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    flagged = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    groups = pd.Series(["A"] * 8 + ["B"] * 6).astype("category")
    t = group_metrics(y, flagged, groups)
    assert t.loc["A", "recall"] == 1.0 and t.loc["B", "recall"] == 0.5
    assert t.loc["A", "precision"] == pytest.approx(2 / 3)
    assert t.loc["A", "fpr"] == pytest.approx(1 / 6) and t.loc["B", "fpr"] == 0.0

    gender = pd.Series(["F"] * 8 + ["M"] * 6)
    ages = pd.Series([25] * 8 + [75] * 6)
    result = audit(y, flagged, gender, ages)
    assert set(result["tables"]) == {"gender", "age band"}
    recall_gap = next(g for g in result["gaps"] if g.attribute == "gender" and g.metric == "recall")
    assert recall_gap.best == "female" and recall_gap.difference == pytest.approx(0.5)
    assert recall_gap.ci_overlap  # 2 fraud per group: far too few to call it real
    text = render_tables(result)
    assert "### By gender" in text and "Reading the results" in text
