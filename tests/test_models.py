"""Tests for the model pipelines and evaluation (tiny data, no MLflow or real dataset)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skops.io as sio
from matplotlib.figure import Figure

from fraudlens.features.feature_names import FEATURE_NAMES
from fraudlens.models.estimators import (
    MODEL_NAMES,
    SKOPS_TRUSTED_TYPES,
    build_pipeline,
    feature_importance,
    fit_kwargs,
    prepare_features,
)
from fraudlens.models.evaluate import (
    evaluate,
    plot_confusion,
    plot_cost_curve,
    plot_feature_importance,
    plot_pr_comparison,
    plot_pr_curve,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_features_sql.csv"
SMALL = {
    "logreg": {"max_iter": 500},
    "xgboost": {"n_estimators": 10, "max_depth": 3},
    "lightgbm": {"n_estimators": 10, "num_leaves": 7, "min_child_samples": 2},
}


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(FIXTURE)
    return prepare_features(df), df["is_fraud"].to_numpy(), df["amt"].to_numpy()


@pytest.fixture(scope="module")
def fitted(data):
    X, y, _ = data
    out = {}
    for name in MODEL_NAMES:
        pipe = build_pipeline(name, SMALL[name], seed=42, scale_pos_weight=20.0)
        pipe.fit(X, y, **fit_kwargs(name))
        out[name] = pipe
    return out


def test_prepare_features_order_and_types(data) -> None:
    X, _, _ = data
    assert list(X.columns) == FEATURE_NAMES
    assert X["category"].map(type).eq(str).all()
    assert X.drop(columns="category").dtypes.eq("float64").all()
    with pytest.raises(KeyError, match="amt"):
        prepare_features(X.drop(columns="amt"))


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_scores_and_handles_unseen_category(fitted, data, name: str) -> None:
    X, _, _ = data
    proba = fitted[name].predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all() and ((proba >= 0) & (proba <= 1)).all()
    unseen = X.head(3).assign(category="never_seen_category")
    assert np.isfinite(fitted[name].predict_proba(unseen)).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_feature_importance_covers_all_features(fitted, name: str) -> None:
    imp = feature_importance(fitted[name])
    assert set(imp.index) == set(FEATURE_NAMES)
    assert imp.sum() == pytest.approx(1.0)
    assert (imp >= 0).all()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_skops_trusted_types_are_complete(fitted, data, name: str) -> None:
    """If a library upgrade adds a type, saving with MLflow would fail: catch it here."""
    blob = sio.dumps(fitted[name])
    assert set(sio.get_untrusted_types(data=blob)) <= set(SKOPS_TRUSTED_TYPES[name])
    loaded = sio.loads(blob, trusted=SKOPS_TRUSTED_TYPES[name])
    X, _, _ = data
    np.testing.assert_allclose(loaded.predict_proba(X), fitted[name].predict_proba(X))


def test_unknown_model_name() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("random_forest", {}, seed=1, scale_pos_weight=1.0)


def test_evaluation_metrics_and_plots(fitted, data) -> None:
    X, y, amt = data
    scores = fitted["lightgbm"].predict_proba(X)[:, 1]
    ev = evaluate(y, scores, amt, false_positive_cost=5.0)
    m = ev.metrics("val")
    for key in ("val_pr_auc", "val_roc_auc", "val_cost", "val_threshold", "val_f1max_f1"):
        assert key in m
    assert ev.cost_flag_nothing == pytest.approx(amt[y == 1].sum())
    assert ev.cost_point.total_cost <= ev.cost_flag_nothing  # flag-nothing is a candidate
    figs = [
        plot_pr_curve(y, scores, ev, "t"),
        plot_confusion(ev.cost_point, "t"),
        plot_cost_curve(y, scores, amt, 5.0, ev, "t"),
        plot_feature_importance(feature_importance(fitted["lightgbm"]), "t"),
        plot_pr_comparison({"a": (y, scores), "b": (y, scores[::-1])}, "t"),
    ]
    assert all(isinstance(f, Figure) for f in figs)
    with pytest.raises(ValueError, match="at most"):
        plot_pr_comparison({str(i): (y, scores) for i in range(4)}, "t")


def test_evaluation_without_fraud_is_nan_safe() -> None:
    ev = evaluate(np.zeros(5, dtype=int), np.linspace(0, 1, 5), np.ones(5), 5.0)
    assert ev.cost_flag_nothing == 0 and np.isnan(ev.cost_saving)


def test_training_refuses_a_split_without_fraud(monkeypatch: pytest.MonkeyPatch) -> None:
    from fraudlens.config import load_config
    from fraudlens.models import train as train_mod

    df = pd.read_csv(FIXTURE).assign(split="validation")
    df.loc[:149, "split"] = "train"
    df.loc[150:, "is_fraud"] = 0  # validation without a single fraud
    monkeypatch.setattr(train_mod, "load_features", lambda *a, **k: df)
    with pytest.raises(ValueError, match="validation split .* no fraud"):
        train_mod.load_split_data(
            load_config(Path(__file__).parents[1] / "configs" / "config.yaml")
        )
