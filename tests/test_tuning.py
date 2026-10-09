"""Tests for the Optuna tuning helpers (tiny fixture data, two trials per model)."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd
import pytest

from fraudlens.models.estimators import build_pipeline, fit_kwargs, prepare_features
from fraudlens.models.tuning import (
    META_KEY,
    SEARCH_SPACES,
    best_params_payload,
    make_objective,
    params_from_payload,
    tune,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_features_sql.csv"


@pytest.fixture(scope="module")
def split():
    df = pd.read_csv(FIXTURE).sort_values("trans_ts")
    # Time-ordered split with fraud on both sides.
    cut = int(len(df) * 0.6)
    train, val = df.iloc[:cut], df.iloc[cut:]
    assert train["is_fraud"].sum() > 0 and val["is_fraud"].sum() > 0
    return (
        prepare_features(train),
        train["is_fraud"].to_numpy(),
        prepare_features(val),
        val["is_fraud"].to_numpy(),
    )


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_tune_runs_and_best_params_build_a_pipeline(split, name: str) -> None:
    X_tr, y_tr, X_val, y_val = split
    study = tune(name, X_tr, y_tr, X_val, y_val, n_trials=2, seed=0, prune_period=5)
    assert len(study.trials) == 2
    assert 0.0 <= study.best_value <= 1.0
    assert study.best_trial.user_attrs["val_pr_auc"] == study.best_value

    payload = best_params_payload({name: study}, meta={"source": "test"})
    params = params_from_payload(json.loads(json.dumps(payload)), name)  # JSON round trip
    pipe = build_pipeline(name, params, seed=0, scale_pos_weight=10.0)
    pipe.fit(X_tr, y_tr, **fit_kwargs(name))
    assert pipe.predict_proba(X_val).shape == (len(X_val), 2)
    assert payload[META_KEY]["studies"][name]["trials"] == {"COMPLETE": 2}


def test_search_spaces_are_seeded_and_reproducible() -> None:
    def sample(name: str) -> list[dict]:
        study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7))
        return [SEARCH_SPACES[name](study.ask()) for _ in range(3)]

    for name in SEARCH_SPACES:
        assert sample(name) == sample(name)


def test_pruned_trial_raises(split, monkeypatch: pytest.MonkeyPatch) -> None:
    """The pruning callbacks stop a trial as soon as Optuna says to prune."""
    X_tr, y_tr, X_val, y_val = split
    monkeypatch.setattr(optuna.trial.Trial, "should_prune", lambda self: True)
    for name in ("xgboost", "lightgbm"):
        study = optuna.create_study(direction="maximize")
        objective = make_objective(name, X_tr, y_tr, X_val, y_val, seed=0, prune_period=1)
        study.optimize(objective, n_trials=1)
        assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_unknown_model_cannot_be_tuned(split) -> None:
    with pytest.raises(ValueError, match="cannot tune"):
        make_objective("logreg", *split, seed=0)


def test_params_from_payload_reports_available_models() -> None:
    with pytest.raises(KeyError, match="available"):
        params_from_payload({"xgboost": {}, META_KEY: {}}, "lightgbm")
