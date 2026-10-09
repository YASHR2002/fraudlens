"""Tests for the promotion gate (pure logic) and the registry workflow (local SQLite MLflow)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from fraudlens.config import PromotionConfig
from fraudlens.models.estimators import build_pipeline, fit_kwargs, prepare_features
from fraudlens.models.promote import choose_winner, decide, evaluate_gate, run_promotion
from fraudlens.models.registry import PENDING, VersionInfo, champion, pending_versions

RULES = PromotionConfig(min_recall=0.80, min_precision=0.50, require_pr_auc_at_least_champion=True)


def candidate(version: str, pr_auc=0.98, precision=0.9, recall=0.97, cost=3000.0, family="x"):
    return VersionInfo(
        version=version, family=family, run_id="", threshold=0.5, val_pr_auc=pr_auc,
        val_precision=precision, val_recall=recall, val_cost=cost, status=PENDING,
    )  # fmt: skip


# --- pure gate logic ---------------------------------------------------------------------


def test_passes_all_rules_without_champion() -> None:
    r = evaluate_gate(candidate("1"), RULES, champion_pr_auc=None)
    assert r.passed
    assert any(reason.startswith("SKIP") for reason in r.reasons)


@pytest.mark.parametrize(
    ("kwargs", "failing"),
    [
        ({"recall": 0.79}, "recall"),
        ({"precision": 0.49}, "precision"),
    ],
)
def test_threshold_rules_fail(kwargs: dict, failing: str) -> None:
    r = evaluate_gate(candidate("1", **kwargs), RULES, champion_pr_auc=None)
    assert not r.passed
    assert any(reason.startswith("FAIL") and failing in reason for reason in r.reasons)


def test_rules_are_inclusive_at_the_boundary() -> None:
    r = evaluate_gate(candidate("1", recall=0.80, precision=0.50, pr_auc=0.9), RULES, 0.9)
    assert r.passed


def test_must_match_or_beat_champion_pr_auc() -> None:
    assert not evaluate_gate(candidate("2", pr_auc=0.979), RULES, champion_pr_auc=0.98).passed
    assert evaluate_gate(candidate("2", pr_auc=0.98), RULES, champion_pr_auc=0.98).passed


def test_champion_rule_can_be_disabled() -> None:
    rules = RULES.model_copy(update={"require_pr_auc_at_least_champion": False})
    assert evaluate_gate(candidate("2", pr_auc=0.5), rules, champion_pr_auc=0.98).passed


def test_winner_is_lowest_cost_among_passing() -> None:
    results = [
        evaluate_gate(candidate("1", cost=3000), RULES, None),
        evaluate_gate(candidate("2", cost=2900), RULES, None),
        evaluate_gate(candidate("3", cost=100, recall=0.5), RULES, None),  # cheap but fails
    ]
    assert choose_winner(results).version == "2"


def test_cost_tie_breaks_on_pr_auc_then_newer_version() -> None:
    a = evaluate_gate(candidate("1", cost=3000, pr_auc=0.98), RULES, None)
    b = evaluate_gate(candidate("2", cost=3000, pr_auc=0.99), RULES, None)
    c = evaluate_gate(candidate("3", cost=3000, pr_auc=0.99), RULES, None)
    assert choose_winner([a, b]).version == "2"
    assert choose_winner([b, c]).version == "3"


def test_no_winner_when_all_fail() -> None:
    d = decide([candidate("1", recall=0.1), candidate("2", precision=0.1)], RULES, None)
    assert d.winner is None and not d.promoted


# --- registry workflow on a local SQLite MLflow (no server needed) -----------------------

FIXTURE = Path(__file__).parent / "fixtures" / "sample_features_sql.csv"


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("test")
    yield MlflowClient()
    mlflow.set_tracking_uri(None)
    mlflow.set_registry_uri(None)


def _register(name: str, pr_auc: float, precision: float, recall: float, cost: float) -> str:
    df = pd.read_csv(FIXTURE)
    X, y = prepare_features(df), df["is_fraud"].to_numpy()
    pipe = build_pipeline("lightgbm", {"n_estimators": 5, "num_leaves": 4,
                                       "min_child_samples": 2}, 0, 10.0)  # fmt: skip
    pipe.fit(X, y, **fit_kwargs("lightgbm"))
    with mlflow.start_run() as run:
        info = mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    tags = {
        "model_family": "lightgbm", "run_id": run.info.run_id, "threshold": "0.5",
        "val_pr_auc": str(pr_auc), "val_precision": str(precision), "val_recall": str(recall),
        "val_cost": str(cost), "promotion_status": PENDING,
    }  # fmt: skip
    return mlflow.register_model(info.model_uri, name, tags=tags).version


def test_promotion_workflow_moves_alias_and_records_reasons(local_mlflow: MlflowClient) -> None:
    client, name = local_mlflow, "detector"
    assert champion(client, name, "champion") is None

    v1 = _register(name, pr_auc=0.98, precision=0.9, recall=0.97, cost=3000)
    v2 = _register(name, pr_auc=0.97, precision=0.3, recall=0.99, cost=1000)  # low precision
    decision = run_promotion(client, name, "champion", RULES)
    assert decision.winner.version == str(v1)
    assert champion(client, name, "champion").version == str(v1)
    assert pending_versions(client, name) == []
    rejected = client.get_model_version(name, v2).tags
    assert rejected["promotion_status"] == "rejected"
    assert "FAIL: precision" in rejected["promotion_reason"]
    run_tags = client.get_run(client.get_model_version(name, v1).run_id).data.tags
    assert run_tags["promotion_decision"] == "champion"

    # A weaker challenger cannot replace the champion (PR-AUC rule) ...
    v3 = _register(name, pr_auc=0.95, precision=0.9, recall=0.97, cost=500)
    assert not run_promotion(client, name, "champion", RULES).promoted
    assert champion(client, name, "champion").version == str(v1)
    assert "champion's" in client.get_model_version(name, v3).tags["promotion_reason"]

    # ... a better one does, and the old champion is marked retired.
    v4 = _register(name, pr_auc=0.99, precision=0.9, recall=0.97, cost=2500)
    assert run_promotion(client, name, "champion", RULES).winner.version == str(v4)
    assert champion(client, name, "champion").version == str(v4)
    old = client.get_model_version(name, v1).tags
    assert old["promotion_status"] == "retired" and old["retired_by_version"] == str(v4)

    # Nothing pending: a no-op with a note.
    assert run_promotion(client, name, "champion", RULES).notes == [
        "No pending versions to evaluate."
    ]
