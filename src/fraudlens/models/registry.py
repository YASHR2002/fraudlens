"""MLflow Model Registry helpers: register versions, read their tags, manage the champion alias.

Every candidate is a version of one registered model. Each version carries, as tags, what the
promotion gate and the scoring service need without recomputing anything:

* ``threshold``: the cost-optimal decision threshold chosen on validation;
* ``val_pr_auc``, ``val_precision``, ``val_recall``, ``val_cost``: validation metrics;
* ``model_family``, ``run_id``;
* after the gate: ``promotion_status`` (``champion`` / ``rejected``) and ``promotion_reason``.

Aliases (not the deprecated stages) mark the production version: ``@champion``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from fraudlens.models.train import TrainResult

logger = logging.getLogger(__name__)

PENDING = "pending"


@dataclass(frozen=True)
class VersionInfo:
    """The registry facts about one model version that promotion and serving use."""

    version: str
    family: str
    run_id: str
    threshold: float
    val_pr_auc: float
    val_precision: float
    val_recall: float
    val_cost: float
    status: str

    @classmethod
    def from_model_version(cls, mv: ModelVersion) -> VersionInfo:
        t = mv.tags
        missing = [k for k in ("threshold", "val_pr_auc", "val_precision", "val_recall", "val_cost")
                   if k not in t]  # fmt: skip
        if missing:
            raise ValueError(f"model version {mv.version} lacks required tags {missing}")
        return cls(
            version=str(mv.version),
            family=t.get("model_family", "unknown"),
            run_id=mv.run_id or t.get("run_id", ""),
            threshold=float(t["threshold"]),
            val_pr_auc=float(t["val_pr_auc"]),
            val_precision=float(t["val_precision"]),
            val_recall=float(t["val_recall"]),
            val_cost=float(t["val_cost"]),
            status=t.get("promotion_status", PENDING),
        )


def register_result(result: TrainResult, model_name: str) -> str:
    """Register a trained model as a new version and tag it. Returns the version number."""
    cp = result.val.cost_point
    tags = {
        "model_family": result.name,
        "run_id": result.run_id,
        "threshold": repr(cp.threshold),
        "val_pr_auc": repr(result.val.pr_auc),
        "val_precision": repr(cp.precision),
        "val_recall": repr(cp.recall),
        "val_cost": repr(cp.total_cost),
        "promotion_status": PENDING,
    }
    mv = mlflow.register_model(result.model_uri, model_name, tags=tags)
    logger.info("Registered %s as %s version %s", result.name, model_name, mv.version)
    return str(mv.version)


def champion(client: MlflowClient, model_name: str, alias: str) -> VersionInfo | None:
    """The version currently holding ``alias``, or None if there is no champion yet."""
    try:
        return VersionInfo.from_model_version(client.get_model_version_by_alias(model_name, alias))
    except MlflowException as exc:
        if "not found" in str(exc).lower() or "RESOURCE_DOES_NOT_EXIST" in str(exc):
            return None
        raise


def pending_versions(client: MlflowClient, model_name: str) -> list[VersionInfo]:
    """Registered versions not yet judged by the promotion gate."""
    versions = client.search_model_versions(f"name = '{model_name}'")
    infos = [VersionInfo.from_model_version(mv) for mv in versions]
    return sorted((v for v in infos if v.status == PENDING), key=lambda v: int(v.version))
