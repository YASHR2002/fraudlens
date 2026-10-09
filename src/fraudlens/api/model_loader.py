"""Load the production model for the API, with everything it needs to describe itself.

``MODEL_SOURCE`` chooses where from:

* ``mlflow`` (default): the registered model's ``@champion`` version from the MLflow server.
* ``local``: an exported copy in ``models/champion/`` (``fraudlens export-model``, Phase 9).
* ``huggingface`` / ``s3``: added in Phases 11 and 12.

The decision threshold comes from the ``FRAUD_THRESHOLD`` environment variable if set,
otherwise from the model version's ``threshold`` tag.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

from fraudlens.config import AppConfig, EnvSettings
from fraudlens.explain.shap_explainer import ShapExplainer

logger = logging.getLogger(__name__)

METRIC_TAGS = ("val_pr_auc", "val_precision", "val_recall", "val_cost", "test_pr_auc",
               "test_precision", "test_recall", "test_cost")  # fmt: skip
LOCAL_METADATA = "metadata.json"


class ModelLoadError(RuntimeError):
    """The model could not be loaded from the configured source."""


@dataclass
class ModelBundle:
    """The served model and the facts the API reports about it."""

    pipeline: Pipeline
    model_threshold: float  # from the model version
    name: str
    version: str
    family: str
    source: str
    alias: str | None = None
    run_id: str | None = None
    trained_utc: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    global_importance: dict[str, float] | None = None
    fairness: dict[str, dict[str, float]] | None = None
    threshold_override: float | None = None  # FRAUD_THRESHOLD

    @property
    def threshold(self) -> float:
        return (
            self.threshold_override if self.threshold_override is not None else self.model_threshold
        )

    @property
    def threshold_source(self) -> str:
        return "env" if self.threshold_override is not None else "model"

    @cached_property
    def explainer(self) -> ShapExplainer:
        """SHAP explainer, built on first use (only /predict_explained needs it)."""
        return ShapExplainer(self.pipeline)

    def metadata(self) -> dict[str, Any]:
        """Everything except the pipeline (written next to an exported model)."""
        return {
            "name": self.name, "version": self.version, "family": self.family,
            "alias": self.alias, "run_id": self.run_id, "trained_utc": self.trained_utc,
            "threshold": self.model_threshold, "metrics": self.metrics,
            "global_importance": self.global_importance, "fairness": self.fairness,
        }  # fmt: skip


def _fairness_from_mlflow(client: Any, experiment: str, version: str) -> dict | None:
    """Group metrics from the latest fairness-audit run for this version (best effort)."""
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs([exp.experiment_id],
                              f"tags.stage = 'fairness_audit' and tags.model_version = '{version}'",
                              order_by=["attributes.start_time DESC"], max_results=1)  # fmt: skip
    if not runs:
        return None
    out: dict[str, dict[str, float]] = {}
    for key, value in runs[0].data.metrics.items():
        group, metric = key.rsplit(".", 1)
        out.setdefault(group, {})[metric] = value
    return out


def load_from_mlflow(config: AppConfig, env: EnvSettings) -> ModelBundle:
    """The @champion version from the MLflow tracking server / registry."""
    import mlflow
    from mlflow import MlflowClient

    t = config.training
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(t.registered_model_name, t.champion_alias)
        pipeline = mlflow.sklearn.load_model(
            f"models:/{t.registered_model_name}@{t.champion_alias}"
        )
    except Exception as exc:  # noqa: BLE001 - surface any MLflow failure as a load error
        raise ModelLoadError(f"cannot load @{t.champion_alias} from {env.mlflow_tracking_uri}: "
                             f"{exc}") from exc  # fmt: skip
    run = client.get_run(mv.run_id) if mv.run_id else None
    importance = None
    if run:
        try:
            path = client.download_artifacts(mv.run_id, "shap/mean_abs_shap.json")
            importance = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - optional extra
            logger.info("No global SHAP importance logged for run %s", mv.run_id)
    try:
        fairness = _fairness_from_mlflow(client, t.experiment_name, str(mv.version))
    except Exception:  # noqa: BLE001 - optional extra
        fairness = None
    return ModelBundle(
        pipeline=pipeline,
        model_threshold=float(mv.tags["threshold"]),
        name=t.registered_model_name,
        version=str(mv.version),
        family=mv.tags.get("model_family", "unknown"),
        source="mlflow",
        alias=t.champion_alias,
        run_id=mv.run_id,
        trained_utc=(
            dt.datetime.fromtimestamp(run.info.start_time / 1000, dt.UTC).isoformat(
                timespec="seconds"
            )
            if run
            else None
        ),  # fmt: skip
        metrics={k: float(mv.tags[k]) for k in METRIC_TAGS if k in mv.tags},
        global_importance=importance,
        fairness=fairness,
        threshold_override=env.fraud_threshold,
    )


def load_from_local(path: Path, env: EnvSettings) -> ModelBundle:
    """An exported model directory: an MLflow model plus ``metadata.json``."""
    import mlflow

    meta_path = path / LOCAL_METADATA
    if not meta_path.is_file():
        raise ModelLoadError(f"{meta_path} not found; run `fraudlens export-model` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pipeline = mlflow.sklearn.load_model(str(path / "model"))
    return ModelBundle(
        pipeline=pipeline,
        model_threshold=float(meta["threshold"]),
        name=meta["name"],
        version=str(meta["version"]),
        family=meta["family"],
        source="local",
        alias=meta.get("alias"),
        run_id=meta.get("run_id"),
        trained_utc=meta.get("trained_utc"),
        metrics=meta.get("metrics") or {},
        global_importance=meta.get("global_importance"),
        fairness=meta.get("fairness"),
        threshold_override=env.fraud_threshold,
    )


def load_model_bundle(config: AppConfig, env: EnvSettings) -> ModelBundle:
    """Load the model from ``MODEL_SOURCE``."""
    if env.model_source == "mlflow":
        return load_from_mlflow(config, env)
    if env.model_source == "local":
        return load_from_local(config.resolve(config.paths.models) / "champion", env)
    raise ModelLoadError(
        f"MODEL_SOURCE={env.model_source!r} is not available yet (added in Phase 11/12)."
    )


def export_champion(config: AppConfig, env: EnvSettings, out_dir: Path) -> dict[str, Any]:
    """Copy the @champion model and its metadata out of MLflow into ``out_dir``.

    The result (``model/`` + ``metadata.json``) is what ``MODEL_SOURCE=local`` loads, so the API
    can run without the MLflow server (monitoring profile, Hugging Face, AWS).
    """
    import shutil

    import mlflow

    bundle = load_from_mlflow(config, env)
    t = config.training
    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{t.registered_model_name}@{t.champion_alias}",
        dst_path=str(tmp / "model"),
    )
    meta = {**bundle.metadata(), "exported_utc": dt.datetime.now(dt.UTC).isoformat(
        timespec="seconds")}  # fmt: skip
    (tmp / LOCAL_METADATA).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    load_from_local(tmp, env)  # verify the export loads before replacing the previous one
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp.replace(out_dir)
    return meta
