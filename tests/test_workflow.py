"""End-to-end ML workflow through the real CLI, offline.

A throwaway project in ``tmp_path``: the synthetic fixture split by time into train /
validation / test, its SQL features, a local SQLite MLflow, and no ``.env`` (so no LLM key and no
network). Runs: train -> promote -> evaluate --final -> export-model -> explain -> shap-global
-> fairness-audit -> drift-report -> build-state, checking each step's outputs. This catches
wiring bugs between modules that unit tests cannot (e.g. a missing import in a command).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from typer.testing import CliRunner

from fraudlens.cli import app
from fraudlens.config import get_config, get_env
from fraudlens.data.convert import convert_csv_to_parquet
from fraudlens.features.build import FEATURES_SCHEMA
from fraudlens.features.feature_names import FEATURE_TABLE_COLUMNS

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
runner = CliRunner()


def run(*args: str) -> str:
    result = runner.invoke(app, list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A self-contained project folder (configs/, data/, sql/) built from the fixture."""
    root = tmp_path_factory.mktemp("project")
    raw = pd.read_csv(FIXTURES / "sample_transactions.csv", dtype={"zip": str})
    raw["ts"] = pd.to_datetime(raw["trans_date_trans_time"])
    raw = raw.sort_values("ts", kind="stable")
    days = raw["ts"].dt.normalize()
    # Split by whole days, like the real time-based split, choosing the two cut days so that
    # train, validation and test each contain fraud (the real data has hundreds in each).
    fraud_days = days[raw["is_fraud"] == 1].tolist()
    train_end = fraud_days[3].date()  # 4 frauds in train
    val_end = fraud_days[5].date()  # 2 more in validation, the rest in test
    is_test = days > pd.Timestamp(val_end)
    assert raw.loc[is_test, "is_fraud"].sum() >= 2

    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    (root / "data" / "raw").mkdir(parents=True)
    for name, part in (("train", raw[~is_test]), ("test", raw[is_test])):
        csv = root / "data" / "raw" / ("fraudTrain.csv" if name == "train" else "fraudTest.csv")
        part.drop(columns="ts").rename(columns={"Unnamed: 0": ""}).to_csv(
            csv, index=False, lineterminator="\n"
        )
        convert_csv_to_parquet(csv, processed / f"transactions_{name}.parquet")

    feats = pd.read_csv(FIXTURES / "sample_features_sql.csv", parse_dates=["trans_ts"])
    feats["source"] = feats["trans_num"].map(
        dict(zip(raw["trans_num"], is_test.map({True: "test", False: "train"}), strict=True))
    )
    table = pa.Table.from_pandas(feats[FEATURE_TABLE_COLUMNS], preserve_index=False)
    pq.write_table(table.cast(FEATURES_SCHEMA), processed / "features.parquet")

    config = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    config["split"] = {"train_end": str(train_end), "validation_end": str(val_end)}
    config["training"]["models"] = {
        "logreg": {"max_iter": 500},
        "xgboost": {"n_estimators": 15, "max_depth": 3},
        "lightgbm": {"n_estimators": 15, "num_leaves": 7, "min_child_samples": 2},
    }
    config["shap"]["global_sample_size"] = 50
    config["promotion"] = {"min_recall": 0.0, "min_precision": 0.0,
                           "require_pr_auc_at_least_champion": True}  # fmt: skip
    (root / "configs").mkdir()
    (root / "configs" / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (root / "sql").mkdir()
    return root


@pytest.fixture(autouse=True)
def isolated(project: Path, monkeypatch: pytest.MonkeyPatch):
    """Run inside the throwaway project: its config, a SQLite MLflow, no .env, no API key."""
    monkeypatch.chdir(project)  # no .env here, so no GOOGLE_API_KEY is loaded
    monkeypatch.setenv("FRAUDLENS_CONFIG", str(project / "configs" / "config.yaml"))
    uri = f"sqlite:///{(project / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    get_config.cache_clear()
    get_env.cache_clear()
    yield
    get_config.cache_clear()
    get_env.cache_clear()
    mlflow.set_tracking_uri(None)


def test_full_workflow(project: Path) -> None:
    reports = project / "reports"

    out = run("download-data")  # files already in data/raw: detected, no download
    assert "fraudTrain.csv" in out
    out = run("validate-data")
    assert "All error-level checks passed" in out and (reports / "data_quality.md").is_file()
    out = run("convert-data")
    assert "transactions_train.parquet" in out and "transactions_test.parquet" in out

    out = run("show-config")
    assert "train_end" in out and "**********" not in out.split("google_api_key")[-1][:5]

    # Baselines and tuned candidates (best params come from config here).
    out = run("train", "--models", "logreg,lightgbm")
    assert "LightGBM" in out and (reports / "baseline_comparison.md").is_file()
    (project / "configs" / "best_params.json").write_text(
        '{"lightgbm": {"n_estimators": 20, "num_leaves": 7, "min_child_samples": 2,'
        ' "subsample_freq": 1}, "xgboost": {"n_estimators": 20, "max_depth": 3},'
        f' "_meta": {{"source": "test", "split": {{"train_end": "{get_config().split.train_end}",'
        f' "validation_end": "{get_config().split.validation_end}"}}}}}}',
        encoding="utf-8",
    )
    out = run("train", "--use-best-params")
    assert "Registered xgboost" in out and "Registered lightgbm" in out

    out = run("promote")
    assert "WINNER" in out and "FRAUD_THRESHOLD=" in out
    assert "No pending versions" in run("promote")

    out = run("evaluate", "--final")
    assert "# Final evaluation on the test set" in out
    assert (reports / "final_evaluation.md").is_file()
    second = runner.invoke(app, ["evaluate", "--final"])
    assert second.exit_code == 1  # the test set is used once

    out = run("export-model")
    assert (project / "models" / "champion" / "metadata.json").is_file()

    trans_num = pd.read_parquet(project / "data" / "processed" / "features.parquet")[
        "trans_num"
    ].iloc[-1]
    out = run("explain", "--trans-num", trans_num)
    assert "Analyst note (source: fallback)" in out  # no API key in this project
    assert "GOOGLE_API_KEY is not set" in out

    out = run("shap-global")
    assert "Mean |SHAP|" in out

    out = run("fairness-audit")
    assert "### By gender" in out and (reports / "fairness.md").is_file()

    out = run("drift-report", "--reference-size", "60")
    assert "# Drift report" in out and (reports / "drift" / "summary.md").is_file()

    out = run("build-state")
    assert "feature mismatches vs SQL: 0" in out
    assert (project / "data" / "processed" / "card_state.json.gz").is_file()

    # The Kaggle bundle: train + validation only, with a checksum (a stand-in wheel file).
    from fraudlens.models.kaggle_bundle import build_bundle

    wheel = project / "fraudlens-0.0.0-py3-none-any.whl"
    wheel.write_bytes(b"not a real wheel")
    manifest = build_bundle(get_config(), project / "kaggle_upload", wheel, "someone")
    assert set(manifest["rows"]) == {"train", "validation"} and len(manifest["data_sha256"]) == 64


def test_local_model_source_after_export(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The exported model serves without MLflow (runs after test_full_workflow)."""
    if not (project / "models" / "champion" / "metadata.json").is_file():
        pytest.skip("needs the export from test_full_workflow")
    from fraudlens.api.model_loader import load_model_bundle

    monkeypatch.setenv("MODEL_SOURCE", "local")
    get_env.cache_clear()
    bundle = load_model_bundle(get_config(), get_env())
    assert bundle.source == "local" and 0 <= bundle.threshold <= 1
    assert bundle.metrics.get("test_pr_auc") is not None

    # The production entry point loads model, state, demo set and LLM settings the same way.
    from fraudlens.api.main import app as production_app
    from fraudlens.api.main import build_services

    services = build_services()
    assert services.bundle.source == "local" and services.demo is not None
    assert services.llm is not None and not services.llm.enabled  # no key in this project
    assert production_app.title == "FraudLens scoring API"


def test_unavailable_model_source(monkeypatch: pytest.MonkeyPatch) -> None:
    from fraudlens.api.model_loader import ModelLoadError, load_model_bundle

    monkeypatch.setenv("MODEL_SOURCE", "s3")
    get_env.cache_clear()
    with pytest.raises(ModelLoadError, match="not available yet"):
        load_model_bundle(get_config(), get_env())
