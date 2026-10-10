"""FraudLens command-line interface: ``python -m fraudlens <command>``.

Single entry point for the local workflow (cross-platform, no Makefile).
Commands are added phase by phase.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from fraudlens import __version__
from fraudlens.config import get_config, get_env
from fraudlens.logging_utils import log_duration, setup_logging

app = typer.Typer(
    name="fraudlens",
    help="FraudLens: explainable real-time credit card fraud detection.",
    no_args_is_help=True,
    add_completion=False,
)
logger = logging.getLogger("fraudlens.cli")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fraudlens {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    log_level: Annotated[
        str, typer.Option("--log-level", "-l", help="DEBUG, INFO, WARNING or ERROR.")
    ] = "INFO",
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
        ),
    ] = False,
) -> None:
    """FraudLens: explainable real-time credit card fraud detection."""
    # Windows consoles default to cp1252, and MLflow prints emoji at the end of each run
    # (which would crash with UnicodeEncodeError). Force UTF-8 for this process.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an MLflow banner
    os.environ.setdefault("MPLBACKEND", "Agg")  # headless plotting (no Tk windows)
    setup_logging(log_level.upper())


@app.command("show-config")
def show_config() -> None:
    """Validate and print the active configuration (secrets are masked)."""
    with log_duration("load and validate configuration", logger):
        config = get_config()
        env = get_env()
    typer.echo(f"Project root: {config.project_root}")
    typer.echo(config.model_dump_json(indent=2))
    # SecretStr fields serialise as '**********', so nothing sensitive is printed.
    typer.echo(env.model_dump_json(indent=2))


# --------------------------------------------------------------------------- Phase 1: data


def _raw_files() -> tuple[Path, Path]:
    config = get_config()
    raw = config.resolve(config.paths.raw)
    return raw / config.data.train_file, raw / config.data.test_file


def _processed_files() -> dict[str, Path]:
    config = get_config()
    processed = config.resolve(config.paths.processed)
    return {
        "train": processed / "transactions_train.parquet",
        "test": processed / "transactions_test.parquet",
    }


def _fail(message: str) -> typer.Exit:
    logger.error(message)
    return typer.Exit(code=1)


@app.command("download-data")
def download_data(
    force: Annotated[bool, typer.Option(help="Download even if files exist.")] = False,
) -> None:
    """Download the dataset from Kaggle, or detect files placed manually in data/raw."""
    from fraudlens.data.download import DownloadError, download_dataset

    config = get_config()
    with log_duration("get raw dataset", logger):
        try:
            paths = download_dataset(
                config.data.kaggle_dataset,
                config.resolve(config.paths.raw),
                [config.data.train_file, config.data.test_file],
                force=force,
            )
        except DownloadError as exc:
            raise _fail(str(exc)) from exc
    for p in paths:
        typer.echo(f"{p}  ({p.stat().st_size / 1024**2:,.0f} MB)")


@app.command("validate-data")
def validate_data() -> None:
    """Check schema, nulls, duplicates, value ranges, fraud rate and dates; write a report."""
    from fraudlens.data.io import SchemaError
    from fraudlens.data.validate import DataQualityError
    from fraudlens.data.validate import validate_data as run_validation

    config = get_config()
    train, test = _raw_files()
    report = config.resolve(config.paths.reports) / "data_quality.md"
    with log_duration("validate raw data", logger):
        try:
            checks = run_validation(train, test, report, config.data.csv_block_size_mb)
        except (SchemaError, DataQualityError, FileNotFoundError) as exc:
            raise _fail(str(exc)) from exc
    flagged = sum(not c.passed for c in checks)
    typer.echo(f"All error-level checks passed ({flagged} warnings flagged). Report: {report}")


@app.command("convert-data")
def convert_data() -> None:
    """Convert the raw CSVs to typed, compressed Parquet files (streamed in chunks)."""
    from fraudlens.data.convert import convert_csv_to_parquet
    from fraudlens.data.io import SchemaError

    config = get_config()
    train, test = _raw_files()
    targets = _processed_files()
    with log_duration("convert CSV to Parquet", logger):
        for source, target in ((train, targets["train"]), (test, targets["test"])):
            try:
                stats = convert_csv_to_parquet(source, target, config.data.csv_block_size_mb)
            except (SchemaError, FileNotFoundError) as exc:
                raise _fail(str(exc)) from exc
            typer.echo(
                f"{target.name}: {stats.rows:,} rows, "
                f"{stats.csv_bytes / 1024**2:,.0f} MB -> {stats.parquet_bytes / 1024**2:,.0f} MB"
            )


@app.command("load-db")
def load_db() -> None:
    """Create the transactions table and bulk-load both Parquet files with COPY."""
    import psycopg

    from fraudlens.data.load_db import LoadError, load_database

    config = get_config()
    files = _processed_files()
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise _fail(f"Missing Parquet files {missing}; run `fraudlens convert-data` first.")
    create_sql = config.project_root / "sql" / "01_create_tables.sql"
    with log_duration("load PostgreSQL", logger):
        try:
            counts = load_database(get_env().postgres_conninfo, create_sql, files)
        except psycopg.OperationalError as exc:
            raise _fail(
                f"Cannot connect to PostgreSQL ({exc}). "
                "Is it running? `docker compose --profile data up -d`"
            ) from exc
        except LoadError as exc:
            raise _fail(str(exc)) from exc
    typer.echo("Loaded: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()))


@app.command("make-fixture")
def make_fixture(
    path: Annotated[Path, typer.Option(help="Output CSV.")] = Path(
        "tests/fixtures/sample_transactions.csv"
    ),
    rows: Annotated[int, typer.Option(help="Number of rows.")] = 200,
) -> None:
    """Generate the small synthetic test fixture (no real data)."""
    from fraudlens.data.synthetic import write_sample_csv

    with log_duration("generate synthetic fixture", logger):
        out = write_sample_csv(path, n_rows=rows, seed=get_config().seed)
    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()


# ------------------------------------------------------------------------- Phase 2: features


@app.command("build-features")
def build_features(
    export_only: Annotated[
        bool, typer.Option(help="Skip the SQL and only re-export the existing table.")
    ] = False,
) -> None:
    """Build the features table in PostgreSQL (sql/02_features.sql) and export it to Parquet."""
    import psycopg

    from fraudlens.features.build import export_features, run_feature_sql

    config = get_config()
    target = config.resolve(config.paths.processed) / "features.parquet"
    sql_path = config.project_root / "sql" / "02_features.sql"
    try:
        with psycopg.connect(get_env().postgres_conninfo) as conn:
            if not export_only:
                with log_duration("build features table in PostgreSQL", logger):
                    rows = run_feature_sql(conn, sql_path)
                logger.info("features table has %s rows", f"{rows:,}")
            with log_duration("export features to Parquet", logger):
                exported = export_features(conn, target, config.resolve(config.paths.processed))
    except psycopg.OperationalError as exc:
        raise _fail(
            f"Cannot connect to PostgreSQL ({exc}). "
            "Is it running? `docker compose --profile data up -d`"
        ) from exc
    typer.echo(f"{target}: {exported:,} rows, {target.stat().st_size / 1024**2:,.0f} MB")


@app.command("make-parity-fixture")
def make_parity_fixture(
    fixture: Annotated[Path, typer.Option(help="Synthetic raw CSV.")] = Path(
        "tests/fixtures/sample_transactions.csv"
    ),
    out: Annotated[Path, typer.Option(help="Where to write the SQL features CSV.")] = Path(
        "tests/fixtures/sample_features_sql.csv"
    ),
) -> None:
    """Run the SQL feature pipeline on the synthetic fixture (in a throwaway schema).

    The output is committed so the training/serving parity test runs without a database.
    """
    from fraudlens.features.build import build_fixture_features

    config = get_config()
    with log_duration("build SQL features for the fixture", logger):
        rows = build_fixture_features(
            get_env().postgres_conninfo,
            fixture,
            out,
            config.project_root / "sql",
            config.resolve(config.paths.processed),
        )
    typer.echo(f"Wrote {out} ({rows} rows)")


# ------------------------------------------------------------------------- Phase 4: models


@app.command("train")
def train(
    models: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated models: logreg, xgboost, lightgbm. "
            "Default: all three, or xgboost,lightgbm with --use-best-params."
        ),
    ] = None,
    use_best_params: Annotated[
        bool,
        typer.Option(
            "--use-best-params",
            help="Use the Kaggle-tuned parameters (configs/best_params.json) and register the "
            "models in the MLflow Model Registry as promotion candidates.",
        ),
    ] = False,
) -> None:
    """Train models on train, compare on validation, and log every run to MLflow."""
    import json

    import pandas as pd

    from fraudlens.models.estimators import MODEL_NAMES
    from fraudlens.models.registry import register_result
    from fraudlens.models.train import (
        comparison_table,
        configure_mlflow,
        load_split_data,
        train_and_log,
    )
    from fraudlens.models.tuning import META_KEY, params_from_payload

    config = get_config()
    default = "xgboost,lightgbm" if use_best_params else "logreg,xgboost,lightgbm"
    names = [m.strip() for m in (models or default).split(",") if m.strip()]
    unknown = [m for m in names if m not in MODEL_NAMES]
    if unknown:
        raise _fail(f"Unknown models {unknown}; choose from {list(MODEL_NAMES)}")

    stage = "tuned" if use_best_params else "baseline"
    params_by_model = {n: config.training.models[n] for n in names}
    tuning_meta: dict = {}
    if use_best_params:
        path = config.resolve(config.training.best_params_file)
        if not path.is_file():
            raise _fail(f"{path} not found; run the Kaggle tuning notebook first (Phase 5).")
        payload = json.loads(path.read_text(encoding="utf-8"))
        tuning_meta = payload.get(META_KEY, {})
        try:
            params_by_model = {n: params_from_payload(payload, n) for n in names}
        except KeyError as exc:
            raise _fail(str(exc)) from exc
        expected_split = {
            "train_end": str(config.split.train_end),
            "validation_end": str(config.split.validation_end),
        }
        if tuning_meta.get("split") != expected_split:
            raise _fail(f"best_params.json was tuned on split {tuning_meta.get('split')}, "
                        f"but config.yaml has {expected_split}.")  # fmt: skip

    tracking_uri = get_env().mlflow_tracking_uri
    configure_mlflow(tracking_uri, config.training.experiment_name)
    with log_duration("load train and validation splits", logger):
        data = load_split_data(config)
    results, versions = [], {}
    for name in names:
        tags = {"stage": stage}
        if use_best_params:
            tags["tuning.source"] = str(tuning_meta.get("source", "unknown"))
            tags["tuning.data_sha256"] = str(tuning_meta.get("data_sha256", ""))
            tags["tuning.best_val_pr_auc"] = str(
                tuning_meta.get("studies", {}).get(name, {}).get("best_val_pr_auc", "")
            )
        with log_duration(f"train and log {name} ({stage})", logger):
            result = train_and_log(
                name, params_by_model[name], config, data, run_name=f"{name}-{stage}", tags=tags
            )
        results.append(result)
        if use_best_params:
            versions[name] = register_result(result, config.training.registered_model_name)

    table = comparison_table(results)
    report = config.resolve(config.paths.reports) / f"{stage}_comparison.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        f"# {stage.capitalize()} comparison (validation split)\n\n{table.round(4).to_markdown()}\n",
        encoding="utf-8",
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        typer.echo(f"\n{table.round(4).to_string()}")
    for name, version in versions.items():
        typer.echo(f"Registered {name} as {config.training.registered_model_name} v{version}")
    if versions:
        typer.echo("Next: `fraudlens promote` to run the promotion gate.")
    typer.echo(f"\nReport: {report}\nMLflow: {tracking_uri}")


@app.command("promote")
def promote() -> None:
    """Run the promotion gate on pending registered versions; the winner becomes @champion."""
    from mlflow import MlflowClient

    from fraudlens.models.promote import run_promotion
    from fraudlens.models.train import configure_mlflow

    config = get_config()
    t = config.training
    configure_mlflow(get_env().mlflow_tracking_uri, t.experiment_name)
    with log_duration("promotion gate", logger):
        decision = run_promotion(MlflowClient(), t.registered_model_name, t.champion_alias,
                                 config.promotion)  # fmt: skip
    rules = config.promotion
    typer.echo(f"\nPromotion gate for {t.registered_model_name}")
    typer.echo(f"  rules: recall >= {rules.min_recall}, precision >= {rules.min_precision}, "
               f"PR-AUC >= champion's: {rules.require_pr_auc_at_least_champion}")  # fmt: skip
    prev = decision.previous_champion
    typer.echo(f"  current champion: v{prev.version} ({prev.family})" if prev else
               "  current champion: none")  # fmt: skip
    for r in decision.results:
        c = r.candidate
        verdict = "PASS" if r.passed else "FAIL"
        typer.echo(f"\n  v{c.version} {c.family}: {verdict} | PR-AUC {c.val_pr_auc:.4f}, "
                   f"precision {c.val_precision:.3f}, recall {c.val_recall:.3f}, "
                   f"cost ${c.val_cost:,.0f}, threshold {c.threshold:.4f}")  # fmt: skip
        for reason in r.reasons:
            typer.echo(f"      {reason}")
    for note in decision.notes:
        typer.echo(f"\n  {note}")
    if decision.winner:
        w = decision.winner
        typer.echo(f"\nWINNER: v{w.version} ({w.family}) now holds @{t.champion_alias}.")
        typer.echo(f"Decision threshold (stored as a model version tag): {w.threshold:.6f}")
        typer.echo(f"Optional override for the API in .env: FRAUD_THRESHOLD={w.threshold:.6f}")


@app.command("evaluate")
def evaluate_cmd(
    final: Annotated[
        bool, typer.Option("--final", help="Evaluate the champion once on the test set.")
    ] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Re-run even if the test set was already used.")
    ] = False,
) -> None:
    """Evaluate the @champion on the untouched test set (fraudTest), once."""
    from mlflow import MlflowClient

    from fraudlens.models.final_eval import AlreadyEvaluatedError, run_final_evaluation
    from fraudlens.models.train import configure_mlflow

    if not final:
        raise _fail("Only `fraudlens evaluate --final` exists: it uses the test set, once.")
    config = get_config()
    configure_mlflow(get_env().mlflow_tracking_uri, config.training.experiment_name)
    with log_duration("final evaluation on the test set", logger):
        try:
            _, report = run_final_evaluation(config, MlflowClient(), force=force)
        except (AlreadyEvaluatedError, RuntimeError) as exc:
            raise _fail(str(exc)) from exc
    out = config.resolve(config.paths.reports) / "final_evaluation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"Report: {out}")


# ------------------------------------------------------------------- Phase 5: Kaggle tuning


@app.command("kaggle-bundle")
def kaggle_bundle(
    kaggle_user: Annotated[
        str, typer.Option(help="Your Kaggle username (for dataset-metadata.json).")
    ] = "your-kaggle-username",
) -> None:
    """Build data/kaggle_upload/: train+validation features and the fraudlens wheel."""
    import shutil
    import subprocess

    from fraudlens.models.kaggle_bundle import build_bundle

    config = get_config()
    uv = shutil.which("uv")
    if not uv:
        raise _fail("`uv` is needed to build the wheel; see docs/setup_windows.md")
    dist = config.project_root / "dist"
    with log_duration("build fraudlens wheel", logger):
        subprocess.run(
            [uv, "build", "--wheel", "--out-dir", str(dist), "-q"],
            cwd=config.project_root,
            check=True,
        )
    wheels = sorted(dist.glob("fraudlens-*.whl"), key=lambda p: p.stat().st_mtime)
    out_dir = config.resolve(config.paths.raw).parent / "kaggle_upload"
    with log_duration("build Kaggle bundle", logger):
        manifest = build_bundle(config, out_dir, wheels[-1], kaggle_user)
    typer.echo(f"Bundle ready: {out_dir}")
    typer.echo(f"  rows: {manifest['rows']}  fraud: {manifest['fraud']}")
    typer.echo(f"  data sha256: {manifest['data_sha256'][:16]}...")


# --------------------------------------------------------- Phase 7: explainability, fairness


@app.command("explain")
def explain(
    trans_num: Annotated[str, typer.Option("--trans-num", help="Transaction id to explain.")],
) -> None:
    """Score one transaction with the champion; print SHAP factors and the analyst note."""
    import pandas as pd

    from fraudlens.explain.run import explain_transaction, score_sum_check, shap_table
    from fraudlens.models.train import configure_mlflow

    config = get_config()
    configure_mlflow(get_env().mlflow_tracking_uri, config.training.experiment_name)
    with log_duration(f"explain transaction {trans_num}", logger):
        try:
            row, exp, version, note = explain_transaction(config, get_env(), trans_num)
        except (KeyError, RuntimeError) as exc:
            raise _fail(str(exc)) from exc
    flagged = exp.score >= version.threshold
    typer.echo(f"\nTransaction {trans_num} ({row['source']} split, actual label: "
               f"{'FRAUD' if row['is_fraud'] else 'legit'})")  # fmt: skip
    typer.echo(f"Model: v{version.version} ({version.family}) | score {exp.score:.4f} | "
               f"threshold {version.threshold:.4f} | decision: "
               f"{'FLAGGED' if flagged else 'APPROVED'}")  # fmt: skip
    typer.echo(f"Additivity check: sigmoid(base {exp.base_value:.3f} + sum of SHAP) = "
               f"{score_sum_check(exp):.4f}\n")  # fmt: skip
    with pd.option_context("display.width", 220, "display.max_colwidth", 90):
        typer.echo(shap_table(exp).to_string(index=False))
    typer.echo(f"\nAnalyst note (source: {note.explanation_source}"
               + (f", {note.llm_model}, {note.latency_ms:.0f} ms" if note.llm_model else "")
               + ")")  # fmt: skip
    typer.echo(f"  {note.summary}")
    for reason in note.key_reasons:
        typer.echo(f"  - {reason}")
    typer.echo(f"  Recommended action: {note.recommended_action.upper()}")
    if note.error:
        typer.echo(f"  (LLM unavailable, used the template fallback: {note.error[:200]})")


@app.command("shap-global")
def shap_global() -> None:
    """Global SHAP summary plots for the champion, logged to its MLflow run."""
    from fraudlens.explain.run import log_global_shap
    from fraudlens.features.feature_names import label
    from fraudlens.models.train import configure_mlflow

    config = get_config()
    configure_mlflow(get_env().mlflow_tracking_uri, config.training.experiment_name)
    out = config.project_root / "docs" / "screenshots" / "shap"
    with log_duration("global SHAP plots", logger):
        importance, version = log_global_shap(config, out)
    typer.echo(f"Mean |SHAP| for champion v{version.version} (log-odds):")
    for name, value in importance.items():
        typer.echo(f"  {label(name):32s} {value:7.3f}")
    typer.echo(f"Plots: {out} and MLflow run {version.run_id} (artifacts/shap)")


@app.command("fairness-audit")
def fairness_audit() -> None:
    """Recall, precision and false positive rate by gender and age band (test set)."""
    from fraudlens.explain.run import run_fairness_audit
    from fraudlens.models.train import configure_mlflow

    config = get_config()
    configure_mlflow(get_env().mlflow_tracking_uri, config.training.experiment_name)
    with log_duration("fairness audit", logger):
        _, report, _ = run_fairness_audit(config)
    out = config.resolve(config.paths.reports) / "fairness.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    typer.echo(report)
    typer.echo(f"Report: {out}")


# ------------------------------------------------------------------- Phase 8: serving


DEMO_COLUMNS = [
    "trans_num", "trans_ts", "cc_num", "merchant", "category", "amt", "dob", "city_pop",
    "lat", "long", "merch_lat", "merch_long", "is_fraud",
]  # fmt: skip


@app.command("build-state")
def build_state_cmd() -> None:
    """Card-state snapshot for the API (end of fraudTrain) and a parity-checked demo set."""
    import datetime as dt

    import numpy as np
    import pandas as pd

    from fraudlens.features.feature_names import FEATURE_NAMES, NUMERIC_FEATURES
    from fraudlens.features.online import Transaction
    from fraudlens.features.state_io import build_state, save_state

    config = get_config()
    processed = config.resolve(config.paths.processed)
    files = _processed_files()
    with log_duration("build card state from fraudTrain", logger):
        train = pd.read_parquet(files["train"], columns=["cc_num", "trans_ts", "amt", "category"])
        store = build_state(train)
    cutoff = train["trans_ts"].max()
    meta = {
        "built_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": "fraudTrain (train + validation)",
        "cutoff": str(cutoff),
        "cards": len(store.cards),
        "transactions": len(train),
    }
    state_path = processed / "card_state.json.gz"
    save_state(store, state_path, meta)

    # Demo set: each card's FIRST transaction after the snapshot, whose true history is exactly
    # the snapshot, so the API computes the same features as the SQL. Names/street excluded.
    with log_duration("build demo set and check feature parity", logger):
        test = pd.read_parquet(files["test"], columns=DEMO_COLUMNS)
        test = test.sort_values(["trans_ts", "trans_num"], kind="stable")
        demo = test.groupby("cc_num", sort=False).head(1).reset_index(drop=True)
        sql = pd.read_parquet(config.resolve(config.paths.processed) / "features.parquet",
                              columns=["trans_num", *FEATURE_NAMES])  # fmt: skip
        sql = sql.set_index("trans_num").loc[demo["trans_num"]]
        online = pd.DataFrame(
            [store.features(Transaction.from_mapping(r)) for r in demo.to_dict("records")]
        )
        mism = 0
        for col in NUMERIC_FEATURES:
            a = pd.to_numeric(online[col]).to_numpy(dtype=float)
            b = sql[col].to_numpy(dtype=float)
            mism += int((~np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)).sum())
        mism += int((online["category"].astype(str).to_numpy()
                     != sql["category"].astype(str).to_numpy()).sum())  # fmt: skip
        demo.to_parquet(processed / "demo_transactions.parquet", index=False)
    typer.echo(f"State: {state_path} ({len(store.cards):,} cards, cutoff {cutoff}, "
               f"{state_path.stat().st_size / 1024:,.0f} KB)")  # fmt: skip
    typer.echo(f"Demo set: {len(demo):,} transactions ({int(demo['is_fraud'].sum())} fraud), "
               f"feature mismatches vs SQL: {mism}")  # fmt: skip
    if mism:
        raise _fail("Online features from the snapshot do not match the SQL features.")


@app.command("serve")
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port.")] = 8000,
) -> None:
    """Run the scoring API locally (uvicorn, one worker)."""
    import uvicorn

    uvicorn.run("fraudlens.api.main:app", host=host, port=port, access_log=False)


# ------------------------------------------------------------------- Phase 9: monitoring


@app.command("export-model")
def export_model() -> None:
    """Copy the @champion out of MLflow into models/champion/ (for MODEL_SOURCE=local)."""
    from fraudlens.api.model_loader import export_champion

    config = get_config()
    out = config.resolve(config.paths.models) / "champion"
    with log_duration("export champion model", logger):
        meta = export_champion(config, get_env(), out)
    typer.echo(f"Exported {meta['name']} v{meta['version']} ({meta['family']}), threshold "
               f"{meta['threshold']:.4f}, to {out}")  # fmt: skip


GITHUB_URL = "https://github.com/YASHR2002/fraudlens"


def _hf_token() -> str:
    token = get_env().hf_token
    if token is None or not token.get_secret_value():
        typer.echo("HF_TOKEN is not set: add a Hugging Face write token to .env", err=True)
        raise typer.Exit(1)
    return token.get_secret_value()


@app.command("push-model")
def push_model_cmd(
    repo_id: Annotated[
        str | None, typer.Option(help="Model repo, e.g. user/fraudlens-model; default HF_REPO_ID.")
    ] = None,
    private: Annotated[bool, typer.Option(help="Create the repo as private.")] = False,
) -> None:
    """Upload the exported champion, card state and demo set to a Hugging Face model repo."""
    from fraudlens.deploy.hub import push_model

    config, env = get_config(), get_env()
    repo = repo_id or env.hf_repo_id
    if not repo:
        typer.echo("Pass --repo-id or set HF_REPO_ID in .env", err=True)
        raise typer.Exit(1)
    champion = config.resolve(config.paths.models) / "champion"
    with log_duration("push model to the Hugging Face Hub", logger):
        url = push_model(config.resolve(config.paths.processed), champion, repo, _hf_token(),
                         GITHUB_URL, private)  # fmt: skip
    typer.echo(f"Uploaded to https://huggingface.co/{repo} ({url})")


@app.command("replay")
def replay_cmd(
    speed: Annotated[float, typer.Option(help="Transactions per second.")] = 20.0,
    limit: Annotated[int, typer.Option(help="How many test transactions to send.")] = 3000,
    start: Annotated[
        str | None, typer.Option(help="Start at this time (default: start of fraudTest).")
    ] = None,
    explain_every: Annotated[
        int, typer.Option(help="Send every Nth to /predict_explained (0 = never; uses Gemini).")
    ] = 0,
    url: Annotated[str, typer.Option(help="API base URL.")] = "http://127.0.0.1:8000",
) -> None:
    """Replay test transactions against the API in time order (simulated live traffic)."""
    import datetime as dt

    from fraudlens.monitoring.replay import load_replay_rows, replay

    config = get_config()
    rows = load_replay_rows(_processed_files()["test"],
                            dt.datetime.fromisoformat(start) if start else None, limit)  # fmt: skip
    typer.echo(f"Replaying {len(rows):,} transactions to {url} at {speed:g}/s "
               f"(about {len(rows) / speed / 60:.1f} min)...")  # fmt: skip
    with log_duration("replay", logger):
        summary, results = replay(rows, url, speed, explain_every)
    out = config.resolve(config.paths.reports) / "replay" / "predictions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)
    s = summary
    typer.echo(
        f"Sent {s.sent:,} ({s.first_ts} to {s.last_ts}) in {s.seconds:,.0f}s ({s.rate:.1f}/s): "
        f"scored {s.scored:,}, explained {s.explained:,}, "
        f"rejected as out of order {s.rejected_out_of_order:,}, errors {s.errors:,}."
    )
    typer.echo(f"Live performance on these labels: recall {s.recall:.3f}, precision "
               f"{s.precision:.3f} (TP {s.tp}, FP {s.fp}, FN {s.fn}). Results: {out}")  # fmt: skip


@app.command("drift-report")
def drift_report_cmd(
    reference_size: Annotated[int, typer.Option(help="Train rows in the reference.")] = 20_000,
    rebuild_reference: Annotated[bool, typer.Option(help="Re-sample the reference.")] = False,
) -> None:
    """Evidently drift reports: each test month vs a training reference (features + score)."""
    import pandas as pd

    from fraudlens.api.model_loader import load_model_bundle
    from fraudlens.monitoring.drift import (
        REFERENCE_FILE,
        build_reference,
        monthly_drift,
        reference_fraud_rate,
        render_summary,
    )

    config = get_config()
    with log_duration("load champion model", logger):
        pipeline = load_model_bundle(config, get_env()).pipeline
    ref_path = config.resolve(config.paths.reference) / REFERENCE_FILE
    with log_duration("reference sample", logger):
        if ref_path.is_file() and not rebuild_reference:
            reference = pd.read_parquet(ref_path)
        else:
            reference = build_reference(config, pipeline, reference_size, ref_path)
    out_dir = config.resolve(config.paths.reports) / "drift"
    with log_duration("monthly drift reports", logger):
        table = monthly_drift(config, pipeline, reference, out_dir)
    summary = render_summary(table, reference, reference_fraud_rate(config))
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    table.to_csv(out_dir / "drift_by_column.csv", index=False)
    typer.echo(summary)
    typer.echo(f"Reports: {out_dir}")
