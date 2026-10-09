"""Build the folder uploaded to Kaggle as a private dataset for hyperparameter tuning.

Contents:

* ``features_dev.parquet``: the 18 features, label and split for **train and validation only**.
  Test rows are never exported, so tuning cannot touch the test set even by accident.
* the ``fraudlens`` wheel, so the Kaggle notebook uses exactly the local pipelines and metrics;
* ``manifest.json``: split dates, row and fraud counts, the data file's SHA-256, versions;
* ``dataset-metadata.json``: for ``kaggle datasets create`` (private by default).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import shutil
from pathlib import Path

from fraudlens.config import AppConfig
from fraudlens.data.splits import load_features
from fraudlens.features.feature_names import FEATURE_NAMES, TARGET
from fraudlens.models.train import package_versions

logger = logging.getLogger(__name__)

DATA_FILE = "features_dev.parquet"
DATASET_SLUG = "fraudlens-features"


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file (streamed)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(config: AppConfig, out_dir: Path, wheel: Path, kaggle_user: str) -> dict:
    """Write the Kaggle upload folder and return its manifest."""
    if not wheel.is_file():
        raise FileNotFoundError(f"wheel not found: {wheel} (run `uv build --wheel`)")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    df = load_features(config, columns=[*FEATURE_NAMES, TARGET], splits=("train", "validation"))
    if (df["split"] == "test").any():  # defensive: load_features already excludes test
        raise RuntimeError("test rows must never be exported for tuning")
    df = df[[*FEATURE_NAMES, TARGET, "split"]]
    df["split"] = df["split"].astype(str)
    data_path = out_dir / DATA_FILE
    df.to_parquet(data_path, index=False, compression="zstd")
    shutil.copy2(wheel, out_dir / wheel.name)

    counts = df.groupby("split")[TARGET].agg(["size", "sum"])
    manifest = {
        "created_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "data_file": DATA_FILE,
        "data_sha256": sha256_of(data_path),
        "wheel": wheel.name,
        "seed": config.seed,
        "split": {
            "train_end": str(config.split.train_end),
            "validation_end": str(config.split.validation_end),
        },
        "rows": {s: int(counts.loc[s, "size"]) for s in counts.index},
        "fraud": {s: int(counts.loc[s, "sum"]) for s in counts.index},
        "features": FEATURE_NAMES,
        "local_versions": package_versions(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    metadata = {
        "title": "FraudLens features (train + validation)",
        "id": f"{kaggle_user}/{DATASET_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (out_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(
        "Kaggle bundle in %s: %s rows, %.0f MB data", out_dir, f"{len(df):,}",
        data_path.stat().st_size / 2**20,
    )  # fmt: skip
    return manifest
