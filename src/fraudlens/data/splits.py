"""Time-based train / validation / test split, and loading features by split.

The split dates live in ``configs/config.yaml`` (inclusive calendar days of ``trans_ts``):

* train      = fraudTrain rows up to ``split.train_end``
* validation = fraudTrain rows after that, up to ``split.validation_end``
* test       = every fraudTest row (used once, for the final report)
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from fraudlens.config import AppConfig, SplitConfig

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
SPLIT_DTYPE = pd.CategoricalDtype(list(SPLITS), ordered=True)


def _next_midnight(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(day) + pd.Timedelta(days=1)


def assign_split(trans_ts: pd.Series, source: pd.Series, split: SplitConfig) -> pd.Series:
    """Label each row ``train``, ``validation`` or ``test``.

    Raises:
        ValueError: If the split dates are not set, or a fraudTrain row falls after
            ``validation_end`` (the dates would not cover the data).
    """
    if split.train_end is None or split.validation_end is None:
        raise ValueError("split.train_end and split.validation_end must be set in config.yaml")
    train_cut = _next_midnight(split.train_end)
    val_cut = _next_midnight(split.validation_end)
    is_test = (source.astype(str) == "test").to_numpy()
    ts = trans_ts.to_numpy()
    labels = np.select(
        [is_test, ts < train_cut.to_datetime64(), ts < val_cut.to_datetime64()],
        ["test", "train", "validation"],
        default="unassigned",
    )
    if (labels == "unassigned").any():
        n = int((labels == "unassigned").sum())
        raise ValueError(f"{n} fraudTrain rows fall after validation_end={split.validation_end}")
    return pd.Series(pd.Categorical(labels, dtype=SPLIT_DTYPE), index=trans_ts.index, name="split")


def features_path(config: AppConfig) -> Path:
    """Location of the exported features Parquet file."""
    return config.resolve(config.paths.processed) / "features.parquet"


def load_features(
    config: AppConfig,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] = SPLITS,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load features (only the requested columns) with a ``split`` column, filtered to ``splits``.

    ``trans_ts`` and ``source`` are always read because the split is derived from them.
    """
    unknown = set(splits) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}; expected a subset of {SPLITS}")
    path = path or features_path(config)
    needed = list(dict.fromkeys([*(columns or []), "trans_ts", "source"])) if columns else None
    dataset = ds.dataset(path)
    # Test rows are skipped at read time unless requested, so they are never even loaded.
    row_filter = None if "test" in splits else ds.field("source") != "test"
    df = dataset.to_table(columns=needed, filter=row_filter).to_pandas()
    df["split"] = assign_split(df["trans_ts"], df["source"], config.split)
    df = df[df["split"].isin(splits)].reset_index(drop=True)
    logger.info(
        "Loaded %s rows (%s) with %d columns from %s",
        f"{len(df):,}", ", ".join(splits), df.shape[1], path.name,
    )  # fmt: skip
    return df
