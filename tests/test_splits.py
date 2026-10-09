"""Tests for the time-based split."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fraudlens.config import SplitConfig, load_config
from fraudlens.data.splits import assign_split, load_features

SPLIT = SplitConfig(train_end=dt.date(2020, 4, 21), validation_end=dt.date(2020, 6, 21))


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trans_ts": pd.to_datetime(
                [
                    "2019-01-01 00:00:00",
                    "2020-04-21 23:59:59",  # last second of train
                    "2020-04-22 00:00:00",  # first second of validation
                    "2020-06-21 12:13:37",  # last fraudTrain row
                    "2020-06-21 12:14:25",  # first fraudTest row, same calendar day
                ]
            ),
            "source": ["train", "train", "train", "train", "test"],
        }
    )


def test_boundaries_are_inclusive_calendar_days() -> None:
    df = _frame()
    assert assign_split(df["trans_ts"], df["source"], SPLIT).tolist() == [
        "train", "train", "validation", "validation", "test",
    ]  # fmt: skip


def test_rows_after_validation_end_are_an_error() -> None:
    df = _frame()
    early = SplitConfig(train_end=dt.date(2020, 1, 1), validation_end=dt.date(2020, 5, 1))
    with pytest.raises(ValueError, match="after validation_end"):
        assign_split(df["trans_ts"], df["source"], early)


def test_unset_dates_are_an_error() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="must be set"):
        assign_split(df["trans_ts"], df["source"], SplitConfig())


def test_load_features_never_reads_test_unless_asked(tmp_path: Path) -> None:
    df = _frame().assign(amt=[1.0, 2.0, 3.0, 4.0, 5.0])
    path = tmp_path / "features.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    config = load_config(Path(__file__).parents[1] / "configs" / "config.yaml")

    dev = load_features(config, columns=["amt"], splits=("train", "validation"), path=path)
    assert dev["split"].tolist() == ["train", "train", "validation", "validation"]
    assert "test" not in set(dev["source"])

    test = load_features(config, columns=["amt"], splits=("test",), path=path)
    assert test["amt"].tolist() == [5.0]

    with pytest.raises(ValueError, match="unknown splits"):
        load_features(config, splits=("holdout",), path=path)
