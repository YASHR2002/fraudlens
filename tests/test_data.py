"""Phase 1 tests: schema checks, data quality rules, Parquet conversion, COPY formatting.

All tests run on the synthetic fixture; no real data, Docker or network needed.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fraudlens.config import EnvSettings
from fraudlens.data.convert import convert_csv_to_parquet
from fraudlens.data.download import missing_files
from fraudlens.data.io import SchemaError, check_header, iter_raw_batches
from fraudlens.data.load_db import COPY_COLUMNS, batch_to_copy_csv
from fraudlens.data.schema import PROCESSED_COLUMNS
from fraudlens.data.synthetic import generate_sample_transactions
from fraudlens.data.validate import DataQualityError, profile_file, run_checks, validate_data

FIXTURE = Path(__file__).parent / "fixtures" / "sample_transactions.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    df = pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)
    return df.rename(columns={"Unnamed: 0": ""})  # pandas renames the empty index header


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False, lineterminator="\n")
    return path


@pytest.fixture
def split_files(raw_df: pd.DataFrame, tmp_path: Path) -> tuple[Path, Path]:
    """Fixture split in time order into a 'train' and a 'test' file, like the real data."""
    return _write(raw_df.iloc[:150], tmp_path / "train.csv"), _write(
        raw_df.iloc[150:], tmp_path / "test.csv"
    )


# --- fixture ---------------------------------------------------------------------------


def test_fixture_is_reproducible_from_the_generator() -> None:
    generated = generate_sample_transactions(seed=42).to_csv(index=False, lineterminator="\n")
    assert FIXTURE.read_text(encoding="utf-8") == generated


def test_fixture_has_fraud_and_multi_transaction_cards(raw_df: pd.DataFrame) -> None:
    assert len(raw_df) == 200
    assert 0 < raw_df["is_fraud"].astype(int).sum() < len(raw_df)
    assert raw_df.groupby("cc_num").size().min() > 5


# --- schema ----------------------------------------------------------------------------


def test_header_check_passes_on_expected_schema() -> None:
    check_header(FIXTURE)


def test_missing_column_raises_schema_error(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    path = _write(raw_df.drop(columns="amt"), tmp_path / "bad.csv")
    with pytest.raises(SchemaError, match="amt"):
        check_header(path)


def test_unparseable_value_raises_schema_error(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    raw_df.loc[3, "amt"] = "not-a-number"
    path = _write(raw_df, tmp_path / "bad.csv")
    with pytest.raises(SchemaError, match="declared type"):
        list(iter_raw_batches(path))


# --- data quality rules ----------------------------------------------------------------


def test_profile_counts(split_files: tuple[Path, Path]) -> None:
    prof = profile_file(split_files[0])
    assert prof.rows == 150
    assert prof.duplicate_trans_num == 0
    assert prof.out_of_order == 0
    assert prof.short_zip > 0  # fixture mimics zips that lost their leading zero


def test_clean_split_passes_all_error_checks(
    split_files: tuple[Path, Path], tmp_path: Path
) -> None:
    report = tmp_path / "dq.md"
    checks = validate_data(*split_files, report)
    assert all(c.passed for c in checks if c.severity == "error")
    text = report.read_text(encoding="utf-8")
    assert "Overall status: **PASSED**" in text
    assert "## Transactions and fraud by month" in text


@pytest.mark.parametrize(
    ("column", "value", "failing_check"),
    [
        ("amt", "-5.00", "amount > 0"),
        ("lat", "123.0", "valid lat/long"),
        ("gender", "X", "gender in"),
        ("is_fraud", "2", "is_fraud in"),
    ],
)
def test_bad_values_fail_error_checks(
    raw_df: pd.DataFrame, tmp_path: Path, column: str, value: str, failing_check: str
) -> None:
    raw_df.loc[5, column] = value
    train = _write(raw_df.iloc[:150], tmp_path / "train.csv")
    test = _write(raw_df.iloc[150:], tmp_path / "test.csv")
    with pytest.raises(DataQualityError, match=failing_check.split()[0]):
        validate_data(train, test, tmp_path / "dq.md")
    assert "**FAILED**" in (tmp_path / "dq.md").read_text(encoding="utf-8")


def test_duplicates_and_overlap_are_detected(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    train = _write(pd.concat([raw_df.iloc[:150], raw_df.iloc[[0]]]), tmp_path / "train.csv")
    test = _write(raw_df.iloc[140:], tmp_path / "test.csv")  # overlaps train in time and ids
    failed = {c.name for c in run_checks(profile_file(train), profile_file(test)) if not c.passed}
    assert "train.csv: trans_num unique" in failed
    assert "train/test: no shared trans_num" in failed
    assert "train/test: test starts after train ends" in failed
    assert "train.csv: sorted by time" in failed


# --- conversion ------------------------------------------------------------------------


def test_convert_produces_typed_parquet(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    stats = convert_csv_to_parquet(FIXTURE, target)
    assert stats.rows == 200
    assert not target.with_suffix(".parquet.tmp").exists()

    df = pd.read_parquet(target)
    assert list(df.columns) == PROCESSED_COLUMNS
    assert (df["zip"].str.len() == 5).all()
    assert isinstance(df["category"].dtype, pd.CategoricalDtype)
    assert str(df["is_fraud"].dtype) == "int8"
    assert str(df["city_pop"].dtype) == "int32"
    assert str(df["amt"].dtype) == "float64"  # money stays exact enough for parity tests
    assert pd.api.types.is_datetime64_any_dtype(df["trans_ts"])
    assert stats.pandas_optimized_bytes < stats.pandas_default_bytes


def test_convert_keeps_values(raw_df: pd.DataFrame, tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    convert_csv_to_parquet(FIXTURE, target)
    df = pd.read_parquet(target)
    assert df["trans_num"].tolist() == raw_df["trans_num"].tolist()
    assert df["amt"].round(2).tolist() == raw_df["amt"].astype(float).tolist()
    assert df["zip"].tolist() == raw_df["zip"].str.zfill(5).tolist()


# --- COPY formatting -------------------------------------------------------------------


def test_copy_csv_has_source_and_decoded_values(tmp_path: Path) -> None:
    target = tmp_path / "out.parquet"
    convert_csv_to_parquet(FIXTURE, target)
    batch = next(pq.ParquetFile(target).iter_batches(batch_size=10))
    rows = list(csv.reader(io.StringIO(batch_to_copy_csv(batch, "train").decode())))
    assert len(rows) == 10
    assert all(len(r) == len(COPY_COLUMNS) for r in rows)
    first = dict(zip(COPY_COLUMNS, rows[0], strict=True))
    assert first["source"] == "train"
    assert len(first["zip"]) == 5
    assert first["trans_ts"].startswith("2019-01-01")


# --- misc ------------------------------------------------------------------------------


def test_missing_files_detects_absent_and_empty(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x", encoding="utf-8")
    (tmp_path / "b.csv").touch()
    assert missing_files(tmp_path, ["a.csv", "b.csv", "c.csv"]) == ["b.csv", "c.csv"]


def test_empty_env_values_are_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_THRESHOLD", "")
    assert EnvSettings(_env_file=None).fraud_threshold is None
