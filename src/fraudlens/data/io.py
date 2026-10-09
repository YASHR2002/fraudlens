"""Memory-friendly reading of the raw CSV files."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

from fraudlens.data.schema import EXPECTED_HEADER, RAW_COLUMNS, RAW_SCHEMA

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """The raw file does not have the expected columns or types."""


def read_header(path: Path) -> list[str]:
    """Return the column names in the first line of a CSV file."""
    with path.open(newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def check_header(path: Path) -> None:
    """Raise :class:`SchemaError` unless the header matches the expected raw schema exactly."""
    header = read_header(path)
    if header == EXPECTED_HEADER:
        return
    missing = [c for c in EXPECTED_HEADER if c not in header]
    unexpected = [c for c in header if c not in EXPECTED_HEADER]
    raise SchemaError(
        f"{path.name}: header mismatch. Missing: {missing or 'none'}; "
        f"unexpected: {unexpected or 'none'}; "
        f"order differs: {not missing and not unexpected}."
    )


def iter_raw_batches(path: Path, block_size_mb: int = 32) -> Iterator[pa.RecordBatch]:
    """Stream a raw CSV as Arrow record batches with the explicit raw types.

    Only one block (``block_size_mb``) of the file is held in memory at a time. The unnamed
    index column is dropped. A value that cannot be parsed as its declared type raises
    :class:`SchemaError`.

    Args:
        path: Path to ``fraudTrain.csv`` or ``fraudTest.csv``.
        block_size_mb: Size of each block read from disk.
    """
    check_header(path)
    try:
        # Opening the reader already parses the first block, so it belongs inside the try.
        reader = pacsv.open_csv(
            path,
            read_options=pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024),
            convert_options=pacsv.ConvertOptions(
                column_types=RAW_SCHEMA,
                include_columns=RAW_COLUMNS,
                strings_can_be_null=False,
            ),
        )
        yield from reader
    except pa.ArrowInvalid as exc:
        raise SchemaError(f"{path.name}: value does not match the declared type: {exc}") from exc
