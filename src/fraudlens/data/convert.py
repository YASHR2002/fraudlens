"""Convert the raw CSVs to compact, typed Parquet files, streaming block by block."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from fraudlens.data.io import iter_raw_batches
from fraudlens.data.schema import PROCESSED_SCHEMA

logger = logging.getLogger(__name__)

MB = 1024 * 1024


@dataclass(frozen=True)
class ConversionStats:
    """Size and memory figures for one converted file."""

    source: Path
    target: Path
    rows: int
    csv_bytes: int
    parquet_bytes: int
    pandas_default_bytes: int  # in-memory size with pandas' default dtypes
    pandas_optimized_bytes: int  # in-memory size with the processed dtypes
    rss_before_bytes: int
    rss_peak_bytes: int


def transform_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Map one raw batch to the processed schema.

    * ``trans_date_trans_time`` is renamed ``trans_ts``.
    * ``zip`` is left-padded to 5 digits (leading zeros were lost in the source CSV).
    * Text columns become dictionary-encoded (pandas ``category``); small ints shrink.
    """
    columns: list[pa.Array] = []
    for target in PROCESSED_SCHEMA:
        source_name = "trans_date_trans_time" if target.name == "trans_ts" else target.name
        arr = batch.column(source_name)
        if target.name == "zip":
            arr = pc.utf8_lpad(arr, width=5, padding="0")
        if pa.types.is_dictionary(target.type):
            arr = pc.dictionary_encode(arr)
        columns.append(arr.cast(target.type))
    return pa.RecordBatch.from_arrays(columns, schema=PROCESSED_SCHEMA)


def _pandas_bytes(batch: pa.RecordBatch) -> int:
    return int(batch.to_pandas().memory_usage(deep=True).sum())


def convert_csv_to_parquet(source: Path, target: Path, block_size_mb: int = 32) -> ConversionStats:
    """Stream ``source`` CSV into a zstd-compressed Parquet file at ``target``.

    Only one block is in memory at a time. The file is written to a temporary name and
    renamed at the end, so an interrupted run never leaves a partial Parquet file behind.
    """
    process = psutil.Process(os.getpid())
    rss_before = rss_peak = process.memory_info().rss
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.tmp")
    rows = default_bytes = optimized_bytes = 0
    try:
        with pq.ParquetWriter(tmp, PROCESSED_SCHEMA, compression="zstd") as writer:
            for raw in iter_raw_batches(source, block_size_mb):
                processed = transform_batch(raw)
                writer.write_batch(processed)
                rows += processed.num_rows
                default_bytes += _pandas_bytes(raw)
                optimized_bytes += _pandas_bytes(processed)
                rss_peak = max(rss_peak, process.memory_info().rss)
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)

    stats = ConversionStats(
        source=source,
        target=target,
        rows=rows,
        csv_bytes=source.stat().st_size,
        parquet_bytes=target.stat().st_size,
        pandas_default_bytes=default_bytes,
        pandas_optimized_bytes=optimized_bytes,
        rss_before_bytes=rss_before,
        rss_peak_bytes=rss_peak,
    )
    logger.info(
        "%s -> %s: %s rows | disk %.0f MB CSV -> %.0f MB Parquet (%.1fx smaller) | "
        "in-memory pandas %.0f MB default dtypes -> %.0f MB optimized | "
        "process RSS %.0f MB before, %.0f MB peak",
        source.name, target.name, f"{rows:,}",
        stats.csv_bytes / MB, stats.parquet_bytes / MB, stats.csv_bytes / stats.parquet_bytes,
        default_bytes / MB, optimized_bytes / MB,
        rss_before / MB, rss_peak / MB,
    )  # fmt: skip
    return stats
