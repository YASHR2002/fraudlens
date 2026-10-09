"""Load the processed Parquet files into PostgreSQL with COPY."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from fraudlens.data.schema import PROCESSED_COLUMNS

logger = logging.getLogger(__name__)

TABLE = "transactions"
COPY_COLUMNS: list[str] = ["source", *PROCESSED_COLUMNS]


class LoadError(RuntimeError):
    """Rows in PostgreSQL do not match the source files."""


def batch_to_copy_csv(batch: pa.RecordBatch, source: str) -> bytes:
    """Render a processed batch as headerless CSV in ``COPY_COLUMNS`` order.

    Dictionary columns are decoded back to plain text; ``source`` is prepended.
    """
    arrays: list[pa.Array] = [pa.array([source] * batch.num_rows, pa.string())]
    for name in PROCESSED_COLUMNS:
        arr = batch.column(name)
        if pa.types.is_dictionary(arr.type):
            arr = arr.dictionary_decode()
        arrays.append(arr)
    table = pa.Table.from_arrays(arrays, names=COPY_COLUMNS)
    buf = io.BytesIO()
    pacsv.write_csv(table, buf, write_options=pacsv.WriteOptions(include_header=False))
    return buf.getvalue()


def copy_parquet(conn: psycopg.Connection, parquet_path: Path, source: str) -> int:
    """Stream one Parquet file into the transactions table. Returns rows sent."""
    rows = 0
    cols = ", ".join(COPY_COLUMNS)
    pf = pq.ParquetFile(parquet_path)
    with (
        conn.cursor() as cur,
        cur.copy(f"COPY {TABLE} ({cols}) FROM STDIN WITH (FORMAT csv)") as cp,
    ):
        for batch in pf.iter_batches(batch_size=100_000):
            cp.write(batch_to_copy_csv(batch, source))
            rows += batch.num_rows
    logger.info("Copied %s rows from %s as source=%s", f"{rows:,}", parquet_path.name, source)
    return rows


def load_database(conninfo: str, create_sql: Path, files: dict[str, Path]) -> dict[str, int]:
    """Recreate the transactions table and load each Parquet file with its ``source`` label.

    Args:
        conninfo: libpq connection string.
        create_sql: Path to ``sql/01_create_tables.sql``.
        files: Mapping of source label (``train`` / ``test``) to Parquet path.

    Returns:
        Row counts per source, as counted in PostgreSQL.

    Raises:
        LoadError: If the counts in PostgreSQL differ from the Parquet row counts.
    """
    expected = {src: pq.ParquetFile(path).metadata.num_rows for src, path in files.items()}
    with psycopg.connect(conninfo) as conn:
        logger.info("Creating table from %s", create_sql.name)
        conn.execute(create_sql.read_text(encoding="utf-8"))
        for src, path in files.items():
            copy_parquet(conn, path, src)
        # Refresh planner statistics so feature queries use the (cc_num, trans_ts) index.
        conn.execute(f"ANALYZE {TABLE}")
        conn.commit()
        rows = conn.execute(f"SELECT source, COUNT(*) FROM {TABLE} GROUP BY source").fetchall()
    actual = {src: int(n) for src, n in rows}
    if actual != expected:
        raise LoadError(f"Row count mismatch: PostgreSQL {actual}, Parquet {expected}")
    logger.info("Row counts match the Parquet files: %s", {k: f"{v:,}" for k, v in actual.items()})
    return actual
