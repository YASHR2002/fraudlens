"""Build the features table in PostgreSQL and export it to Parquet."""

from __future__ import annotations

import logging
from pathlib import Path

import psycopg
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from fraudlens.features.feature_names import FEATURE_TABLE_COLUMNS

logger = logging.getLogger(__name__)

# Explicit export types: counts as int32, flags as int8, text as dictionary (pandas category).
FEATURES_SCHEMA = pa.schema(
    [
        ("trans_num", pa.string()),
        ("cc_num", pa.int64()),
        ("trans_ts", pa.timestamp("s")),
        ("source", pa.dictionary(pa.int8(), pa.string())),
        ("is_fraud", pa.int8()),
        ("gender", pa.dictionary(pa.int8(), pa.string())),
        ("amt", pa.float64()),
        ("log_amt", pa.float64()),
        ("category", pa.dictionary(pa.int8(), pa.string())),
        ("hour", pa.int8()),
        ("day_of_week", pa.int8()),
        ("is_night", pa.int8()),
        ("age_at_txn", pa.int16()),
        ("log_city_pop", pa.float64()),
        ("distance_km", pa.float64()),
        ("card_txn_count_1h", pa.int32()),
        ("card_txn_count_24h", pa.int32()),
        ("card_txn_count_7d", pa.int32()),
        ("card_amt_sum_24h", pa.float64()),
        ("card_avg_amt_prior", pa.float64()),
        ("amt_to_card_avg_ratio", pa.float64()),
        ("secs_since_last_txn", pa.float64()),
        ("is_new_category_for_card", pa.int8()),
        ("card_txn_number", pa.int32()),
    ]
)
if FEATURES_SCHEMA.names != FEATURE_TABLE_COLUMNS:
    raise RuntimeError("FEATURES_SCHEMA is out of sync with feature_names.FEATURE_TABLE_COLUMNS")

# CSV parse types: dictionary columns are read as plain text, then encoded.
_CSV_TYPES = {
    f.name: f.type.value_type if pa.types.is_dictionary(f.type) else f.type for f in FEATURES_SCHEMA
}


def run_feature_sql(conn: psycopg.Connection, sql_path: Path) -> int:
    """Execute ``sql/02_features.sql`` and return the number of feature rows."""
    logger.info("Running %s (a few minutes on the full dataset)", sql_path.name)
    conn.execute(sql_path.read_text(encoding="utf-8"))
    conn.commit()
    (rows,) = conn.execute("SELECT COUNT(*) FROM features").fetchone()
    return int(rows)


def export_features(conn: psycopg.Connection, target: Path, tmp_dir: Path) -> int:
    """Stream the features table (ordered by time) into a typed Parquet file.

    PostgreSQL writes CSV via ``COPY TO STDOUT`` into a temporary file, which Arrow then reads
    with the explicit schema. Returns the number of rows written.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = tmp_dir / "features_export.csv.tmp"
    tmp_parquet = target.with_suffix(".parquet.tmp")
    cols = ", ".join(FEATURE_TABLE_COLUMNS)
    query = (
        f"COPY (SELECT {cols} FROM features ORDER BY trans_ts, trans_num) TO STDOUT (FORMAT csv)"
    )
    try:
        with tmp_csv.open("wb") as fh, conn.cursor() as cur, cur.copy(query) as cp:
            for chunk in cp:
                fh.write(chunk)
        rows = 0
        # Open the CSV ourselves so it is closed before the cleanup below (Windows locks open
        # files, and Arrow's reader would otherwise keep it open until garbage collection).
        with (
            tmp_csv.open("rb") as src,
            pq.ParquetWriter(tmp_parquet, FEATURES_SCHEMA, compression="zstd") as writer,
        ):
            reader = pacsv.open_csv(
                src,
                read_options=pacsv.ReadOptions(
                    column_names=FEATURE_TABLE_COLUMNS, block_size=32 << 20
                ),
                convert_options=pacsv.ConvertOptions(column_types=_CSV_TYPES),
            )
            for batch in reader:
                arrays = [
                    batch.column(f.name).dictionary_encode().cast(f.type)
                    if pa.types.is_dictionary(f.type)
                    else batch.column(f.name)
                    for f in FEATURES_SCHEMA
                ]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=FEATURES_SCHEMA))
                rows += batch.num_rows
        tmp_parquet.replace(target)
    finally:
        tmp_csv.unlink(missing_ok=True)
        tmp_parquet.unlink(missing_ok=True)
    logger.info("Exported %s feature rows to %s", f"{rows:,}", target)
    return rows


FIXTURE_SCHEMA = "parity_fixture"


def _with_search_path(conninfo: str, schema: str) -> str:
    sep = "&" if "?" in conninfo else "?"
    return f"{conninfo}{sep}options=-csearch_path%3D{schema}"


def build_fixture_features(
    conninfo: str, fixture_csv: Path, out_csv: Path, sql_dir: Path, tmp_dir: Path
) -> int:
    """Run the real SQL pipeline on the synthetic fixture and save its features as CSV.

    The fixture is loaded into a throwaway schema (``parity_fixture``) so the real tables are
    untouched. The CSV is committed and used by the parity test, which then needs no database.
    """
    from fraudlens.data.convert import convert_csv_to_parquet
    from fraudlens.data.load_db import load_database

    tmp_dir.mkdir(parents=True, exist_ok=True)
    raw_parquet = tmp_dir / "fixture_transactions.parquet"
    features_parquet = tmp_dir / "fixture_features.parquet"
    convert_csv_to_parquet(fixture_csv, raw_parquet)
    with psycopg.connect(conninfo, autocommit=True) as admin:
        admin.execute(f"DROP SCHEMA IF EXISTS {FIXTURE_SCHEMA} CASCADE")
        admin.execute(f"CREATE SCHEMA {FIXTURE_SCHEMA}")
    try:
        scoped = _with_search_path(conninfo, FIXTURE_SCHEMA)
        load_database(scoped, sql_dir / "01_create_tables.sql", {"train": raw_parquet})
        with psycopg.connect(scoped) as conn:
            run_feature_sql(conn, sql_dir / "02_features.sql")
            rows = export_features(conn, features_parquet, tmp_dir)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pq.read_table(features_parquet).to_pandas().to_csv(
            out_csv, index=False, lineterminator="\n", float_format="%.17g"
        )
    finally:
        with psycopg.connect(conninfo, autocommit=True) as admin:
            admin.execute(f"DROP SCHEMA IF EXISTS {FIXTURE_SCHEMA} CASCADE")
        raw_parquet.unlink(missing_ok=True)
        features_parquet.unlink(missing_ok=True)
    logger.info("Wrote %s SQL feature rows for the fixture to %s", rows, out_csv)
    return rows
