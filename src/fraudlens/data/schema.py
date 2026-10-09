"""Single source of truth for the raw Sparkov schema and column roles."""

from __future__ import annotations

import pyarrow as pa

# The CSVs start with an unnamed pandas index column (header is an empty string).
RAW_INDEX_COLUMN = ""

# Raw columns in file order, with the types they are parsed as. Types are explicit so a
# changed file fails loudly instead of being silently re-inferred.
RAW_SCHEMA = pa.schema(
    [
        ("trans_date_trans_time", pa.timestamp("s")),
        ("cc_num", pa.int64()),  # up to 19 digits; fits in int64 (max ~9.2e18)
        ("merchant", pa.string()),
        ("category", pa.string()),
        ("amt", pa.float64()),
        ("first", pa.string()),
        ("last", pa.string()),
        ("gender", pa.string()),
        ("street", pa.string()),
        ("city", pa.string()),
        ("state", pa.string()),
        ("zip", pa.string()),  # read as text: leading zeros were lost in the source
        ("lat", pa.float64()),
        ("long", pa.float64()),
        ("city_pop", pa.int64()),
        ("job", pa.string()),
        ("dob", pa.date32()),
        ("trans_num", pa.string()),
        ("unix_time", pa.int64()),
        ("merch_lat", pa.float64()),
        ("merch_long", pa.float64()),
        ("is_fraud", pa.int64()),
    ]
)
RAW_COLUMNS: list[str] = RAW_SCHEMA.names
EXPECTED_HEADER: list[str] = [RAW_INDEX_COLUMN, *RAW_COLUMNS]

# Processed (Parquet / PostgreSQL) schema. Types are chosen to save memory where that is
# lossless: low-cardinality text becomes dictionary-encoded (pandas "category"), small
# integers shrink. Money and coordinates stay float64: float32 would change amounts
# (107.23 -> 107.2300034) and break exact training/serving feature parity later.
PROCESSED_SCHEMA = pa.schema(
    [
        ("trans_num", pa.string()),
        ("trans_ts", pa.timestamp("s")),
        ("cc_num", pa.int64()),
        ("merchant", pa.dictionary(pa.int16(), pa.string())),
        ("category", pa.dictionary(pa.int8(), pa.string())),
        ("amt", pa.float64()),
        ("first", pa.dictionary(pa.int16(), pa.string())),
        ("last", pa.dictionary(pa.int16(), pa.string())),
        ("gender", pa.dictionary(pa.int8(), pa.string())),
        ("street", pa.dictionary(pa.int16(), pa.string())),
        ("city", pa.dictionary(pa.int16(), pa.string())),
        ("state", pa.dictionary(pa.int8(), pa.string())),
        ("zip", pa.dictionary(pa.int16(), pa.string())),
        ("lat", pa.float64()),
        ("long", pa.float64()),
        ("city_pop", pa.int32()),
        ("job", pa.dictionary(pa.int16(), pa.string())),
        ("dob", pa.date32()),
        ("unix_time", pa.int64()),
        ("merch_lat", pa.float64()),
        ("merch_long", pa.float64()),
        ("is_fraud", pa.int8()),
    ]
)
PROCESSED_COLUMNS: list[str] = PROCESSED_SCHEMA.names

# --- Column roles (see docs/decisions.md) ---
# Personally identifying: never model features. cc_num is used only as a grouping key.
PII_COLUMNS = ("first", "last", "street", "cc_num", "trans_num")
# Protected attribute: excluded from features, used only for the fairness audit.
PROTECTED_COLUMNS = ("gender",)
TARGET_COLUMN = "is_fraud"
TIMESTAMP_COLUMN = "trans_ts"

# Allowed values / ranges used by validation.
VALID_GENDERS = frozenset({"F", "M"})
LAT_RANGE = (-90.0, 90.0)
LONG_RANGE = (-180.0, 180.0)
