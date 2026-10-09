"""Training-serving parity: the Python online features must equal the SQL features.

The model is trained on features computed in PostgreSQL (``sql/02_features.sql``) but the API
computes them in Python (``fraudlens.features.online``). Any difference between the two is
training-serving skew: the model would silently see different inputs in production.

* ``test_fixture_parity`` runs everywhere: it replays the synthetic fixture through Python and
  compares against the SQL output for the same fixture, committed as
  ``tests/fixtures/sample_features_sql.csv`` (regenerate with
  ``fraudlens make-parity-fixture``).
* ``test_real_data_parity`` replays the full history of a random sample of real cards and
  compares against ``data/processed/features.parquet``; it is skipped when the data is absent.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pytest

from fraudlens.features.feature_names import FEATURE_NAMES, NUMERIC_FEATURES
from fraudlens.features.online import CardStateStore, Transaction, replay

FIXTURES = Path(__file__).parent / "fixtures"
PROCESSED = Path(__file__).parents[1] / "data" / "processed"
RTOL = 1e-9
ATOL = 1e-9


def _raw_fixture() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES / "sample_transactions.csv", dtype={"zip": str})
    df["trans_ts"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"]).dt.date
    return df


def _transactions(df: pd.DataFrame) -> list[Transaction]:
    return [Transaction.from_mapping(r) for r in df.to_dict("records")]


def _online_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replay ``df`` in time order (ties kept in file order) and return Python features."""
    ordered = df.sort_values("trans_ts", kind="stable")
    feats = pd.DataFrame(list(replay(_transactions(ordered))))
    feats.insert(0, "trans_num", ordered["trans_num"].to_numpy())
    return feats


def assert_features_match(online: pd.DataFrame, sql: pd.DataFrame) -> None:
    """Compare feature by feature; NaN/None (no history) must match NaN/NULL."""
    merged = online.merge(sql, on="trans_num", suffixes=("_py", "_sql"), validate="1:1")
    assert len(merged) == len(online) == len(sql)
    problems = []
    for name in FEATURE_NAMES:
        py, sq = merged[f"{name}_py"], merged[f"{name}_sql"]
        if name in NUMERIC_FEATURES:
            py_f = pd.to_numeric(py, errors="raise").astype(float).to_numpy()
            sq_f = pd.to_numeric(sq, errors="raise").astype(float).to_numpy()
            ok = np.isclose(py_f, sq_f, rtol=RTOL, atol=ATOL, equal_nan=True)
        else:
            ok = (py.astype(str) == sq.astype(str)).to_numpy()
        if not ok.all():
            bad = merged.loc[~ok, ["trans_num", f"{name}_py", f"{name}_sql"]].head(3)
            problems.append(f"{name}: {int((~ok).sum())} mismatches, e.g.\n{bad.to_string()}")
    assert not problems, "Training-serving skew detected:\n" + "\n".join(problems)


# --- synthetic fixture (always runs) ---------------------------------------------------


def test_fixture_contains_edge_cases() -> None:
    """The fixture must exercise ties and exact window edges, or parity proves little."""
    df = _raw_fixture()[["trans_num", "cc_num", "trans_ts"]]
    pairs = df.merge(df, on="cc_num", suffixes=("_a", "_b"))
    pairs = pairs[pairs["trans_num_a"] != pairs["trans_num_b"]]
    gaps = set((pairs["trans_ts_b"] - pairs["trans_ts_a"]).dt.total_seconds())
    for gap, what in [(0, "same-second ties"), (3_600, "1-hour"), (86_400, "24-hour"),
                      (604_800, "7-day")]:  # fmt: skip
        assert gap in gaps, f"fixture has no pair of card transactions exactly {what} apart"


def test_fixture_parity() -> None:
    sql = pd.read_csv(FIXTURES / "sample_features_sql.csv")
    assert_features_match(_online_features(_raw_fixture()), sql)


def test_tie_order_does_not_change_features() -> None:
    """Same-second transactions must get identical history whichever arrives first."""
    df = _raw_fixture()
    forward = _online_features(df).set_index("trans_num").sort_index()
    reversed_ties = df.iloc[::-1]  # stable sort then keeps ties in reverse file order
    backward = _online_features(reversed_ties).set_index("trans_num").sort_index()
    pd.testing.assert_frame_equal(forward, backward)


# --- real data (skipped without data/processed) ----------------------------------------


@pytest.mark.skipif(
    not (PROCESSED / "features.parquet").exists()
    or not (PROCESSED / "transactions_train.parquet").exists(),
    reason="real data not built (run convert-data and build-features)",
)
def test_real_data_parity() -> None:
    features = ds.dataset(PROCESSED / "features.parquet")
    cards = pc.unique(features.to_table(columns=["cc_num"])["cc_num"]).to_numpy()
    sample = np.random.default_rng(42).choice(cards, size=40, replace=False)
    card_filter = pc.field("cc_num").isin(sample.tolist())

    raw = ds.dataset(
        [PROCESSED / "transactions_train.parquet", PROCESSED / "transactions_test.parquet"]
    )
    cols = ["trans_num", "cc_num", "trans_ts", "amt", "category", "dob", "city_pop", "lat",
            "long", "merch_lat", "merch_long"]  # fmt: skip
    txns = raw.to_table(columns=cols, filter=card_filter).to_pandas()
    txns["category"] = txns["category"].astype(str)
    sql = features.to_table(columns=["trans_num", *FEATURE_NAMES], filter=card_filter).to_pandas()
    sql["category"] = sql["category"].astype(str)

    assert len(txns) == len(sql) > 10_000  # 40 cards' full two-year history
    assert_features_match(_online_features(txns), sql)


# --- smoke check of the store API ------------------------------------------------------


def test_features_does_not_mutate_state() -> None:
    txn = _transactions(_raw_fixture().head(1))[0]
    store = CardStateStore()
    first = store.features(txn)
    assert store.features(txn) == first
    assert store.cards == {}
    store.record(txn)
    later = Transaction.from_mapping(
        {**txn.__dict__, "trans_ts": txn.trans_ts + dt.timedelta(minutes=5)}
    )
    assert store.features(later)["card_txn_count_1h"] == 1
