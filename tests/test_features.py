"""Unit tests for the online feature calculations (edge cases spelled out by hand)."""

from __future__ import annotations

import datetime as dt
import math

import pytest

from fraudlens.features.feature_names import (
    FEATURE_NAMES,
    FEATURE_TABLE_COLUMNS,
    FEATURES,
    get_feature,
    label,
)
from fraudlens.features.online import (
    CardStateStore,
    OutOfOrderError,
    Transaction,
    age_in_years,
    haversine_km,
    static_features,
)

T0 = dt.datetime(2020, 3, 1, 12, 0, 0)


def txn(
    seconds: float = 0, amt: float = 50.0, category: str = "grocery_pos", card: int = 1
) -> Transaction:
    return Transaction(
        cc_num=card,
        trans_ts=T0 + dt.timedelta(seconds=seconds),
        amt=amt,
        category=category,
        dob=dt.date(1980, 6, 15),
        city_pop=1000,
        lat=40.0,
        long=-75.0,
        merch_lat=40.5,
        merch_long=-75.5,
    )


# --- static features -------------------------------------------------------------------


def test_haversine_known_distances() -> None:
    assert haversine_km(40.0, -75.0, 40.0, -75.0) == 0.0
    # One degree of latitude is about 111.19 km with R = 6371 km.
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.01)
    # New York to Los Angeles, roughly 3,936 km.
    assert haversine_km(40.7128, -74.0060, 34.0522, -118.2437) == pytest.approx(3936, rel=0.01)


@pytest.mark.parametrize(
    ("on", "dob", "expected"),
    [
        (dt.date(2020, 6, 14), dt.date(1980, 6, 15), 39),  # day before birthday
        (dt.date(2020, 6, 15), dt.date(1980, 6, 15), 40),  # on birthday
        (dt.date(2019, 2, 28), dt.date(1988, 2, 29), 30),  # leap-day birthday, non-leap year
        (dt.date(2019, 3, 1), dt.date(1988, 2, 29), 31),
        (dt.date(2020, 2, 29), dt.date(1988, 2, 29), 32),
    ],
)
def test_age_in_years(on: dt.date, dob: dt.date, expected: int) -> None:
    assert age_in_years(on, dob) == expected


@pytest.mark.parametrize(("hour", "night"), [(5, 1), (6, 0), (21, 0), (22, 1), (0, 1)])
def test_is_night_boundaries(hour: int, night: int) -> None:
    t = txn()
    t = Transaction(**{**t.__dict__, "trans_ts": T0.replace(hour=hour, minute=59)})
    assert static_features(t)["is_night"] == night


def test_static_values() -> None:
    f = static_features(txn(amt=99.0))
    assert f["log_amt"] == pytest.approx(math.log(100.0))
    assert f["day_of_week"] == 7  # 2020-03-01 was a Sunday (ISO 7)
    assert f["age_at_txn"] == 39


# --- history features ------------------------------------------------------------------


def test_first_transaction_has_no_history() -> None:
    f = CardStateStore().process(txn())
    assert f["card_txn_number"] == 0
    assert f["card_txn_count_1h"] == f["card_txn_count_24h"] == f["card_txn_count_7d"] == 0
    assert f["card_amt_sum_24h"] == 0.0
    assert f["card_avg_amt_prior"] is None
    assert f["amt_to_card_avg_ratio"] is None
    assert f["secs_since_last_txn"] is None
    assert f["is_new_category_for_card"] == 1


def test_averages_ratio_and_gap() -> None:
    store = CardStateStore()
    store.process(txn(0, amt=10.0))
    store.process(txn(60, amt=30.0))
    f = store.process(txn(120, amt=100.0))
    assert f["card_txn_number"] == 2
    assert f["card_avg_amt_prior"] == pytest.approx(20.0)
    assert f["amt_to_card_avg_ratio"] == pytest.approx(5.0)
    assert f["secs_since_last_txn"] == 60.0
    assert f["card_amt_sum_24h"] == pytest.approx(40.0)


def test_window_edges_are_inclusive_at_the_far_end() -> None:
    store = CardStateStore()
    store.process(txn(0))
    exactly_1h = store.features(txn(3_600))
    just_over_1h = store.features(txn(3_601))
    assert exactly_1h["card_txn_count_1h"] == 1  # [t - 1h, t) includes t - 1h
    assert just_over_1h["card_txn_count_1h"] == 0
    assert store.features(txn(86_400))["card_txn_count_24h"] == 1
    assert store.features(txn(86_401))["card_txn_count_24h"] == 0
    assert store.features(txn(7 * 86_400))["card_txn_count_7d"] == 1
    assert store.features(txn(7 * 86_400 + 1))["card_txn_count_7d"] == 0
    # All-time features never expire.
    assert store.features(txn(30 * 86_400))["card_txn_number"] == 1


def test_same_second_transactions_are_not_earlier_than_each_other() -> None:
    store = CardStateStore()
    store.process(txn(0, amt=10.0, category="travel"))
    a = store.process(txn(100, amt=20.0, category="misc_net"))
    b = store.process(txn(100, amt=40.0, category="misc_net"))
    for f in (a, b):
        assert f["card_txn_number"] == 1
        assert f["card_avg_amt_prior"] == pytest.approx(10.0)
        assert f["secs_since_last_txn"] == 100.0
        assert f["is_new_category_for_card"] == 1
    c = store.process(txn(160, amt=1.0, category="misc_net"))
    assert c["card_txn_number"] == 3
    assert c["secs_since_last_txn"] == 60.0
    assert c["is_new_category_for_card"] == 0


def test_cards_are_independent() -> None:
    store = CardStateStore()
    store.process(txn(0, card=1))
    assert store.features(txn(10, card=2))["card_txn_number"] == 0


def test_out_of_order_transaction_is_rejected() -> None:
    store = CardStateStore()
    store.process(txn(100))
    with pytest.raises(OutOfOrderError):
        store.process(txn(50))


def test_old_entries_are_evicted_but_counted_all_time() -> None:
    store = CardStateStore()
    for i in range(5):
        store.process(txn(i * 86_400))  # one per day
    store.process(txn(20 * 86_400))
    assert len(store.cards[1].recent) == 1  # only the last week is kept in memory
    assert store.cards[1].n_total == 6


# --- catalogue -------------------------------------------------------------------------


def test_catalogue_is_consistent() -> None:
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 18
    assert "gender" not in FEATURE_NAMES  # protected attribute: audit only
    for pii in ("first", "last", "street", "cc_num", "trans_num"):
        assert pii not in FEATURE_NAMES
    assert set(FEATURE_NAMES) <= set(FEATURE_TABLE_COLUMNS)
    assert all(f.description.endswith(".") for f in FEATURES)
    assert label("distance_km") == "Distance to merchant"
    assert get_feature("amt").unit == "USD"
    assert set(CardStateStore().features(txn())) == set(FEATURE_NAMES)
