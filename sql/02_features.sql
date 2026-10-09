-- Behavioural features for every transaction, materialised as the `features` table.
-- Run by `fraudlens build-features`; must match src/fraudlens/features/online.py exactly
-- (checked by tests/test_feature_parity.py).
--
-- No leakage: every history feature looks only at transactions on the same card that are
-- STRICTLY EARLIER than the current one. Frames end at CURRENT ROW with EXCLUDE GROUP, which
-- drops the current row and any other transaction in the same second, so ties are never
-- counted as "before" each other, whatever order they arrive in. Windows include their far
-- edge: the 1-hour window is [t - 1 hour, t). Train and test are computed together in time
-- order, so a test transaction sees its card's real earlier history (legitimate past data).
-- Nothing here reads is_fraud of any row, and gender / names / street / trans_num are never
-- used as inputs (gender is carried through only for the fairness audit).

DROP TABLE IF EXISTS features;

CREATE TABLE features AS
WITH history AS (
    SELECT
        t.trans_num,
        t.cc_num,
        t.trans_ts,
        t.source,
        t.is_fraud,
        t.gender,
        t.amt,
        t.category,
        t.dob,
        t.city_pop,
        t.lat,
        t.long,
        t.merch_lat,
        t.merch_long,
        COUNT(*)      OVER card_1h       AS card_txn_count_1h,
        COUNT(*)      OVER card_24h      AS card_txn_count_24h,
        COUNT(*)      OVER card_7d       AS card_txn_count_7d,
        SUM(t.amt)    OVER card_24h      AS card_amt_sum_24h,
        COUNT(*)      OVER card_all      AS card_txn_number,
        AVG(t.amt)    OVER card_all      AS card_avg_amt_prior,
        MAX(t.trans_ts) OVER card_all    AS prev_trans_ts,
        COUNT(*)      OVER card_cat_all  AS card_category_txns_prior
    FROM transactions AS t
    WINDOW
        card         AS (PARTITION BY t.cc_num ORDER BY t.trans_ts),
        card_1h      AS (card RANGE BETWEEN INTERVAL '1 hour'  PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        card_24h     AS (card RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        card_7d      AS (card RANGE BETWEEN INTERVAL '7 days'  PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        card_all     AS (card RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP),
        card_cat_all AS (PARTITION BY t.cc_num, t.category ORDER BY t.trans_ts
                         RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)
)
SELECT
    -- identifiers, label and audit-only columns (never model inputs)
    trans_num,
    cc_num,
    trans_ts,
    source,
    is_fraud,
    gender,
    -- transaction features
    amt,
    LN(1 + amt)                                             AS log_amt,
    category,
    EXTRACT(HOUR FROM trans_ts)::INTEGER                    AS hour,
    EXTRACT(ISODOW FROM trans_ts)::INTEGER                  AS day_of_week,
    (EXTRACT(HOUR FROM trans_ts) >= 22
        OR EXTRACT(HOUR FROM trans_ts) < 6)::INTEGER        AS is_night,
    DATE_PART('year', AGE(trans_ts::DATE, dob))::INTEGER    AS age_at_txn,
    LN(1 + city_pop)                                        AS log_city_pop,
    -- haversine distance, mean Earth radius 6371 km
    2 * 6371.0 * ASIN(SQRT(
        POWER(SIN(RADIANS(merch_lat - lat) / 2), 2)
        + COS(RADIANS(lat)) * COS(RADIANS(merch_lat))
          * POWER(SIN(RADIANS(merch_long - long) / 2), 2)
    ))                                                      AS distance_km,
    -- card history features (strictly earlier transactions only)
    card_txn_count_1h::INTEGER                              AS card_txn_count_1h,
    card_txn_count_24h::INTEGER                             AS card_txn_count_24h,
    card_txn_count_7d::INTEGER                              AS card_txn_count_7d,
    COALESCE(card_amt_sum_24h, 0)                           AS card_amt_sum_24h,
    card_avg_amt_prior,                                     -- NULL when no history
    amt / NULLIF(card_avg_amt_prior, 0)                     AS amt_to_card_avg_ratio,
    EXTRACT(EPOCH FROM trans_ts - prev_trans_ts)::DOUBLE PRECISION
                                                            AS secs_since_last_txn,
    (card_category_txns_prior = 0)::INTEGER                 AS is_new_category_for_card,
    card_txn_number::INTEGER                                AS card_txn_number
FROM history;

ALTER TABLE features ADD PRIMARY KEY (trans_num);
CREATE INDEX idx_features_card_ts ON features (cc_num, trans_ts);
ANALYZE features;
