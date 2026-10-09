-- Raw transactions from both Sparkov files, in one table.
-- Run by `fraudlens load-db`, which then bulk-loads the processed Parquet files with COPY.
-- Re-running rebuilds the table from scratch (anything derived from it is rebuilt too).

DROP TABLE IF EXISTS transactions CASCADE;

CREATE TABLE transactions (
    trans_num   TEXT             PRIMARY KEY,
    source      TEXT             NOT NULL CHECK (source IN ('train', 'test')),
    trans_ts    TIMESTAMP        NOT NULL,  -- event time; source data has no timezone
    cc_num      BIGINT           NOT NULL,  -- card id: grouping key only, never a feature
    merchant    TEXT             NOT NULL,
    category    TEXT             NOT NULL,
    amt         DOUBLE PRECISION NOT NULL CHECK (amt > 0),
    first       TEXT             NOT NULL,  -- PII: never a feature
    last        TEXT             NOT NULL,  -- PII: never a feature
    gender      CHAR(1)          NOT NULL,  -- protected: fairness audit only
    street      TEXT             NOT NULL,  -- PII: never a feature
    city        TEXT             NOT NULL,
    state       CHAR(2)          NOT NULL,
    zip         CHAR(5)          NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    long        DOUBLE PRECISION NOT NULL,
    city_pop    INTEGER          NOT NULL,
    job         TEXT             NOT NULL,
    dob         DATE             NOT NULL,
    unix_time   BIGINT           NOT NULL,  -- inconsistent with trans_ts in the source; unused
    merch_lat   DOUBLE PRECISION NOT NULL,
    merch_long  DOUBLE PRECISION NOT NULL,
    is_fraud    SMALLINT         NOT NULL CHECK (is_fraud IN (0, 1))
);

-- Per-card history in time order: drives every window feature in sql/02_features.sql.
CREATE INDEX idx_transactions_card_ts ON transactions (cc_num, trans_ts);
CREATE INDEX idx_transactions_source ON transactions (source);

COMMENT ON TABLE transactions IS
    'Sparkov credit card transactions (fraudTrain = train, fraudTest = test).';
