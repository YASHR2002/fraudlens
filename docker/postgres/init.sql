-- Runs once, when the PostgreSQL data volume is first created.
-- Tables and indexes are created by `fraudlens load-db` (sql/01_create_tables.sql) so that
-- schema changes don't require recreating the volume.

-- Store and display timestamps in UTC (Sparkov times have no timezone). The server is also
-- started with timezone=UTC; this keeps the setting even if the server flags change.
DO $$
BEGIN
    EXECUTE format('ALTER DATABASE %I SET timezone TO %L', current_database(), 'UTC');
END
$$;
