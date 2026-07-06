-- Migration 001: quotes hypertable
--
-- A quote is a point-in-time snapshot, not an OHLCV bar, so it's kept
-- separate from candles even though both are time-series. Storing them
-- together would force one schema to awkwardly serve two different shapes
-- of data (a quote has no "period" it summarizes; a candle does).
--
-- symbol + quoted_at is deliberately NOT a primary key: Finnhub's quote
-- endpoint can return the same quoted_at timestamp on repeated polls during
-- illiquid periods (no new trade has occurred), and we want every poll
-- recorded for audit/debugging, not silently deduplicated at the DB layer.
-- Deduplication, if ever needed, belongs in a query, not a constraint.

CREATE TABLE IF NOT EXISTS quotes (
    id              BIGSERIAL,
    symbol          TEXT NOT NULL,
    current_price   DOUBLE PRECISION NOT NULL,
    change          DOUBLE PRECISION NOT NULL,
    percent_change  DOUBLE PRECISION NOT NULL,
    high_of_day     DOUBLE PRECISION NOT NULL,
    low_of_day      DOUBLE PRECISION NOT NULL,
    open_price      DOUBLE PRECISION NOT NULL,
    previous_close  DOUBLE PRECISION NOT NULL,
    quoted_at       TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol, quoted_at, id)
);

-- Turn it into a hypertable partitioned on quoted_at. chunk_time_interval
-- of 1 day is reasonable for per-minute quote polling volume; revisit if
-- polling frequency changes by an order of magnitude.
SELECT create_hypertable(
    'quotes',
    'quoted_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- The access pattern that matters most: "give me recent quotes for symbol X".
CREATE INDEX IF NOT EXISTS idx_quotes_symbol_quoted_at
    ON quotes (symbol, quoted_at DESC);

