-- Migration 005: event_outcomes -- forward returns following a news event
--
-- One row per (article, symbol, horizon) -- an article can mention several
-- symbols, and we care about multiple time horizons (e.g. 1h vs 24h) per
-- symbol, since a "true" reaction and an initial overreaction can look very
-- different depending on the window measured.
--
-- entry_price/exit_price are stored alongside return_pct (not just the
-- computed return) so outcomes are auditable -- if a return looks wrong,
-- it should be checkable against the actual two quotes used, not just
-- trusted as a black-box number.

CREATE TABLE IF NOT EXISTS event_outcomes (
    article_uuid    TEXT NOT NULL REFERENCES news_articles(uuid) ON DELETE CASCADE,
    symbol          TEXT NOT NULL,
    horizon_minutes INTEGER NOT NULL,
    entry_price     DOUBLE PRECISION NOT NULL,
    entry_quoted_at TIMESTAMPTZ NOT NULL,
    exit_price      DOUBLE PRECISION NOT NULL,
    exit_quoted_at  TIMESTAMPTZ NOT NULL,
    return_pct      DOUBLE PRECISION NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (article_uuid, symbol, horizon_minutes)
);

CREATE INDEX IF NOT EXISTS idx_event_outcomes_symbol
    ON event_outcomes (symbol);

