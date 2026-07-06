-- Migration 002: news_articles + article_entities
--
-- Split into two tables rather than one denormalized table because an
-- article's entities (symbols mentioned, with per-symbol sentiment) are a
-- one-to-many relationship -- one article often mentions several tickers,
-- each with its own sentiment_score and match_score. Flattening that into
-- the articles table would mean either duplicating the article row per
-- entity (breaks "one row per article") or cramming a JSON blob in (breaks
-- the ability to efficiently query "all articles that mention AAPL").
--
-- uuid from Marketaux is the natural primary key -- it's already a unique
-- identifier assigned by the source, so no reason to introduce a surrogate
-- key on top of it.

CREATE TABLE IF NOT EXISTS news_articles (
    uuid            TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    description     TEXT NOT NULL,
    url             TEXT NOT NULL,
    source          TEXT NOT NULL,
    published_at    TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published_at
    ON news_articles (published_at DESC);

CREATE TABLE IF NOT EXISTS article_entities (
    article_uuid    TEXT NOT NULL REFERENCES news_articles(uuid) ON DELETE CASCADE,
    symbol          TEXT NOT NULL,
    name            TEXT NOT NULL,
    sentiment_score DOUBLE PRECISION,
    match_score     DOUBLE PRECISION,
    PRIMARY KEY (article_uuid, symbol)
);

-- The overwhelmingly common query: "give me recent articles mentioning
-- symbol X", joined back to news_articles for published_at ordering.
CREATE INDEX IF NOT EXISTS idx_article_entities_symbol
    ON article_entities (symbol);
