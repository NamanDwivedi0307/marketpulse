-- Migration 003: FinBERT sentiment columns on news_articles
--
-- Added as columns on the existing table, not a separate table, because
-- sentiment is a 1:1 property of the article as a whole (computed once
-- from title+description), unlike article_entities which is genuinely
-- one-to-many. sentiment_scored_at being NULL is how we distinguish
-- "not yet scored" from "scored as neutral" -- a NULL label would be
-- ambiguous between those two states otherwise.

ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS sentiment_label TEXT,
    ADD COLUMN IF NOT EXISTS sentiment_confidence DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS positive_prob DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS negative_prob DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS neutral_prob DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS sentiment_scored_at TIMESTAMPTZ;

-- Supports "give me the next batch of unscored articles" efficiently.
CREATE INDEX IF NOT EXISTS idx_news_articles_unscored
    ON news_articles (published_at)
    WHERE sentiment_scored_at IS NULL;

