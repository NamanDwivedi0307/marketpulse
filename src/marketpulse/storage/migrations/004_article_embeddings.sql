-- Migration 004: article embeddings for historical similarity search
--
-- Embedding dimension is 384 -- this must match all-MiniLM-L6-v2's output
-- size exactly, since pgvector requires a fixed dimension per column. If
-- the embedding model is ever changed to one with a different output size,
-- this column (and every stored embedding) needs to be rebuilt from
-- scratch, not just added to.
--
-- ivfflat index with cosine distance: cosine similarity is the standard
-- choice for sentence-transformer embeddings, since these models are
-- trained/normalized such that direction (not magnitude) carries meaning.
-- ivfflat trades a small amount of recall for large speed gains at scale --
-- fine here, since this is approximate nearest-neighbor search feeding a
-- statistical aggregate, not a single hard lookup.

CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE news_articles
    ADD COLUMN IF NOT EXISTS embedding vector(384);

-- ivfflat requires at least some rows to build meaningful clusters; with
-- very few rows (e.g. a fresh dev DB), a plain sequential scan is actually
-- fine and Postgres will use one automatically until the index is useful.
CREATE INDEX IF NOT EXISTS idx_news_articles_embedding
    ON news_articles USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

