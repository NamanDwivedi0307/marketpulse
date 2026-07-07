# MarketPulse

A real-time AI-driven financial event and price-movement analysis platform.
Ingests live market data and financial news, scores sentiment with FinBERT,
embeds articles for semantic search, and retrieves historically similar
events to surface what sentiment those past events carried as a precedent
for a new one.

## What's actually built (not aspirational)

**Ingestion**
- Async Finnhub client — real-time quotes and historical candles, token-bucket
  rate limiting (60/min free tier), retry-with-backoff on transient failures
  only, typed pydantic response models. Proven against the live API.
- Async Marketaux client — financial news search, daily-budget rate limiting
  (100/day free tier), same retry/typing discipline. Proven against the live API.
- Continuous quote poller — concurrent multi-symbol polling on a fixed
  interval, graceful shutdown on SIGINT/SIGTERM, per-symbol failure isolation.
- Continuous news poller — single batched request per cycle (not one per
  symbol, to respect the daily quota), auto-stops cleanly when the quota is
  exhausted rather than looping on failures.

**Storage**
- TimescaleDB (Postgres + pgvector), running in Docker.
- Hand-rolled migration runner (`storage/migrator.py`) — simple, transactional,
  tracks applied migrations; intentionally not Alembic at this project size.
- `quotes` — hypertable partitioned by time, one row per poll.
- `news_articles` / `article_entities` — normalized article + per-symbol
  entity/sentiment relationship.
- Sentiment and embedding columns added via later migrations, backfillable
  independently of ingestion.

**ML / AI**
- FinBERT (`ProsusAI/finbert`) sentiment scoring on article title+description,
  batched inference, GPU-accelerated when available. Proven on real financial
  headlines.
- `all-MiniLM-L6-v2` sentence embeddings stored in pgvector (384-dim,
  cosine distance via `ivfflat` index).
- Historical event matching: embed a new piece of news, retrieve the most
  semantically similar past articles via pgvector's `<=>` operator, and
  report their sentiment as a statistical precedent (majority label +
  agreement ratio). Proven end to end on real ingested news.

## What's not built yet

- Forward-return labeling (i.e. "similar past events preceded a +X% price
  move," not just "+/- sentiment") — needs a meaningfully longer quote-polling
  history before it's trainable.
- Price forecasting model (TFT/LSTM baseline).
- Fusion layer combining sentiment + historical match + price forecast into
  a single confidence score.
- HTTP API layer to serve predictions.
- ASX/NSE coverage (currently US equities via Finnhub only).

See the original architecture discussion for the full intended scope; this
section reflects what's actually implemented today.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Docker (for TimescaleDB)
- Free API keys: [Finnhub](https://finnhub.io/register),
  [Marketaux](https://www.marketaux.com/account/dashboard)

## Setup (WSL / Ubuntu)

```bash
git clone <your-repo-url> marketpulse
cd marketpulse

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --extra dev

cp .env.example .env
# edit .env: add FINNHUB_API_KEY, MARKETAUX_API_KEY, DATABASE_PASSWORD

docker compose up -d
docker exec -it marketpulse-timescaledb psql -U marketpulse -d marketpulse \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Running checks

```bash
uv run pytest          # test suite (integration tests skip cleanly if DB is down)
uv run ruff check .    # lint
uv run mypy src        # strict type checking
```

All three should pass clean before any commit.

## Running the system

```bash
# Continuously ingest live quotes for a watchlist
uv run python scripts/run_poller.py AAPL MSFT GOOGL --interval 60

# Continuously ingest news for the same watchlist (mind the 100/day quota)
uv run python scripts/run_news_poller.py AAPL MSFT GOOGL --interval 1800

# Score sentiment on any articles not yet scored
uv run python scripts/score_pending_sentiment.py

# Generate embeddings for any articles not yet embedded
uv run python scripts/embed_pending_articles.py

# Ask: "what happened last time something like this occurred?"
uv run python scripts/find_similar_events.py "Nvidia announces new chip architecture" 5
```

## Project layout
 
## Design principles

- Every setting is read once, validated at startup, and fails with a clear
  message if a required key is missing — never a bare 401/auth error three
  layers deep.
- Free-tier API limits are real constraints, built into the client layer
  (token-bucket rate limiting, quota-aware stop behavior) rather than
  discovered by getting throttled in production.
- Retries only happen on genuinely transient failures (timeouts, 5xx) —
  never on bad auth or bad requests, where retrying just burns budget on
  the same guaranteed failure.
- All external API responses are parsed into typed pydantic models at the
  ingestion boundary, so a shape change in a third-party API surfaces as a
  clear validation error at the call site, not a KeyError somewhere else.
- Point-in-time correctness (not letting data leak in that wasn't actually
  available at the timestamp being modeled) is treated as a first-class
  requirement for anything downstream that will eventually do backtesting —
  the single most common way systems like this quietly produce false
  confidence.
