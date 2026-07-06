# MarketPulse

Real-time AI-driven financial event and price-movement analysis platform.
Ingests live market data and financial news, matches breaking events against
a historical corpus of similar past events, and produces a fused prediction
of short-term price movement with a confidence score.

## Status

Phase 1 (MVP) — foundation. Configuration, logging, and project scaffolding
are in place. Ingestion, storage, and modeling layers are being built
incrementally; see `docs/roadmap.md` (coming soon) for the phase breakdown.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Free API keys (see below) — no paid tier required for Phase 1

## Setup (WSL / Ubuntu)

```bash
git clone <your-repo-url> marketpulse
cd marketpulse

# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# install dependencies (including dev tools)
uv sync --extra dev

# configure environment
cp .env.example .env
# then edit .env and add your free API keys:
#   Finnhub:    https://finnhub.io/register
#   Marketaux:  https://www.marketaux.com/account/dashboard
```

## Running checks

```bash
uv run pytest          # test suite
uv run ruff check .    # lint
uv run mypy src        # strict type checking
```

All three should pass clean before any commit. This isn't optional ceremony —
strict mypy on the config/ingestion layers catches real bugs (wrong type
flowing into an API client, silently-optional fields) before they turn into
a live incident three weeks from now.

## Project layout

```
src/marketpulse/
    config/      settings, environment validation
    ingestion/   external API clients (prices, news)
    storage/     database access layer
    models/      ML models: sentiment, historical matching, forecasting
    api/         FastAPI serving layer
    utils/       logging and shared helpers
tests/           mirrors src/ structure
```

## Design principles for this codebase

- Every setting is read once, in `config/settings.py`, and validated at
  startup. No `os.getenv` calls scattered through the codebase.
- No network call happens without an explicit `require_*()` check that fails
  with a clear message if a key is missing — never a bare 401 from three
  layers deep.
- Free-tier API limits are treated as real constraints, not an afterthought:
  rate limiting and backoff are built into the client layer, not bolted on
  after getting throttled in production.
- Point-in-time correctness (no using data that wasn't actually available at
  the timestamp being modeled) is treated as a first-class requirement, not
  a backtest detail, because it is the single most common way systems like
  this quietly produce false confidence.
