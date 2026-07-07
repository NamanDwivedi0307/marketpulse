# MarketPulse API image.
#
# Multi-stage build: the builder stage has uv and compiles/installs
# dependencies into a venv; the runtime stage copies only that venv plus
# source, keeping the final image free of build tooling and uv itself.
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src

# --no-dev: production image doesn't need pytest/ruff/mypy.
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY scripts ./scripts

ENV PATH="/app/.venv/bin:$PATH" \
    ENVIRONMENT=production

EXPOSE 8000

CMD ["uvicorn", "marketpulse.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

