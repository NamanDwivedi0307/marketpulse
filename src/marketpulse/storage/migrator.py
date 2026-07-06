"""A deliberately small migration runner.

Not Alembic -- for a single-developer project, a full migration framework
is more ceremony than value. What matters: every migration file runs
exactly once, in filename order, and a partial failure doesn't leave the
tracking table lying about what's applied.
"""

from __future__ import annotations

from pathlib import Path

import asyncpg
import structlog

logger = structlog.get_logger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

_TRACKING_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename    TEXT PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


async def run_migrations(pool: asyncpg.Pool) -> list[str]:
    async with pool.acquire() as conn:
        await conn.execute(_TRACKING_TABLE_DDL)

        applied_rows = await conn.fetch("SELECT filename FROM schema_migrations")
        already_applied = {row["filename"] for row in applied_rows}

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        if not migration_files:
            raise RuntimeError(f"No migration files found in {MIGRATIONS_DIR}")

        newly_applied: list[str] = []
        for path in migration_files:
            if path.name in already_applied:
                continue

            sql = path.read_text()
            async with conn.transaction():
                await conn.execute(sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
                )
            logger.info("migration_applied", filename=path.name)
            newly_applied.append(path.name)

        return newly_applied
