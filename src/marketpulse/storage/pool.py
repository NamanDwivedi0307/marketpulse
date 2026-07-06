"""Async connection pool for TimescaleDB/Postgres.

One pool per process, created explicitly at startup and closed explicitly at
shutdown -- not a global created at import time, which would make it
impossible to point tests at a different database or to control connection
lifecycle in a long-running service.
"""

from __future__ import annotations

import asyncpg

from marketpulse.config.settings import DatabaseSettings


async def create_pool(settings: DatabaseSettings) -> asyncpg.Pool:
    pool = await asyncpg.create_pool(
        dsn=settings.dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    if pool is None:
        # asyncpg.create_pool can return None if closed before init completes;
        # treat that as a hard failure rather than letting a None pool leak
        # into calling code that assumes it always gets a usable pool back.
        raise RuntimeError("Failed to create database connection pool")
    return pool

