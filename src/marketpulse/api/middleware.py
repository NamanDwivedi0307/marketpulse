"""API middleware: request logging and basic rate limiting.

Rate limiting here is a simple in-memory sliding window, keyed by client
IP -- adequate for a single-instance API. A multi-instance deployment would
need a shared store (Redis) since each instance would otherwise track its
own independent counter, but that's real infrastructure this project
doesn't have yet, and adding it now would be complexity without a need.
"""

from __future__ import annotations

import time
from collections import defaultdict

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        duration_ms = round((time.monotonic() - start) * 1000, 2)

        logger.info(
            "api_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            client=request.client.host if request.client else "unknown",
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter: max_requests per window_seconds, per client IP."""

    def __init__(
        self, app: ASGIApp, max_requests: int = 30, window_seconds: float = 60.0
    ) -> None:
        super().__init__(app)
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests_by_client: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Health checks are exempt -- monitoring/orchestration tools poll
        # these frequently and legitimately, and limiting them risks a
        # false "unhealthy" signal during normal operation.
        if request.url.path == "/health":
            return await call_next(request)

        client_id = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window_start = now - self._window_seconds

        recent = [t for t in self._requests_by_client[client_id] if t > window_start]
        recent.append(now)
        self._requests_by_client[client_id] = recent

        if len(recent) > self._max_requests:
            logger.warning("rate_limit_exceeded", client=client_id, path=request.url.path)
            return Response(
                content='{"detail":"Rate limit exceeded. Try again shortly."}',
                status_code=429,
                media_type="application/json",
            )

        return await call_next(request)
