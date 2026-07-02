"""Small FastAPI-compatible rate-limit dependency.

SlowAPI route decorators currently do not attach cleanly to this project's
FastAPI 0.121 router stack, so video routes use this dependency instead. It is
purposefully narrow: fixed-window limits, IP-based keys, and endpoint-specific
limits from ``settings.RATE_LIMIT_ENDPOINTS``.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock

from fastapi import HTTPException, Request, status

from app.core.config import settings

_WINDOW_SECONDS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "day": 86_400,
}


@dataclass(frozen=True)
class RateLimitRule:
    """Parsed fixed-window rate-limit rule."""

    limit: int
    window_s: int


_hits: dict[tuple[str, str, int], deque[float]] = defaultdict(deque)
_lock = Lock()


def parse_rate_limit_rule(raw_rule: str) -> RateLimitRule:
    """Parse a SlowAPI-style limit such as ``60 per minute``.

    Args:
        raw_rule: Human-readable rate-limit string.

    Returns:
        Parsed rule with an integer limit and window length.

    Raises:
        ValueError: If the rule is not in a supported format.
    """
    parts = raw_rule.strip().lower().replace("/", " per ").split()
    if len(parts) < 3 or parts[1] != "per":
        raise ValueError(f"unsupported rate-limit rule: {raw_rule!r}")

    limit = int(parts[0])
    unit = parts[2].removesuffix("s")
    window_s = _WINDOW_SECONDS.get(unit)
    if limit < 1 or window_s is None:
        raise ValueError(f"unsupported rate-limit rule: {raw_rule!r}")

    return RateLimitRule(limit=limit, window_s=window_s)


def _client_key(request: Request) -> str:
    """Return the best-effort client key for rate limiting."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _video_rules() -> list[RateLimitRule]:
    """Return configured video endpoint rate-limit rules."""
    raw_rules = settings.RATE_LIMIT_ENDPOINTS.get("videos", ["60 per minute"])
    return [parse_rate_limit_rule(rule) for rule in raw_rules]


async def enforce_video_rate_limit(request: Request) -> None:
    """FastAPI dependency that rate-limits ``/videos`` routes."""
    client_key = _client_key(request)
    now = time.monotonic()

    with _lock:
        for rule in _video_rules():
            bucket_key = ("videos", client_key, rule.window_s)
            bucket = _hits[bucket_key]
            cutoff = now - rule.window_s
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= rule.limit:
                retry_after = max(1, int(rule.window_s - (now - bucket[0])))
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="video route rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )

        for rule in _video_rules():
            _hits[("videos", client_key, rule.window_s)].append(now)


def reset_rate_limit_state() -> None:
    """Clear in-memory buckets; intended for tests only."""
    with _lock:
        _hits.clear()
