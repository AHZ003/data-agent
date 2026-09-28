"""API-key auth and per-key token-bucket rate limiting.

Keys come from DATAAGENT_API_KEYS as comma-separated `name:key` pairs.
If unset, auth is off (local development) and every caller is "anon".
Each key gets a bucket of RATE_LIMIT_BURST requests refilled at
RATE_LIMIT_PER_MINUTE; only analyze/resume calls (the ones that spend
LLM tokens) draw from it.
"""

from __future__ import annotations

import hmac
import os
import threading
import time
from dataclasses import dataclass

from fastapi import Header, HTTPException


def _keys() -> dict[str, str]:
    raw = os.getenv("DATAAGENT_API_KEYS", "")
    pairs = [p.split(":", 1) for p in raw.split(",") if ":" in p]
    return {key.strip(): name.strip() for name, key in pairs}


def require_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency: returns the caller's key name."""
    keys = _keys()
    if not keys:
        return "anon"
    for key, name in keys.items():
        if x_api_key and hmac.compare_digest(x_api_key, key):
            return name
    raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, per_minute: float | None = None, burst: int | None = None, clock=time.monotonic):
        self.rate = (per_minute or float(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))) / 60.0
        self.burst = burst or int(os.getenv("RATE_LIMIT_BURST", "10"))
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            b = self._buckets.setdefault(key, _Bucket(self.burst, now))
            b.tokens = min(self.burst, b.tokens + (now - b.updated) * self.rate)
            b.updated = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True
            return False
