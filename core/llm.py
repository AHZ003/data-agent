"""Shared helpers for calling the LLM provider.

Rate limits vs. quota exhaustion
--------------------------------
In the interactive app a 429 is treated as terminal (see the
2026-04-13 postmortem: retrying a quota storm burned 2,703 coder
spans). Batch benchmark runs are different: with --concurrency they
routinely hit per-minute rate limits that clear in seconds. Setting
DATAAGENT_LLM_BACKOFF_RETRIES > 0 turns on bounded exponential backoff
for those runs only; after the last retry the 429 propagates and the
normal terminal handling applies.
"""

from __future__ import annotations

import os
import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def backoff_retries() -> int:
    return int(os.getenv("DATAAGENT_LLM_BACKOFF_RETRIES", "0"))


def with_rate_limit_backoff(
    call: Callable[[], T],
    is_rate_limit: Callable[[Exception], bool],
    retries: int | None = None,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying rate-limit errors with jittered exponential backoff."""
    retries = backoff_retries() if retries is None else retries
    for attempt in range(retries + 1):
        try:
            return call()
        except Exception as e:
            if attempt >= retries or not is_rate_limit(e):
                raise
            delay = min(max_delay, base_delay * 2 ** attempt)
            sleep(delay * random.uniform(0.5, 1.0))
    raise AssertionError("unreachable")
