"""The one place DataAgent calls a language model.

`generate()` / `generate_stream()` route on the model name:

  gemini-*          Google Gemini (google-genai SDK)
  ollama/<model>    a local model served by Ollama (OLLAMA_HOST)

so any agent can run on any backend (per-agent override:
DATAAGENT_MODEL_<AGENT>, see config.model_for). Every call records tokens
and cost on the open tracing span and usage scopes. Local models have no
API bill; their cost is amortized hardware time, OLLAMA_USD_PER_HOUR x
generation seconds (default 0 — state the rate you assume when reporting).

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

import json
import os
import random
import time
import urllib.request
from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Callable, Iterator, Optional, TypeVar

from google import genai
from google.genai import types as genai_types

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


# ── Per-session API key (bring-your-own-key in the public demo) ─────────

_api_key: ContextVar[Optional[str]] = ContextVar("dataagent_api_key", default=None)


def current_api_key() -> str:
    """The key for this call: a session override, else GOOGLE_API_KEY."""
    from config import GOOGLE_API_KEY
    return _api_key.get() or GOOGLE_API_KEY


@contextmanager
def api_key_override(key: Optional[str]) -> Iterator[None]:
    """Use `key` for every LLM call made in this context (no-op if empty)."""
    token = _api_key.set(key.strip() if key and key.strip() else None)
    try:
        yield
    finally:
        _api_key.reset(token)


# ── Generation ──────────────────────────────────────────────────────────

def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in ("resource_exhausted", "429", "quota", "rate limit", "rate_limit"))


def generate(
    prompt,
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_output_tokens: int = 1024,
    retry_rate_limits: bool = False,
) -> str:
    """Text completion. `retry_rate_limits` enables the batch backoff policy."""
    from config import MODEL_NAME

    model = model or MODEL_NAME
    if model.startswith("ollama/"):
        return _ollama(prompt, model, temperature, max_output_tokens)
    client = genai.Client(api_key=current_api_key())
    call = lambda: client.models.generate_content(  # noqa: E731
        model=model, contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens),
    )
    response = with_rate_limit_backoff(call, _is_rate_limit) if retry_rate_limits else call()
    _record(response, model)
    return response.text


def generate_stream(
    prompt, *, model: Optional[str] = None, temperature: float = 0.0, max_output_tokens: int = 1024,
) -> Iterator[str]:
    """Token stream. Ollama models yield the whole answer as one chunk."""
    from config import MODEL_NAME

    model = model or MODEL_NAME
    if model.startswith("ollama/"):
        yield _ollama(prompt, model, temperature, max_output_tokens)
        return
    client = genai.Client(api_key=current_api_key())
    last = None
    for chunk in client.models.generate_content_stream(
        model=model, contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens),
    ):
        last = chunk
        text = getattr(chunk, "text", None)
        if text:
            yield text
    # Streamed usage metadata is cumulative; the last chunk has the total.
    _record(last, model)


def _record(response, model: str) -> None:
    from core import tracing
    tracing.record_usage(response, model)


def _ollama(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from core import tracing

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    body = json.dumps({"model": model.split("/", 1)[1], "prompt": prompt, "stream": False,
                       "options": {"temperature": temperature, "num_predict": max_tokens}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=float(os.getenv("OLLAMA_TIMEOUT", "300"))) as resp:
        data = json.loads(resp.read())
    seconds = time.monotonic() - start
    usd = seconds * float(os.getenv("OLLAMA_USD_PER_HOUR", "0")) / 3600
    usage = SimpleNamespace(prompt_token_count=data.get("prompt_eval_count", 0),
                            candidates_token_count=data.get("eval_count", 0), thoughts_token_count=0)
    tracing.record_usage(SimpleNamespace(usage_metadata=usage), model, usd=usd)
    return data.get("response", "")

