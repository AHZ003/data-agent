"""VCR-style cassette recording/replay for Gemini calls in tests.

Goal: let agent tests run offline, against frozen LLM responses, so
they're (a) deterministic, (b) free, (c) executable in CI without a
Gemini API key, and (d) reproducible across machines.

How it works:

- A **cassette** is a JSON file under `tests/cassettes/<name>.json` that
  maps `sha256(prompt)[:24]` → recorded response text.
- In **replay mode** (default when `DATAAGENT_CASSETTE_MODE=replay` or
  the env var is unset in a test context), `match(prompt)` looks up the
  cassette and returns the stored text — or raises `CassetteMiss`.
- In **record mode** (`DATAAGENT_CASSETTE_MODE=record`), the caller is
  expected to make the real API call and then store the result via
  `record(cassette, prompt, text)`. Tests under record mode are NOT
  meant to run in CI — they're a local one-time capture step.

Agents don't need to know about cassettes at all. Tests monkey-patch
`client.models.generate_content` to route through a cassette, using
`fake_response(text)` to fabricate a minimal object with a `.text`
attribute (which is the only field callers use).

This gives us the cheap version of VCR.py without adding another
dependency and without having to intercept HTTP at the transport
layer — the google-genai SDK is pinned so its protocol is stable.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


CASSETTE_DIR = Path(__file__).resolve().parent.parent / "tests" / "cassettes"


class CassetteMiss(RuntimeError):
    """Raised when replay mode can't find a stored response for a prompt.

    Always include the name + sha in the message so a failing CI run
    tells you exactly which cassette to re-record.
    """


def _sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:24]


def load(name: str) -> Dict[str, str]:
    path = CASSETTE_DIR / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save(name: str, data: Dict[str, str]) -> None:
    CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
    path = CASSETTE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def record(name: str, prompt: str, text: str) -> None:
    """Append a (prompt, text) pair to a cassette on disk.

    Intentionally idempotent on prompt sha, so re-recording the same
    prompt with a refreshed model version just overwrites the entry.
    """
    data = load(name)
    data[_sha(prompt)] = text
    save(name, data)


def match(name: str, prompt: str) -> str:
    data = load(name)
    sha = _sha(prompt)
    if sha not in data:
        raise CassetteMiss(
            f"Cassette '{name}' has no entry for prompt sha {sha}. "
            "Re-record with DATAAGENT_CASSETTE_MODE=record, or set "
            "DATAAGENT_CASSETTE_MODE=record_append to add this prompt."
        )
    return data[sha]


@dataclass
class FakeResponse:
    """Minimal stand-in for google-genai's GenerateContentResponse.

    The agents only read `.text`, so that's all we need to provide.
    If a future agent starts reading `.usage_metadata` or similar,
    extend this class — don't try to mock the whole SDK.
    """
    text: str


def fake_response(text: str) -> FakeResponse:
    return FakeResponse(text=text)


def make_replay_client(cassette_name: str):
    """Return an object shaped like the bits of `genai.Client` the
    agents use (`.models.generate_content(...)`), routed through a
    cassette. Tests monkey-patch the agent module's `genai.Client`
    constructor to return this, e.g.:

        from core import llm_cassette
        monkeypatch.setattr(
            coder_agent.genai, "Client",
            lambda api_key=None: llm_cassette.make_replay_client("coder_basic"),
        )
    """
    class _Models:
        def generate_content(self, *, model: str, contents: str, **kwargs: Any):
            text = match(cassette_name, contents)
            return fake_response(text)

        def generate_content_stream(self, *, model: str, contents: str, **kwargs: Any):
            text = match(cassette_name, contents)
            # Yield the full text as a single chunk — good enough for
            # tests that want to exercise the streaming code path.
            yield fake_response(text)

    class _Client:
        models = _Models()

    return _Client()
