"""Text embedders: Gemini (cached on disk) and an offline hashing fallback."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from retrieval.text import char_ngrams, tokenize

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "outputs" / "cache" / "embeddings"


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) array of L2-normalized vectors."""


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)


class HashingEmbedder:
    """Deterministic, offline: hashed word + char-trigram features.

    Not semantic (no synonyms), but it catches morphology and partial
    matches BM25 misses ("cntry" ~ "country"), needs no API key, and makes
    every retrieval test and the offline recall eval reproducible.
    """

    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim)
        for feat in [f"w:{t}" for t in tokenize(text)] + [f"c:{g}" for g in char_ngrams(text)]:
            h = int.from_bytes(hashlib.md5(feat.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim))
        return _normalize(np.stack([self._vec(t) for t in texts]))


class GeminiEmbedder:
    """Gemini embeddings with a content-addressed on-disk cache.

    Schema and memory documents are embedded once per content hash, so
    re-running a benchmark costs no embedding calls.
    """

    def __init__(self, model: str = "gemini-embedding-001", dim: int = 768, cache_dir: Path = CACHE_DIR):
        self.model, self.dim = model, dim
        self.name = f"{model}-{dim}"
        self._path = cache_dir / f"{self.name}.jsonl"
        self._lock = threading.Lock()
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                rec = json.loads(line)
                self._cache[rec["k"]] = rec["v"]

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:24]

    def embed(self, texts: list[str]) -> np.ndarray:
        from google import genai
        from google.genai import types

        from core.llm import current_api_key

        keys = [self._key(t) for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in self._cache))
        if missing:
            client = genai.Client(api_key=current_api_key())
            new = {}
            for i in range(0, len(missing), 100):
                batch = missing[i : i + 100]
                resp = client.models.embed_content(
                    model=self.model, contents=batch,
                    config=types.EmbedContentConfig(output_dimensionality=self.dim),
                )
                for t, e in zip(batch, resp.embeddings):
                    new[self._key(t)] = list(e.values)
            with self._lock:
                self._cache.update(new)
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with open(self._path, "a") as f:
                    for k, v in new.items():
                        f.write(json.dumps({"k": k, "v": v}) + "\n")
        if not texts:
            return np.zeros((0, self.dim))
        return _normalize(np.array([self._cache[k] for k in keys], dtype=float))


def default_embedder() -> Embedder:
    """Gemini when a key is configured and DATAAGENT_EMBEDDER != 'hashing'."""
    import os

    from core.llm import current_api_key
    if os.getenv("DATAAGENT_EMBEDDER", "gemini") == "hashing" or not current_api_key():
        return HashingEmbedder()
    return GeminiEmbedder()
