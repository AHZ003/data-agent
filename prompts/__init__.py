"""Versioned prompt registry.

Prompts live as individual Markdown files under `prompts/` with a YAML
frontmatter block:

    ---
    name: coder_agent
    version: 2
    description: One-line summary of the change.
    ---
    <prompt body goes here, verbatim>

Why version prompts as files?

- **Reproducibility.** The eval harness stamps `prompt_versions()` into
  `summary.json` for every run, so a regression can be tied back to a
  specific prompt revision — not just a git sha.
- **Diffability.** Prompts are the single biggest lever on agent
  behavior. They deserve normal code-review: split across lines, diffed
  in PRs, not collapsed inside a triple-quoted Python literal.
- **A/B-ability.** Future: a `DATAAGENT_PROMPT_VARIANT=experimental`
  env var can point the loader at `prompts/variants/coder_agent.md`
  instead of the default, so we can bake prompt variants into eval runs
  the same way we bake model variants.

The loader is deliberately trivial: no Jinja, no composition, no
templating DSL. Any `{placeholder}` substitution is left to the caller
(`.format(...)`) just like it was when prompts lived in config.py.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

_PROMPTS_DIR = Path(__file__).resolve().parent


def _parse(raw: str) -> Tuple[Dict[str, str], str]:
    """Split a prompt file into (frontmatter_dict, body)."""
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    fm_block = raw[3:end].strip()
    body = raw[end + 4 :].lstrip("\n")
    meta: Dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    return meta, body


@lru_cache(maxsize=None)
def _load(name: str) -> Tuple[Dict[str, str], str, str]:
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    raw = path.read_text()
    meta, body = _parse(raw)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]
    return meta, body, sha


def get(name: str) -> str:
    """Return just the prompt body. Callers use this like a string."""
    _, body, _ = _load(name)
    return body


def version_info(name: str) -> Dict[str, str]:
    """Return {version, sha} for a single prompt."""
    meta, _, sha = _load(name)
    return {"version": meta.get("version", "?"), "sha": sha}


# Names tracked by the registry — order matches config.py so `version_map`
# is stable across runs. Add new prompts here when you add a new .md file.
REGISTRY = (
    "schema_agent",
    "coder_agent",
    "planner_agent",
    "critic_agent",
    "storyteller_agent",
    "suggested_questions",
    "vision_extraction",
)


def version_map() -> Dict[str, Dict[str, str]]:
    """Return the full {name: {version, sha}} map for all registered prompts.

    Called by the eval runner to stamp summary.json with the exact
    prompt revisions used in a run.
    """
    return {name: version_info(name) for name in REGISTRY}
