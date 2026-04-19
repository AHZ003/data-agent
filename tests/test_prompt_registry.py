"""Smoke tests for the versioned prompt registry.

These pin two invariants:

1. Every name in `prompts.REGISTRY` actually resolves to a file with a
   non-empty body and a parseable version. If someone adds a prompt
   name to REGISTRY but forgets the .md file, CI fails here instead
   of at runtime inside an agent.

2. The back-compat constants in `config.py` still round-trip through
   the loader. If a refactor accidentally makes
   `config.CODER_AGENT_SYSTEM_PROMPT` drift from
   `prompts.get("coder_agent")`, agents will silently use a stale
   string. This test catches that.
"""

import config
from prompts import REGISTRY, get, version_info, version_map


def test_every_registered_prompt_loads():
    vm = version_map()
    assert set(vm.keys()) == set(REGISTRY), (
        "version_map and REGISTRY must stay in sync"
    )
    for name in REGISTRY:
        body = get(name)
        assert body.strip(), f"{name} has empty body"
        info = version_info(name)
        assert info["sha"] and len(info["sha"]) == 12
        assert info["version"], f"{name} missing version in frontmatter"


def test_config_constants_match_registry():
    """Back-compat: `from config import X` must equal `prompts.get(...)`."""
    pairs = [
        ("schema_agent", config.SCHEMA_AGENT_SYSTEM_PROMPT),
        ("coder_agent", config.CODER_AGENT_SYSTEM_PROMPT),
        ("planner_agent", config.PLANNER_AGENT_SYSTEM_PROMPT),
        ("critic_agent", config.CRITIC_AGENT_SYSTEM_PROMPT),
        ("storyteller_agent", config.STORYTELLER_AGENT_SYSTEM_PROMPT),
        ("suggested_questions", config.SUGGESTED_QUESTIONS_PROMPT),
        ("vision_extraction", config.VISION_EXTRACTION_PROMPT),
    ]
    for name, const in pairs:
        assert const == get(name), f"config.{name} drifted from prompts/{name}.md"


def test_coder_prompt_still_has_placeholders():
    """The coder prompt is `.format()`ed with table_name / max_rows.
    Removing these placeholders would cause a runtime KeyError in
    coder_agent — pin them here so prompt edits catch it.
    """
    body = get("coder_agent")
    assert "{table_name}" in body
    assert "{max_rows}" in body
