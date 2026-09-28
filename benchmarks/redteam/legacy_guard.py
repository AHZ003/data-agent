"""The original regex SQL guard (commit 608bfe1), kept only as the
red-team "before" baseline. Do not use it anywhere else."""

import re


class LegacyGuardError(ValueError):
    pass


_BLOCKED_KEYWORDS = (
    "drop", "delete", "update", "insert", "replace",
    "attach", "detach", "pragma", "create", "alter",
    "truncate", "vacuum", "reindex", "exec",
)
_ALLOWED_PREFIXES = ("select", "with")


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    return sql


def legacy_guard(sql: str) -> str:
    if not sql or not sql.strip():
        raise LegacyGuardError("empty query")
    cleaned = _strip_comments(sql).strip().rstrip(";").strip()
    if not cleaned:
        raise LegacyGuardError("empty query after stripping comments")
    if ";" in cleaned:
        raise LegacyGuardError("multiple statements are not allowed")
    head = cleaned.split(None, 1)[0].lower()
    if head not in _ALLOWED_PREFIXES:
        raise LegacyGuardError(f"only SELECT/WITH queries are allowed (got '{head}')")
    masked = re.sub(r"'[^']*'", "''", cleaned.lower())
    for kw in _BLOCKED_KEYWORDS:
        if re.search(rf"\b{kw}\b", masked):
            raise LegacyGuardError(f"blocked keyword: {kw.upper()}")
    return cleaned
