"""Spider's official difficulty levels (easy / medium / hard / extra).

A line-for-line port of `eval_hardness` from the official Spider
evaluation script (taoyds/spider, evaluation.py). It runs on the parsed
`sql` field shipped in dev.json, so no SQL parsing is needed. Quirks of
the original are kept on purpose (e.g. `count_agg` over WHERE condition
units tests the NOT flag, and over HAVING it sees the 'and'/'or'
connector strings) so that our per-difficulty numbers are comparable
with published ones. tests/test_spider_hardness.py checks the dev-set
distribution against the published 248 / 446 / 174 / 166 split.
"""

from __future__ import annotations

WHERE_OPS = ("not", "between", "=", ">", "<", ">=", "<=", "!=", "in", "like", "is", "exists")
_LIKE = WHERE_OPS.index("like")


def _has_agg(unit) -> bool:
    return unit[0] != 0  # AGG_OPS.index('none') == 0


def _count_agg(units) -> int:
    return len([u for u in units if _has_agg(u)])


def _nested(sql: dict) -> list[dict]:
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if isinstance(cond_unit[3], dict):
            nested.append(cond_unit[3])
        if isinstance(cond_unit[4], dict):
            nested.append(cond_unit[4])
    for key in ("intersect", "except", "union"):
        if sql[key] is not None:
            nested.append(sql[key])
    return nested


def _component1(sql: dict) -> int:
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1
    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([t for t in ao if t == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len([c for c in cond_units if c[1] == _LIKE])
    return count


def _others(sql: dict) -> int:
    count = 0
    agg_count = _count_agg(sql["select"][1])
    agg_count += _count_agg(sql["where"][::2])
    agg_count += _count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += _count_agg(
            [u[1] for u in sql["orderBy"][1] if u[1]] + [u[2] for u in sql["orderBy"][1] if u[2]]
        )
    agg_count += _count_agg(sql["having"])
    if agg_count > 1:
        count += 1
    if len(sql["select"][1]) > 1:
        count += 1
    if len(sql["where"]) > 1:
        count += 1
    if len(sql["groupBy"]) > 1:
        count += 1
    return count


def hardness(sql: dict) -> str:
    c1, c2, others = _component1(sql), len(_nested(sql)), _others(sql)
    if c1 <= 1 and others == 0 and c2 == 0:
        return "easy"
    if (others <= 2 and c1 <= 1 and c2 == 0) or (c1 <= 2 and others < 2 and c2 == 0):
        return "medium"
    if (
        (others > 2 and c1 <= 2 and c2 == 0)
        or (2 < c1 <= 3 and others <= 2 and c2 == 0)
        or (c1 <= 1 and others == 0 and c2 <= 1)
    ):
        return "hard"
    return "extra"
