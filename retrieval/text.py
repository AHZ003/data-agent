"""Tokenization shared by BM25, schema docs and value matching."""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an the of in on at to for from by with and or not is are was were be been being
what which who whom whose how many much list show give find return me all each every
any that this these those there their its it as do does did than then per
""".split())

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORD = re.compile(r"[A-Za-z0-9]+")


def split_identifier(name: str) -> list[str]:
    """'BillingCountry' / 'billing_country' / 'T2.Score' -> ['billing', 'country']"""
    out = []
    for part in _WORD.findall(name):
        out.extend(p.lower() for p in _CAMEL.split(part) if p)
    return out


def _stem(tok: str) -> str:
    # Light plural stripping; enough to match "customers" to "Customer".
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = [_stem(t) for t in split_identifier(text)]
    return toks if keep_stop else [t for t in toks if t not in STOPWORDS]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = f"  {text.lower().strip()} "
    return {s[i : i + n] for i in range(len(s) - n + 1)}
