"""PII detection and masking for what the LLM sees.

The Coder prompt carries sample values and the Storyteller sees result
rows; both leave the machine. PII columns are detected from two signals:

  - the column name (email, phone, ssn, first_name, address, dob, ...);
  - the values: validated patterns (email, phone, US SSN, payment card
    numbers that pass the Luhn check, IPv4, IBAN) matching at least
    `min_share` of a sample.

Detected columns keep their name and type in the prompt (the Coder must
still be able to filter on them) but their sample values become typed
placeholders like <EMAIL>. With DATAAGENT_LOCAL_ONLY_VALUES=1 no raw
value reaches the LLM at all: every sample is dropped, value hints are
disabled, and the Storyteller gets only shape and numeric summaries.

Why not Presidio: its NER backend needs a ~500 MB spaCy model, which
multiplies the Cloud Run image and cold start. Our inputs are structured
columns, where names and validated patterns catch the common cases; NER
would mainly add people/places inside free text. Recorded as a known
limitation in docs/system_card.md.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from typing import Iterable, Optional

from core.datasource import DataSource

_NAME_HINTS = {
    "EMAIL": r"e_?mail",
    "PHONE": r"phone|mobile|\bfax\b|\btel(ephone)?\b|cell_?(phone|no|number)",
    "SSN": r"\bssn\b|social_?security|national_?id|tax_?id",
    "CARD": r"card_?(number|no|num)|\bpan\b|credit_?card|cc_?num",
    "PERSON": r"(first|last|middle|full|given|family|sur)_?name|\bname_(first|last)\b|surname",
    "ADDRESS": r"address|street|postal|zip_?code|\bzip\b|postcode",
    "DOB": r"birth|\bdob\b",
    "IP": r"\bip_?(addr(ess)?)?\b",
    "IBAN": r"iban|account_?(number|no)|routing",
}
_NAME_RES = {k: re.compile(v, re.IGNORECASE) for k, v in _NAME_HINTS.items()}

# Checked in this order: IPs and dates would otherwise look like phones.
_VALUE_RES = {
    "EMAIL": re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$"),
    "SSN": re.compile(r"^(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}$"),
    "IP": re.compile(r"^(25[0-5]|2[0-4]\d|1?\d?\d)(\.(25[0-5]|2[0-4]\d|1?\d?\d)){3}$"),
    "IBAN": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{11,30}$"),
    "PHONE": re.compile(r"^\+?[\d\s().-]{7,20}$"),
}
_DATE_LIKE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")
_CARD_RE = re.compile(r"^\d[\d -]{11,22}\d$")


def luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def value_kind(value) -> Optional[str]:
    """PII type of a single value, or None."""
    if not isinstance(value, str):
        return None
    v = value.strip()
    if _CARD_RE.match(v) and luhn_ok(v):
        return "CARD"
    for kind, rx in _VALUE_RES.items():
        if rx.match(v):
            if kind == "PHONE":
                n_digits = sum(ch.isdigit() for ch in v)
                # Needs separators or +, a phone-like digit count, and not a date.
                if not re.search(r"[\s().+-]", v) or not 8 <= n_digits <= 15 or _DATE_LIKE.match(v):
                    continue
            return kind
    return None


def column_kind(name: str, samples: Iterable, min_share: float = 0.5) -> Optional[str]:
    """PII type of a column from its name, else from its sample values."""
    for kind, rx in _NAME_RES.items():
        if rx.search(name):
            return kind
    vals = [v for v in samples if v is not None]
    if not vals:
        return None
    kinds = [value_kind(v) for v in vals]
    best = max(set(kinds) - {None}, key=kinds.count, default=None)
    return best if best and kinds.count(best) / len(vals) >= min_share else None


@dataclass(frozen=True)
class PIIReport:
    columns: dict[tuple[str, str], str]  # (table, column) -> kind

    def kinds_for(self, table: str) -> dict[str, str]:
        return {c: k for (t, c), k in self.columns.items() if t == table}


def detect(ds: DataSource) -> PIIReport:
    found = {}
    for t in ds.tables:
        for c in t.columns:
            kind = column_kind(c.name, c.sample_values)
            if kind:
                found[(t.name, c.name)] = kind
    return PIIReport(found)


def local_only() -> bool:
    return os.getenv("DATAAGENT_LOCAL_ONLY_VALUES", "0") not in ("0", "", "false")


def mask_datasource(ds: DataSource, report: Optional[PIIReport] = None) -> DataSource:
    """Copy of `ds` safe to put in a prompt: PII samples become placeholders,
    or every sample is dropped in local-only mode."""
    report = report or detect(ds)
    tables = []
    for t in ds.tables:
        cols = []
        for c in t.columns:
            kind = report.columns.get((t.name, c.name))
            if local_only():
                cols.append(replace(c, sample_values=[]))
            elif kind:
                note = f"contains {kind.lower()} (PII, values hidden)"
                cols.append(replace(c, sample_values=[f"<{kind}>"],
                                    description="; ".join(x for x in (c.description, note) if x)))
            else:
                cols.append(c)
        tables.append(replace(t, columns=cols))
    return replace(ds, tables=tables)


def mask_frame(df, report: PIIReport, table_hint: Optional[str] = None):
    """Mask result columns whose name matches a detected PII column (or whose
    values look like PII) before the Storyteller sees them."""
    import pandas as pd

    if df is None:
        return df
    pii_names = {c.lower(): k for (_, c), k in report.columns.items()}
    out = df.copy()
    for col in out.columns:
        kind = pii_names.get(str(col).lower()) or column_kind(str(col), out[col].head(20).tolist())
        if kind:
            out[col] = pd.Series([f"<{kind}>"] * len(out), index=out.index, dtype=object)
    return out
