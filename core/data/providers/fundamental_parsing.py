"""Shared, schema-agnostic helpers for parsing fundamental API responses.

The exact JSON shape returned by the Apify financial / ratio actors is not
fixed across vendors or actor versions. These helpers locate the right values
by matching *candidate* field names (case-insensitive, punctuation-insensitive)
anywhere in a (possibly nested) response, so the pipeline keeps working even if
the actor changes its key names. Unknown fields are left as NULL rather than
raising, so a partial response still stores what it can.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


def norm_key(s: str) -> str:
    """Normalise a key for matching: lowercase, alphanumerics only."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def iter_dicts(obj: Any, path: str = ""):
    """Yield ``(path, dict)`` for every dict nested inside ``obj``."""
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from iter_dicts(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_dicts(v, f"{path}[{i}]")


def _match_one(mapping: dict[str, str], candidates: Iterable[str]) -> str | None:
    """Return the value in ``mapping`` whose normalised key matches a candidate.

    ``mapping`` maps *normalised* keys -> original keys so we avoid re-normalising.
    """
    cand = {norm_key(c) for c in candidates}
    for nk, orig in mapping.items():
        if nk in cand:
            return orig
    return None


def build_lookup(d: dict) -> dict[str, str]:
    """Map normalised-key -> original key for fast candidate lookup."""
    return {norm_key(k): k for k in d.keys()}


def extract_field(
    d: dict, candidates: Iterable[str], lookup: dict[str, str] | None = None
) -> Any:
    """Return the first value in ``d`` matching any candidate key (or None)."""
    lookup = lookup or build_lookup(d)
    orig = _match_one(lookup, candidates)
    if orig is None:
        return None
    val = d[orig]
    if isinstance(val, str):
        return val.strip() if val.strip() else None
    return val


def extract_year(value: Any) -> int | None:
    """Best-effort extraction of a fiscal year integer from a value."""
    if value is None:
        return None
    if isinstance(value, int) and 1900 <= value <= 2100:
        return value
    if isinstance(value, float) and 1900 <= value <= 2100:
        return int(value)
    text = str(value)
    m = re.search(r"(19|20)\d{2}", text)
    if m:
        return int(m.group(0))
    return None


def extract_date(value: Any) -> Any:
    """Return a date-like value (string/date) if present, else None."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def to_float(value: Any) -> float | None:
    """Coerce a numeric-ish value to float, tolerating strings/lists."""
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.\-]", "", value)
        if cleaned in ("", "-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
