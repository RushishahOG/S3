"""NIFTY 500 constituent loader.

Reads the official constituent file (CSV/Excel) supplied by the competition so
the platform never hardcodes ticker lists. Returns lightweight
:class:`Constituent` records carrying the company name and base symbol, which
downstream code resolves to Yahoo Finance symbols via :class:`TickerResolver`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Constituent:
    company_name: str
    base_symbol: str
    series: str = ""
    isin: str = ""


def load_constituents(path: str) -> list[Constituent]:
    """Load constituents from a CSV or Excel file.

    The expected (minimum) columns are ``Company Name`` and ``Symbol``; the
    loader tolerates alternate capitalisation / underscore variants.
    """
    ext = str(path).lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xls"):
        try:
            df = pd.read_excel(path)
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError("openpyxl is required to read Excel constituents.") from exc
    else:
        df = pd.read_csv(path)

    col_map = {str(c).strip().lower(): c for c in df.columns}
    name_col = (
        col_map.get("company name")
        or col_map.get("company_name")
        or col_map.get("name")
        or df.columns[0]
    )
    sym_col = (
        col_map.get("symbol")
        or col_map.get("ticker")
        or col_map.get("code")
        or df.columns[1]
    )
    series_col = col_map.get("series")
    isin_col = col_map.get("isin code") or col_map.get("isin")

    constituents: list[Constituent] = []
    for _, row in df.iterrows():
        raw_sym = str(row[sym_col]).strip()
        if not raw_sym or raw_sym.lower() == "nan":
            continue
        name = str(row[name_col]).strip() if name_col in row else raw_sym
        series = str(row[series_col]).strip() if series_col and series_col in row else ""
        isin = str(row[isin_col]).strip() if isin_col and isin_col in row else ""
        constituents.append(
            Constituent(company_name=name, base_symbol=raw_sym, series=series, isin=isin)
        )

    logger.info("Loaded %d constituents from %s", len(constituents), path)
    return constituents
