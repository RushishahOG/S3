"""Export helpers for backtest results.

Thin wrappers that serialise a DataFrame to CSV / Parquet / Excel bytes so the UI
can offer downloads without scattering I/O logic across the page.
"""

from __future__ import annotations

import io

import pandas as pd

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


def export_dataframe(df: pd.DataFrame, fmt: str) -> bytes:
    """Return ``df`` serialised to ``fmt`` (csv | parquet | excel) as bytes."""
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        df = pd.DataFrame()
    fmt = (fmt or "csv").lower()
    if fmt == "csv":
        return df.to_csv(index=True).encode("utf-8")
    if fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow")
        return buf.getvalue()
    if fmt == "excel":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            df.to_excel(xw, sheet_name="Export", index=True)
        return buf.getvalue()
    raise ValueError(f"Unsupported export format: {fmt}")
