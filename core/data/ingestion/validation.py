"""Data validation for ingested OHLCV records.

Runs *before* data is persisted. Ensures required columns exist, coerces
numeric types, removes duplicate (ticker, date) rows, sorts chronologically and
flags missing business-day gaps. Every issue is recorded as a structured
anomaly so the download report can surface data-quality problems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.data.providers.base_provider import PriceColumns
from core.utils.dates import date_range_business
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    clean: pd.DataFrame
    anomalies: list[dict] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.anomalies


def validate_ohlcv(df: pd.DataFrame) -> ValidationResult:
    """Validate and normalise a long OHLCV frame (one or many tickers)."""
    anomalies: list[dict] = []

    if df is None or df.empty:
        anomalies.append({"ticker": "*", "issue": "empty", "detail": "No rows returned"})
        return ValidationResult(pd.DataFrame(columns=PriceColumns.LONG_COLUMNS), anomalies)

    missing_cols = [c for c in PriceColumns.FIELDS if c not in df.columns]
    if missing_cols:
        anomalies.append(
            {"ticker": "*", "issue": "missing_columns", "detail": ",".join(missing_cols)}
        )

    # Adjusted Close is the primary return series for this platform. Its absence
    # (or all-null values) is flagged explicitly rather than silently ignored.
    if PriceColumns.ADJ_CLOSE not in df.columns:
        anomalies.append(
            {
                "ticker": "*",
                "issue": "adj_close_missing",
                "detail": "Adjusted Close column absent; momentum/return factors will be unreliable.",
            }
        )
    elif df[PriceColumns.ADJ_CLOSE].isna().all():
        anomalies.append(
            {
                "ticker": "*",
                "issue": "adj_close_null",
                "detail": "Adjusted Close present but all values are null.",
            }
        )

    df = df.copy()
    df[PriceColumns.DATE] = pd.to_datetime(df[PriceColumns.DATE])
    for col in PriceColumns.FIELDS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[PriceColumns.TICKER, PriceColumns.DATE])
    before = len(df)
    df = df.drop_duplicates(subset=[PriceColumns.TICKER, PriceColumns.DATE], keep="last")
    dupes = before - len(df)
    if dupes:
        anomalies.append({"ticker": "*", "issue": "duplicates_removed", "detail": str(dupes)})

    df = df.sort_values([PriceColumns.TICKER, PriceColumns.DATE]).reset_index(drop=True)

    # Per-ticker missing business-day detection.
    for ticker, grp in df.groupby(PriceColumns.TICKER):
        if grp.empty:
            continue
        first, last = grp[PriceColumns.DATE].min(), grp[PriceColumns.DATE].max()
        expected = set(date_range_business(first, last).date)
        actual = set(grp[PriceColumns.DATE].dt.date)
        missing = expected - actual
        if missing:
            anomalies.append(
                {
                    "ticker": ticker,
                    "issue": "missing_trading_dates",
                    "detail": f"{len(missing)} missing between {first.date()} and {last.date()}",
                }
            )

    return ValidationResult(clean=df, anomalies=anomalies)
