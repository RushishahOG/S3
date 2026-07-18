"""Date / calendar utilities for feature and factor computation.

All conventions (trading days per year, month->trading-day conversion) are
centralised here so the analytics layers stay free of magic numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config.settings import settings

TD_PER_YEAR = settings.features.trading_days_per_year


def months_to_trading_days(months: int) -> int:
    """Approximate number of trading days in ``months`` business months."""
    return int(round(months * TD_PER_YEAR / 12))


def trading_days_to_months(trading_days: int) -> float:
    return trading_days * 12 / TD_PER_YEAR


def last_n_trading_days(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """Return the last ``n`` dates from a sorted trading-day index."""
    return index[-n:]


def date_range_business(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Inclusive business-day range between two timestamps."""
    return pd.bdate_range(start, end)


def as_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def infer_start_date(end: pd.Timestamp, years: int) -> pd.Timestamp:
    """Compute a history start date ``years`` before ``end``."""
    return end - pd.DateOffset(years=years)


def annualisation_factor(periods_per_year: int = TD_PER_YEAR) -> float:
    return float(np.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Global competition backtesting constraints.
# These are sourced from settings.backtest so the whole application enforces a
# single, configurable window. The end date is FIXED (non-configurable in the
# UI); only the start date is user-selectable (>= MIN_BACKTEST_DATE).
# ---------------------------------------------------------------------------
MIN_BACKTEST_DATE = pd.Timestamp(settings.backtest.min_date)
MAX_BACKTEST_DATE = pd.Timestamp(settings.backtest.max_date)
DEFAULT_BACKTEST_START = pd.Timestamp(settings.backtest.default_start)
DEFAULT_BACKTEST_END = MAX_BACKTEST_DATE


class DateRangeError(ValueError):
    """Raised when a requested date range violates competition constraints."""


def validate_backtest_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate and clamp a backtest range to the global competition limits.

    Returns ``(start, end)`` as ``pd.Timestamp``. Raises ``DateRangeError`` if
    the request cannot be reconciled with the fixed window.
    """
    start = as_timestamp(start)
    end = as_timestamp(end) if end is not None else MAX_BACKTEST_DATE

    if start < MIN_BACKTEST_DATE:
        raise DateRangeError(
            f"Start date {start.date()} is before the permitted minimum "
            f"{MIN_BACKTEST_DATE.date()}."
        )
    if end > MAX_BACKTEST_DATE:
        raise DateRangeError(
            f"End date {end.date()} exceeds the fixed maximum "
            f"{MAX_BACKTEST_DATE.date()}."
        )
    if start > end:
        raise DateRangeError(f"Start date {start.date()} is after end date {end.date()}.")
    return start, end


def clamp_to_bounds(value: str | pd.Timestamp) -> pd.Timestamp:
    """Clamp any date into the permitted backtest window."""
    value = as_timestamp(value)
    if value < MIN_BACKTEST_DATE:
        return MIN_BACKTEST_DATE
    if value > MAX_BACKTEST_DATE:
        return MAX_BACKTEST_DATE
    return value


def default_backtest_range() -> tuple[pd.Timestamp, pd.Timestamp]:
    """The default competition range: 2006-01-01 -> 2026-05-31."""
    return DEFAULT_BACKTEST_START, DEFAULT_BACKTEST_END
