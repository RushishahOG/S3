"""Return Engine: computes return series at multiple frequencies.

This module is the single source of truth for all return calculations.
Every future factor (Momentum, Quality, Value, etc.) must consume returns
from here instead of recalculating independently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Standard trading-day conventions
TRADING_DAYS_PER_YEAR = 252
TRADING_WEEKS_PER_YEAR = 52
TRADING_MONTHS_PER_YEAR = 12


def _ensure_sorted_by_ticker_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame is sorted by [Ticker, Date] ascending."""
    if "Ticker" in df.columns and "Date" in df.columns:
        return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


def _validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def prepare_price_panel(
    df: pd.DataFrame,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Prepare a clean price panel from raw market data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data with columns: Date, Ticker, Adj Close (and optionally
        Open, High, Low, Close, Volume).
    price_col : str
        Column name for adjusted close price.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [Ticker, Date, price_col] sorted by
        [Ticker, Date] and deduplicated on (Ticker, Date).
    """
    _validate_required_columns(df, ["Date", "Ticker", price_col])
    df = df[["Ticker", "Date", price_col]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).drop_duplicates(
        subset=["Ticker", "Date"], keep="last"
    )
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Return Calculations
# ---------------------------------------------------------------------------

def compute_daily_returns(
    price_df: pd.DataFrame,
    price_col: str = "Adj Close",
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute daily returns per ticker.

    Parameters
    ----------
    price_df : pd.DataFrame
        Output of `prepare_price_panel` with columns [Ticker, Date, price_col].
    price_col : str
        Column name for price.
    method : {"simple", "log"}
        Return type.

    Returns
    -------
    pd.DataFrame
        Columns: [Ticker, Date, daily_return] (or daily_log_return).
    """
    ret_col = "daily_return" if method == "simple" else "daily_log_return"

    results = []
    for ticker, g in price_df.groupby("Ticker", sort=False):
        g = g.sort_values("Date")
        price = g[price_col]
        if method == "simple":
            ret = price / price.shift(1) - 1.0
        else:
            ret = np.log(price / price.shift(1))
        results.append(g[["Date"]].assign(**{ret_col: ret, "Ticker": ticker}))

    out = pd.concat(results, ignore_index=True)
    out = out.dropna(subset=[ret_col]).reset_index(drop=True)
    return out


def compute_weekly_returns(
    price_df: pd.DataFrame,
    price_col: str = "Adj Close",
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute weekly returns by resampling to week-end (Friday) prices.

    Parameters
    ----------
    price_df : pd.DataFrame
        Output of `prepare_price_panel`.
    price_col : str
        Column name for price.
    method : {"simple", "log"}
        Return type.

    Returns
    -------
    pd.DataFrame
        Columns: [Ticker, Date, weekly_return] (or weekly_log_return).
    """
    ret_col = "weekly_return" if method == "simple" else "weekly_log_return"

    results = []
    for ticker, g in price_df.groupby("Ticker", sort=False):
        g = g.set_index("Date").sort_index()
        wk_price = g[price_col].resample("W-FRI").last()
        if method == "simple":
            ret = wk_price / wk_price.shift(1) - 1.0
        else:
            ret = np.log(wk_price / wk_price.shift(1))
        results.append(ret.reset_index().rename(columns={price_col: ret_col, "index": "Date"}).assign(Ticker=ticker))

    out = pd.concat(results, ignore_index=True)
    out = out.dropna(subset=[ret_col]).reset_index(drop=True)
    return out


def compute_monthly_returns(
    price_df: pd.DataFrame,
    price_col: str = "Adj Close",
    method: str = "simple",
) -> pd.DataFrame:
    """
    Compute monthly returns by resampling to month-end prices.

    Parameters
    ----------
    price_df : pd.DataFrame
        Output of `prepare_price_panel`.
    price_col : str
        Column name for price.
    method : {"simple", "log"}
        Return type.

    Returns
    -------
    pd.DataFrame
        Columns: [Ticker, Date, monthly_return] (or monthly_log_return).
    """
    ret_col = "monthly_return" if method == "simple" else "monthly_log_return"

    results = []
    for ticker, g in price_df.groupby("Ticker", sort=False):
        g = g.set_index("Date").sort_index()
        mo_price = g[price_col].resample("ME").last()
        if method == "simple":
            ret = mo_price / mo_price.shift(1) - 1.0
        else:
            ret = np.log(mo_price / mo_price.shift(1))
        results.append(ret.reset_index().rename(columns={price_col: ret_col, "index": "Date"}).assign(Ticker=ticker))

    out = pd.concat(results, ignore_index=True)
    out = out.dropna(subset=[ret_col]).reset_index(drop=True)
    return out


def compute_all_returns(
    price_df: pd.DataFrame,
    price_col: str = "Adj Close",
) -> dict[str, pd.DataFrame]:
    """
    Compute all return series in one pass.

    Returns
    -------
    dict
        Keys: "daily_return", "daily_log_return", "weekly_return",
              "weekly_log_return", "monthly_return", "monthly_log_return"
    """
    return {
        "daily_return": compute_daily_returns(price_df, price_col, "simple"),
        "daily_log_return": compute_daily_returns(price_df, price_col, "log"),
        "weekly_return": compute_weekly_returns(price_df, price_col, "simple"),
        "weekly_log_return": compute_weekly_returns(price_df, price_col, "log"),
        "monthly_return": compute_monthly_returns(price_df, price_col, "simple"),
        "monthly_log_return": compute_monthly_returns(price_df, price_col, "log"),
    }


def merge_returns_into_panel(
    price_df: pd.DataFrame,
    returns_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge all return series into a single long-format panel.

    Result columns: Ticker, Date, daily_return, daily_log_return,
                    weekly_return, weekly_log_return,
                    monthly_return, monthly_log_return
    """
    panel = price_df[["Ticker", "Date", "Adj Close"]].copy()
    for name, df in returns_dict.items():
        panel = panel.merge(df, on=["Ticker", "Date"], how="left")
    return panel


# ---------------------------------------------------------------------------
# Annualization helpers
# ---------------------------------------------------------------------------

def annualize_daily(value: float | pd.Series) -> float | pd.Series:
    """Annualize a daily volatility/variance using sqrt(252)."""
    return value * np.sqrt(TRADING_DAYS_PER_YEAR)


def annualize_weekly(value: float | pd.Series) -> float | pd.Series:
    """Annualize a weekly volatility/variance using sqrt(52)."""
    return value * np.sqrt(TRADING_WEEKS_PER_YEAR)


def annualize_monthly(value: float | pd.Series) -> float | pd.Series:
    """Annualize a monthly volatility/variance using sqrt(12)."""
    return value * np.sqrt(TRADING_MONTHS_PER_YEAR)


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------


class ReturnEngine:
    """
    High-level interface for computing return series.

    Usage
    -----
    >>> engine = ReturnEngine()
    >>> daily_simple = engine.compute_daily_returns(prices, "simple")
    >>> weekly_log = engine.compute_weekly_returns(prices, "log")
    >>> all_returns = engine.compute_all_returns(prices)
    >>> panel = engine.merge_returns_into_panel(prices, all_returns)
    """

    def __init__(self) -> None:
        pass

    def prepare_price_panel(
        self,
        df: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> pd.DataFrame:
        return prepare_price_panel(df, price_col)

    def compute_daily_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Adj Close",
        method: str = "simple",
    ) -> pd.DataFrame:
        return compute_daily_returns(price_df, price_col, method)

    def compute_weekly_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Adj Close",
        method: str = "simple",
    ) -> pd.DataFrame:
        return compute_weekly_returns(price_df, price_col, method)

    def compute_monthly_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Adj Close",
        method: str = "simple",
    ) -> pd.DataFrame:
        return compute_monthly_returns(price_df, price_col, method)

    def compute_all_returns(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> dict[str, pd.DataFrame]:
        return compute_all_returns(price_df, price_col)

    def merge_returns_into_panel(
        self,
        price_df: pd.DataFrame,
        returns_dict: dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        return merge_returns_into_panel(price_df, returns_dict)

    def compute_all(
        self,
        price_df: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> pd.DataFrame:
        """
        Compute all returns and merge into a single panel in one call.

        Returns
        -------
        pd.DataFrame
            Columns: [Ticker, Date, Adj Close,
                      daily_return, daily_log_return,
                      weekly_return, weekly_log_return,
                      monthly_return, monthly_log_return]
        """
        panel = self.prepare_price_panel(price_df, price_col)
        returns = self.compute_all_returns(panel, price_col)
        return self.merge_returns_into_panel(panel, returns)


# ---------------------------------------------------------------------------
# Class wrapper for unified API
# ---------------------------------------------------------------------------

class ReturnEngine:
    """
    Unified API for computing returns at all frequencies.

    Example
    -------
    >>> engine = ReturnEngine()
    >>> returns = engine.compute_all(price_df)
    >>> panel = engine.merge_returns(returns)
    """

    def __init__(self, price_col: str = "Adj Close") -> None:
        self.price_col = price_col

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and sort price data."""
        return prepare_price_panel(df, self.price_col)

    def daily(self, price_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
        return compute_daily_returns(price_df, self.price_col, method)

    def weekly(self, price_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
        return compute_weekly_returns(price_df, self.price_col, method)

    def monthly(self, price_df: pd.DataFrame, method: str = "simple") -> pd.DataFrame:
        return compute_monthly_returns(price_df, self.price_col, method)

    def compute_all(self, price_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Compute all 6 return series."""
        return compute_all_returns(price_df, self.price_col)

    def merge(self, price_df: pd.DataFrame, returns_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge returns into a single long-format panel."""
        return merge_returns_into_panel(price_df, returns_dict)

    # Annualization helpers
    @staticmethod
    def annualize_daily(value: float | pd.Series) -> float | pd.Series:
        return annualize_daily(value)

    @staticmethod
    def annualize_weekly(value: float | pd.Series) -> float | pd.Series:
        return annualize_weekly(value)

    @staticmethod
    def annualize_monthly(value: float | pd.Series) -> float | pd.Series:
        return annualize_monthly(value)