"""Risk Engine: computes daily market-data risk & momentum features.

Produces a single, daily-frequency feature set (no monthly / weekly matrix):

* ``beta``              — rolling 12-month beta vs the benchmark (Cov/Var).
* ``momentum_unscaled`` — classic 12-1 momentum: trailing 12-month total return
                          shifted by 1 month (252 + 21 trading days).
* ``momentum_scaled``   — momentum_unscaled divided by annualised volatility
                          (risk-adjusted momentum).
* ``semi_deviation``    — rolling 12-month downside deviation (annualised).

All measures are built on the return series from
`core.feature_engineering.return_engine`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.feature_engineering.return_engine import TRADING_DAYS_PER_YEAR

# Single, daily lookback windows (in trading days).
BETA_WINDOW = int(round(12 * TRADING_DAYS_PER_YEAR / 12))          # 252d
SEMI_DEV_WINDOW = int(round(12 * TRADING_DAYS_PER_YEAR / 12))      # 252d
MOMENTUM_HORIZON = int(round(12 * TRADING_DAYS_PER_YEAR / 12))     # 252d
MOMENTUM_LAG = int(round(1 * TRADING_DAYS_PER_YEAR / 12))          # 21d

ANNUALIZATION = np.sqrt(TRADING_DAYS_PER_YEAR)

# Canonical daily feature columns produced by this engine.
MARKET_RISK_FEATURES = ("beta", "momentum_unscaled", "momentum_scaled", "semi_deviation")


def _rolling_beta(
    asset_ret: pd.Series,
    bench_ret: pd.Series,
    window: int,
) -> pd.Series:
    """Rolling beta = Cov(asset, bench) / Var(bench) over ``window`` periods."""
    asset_ret, bench_ret = asset_ret.align(bench_ret, join="inner")
    cov = asset_ret.rolling(window, min_periods=window).cov(bench_ret)
    var = bench_ret.rolling(window, min_periods=window).var(ddof=1)
    return cov / var


def _rolling_semi_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling semi-deviation (downside deviation); only negative returns count."""
    downside = series.clip(upper=0.0)
    return np.sqrt((downside ** 2).rolling(window, min_periods=window).mean())


def _compute_daily_features(
    returns_df: pd.DataFrame,
    bench_returns: pd.Series,
) -> pd.DataFrame:
    """Compute the four daily features for every ticker."""
    results = []
    for ticker, g in returns_df.groupby("Ticker"):
        g = g.sort_values("Date").set_index("Date")
        ret = g["return"]
        bench = bench_returns.reindex(ret.index).ffill(limit=5)

        vol = _rolling_std(ret, SEMI_DEV_WINDOW) * ANNUALIZATION
        semi = _rolling_semi_std(ret, SEMI_DEV_WINDOW) * ANNUALIZATION
        beta = _rolling_beta(ret, bench, BETA_WINDOW)

        # Momentum (12-1): price now vs price (lag + horizon) ago, via returns.
        # Equivalent to cumulative product of returns over the window.
        mom = _trailing_momentum(ret, MOMENTUM_HORIZON, MOMENTUM_LAG)
        mom_scaled = mom / vol.replace(0, np.nan)

        df = pd.DataFrame({
            "Ticker": ticker,
            "Date": ret.index,
            "beta": beta,
            "momentum_unscaled": mom,
            "momentum_scaled": mom_scaled,
            "semi_deviation": semi,
        })
        results.append(df)

    if not results:
        return pd.DataFrame(columns=["Ticker", "Date", *MARKET_RISK_FEATURES])
    return pd.concat(results).reset_index(drop=True)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation (sample std, ddof=1), annualised-ready."""
    return series.rolling(window, min_periods=window).std(ddof=1)


def _trailing_momentum(ret: pd.Series, horizon: int, lag: int) -> pd.Series:
    """Daily 12-1 momentum: trailing total return over ``horizon`` days,
    shifted forward by ``lag`` days (skip recent short-term reversal)."""
    # Cumulative return from (t - lag - horizon) to (t - lag) using daily returns.
    shift = lag + horizon
    future = (1.0 + ret).rolling(horizon, min_periods=horizon).apply(
        lambda x: float(np.prod(x)), raw=True
    )
    return future.shift(lag)


def compute_all_risk(
    returns_panel: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    Compute the daily market-data risk & momentum features.

    Parameters
    ----------
    returns_panel : pd.DataFrame
        Output of `merge_returns_into_panel` with columns
        [Ticker, Date, daily_return, ...].
    benchmark_returns : pd.DataFrame
        DataFrame with columns [Date, benchmark_return].
    price_col : str
        Price column name (kept for API compatibility; unused).

    Returns
    -------
    pd.DataFrame
        Long-format features with columns:
        [Ticker, Date, beta, momentum_unscaled, momentum_scaled, semi_deviation].
    """
    bench = benchmark_returns.set_index("Date")["benchmark_return"]
    daily_ret = returns_panel[["Ticker", "Date", "daily_return"]].rename(
        columns={"daily_return": "return"}
    )
    return _compute_daily_features(daily_ret, bench)


# ---------------------------------------------------------------------------
# Convenience: compute everything in one call
# ---------------------------------------------------------------------------

def compute_engineered_features(
    price_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    price_col: str = "Adj Close",
) -> pd.DataFrame:
    """
    End-to-end feature engineering: returns + risk.

    Parameters
    ----------
    price_df : pd.DataFrame
        Raw market data with [Date, Ticker, Adj Close, ...]
    benchmark_df : pd.DataFrame
        Benchmark index data with [Date, Adj Close] (or similar).

    Returns
    -------
    pd.DataFrame
        Full engineered feature panel with returns + risk features.
    """
    from core.feature_engineering.return_engine import (
        prepare_price_panel,
        compute_all_returns,
        merge_returns_into_panel,
    )

    # Prepare price panel
    price_panel = prepare_price_panel(price_df, price_col)

    # Compute all returns
    returns_dict = compute_all_returns(price_panel, price_col)
    returns_panel = merge_returns_into_panel(price_panel, returns_dict)

    # Prepare benchmark returns (daily simple)
    bench_price = benchmark_df[["Date", price_col]].copy()
    bench_price["Date"] = pd.to_datetime(bench_price["Date"])
    bench_price = bench_price.sort_values("Date").set_index("Date")
    bench_ret = bench_price[price_col].pct_change().dropna().reset_index()
    bench_ret.columns = ["Date", "benchmark_return"]

    # Compute risk
    risk = compute_all_risk(returns_panel, bench_ret, price_col)

    # Merge returns + risk
    engineered = returns_panel.merge(risk, on=["Ticker", "Date"], how="left")
    return engineered


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------


class RiskEngine:
    """
    High-level interface for computing daily market-data risk features.

    Usage
    -----
    >>> risk_engine = RiskEngine()
    >>> risk_features = risk_engine.compute(returns_panel, benchmark_returns)
    """

    def __init__(self) -> None:
        self._feature_metadata = []

    def compute(
        self,
        returns_panel: pd.DataFrame,
        benchmark_returns: pd.DataFrame,
        price_col: str = "Adj Close",
    ) -> pd.DataFrame:
        """Compute the daily risk & momentum features (see ``compute_all_risk``)."""
        return compute_all_risk(returns_panel, benchmark_returns, price_col)

    def compute_rolling_volatility(
        self,
        returns_df: pd.DataFrame,
        price_df: pd.DataFrame,
        benchmark_df: pd.DataFrame | None = None,
        freq: str = "daily",
        return_col: str = "return",
    ) -> pd.DataFrame:
        """Compute daily risk features from a long-format returns frame."""
        if benchmark_df is None:
            benchmark_df = price_df[price_df["Ticker"] == price_df["Ticker"].iloc[0]][["Date", "Adj Close"]].copy()
        bench = benchmark_df.set_index("Date")["Adj Close"].pct_change().dropna().reset_index()
        bench.columns = ["Date", "benchmark_return"]

        if return_col not in returns_df.columns:
            for cand in ("daily_return", "return", "daily_log_return"):
                if cand in returns_df.columns:
                    return_col = cand
                    break
        df = returns_df.rename(columns={return_col: "return"}) if return_col in returns_df.columns else returns_df
        return _compute_daily_features(df, bench.set_index("Date")["benchmark_return"])

    def get_feature_metadata(self) -> list:
        """Return metadata for all computed daily features."""
        from dataclasses import dataclass

        @dataclass
        class FeatureSpec:
            key: str
            description: str
            factor_category: str
            frequency: str
            lookback_months: int
            formula: str

        return [
            FeatureSpec("beta", "12-month beta vs NIFTY 500", "Risk", "daily", 12, "Cov(stock, bench)/Var(bench)"),
            FeatureSpec("momentum_unscaled", "12-1 trailing total return", "Momentum", "daily", 12, "P_t / P_{t-273} - 1"),
            FeatureSpec("momentum_scaled", "Momentum scaled by volatility", "Momentum", "daily", 12, "momentum_unscaled / annualised_std"),
            FeatureSpec("semi_deviation", "12-month downside deviation (annualised)", "Risk", "daily", 12, "sqrt(mean(min(ret,0)^2)) × √252"),
        ]