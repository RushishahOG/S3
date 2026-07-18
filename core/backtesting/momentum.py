"""Momentum signal generation (ARQM Gate 1).

The spec requires momentum factors but the platform has **no persisted momentum
table** -- only raw prices. Per the agreed design, the backtest engine derives
momentum *signals* directly from the adjusted-price panel at each rebalance date
(point-in-time, no look-ahead). This keeps the engine self-contained while still
consuming only already-stored (processed) price data.

Four factors are supported (all trailing-return style):

* ``scaled_momentum`` -- total return over ``horizon`` months, shifted by
  ``lag`` months (classic 12-1 momentum). Higher is better.
* ``roc`` -- rate of change = P_t / P_{t-lag-horizon} - 1. Higher is better.
* ``relative_strength`` -- stock trailing return minus benchmark trailing return
  over the same horizon. Higher is better.
* ``momentum_lag`` -- trailing return over ``horizon`` with an extended ``lag``
  (skips the most recent short-term reversal). Higher is better.

All returns are computed from the adjusted close so they are total-return
momentum. Windows are expressed in *months* and converted to trading days
(21/day) for the price lookup; the function uses the actual price observed on or
before the target date (as-of semantics) so there is never any future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config.backtest_schema import MomentumConfig, MomentumFactorConfig
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_MONTH = 21


def _asof_price(prices: pd.DataFrame, ticker: str, date: pd.Timestamp, offset_days: int) -> float | None:
    """Price ``offset_days`` before ``date`` (or None if unavailable)."""
    if ticker not in prices.columns:
        return None
    sub = prices[ticker]
    target = date - pd.Timedelta(days=offset_days)
    hist = sub.loc[:target]
    if hist.empty:
        return None
    return float(hist.iloc[-1])


def _trailing_return(prices: pd.DataFrame, ticker: str, date: pd.Timestamp,
                     horizon_months: int, lag_months: int) -> float | None:
    """Total return from (date - lag - horizon) to (date - lag)."""
    h_days = horizon_months * TRADING_DAYS_PER_MONTH
    l_days = lag_months * TRADING_DAYS_PER_MONTH
    p_end = _asof_price(prices, ticker, date, l_days)
    p_start = _asof_price(prices, ticker, date, l_days + h_days)
    if p_start is None or p_end is None or p_start == 0:
        return None
    return p_end / p_start - 1.0


def _momentum_matrix(prices: pd.DataFrame, tickers: list[str],
                      date: pd.Timestamp, config: MomentumConfig) -> pd.DataFrame:
    """Vectorized trailing-return matrix for all enabled factors / tickers.

    For each enabled factor we compute the as-of price ``lookback`` trading days
    before ``date`` (and, when ``lag_months > 0``, the price at
    ``lag + horizon`` days before) using an integer-position lookup on the sorted
    price panel -- O(log n) per ticker instead of the O(n) ``loc[:target]``
    slice that the per-ticker loop used. Returns a ticker-indexed DataFrame with
    one column per enabled factor.
    """
    if date not in prices.index:
        # Find the last trading day on/before date.
        mask = prices.index <= date
        if not mask.any():
            return pd.DataFrame(index=tickers)
        date = prices.index[mask][-1]

    pos = prices.index.get_indexer([date])[0]
    if pos < 0:
        return pd.DataFrame(index=tickers)

    cols = [t for t in tickers if t in prices.columns]
    if not cols:
        return pd.DataFrame(index=tickers)
    panel = prices[cols].to_numpy()

    out: dict[str, np.ndarray] = {}
    for fcfg in config.factors:
        if not fcfg.enabled:
            continue
        h_days = fcfg.horizon_months * TRADING_DAYS_PER_MONTH
        l_days = fcfg.lag_months * TRADING_DAYS_PER_MONTH
        end_pos = pos - l_days
        start_pos = pos - (l_days + h_days)
        if end_pos < 0 or start_pos < 0:
            out[fcfg.name] = np.full(len(cols), np.nan)
            continue
        p_end = panel[end_pos]
        p_start = panel[start_pos]
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where((p_start == 0) | np.isnan(p_start) | np.isnan(p_end),
                           np.nan, p_end / p_start - 1.0)
        out[fcfg.name] = ret

    mat = pd.DataFrame(out, index=cols)
    mat = mat.reindex(tickers)
    return mat


def compute_momentum_signals(
    prices: pd.DataFrame,
    config: MomentumConfig,
    date: pd.Timestamp,
    tickers: list[str],
    benchmark_returns: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute every enabled momentum factor for ``tickers`` as-of ``date``.

    Returns a DataFrame indexed by ticker with one column per factor (raw
    trailing returns). Normalization/combining happens later in the gate.
    """
    mat = _momentum_matrix(prices, tickers, date, config)

    if benchmark_returns is not None and "relative_strength" in mat.columns:
        bench_ret = _trailing_benchmark_return(benchmark_returns, date,
                                               next(f for f in config.factors if f.name == "relative_strength"))
        mat["relative_strength"] = mat["relative_strength"] - bench_ret if bench_ret is not None else mat["relative_strength"]

    return mat


def _trailing_benchmark_return(benchmark_returns: pd.Series, date: pd.Timestamp,
                               fcfg: MomentumFactorConfig) -> float | None:
    """Benchmark cumulative return over the factor's horizon/lag window."""
    h_days = fcfg.horizon_months * TRADING_DAYS_PER_MONTH
    l_days = fcfg.lag_months * TRADING_DAYS_PER_MONTH
    start = date - pd.Timedelta(days=l_days + h_days)
    end = date - pd.Timedelta(days=l_days)
    window = benchmark_returns.loc[start:end]
    if window.empty or window.isna().all():
        return None
    return float((1.0 + window).prod() - 1.0)
