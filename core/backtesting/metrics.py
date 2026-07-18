"""Performance & risk metrics (ARQM backtest engine).

Pure functions over return Series. All metrics follow standard institutional
definitions and are computed on *daily* portfolio returns unless noted. NaN
handling is explicit: a metric that cannot be computed (e.g. negative returns for
Sharpe) returns NaN rather than a misleading number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.utils.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


def daily_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().fillna(0.0)


def _annualize(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    if ret.empty:
        return np.nan
    n = ret.notna().sum()
    if n == 0:
        return np.nan
    mean = ret.mean()
    return float((1.0 + mean) ** periods - 1.0)


def _annualized_vol(ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    if ret.empty:
        return np.nan
    sd = ret.std()
    if sd is None or pd.isna(sd):
        return np.nan
    return float(sd * np.sqrt(periods))


def sharpe(ret: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    if ret.empty:
        return np.nan
    excess = ret - rf_annual / periods
    sd = excess.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return np.nan
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(ret: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    if ret.empty:
        return np.nan
    excess = ret - rf_annual / periods
    downside = excess[excess < 0]
    dd = downside.std()
    if dd is None or pd.isna(dd) or dd == 0:
        return np.nan
    return float(excess.mean() / dd * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return float(dd.min())


def calmar(ret: pd.Series, nav: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(nav)
    if pd.isna(mdd) or mdd == 0:
        return np.nan
    ann = _annualize(ret, periods)
    return float(ann / abs(mdd))


def treynor(ret: pd.Series, bench_ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    beta = beta_coefficient(ret, bench_ret)
    if pd.isna(beta) or beta == 0:
        return np.nan
    ann = _annualize(ret, periods)
    return float(ann / beta)


def beta_coefficient(ret: pd.Series, bench_ret: pd.Series) -> float:
    if ret.empty or bench_ret.empty:
        return np.nan
    df = pd.concat([ret, bench_ret], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    cov = df.iloc[:, 0].cov(df.iloc[:, 1])
    var = df.iloc[:, 1].var()
    if pd.isna(var) or var == 0:
        return np.nan
    return float(cov / var)


def alpha_annual(ret: pd.Series, bench_ret: pd.Series, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    beta = beta_coefficient(ret, bench_ret)
    if pd.isna(beta):
        return np.nan
    ann_ret = _annualize(ret, periods)
    ann_bench = _annualize(bench_ret, periods)
    return float(ann_ret - (rf_annual + beta * (ann_bench - rf_annual)))


def information_ratio(ret: pd.Series, bench_ret: pd.Series, periods: int = TRADING_DAYS) -> float:
    if ret.empty or bench_ret.empty:
        return np.nan
    df = pd.concat([ret, bench_ret], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    te = (df.iloc[:, 0] - df.iloc[:, 1]).std()
    if pd.isna(te) or te == 0:
        return np.nan
    return float((df.iloc[:, 0] - df.iloc[:, 1]).mean() / te * np.sqrt(periods))


def ulcer_index(nav: pd.Series) -> float:
    if nav.empty:
        return np.nan
    roll_max = nav.cummax()
    dd = (nav / roll_max - 1.0) * 100.0
    downside = dd[dd < 0]
    if downside.empty:
        return 0.0
    return float(np.sqrt((downside ** 2).mean()))


def hit_ratio(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return np.nan
    wins = (trade_returns > 0).sum()
    return float(wins / len(trade_returns))


def profit_factor(trade_returns: pd.Series) -> float:
    if trade_returns.empty:
        return np.nan
    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def rolling_vol(ret: pd.Series, window: int = 63, periods: int = TRADING_DAYS) -> pd.Series:
    return ret.rolling(window).std() * np.sqrt(periods)


def rolling_sharpe(ret: pd.Series, window: int = 63, rf_annual: float = 0.0, periods: int = TRADING_DAYS) -> pd.Series:
    excess = ret - rf_annual / periods
    return excess.rolling(window).mean() / ret.rolling(window).std() * np.sqrt(periods)


def rolling_beta(ret: pd.Series, bench_ret: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([ret, bench_ret], axis=1).dropna()
    df.columns = ["r", "b"]
    return df["r"].rolling(window).cov(df["b"]) / df["b"].rolling(window).var()


def compute_all_metrics(
    nav: pd.Series,
    bench_nav: pd.Series,
    rf_annual: float = 0.0,
) -> dict[str, float]:
    """Compute the full metric suite for the engine's output tables."""
    ret = daily_returns(nav)
    bench_ret = daily_returns(bench_nav) if not bench_nav.empty else ret * np.nan
    metrics = {
        "total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0) if not nav.empty else np.nan,
        "annual_return": _annualize(ret),
        "annual_volatility": _annualized_vol(ret),
        "sharpe": sharpe(ret, rf_annual),
        "sortino": sortino(ret, rf_annual),
        "calmar": calmar(ret, nav),
        "treynor": treynor(ret, bench_ret),
        "beta": beta_coefficient(ret, bench_ret),
        "alpha_annual": alpha_annual(ret, bench_ret, rf_annual),
        "information_ratio": information_ratio(ret, bench_ret),
        "max_drawdown": max_drawdown(nav),
        "ulcer_index": ulcer_index(nav),
    }
    return metrics
