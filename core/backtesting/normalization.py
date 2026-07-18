"""Cross-sectional normalization helpers for the ARQM backtest engine.

Every normalization is applied *cross-sectionally* (across stocks) at a single
rebalance date, turning raw factor values into comparable scores. All functions
are pure and vectorized over a pandas Series so the engine can call them inside
tight loops without per-stock Python overhead.

Missing values (NaN) are propagated as NaN and never coerced to zero -- the
engine treats NaN as "factor unavailable" and excludes the stock (or uses the
configured fallback) rather than inventing a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config.backtest_schema import Normalization


def normalize(series: pd.Series, method: Normalization) -> pd.Series:
    """Dispatch to the requested normalization method.

    Parameters
    ----------
    series : pd.Series
        Raw factor values (one per stock) for a single rebalance date.
    method : Normalization
        One of ``zscore``, ``robust_zscore``, ``percentile``, ``minmax``.

    Returns
    -------
    pd.Series
        Normalized scores. Z/robust-z are roughly standard-normal; percentile is
        0..1; minmax is 0..1. NaN in -> NaN out.
    """
    if method == "zscore":
        return zscore(series)
    if method == "robust_zscore":
        return robust_zscore(series)
    if method == "percentile":
        return percentile_rank(series)
    if method == "minmax":
        return minmax(series)
    raise ValueError(f"Unknown normalization method: {method}")


def zscore(series: pd.Series) -> pd.Series:
    """Standard Z-score: (x - mean) / std across the available (non-null) cross-section."""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean()
    sd = s.std()
    if sd is None or pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def robust_zscore(series: pd.Series) -> pd.Series:
    """Robust Z-score using median and IQR (MAD-based), resistant to outliers."""
    s = pd.to_numeric(series, errors="coerce")
    med = s.median()
    iqr = s.quantile(0.75) - s.quantile(0.25)
    if pd.isna(iqr) or iqr == 0:
        mad = (s - med).abs().median()
        scale = 1.4826 * mad if mad and not pd.isna(mad) else np.nan
    else:
        scale = iqr / 1.349
    if pd.isna(scale) or scale == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - med) / scale


def percentile_rank(series: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1] using average rank; NaN stays NaN."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(pct=True, method="average")


def minmax(series: pd.Series) -> pd.Series:
    """Min-max scale to [0, 1]; constant series -> NaN (undefined)."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


def score_lower_is_better(series: pd.Series, method: Normalization) -> pd.Series:
    """Normalize a *risk* factor where lower is better (e.g. volatility).

    We invert the percentile / minmax orientation so that a low raw value maps to
    a high score; for z/robust-z we simply negate.
    """
    if method in ("percentile", "minmax"):
        return 1.0 - normalize(series, method)
    return -normalize(series, method)
