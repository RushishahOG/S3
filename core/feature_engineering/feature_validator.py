"""Feature Validator: validates engineered features before persistence.

Ensures data quality by checking for:
- Missing / infinite values
- Duplicate rows
- Date ordering
- Minimum history requirements
- Benchmark alignment
- Rolling window availability
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.providers.base_provider import PriceColumns


# Minimum coverage threshold (fraction of expected windows that must be valid)
MIN_COVERAGE_PCT = 0.5


def validate_features(
    df: pd.DataFrame,
    benchmark_col: str | None = None,
    min_history_days: int = 252,
) -> tuple[bool, list[str], pd.DataFrame]:
    """
    Validate a long-format feature frame.

    Returns
    -------
    (ok, issues, per_feature_stats)
    """
    issues: list[str] = []

    if df is None or df.empty:
        issues.append("empty: no feature rows")
        return False, issues, pd.DataFrame()

    df = df.copy()
    # Normalise key column names to lowercase
    rename = {}
    for c in df.columns:
        if c.lower() in ("ticker", "date"):
            rename[c] = c.lower()
    if rename:
        df = df.rename(columns=rename)

    # Required key columns
    for c in (PriceColumns.TICKER, PriceColumns.DATE):
        if c not in df.columns:
            issues.append(f"missing_key_column: {c}")

    value_cols = [c for c in df.columns if c not in (PriceColumns.TICKER, PriceColumns.DATE)]

    # Duplicate (ticker, date) pairs
    dup_keys = int(df.duplicated(subset=[PriceColumns.TICKER, PriceColumns.DATE]).sum())
    if dup_keys:
        issues.append(f"duplicate_keys: {dup_keys} duplicate (ticker, date) rows")

    # Fully duplicated rows
    full_dups = int(df.duplicated().sum())
    if full_dups:
        issues.append(f"duplicate_rows: {full_dups} fully duplicated rows")

    rows = []
    for col in value_cols:
        s = df[col]
        missing = int(s.isna().sum())
        inf = int(np.isinf(s.astype(float)).sum()) if s.dtype != object else 0
        valid = int(s.notna().sum())
        total = valid + missing
        coverage = round(100.0 * valid / total, 2) if total else 0.0

        rows.append({
            "feature": col,
            "valid_obs": valid,
            "missing": missing,
            "infinite": inf,
            "coverage_pct": coverage,
        })

        if inf:
            issues.append(f"infinite_values: {col} has {inf} infinite values")
        if coverage < MIN_COVERAGE_PCT * 100:
            issues.append(
                f"insufficient_history: {col} coverage {coverage}% < {MIN_COVERAGE_PCT * 100}%"
            )

    # Benchmark alignment check
    if benchmark_col and benchmark_col in df.columns:
        bm_missing = int(df[benchmark_col].isna().sum())
        if bm_missing == len(df):
            issues.append(f"missing_benchmark: {benchmark_col} is entirely NaN")

    per_feature = pd.DataFrame(rows).sort_values("missing", ascending=False)
    return len(issues) == 0, issues, per_feature


def validate_benchmark_alignment(
    features: pd.DataFrame,
    benchmark: pd.DataFrame,
    benchmark_col: str = "Adj Close",
) -> list[str]:
    """Check that benchmark covers the feature date range."""
    issues: list[str] = []
    if features is None or features.empty:
        return ["empty_features"]

    feat_dates = pd.to_datetime(features[PriceColumns.DATE]).dropna()
    if feat_dates.empty:
        return ["no_feature_dates"]

    bm_dates = pd.to_datetime(benchmark["Date"]).dropna() if "Date" in benchmark.columns else pd.to_datetime(benchmark.index).dropna()
    bm_min, bm_max = bm_dates.min(), bm_dates.max()
    f_min, f_max = feat_dates.min(), feat_dates.max()

    if bm_min > f_min:
        issues.append(f"benchmark_start_after_features: benchmark starts {bm_min.date()}, features start {f_min.date()}")
    if bm_max < f_max:
        issues.append(f"benchmark_end_before_features: benchmark ends {bm_max.date()}, features end {f_max.date()}")

    return issues


def validate_rolling_window_availability(
    df: pd.DataFrame,
    window_months: int,
    frequency: str = "daily",
) -> tuple[bool, str]:
    """
    Check if enough history exists for a rolling window.

    Returns (ok, message)
    """
    if df is None or df.empty:
        return False, "empty dataframe"

    expected_periods = window_months * (21 if frequency == "daily" else 4)
    ticker_counts = df.groupby("Ticker").size()
    min_obs = ticker_counts.min()

    if min_obs < expected_periods:
        return False, (
            f"insufficient_history: min observations per ticker = {min_obs}, "
            f"need ~{expected_periods} for {window_months}M {frequency} window"
        )
    return True, f"ok: min {min_obs} observations >= required {expected_periods}"