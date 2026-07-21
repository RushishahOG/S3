"""Monte Carlo statistics: aggregation, percentiles, probabilities, risk summary.

All functions operate on the per-simulation metrics DataFrame produced by the
engine and return plain Python objects / small DataFrames for the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.monte_carlo.types import METRIC_COLUMNS, SimulationConfig

_PERCENTILES = [5, 10, 25, 50, 75, 90, 95]

_AGG_STATS = ["mean", "median", "std", "min", "max"] + [f"p{p}" for p in _PERCENTILES]


def compute_aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-metric summary statistics including the requested percentiles."""
    rows = {}
    for col in METRIC_COLUMNS:
        if col not in metrics_df.columns:
            continue
        series = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if series.empty:
            rows[col] = {stat: np.nan for stat in _AGG_STATS}
            continue
        row = {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
        }
        for p in _PERCENTILES:
            row[f"p{p}"] = np.percentile(series, p)
        rows[col] = row

    agg = pd.DataFrame(rows, index=_AGG_STATS).T
    agg.index.name = "metric"
    return agg


def compute_probabilities(metrics_df: pd.DataFrame) -> dict[str, float]:
    """Probability that a simulated run satisfies each threshold condition."""
    n = len(metrics_df)
    if n == 0:
        return {}

    def frac(mask) -> float:
        mask = mask.fillna(False)
        return float(mask.sum() / n)

    cagr = metrics_df["cagr"]
    mdd = metrics_df["max_drawdown"]
    sharpe = metrics_df["sharpe"]
    sortino = metrics_df["sortino"]
    total_return = metrics_df["total_return"]

    return {
        "P(profit)": frac(total_return > 0),
        "P(loss)": frac(total_return < 0),
        "P(cagr > 15%)": frac(cagr > 0.15),
        "P(cagr > 18%)": frac(cagr > 0.18),
        "P(cagr > 20%)": frac(cagr > 0.20),
        "P(cagr > 25%)": frac(cagr > 0.25),
        "P(max_drawdown > 20%)": frac(mdd < -0.20),
        "P(max_drawdown > 30%)": frac(mdd < -0.30),
        "P(max_drawdown > 40%)": frac(mdd < -0.40),
        "P(sharpe > 1)": frac(sharpe > 1.0),
        "P(sharpe > 1.5)": frac(sharpe > 1.5),
        "P(sharpe > 2)": frac(sharpe > 2.0),
        "P(sortino > 2)": frac(sortino > 2.0),
    }


def compute_risk_summary(
    metrics_df: pd.DataFrame, config: SimulationConfig
) -> tuple[dict, dict]:
    """Risk KPIs and percentile-based confidence intervals for key metrics."""
    cagr = metrics_df["cagr"]
    final = metrics_df["final_value"]
    mdd = metrics_df["max_drawdown"]
    sharpe = metrics_df["sharpe"]
    sortino = metrics_df["sortino"]
    total_return = metrics_df["total_return"]

    var_95 = float(np.percentile(total_return, 5)) if len(total_return) else np.nan
    tail = total_return[total_return <= var_95]
    cvar_95 = float(tail.mean()) if len(tail) else np.nan

    risk_summary = {
        "probability_of_profit": float((total_return > 0).mean()),
        "probability_of_loss": float((total_return < 0).mean()),
        "worst_drawdown": float(mdd.min()),
        "median_cagr": float(cagr.median()),
        "best_cagr": float(cagr.max()),
        "expected_cagr": float(cagr.mean()),
        "expected_final_portfolio": float(final.mean()),
        "expected_sharpe": float(sharpe.mean()),
        "expected_sortino": float(sortino.mean()),
        "worst_case_95_cagr": float(np.percentile(cagr, 5)),
        "best_case_95_cagr": float(np.percentile(cagr, 95)),
        "worst_case_95_final": float(np.percentile(final, 5)),
        "best_case_95_final": float(np.percentile(final, 95)),
        "var_95": var_95,
        "cvar_95": cvar_95,
    }

    def ci(series: pd.Series) -> dict:
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return {"ci95": (np.nan, np.nan), "ci99": (np.nan, np.nan)}
        return {
            "ci95": (float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))),
            "ci99": (float(np.percentile(s, 0.5)), float(np.percentile(s, 99.5))),
        }

    confidence_intervals = {
        "cagr": ci(cagr),
        "final_value": ci(final),
        "sharpe": ci(sharpe),
        "max_drawdown": ci(mdd),
    }

    return risk_summary, confidence_intervals
