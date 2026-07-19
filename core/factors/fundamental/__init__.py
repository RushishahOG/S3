"""Fundamental Quality Factor Engineering package.

Exports:
    FundamentalQualityEngine: Engine to compute 16 Quality factors from screener data.
    QUALITY_FACTOR_FUNCTIONS: Dictionary of available quality factor compute functions.
    compute_*: Individual factor computation functions.
"""

from .engine import (
    FundamentalQualityEngine,
    QUALITY_FACTOR_FUNCTIONS,
    compute_cash_roce,
    compute_dividend_payout_ratio,
    compute_dividend_payout_ratio_cumulative,
    compute_equity_to_total_capital,
    compute_interest_coverage_ratio,
    compute_ocf_to_ebitda,
    compute_roa,
    compute_roce,
    compute_roe,
    compute_sustainable_growth_rate,
)

__all__ = [
    "FundamentalQualityEngine",
    "QUALITY_FACTOR_FUNCTIONS",
    "compute_cash_roce",
    "compute_dividend_payout_ratio",
    "compute_dividend_payout_ratio_cumulative",
    "compute_equity_to_total_capital",
    "compute_interest_coverage_ratio",
    "compute_ocf_to_ebitda",
    "compute_roa",
    "compute_roce",
    "compute_roe",
    "compute_sustainable_growth_rate",
]