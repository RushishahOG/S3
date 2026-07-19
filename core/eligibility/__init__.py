"""Eligibility analysis package.

The :mod:`core.eligibility.registry` is the single integration point for new
factors; the :class:`core.eligibility.analyzer.EligibilityAnalyzer` consumes it
and never needs to change when a factor is added.
"""

from .analyzer import (
    EligibilityAnalyzer,
    EligibilityResult,
    REBALANCE_FREQUENCIES,
)
from .registry import (
    FactorLookback,
    list_factors,
    max_lookback_for,
    register_factor,
    register_default_factors,
)

__all__ = [
    "EligibilityAnalyzer",
    "EligibilityResult",
    "REBALANCE_FREQUENCIES",
    "FactorLookback",
    "list_factors",
    "max_lookback_for",
    "register_factor",
    "register_default_factors",
]