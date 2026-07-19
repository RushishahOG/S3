"""Fundamental Quality Factor Engineering package.

Exports:
    FundamentalQualityEngine: Engine to compute 16 Quality factors from screener data.
    QUALITY_FACTOR_FUNCTIONS: Dictionary of available quality factor compute functions.
"""

from .engine import (
    FundamentalQualityEngine,
    QUALITY_FACTOR_FUNCTIONS,
)

__all__ = ["FundamentalQualityEngine", "QUALITY_FACTOR_FUNCTIONS"]