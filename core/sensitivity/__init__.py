"""Sensitivity Analysis engine for the ARQM Research Lab.

This package provides the computation layer behind the Sensitivity Analysis UI:
parameter cataloguing, grid construction, cached backtest evaluation, and all
of the post-hoc analytics (sensitivity scores, stability, parameter importance,
correlation, interaction, robustness and recommendations).

The engine is intentionally decoupled from Streamlit so it can be unit-tested
and reused by other callers.
"""

from __future__ import annotations

from core.sensitivity.engine import (
    SensitivityResult,
    build_catalog,
    run_sensitivity,
)

__all__ = ["SensitivityResult", "build_catalog", "run_sensitivity"]
