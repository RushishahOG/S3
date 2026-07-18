"""Factor eligibility registry.

A single, framework-agnostic place where *every* factor declares the maximum
historical lookback (in months) it needs before it can be computed for a stock.

The :class:`~core.eligibility.analyzer.EligibilityAnalyzer` reads exclusively
from this registry, so a new factor (Quality, Value, Growth, ...) only has to
call :func:`register_factor` once at import time to be included in the
eligibility framework automatically - no changes to the analyzer or the UI are
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FactorLookback:
    """Describes the warm-up a factor needs before it is eligible.

    Attributes
    ----------
    name:
        Stable factor key (e.g. ``"Beta"``). Used by the UI for selection.
    description:
        Human readable summary of the factor's horizons.
    max_lookback_months:
        Longest lookback (in months) the factor requires. A stock is only
        eligible for the factor once it has at least this much history.
    features:
        Optional list of representative feature keys the factor produces.
    """

    name: str
    description: str
    max_lookback_months: int
    features: tuple[str, ...] = ()


_REGISTRY: dict[str, FactorLookback] = {}


def register_factor(spec: FactorLookback) -> None:
    """Register (or overwrite) a factor's lookback requirement."""
    _REGISTRY[spec.name] = spec


def get_factor(name: str) -> FactorLookback:
    return _REGISTRY[name]


def list_factors() -> dict[str, FactorLookback]:
    """Return a copy of all registered factors keyed by name."""
    return dict(_REGISTRY)


def max_lookback_for(selected: Iterable[str]) -> int:
    """Return the maximum lookback (months) across ``selected`` factor names.

    Returns ``0`` when nothing is selected (callers should treat this as
    "no warm-up required").
    """
    chosen = [s for s in selected if s in _REGISTRY]
    if not chosen:
        return 0
    return max(_REGISTRY[s].max_lookback_months for s in chosen)


def register_default_factors() -> None:
    """Idempotently register the Version 1 factor set.

    New factors (Quality, Value, Growth, ...) simply call
    :func:`register_factor` once; this function is only responsible for the
    factors shipped in V1.
    """
    register_factor(
        FactorLookback(
            name="Momentum",
            description="12-1 momentum (12M horizon, 1M lag), scaled & unscaled.",
            max_lookback_months=12,
            features=("momentum_unscaled", "momentum_scaled"),
        )
    )
    register_factor(
        FactorLookback(
            name="Beta",
            description="Market beta vs NIFTY 500 (12M daily).",
            max_lookback_months=12,
            features=("beta",),
        )
    )
    register_factor(
        FactorLookback(
            name="Low Volatility",
            description="12-month downside deviation (annualised).",
            max_lookback_months=12,
            features=("semi_deviation",),
        )
    )


# Register the V1 factor set on import so the analyzer/UI always have it.
register_default_factors()
