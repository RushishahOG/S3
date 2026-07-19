"""Optimizer parameter specifications for the Parameter Optimization engine.

This module is the **engine-side** counterpart to
:mod:`core.optimization.param_registry`. Where the registry describes
parameters for *discovery and UI*, this module describes them for *optimization*:
each spec carries the search-space semantics the engine needs to build
candidates (continuous / discrete / categorical / boolean), any sum-group it
belongs to (so a group of weights can be auto-normalised or cross-validated),
and optional dependencies.

The engine is deliberately generic: it knows only about ``(block, field)``
paths into :class:`~core.config.backtest_schema.BacktestParameters`. Adding a
new ARQM module to the optimizer is a pure data change here -- no algorithm,
objective, or engine code needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ParamKind(str, Enum):
    """Search-space kind for a parameter."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass(frozen=True)
class OptimizerParamSpec:
    """Engine-side description of one optimizable parameter."""

    key: str
    name: str
    category: str
    block: str
    field: str
    kind: ParamKind
    current: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    #: Discrete allowed values (for DISCRETE) or categorical options (CATEGORICAL).
    allowed: tuple[Any, ...] | None = None
    #: Sum-group id; parameters sharing it must sum to `group_target` (or 1.0).
    group: str | None = None
    group_target: float = 1.0
    #: If True the engine normalises the group so it sums exactly to group_target.
    normalize_group: bool = False
    #: Keys that must also be optimized/enabled for this parameter to apply.
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    help: str = ""


# --- Sum groups -------------------------------------------------------------
# Weights that must sum to 1.0 and are auto-normalised when optimized together.
_GRP_CAP = "cap_weights"
_GRP_SCORING = "scoring_weights"
_GRP_QUALITY = "quality_pillars"


def _default_specs() -> list[OptimizerParamSpec]:
    """Build the canonical optimizer parameter specifications.

    Each entry is a pure data description; the engine, algorithms and UI all
    consume these. To expose a new ARQM parameter, add one entry here.
    """
    return [
        # --- Market Timing -------------------------------------------------
        OptimizerParamSpec("buy_trigger_pct", "Buy Threshold", "Market Timing", "regime",
                           "buy_trigger_pct", ParamKind.CONTINUOUS, 5.0, -50.0, 50.0, 0.5,
                           help="Regime buy-trigger percentage"),
        OptimizerParamSpec("sell_trigger_pct", "Sell Threshold", "Market Timing", "regime",
                           "sell_trigger_pct", ParamKind.CONTINUOUS, -15.0, -50.0, 0.0, 0.5,
                           help="Regime sell-trigger percentage"),

        # --- Portfolio Allocation (cap segmentation) ----------------------
        OptimizerParamSpec("large_cap_weight", "Large Cap %", "Portfolio Allocation", "cap_segment",
                           "large_cap_weight", ParamKind.CONTINUOUS, 0.60, 0.0, 1.0, 0.05,
                           group=_GRP_CAP, group_target=1.0, normalize_group=True,
                           help="Large cap allocation fraction"),
        OptimizerParamSpec("mid_cap_weight", "Mid Cap %", "Portfolio Allocation", "cap_segment",
                           "mid_cap_weight", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           group=_GRP_CAP, group_target=1.0, normalize_group=True,
                           help="Mid cap allocation fraction"),
        OptimizerParamSpec("small_cap_weight", "Small Cap %", "Portfolio Allocation", "cap_segment",
                           "small_cap_weight", ParamKind.CONTINUOUS, 0.10, 0.0, 1.0, 0.05,
                           group=_GRP_CAP, group_target=1.0, normalize_group=True,
                           help="Small cap allocation fraction"),

        # --- Gate Parameters ----------------------------------------------
        OptimizerParamSpec("mom_top_pct", "Momentum Top %", "Gate Parameters", "momentum",
                           "top_pct", ParamKind.CONTINUOUS, 0.30, 0.05, 1.0, 0.05,
                           help="Momentum top-pct selection threshold"),
        OptimizerParamSpec("mom_top_n", "Momentum Top N", "Gate Parameters", "momentum",
                           "top_n", ParamKind.DISCRETE, 50, 5, 200, 5,
                           allowed=(10, 20, 30, 40, 50, 75, 100, 150),
                           help="Momentum top-N selection count"),
        OptimizerParamSpec("stab_top_pct", "Stability Top %", "Gate Parameters", "stability",
                           "top_pct", ParamKind.CONTINUOUS, 0.50, 0.05, 1.0, 0.05,
                           help="Stability top-pct selection threshold"),
        OptimizerParamSpec("stab_top_n", "Stability Top N", "Gate Parameters", "stability",
                           "top_n", ParamKind.DISCRETE, 50, 5, 200, 5,
                           allowed=(10, 20, 30, 40, 50, 75, 100, 150),
                           help="Stability top-N selection count"),
        OptimizerParamSpec("min_quality_score", "Min Quality Score", "Gate Parameters", "quality",
                           "min_quality_score", ParamKind.CONTINUOUS, 0.0, 0.0, 1.0, 0.05,
                           help="Minimum quality score threshold"),

        # --- Momentum ------------------------------------------------------
        OptimizerParamSpec("mom_lookback", "Lookback Period", "Momentum", "momentum",
                           "horizon_months", ParamKind.DISCRETE, 12, 1, 36, 1,
                           allowed=(3, 6, 9, 12, 18, 24, 36),
                           help="Momentum lookback horizon (months)"),
        OptimizerParamSpec("w_momentum", "Momentum Weight", "Momentum", "scoring",
                           "momentum_weight", ParamKind.CONTINUOUS, 0.40, 0.0, 1.0, 0.05,
                           group=_GRP_SCORING, group_target=1.0, normalize_group=True,
                           help="Final scoring momentum weight"),
        OptimizerParamSpec("w_stability", "Stability Weight", "Momentum", "scoring",
                           "stability_weight", ParamKind.CONTINUOUS, 0.20, 0.0, 1.0, 0.05,
                           group=_GRP_SCORING, group_target=1.0, normalize_group=True,
                           help="Final scoring stability weight"),
        OptimizerParamSpec("w_quality", "Quality Weight", "Momentum", "scoring",
                           "quality_weight", ParamKind.CONTINUOUS, 0.40, 0.0, 1.0, 0.05,
                           group=_GRP_SCORING, group_target=1.0, normalize_group=True,
                           help="Final scoring quality weight"),

        # --- Quality -------------------------------------------------------
        OptimizerParamSpec("w_profitability", "Profitability Weight", "Quality", "quality",
                           "profitability_pillar", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           group=_GRP_QUALITY, group_target=1.0, normalize_group=True,
                           help="Quality profitability pillar weight"),
        OptimizerParamSpec("w_growth", "Growth Weight", "Quality", "quality",
                           "growth_pillar", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           group=_GRP_QUALITY, group_target=1.0, normalize_group=True,
                           help="Quality growth pillar weight"),
        OptimizerParamSpec("w_fin_strength", "Financial Strength Weight", "Quality", "quality",
                           "fin_strength_pillar", ParamKind.CONTINUOUS, 0.15, 0.0, 1.0, 0.05,
                           group=_GRP_QUALITY, group_target=1.0, normalize_group=True,
                           help="Quality financial strength pillar weight"),
        OptimizerParamSpec("w_cashflow", "Cash Flow Quality Weight", "Quality", "quality",
                           "cashflow_pillar", ParamKind.CONTINUOUS, 0.15, 0.0, 1.0, 0.05,
                           group=_GRP_QUALITY, group_target=1.0, normalize_group=True,
                           help="Quality cash flow pillar weight"),
        OptimizerParamSpec("w_shareholder", "Shareholder Quality Weight", "Quality", "quality",
                           "shareholder_pillar", ParamKind.CONTINUOUS, 0.10, 0.0, 1.0, 0.05,
                           group=_GRP_QUALITY, group_target=1.0, normalize_group=True,
                           help="Quality shareholder return pillar weight"),

        # --- Portfolio Construction ---------------------------------------
        OptimizerParamSpec("total_size", "Number of Holdings", "Portfolio Construction", "portfolio",
                           "total_size", ParamKind.DISCRETE, 50, 5, 200, 5,
                           allowed=(10, 15, 20, 25, 30, 40, 50, 75, 100),
                           help="Total portfolio size (number of holdings)"),
        OptimizerParamSpec("max_position_pct", "Maximum Stock Weight", "Portfolio Construction", "portfolio",
                           "max_position_pct", ParamKind.CONTINUOUS, 0.07, 0.01, 1.0, 0.01,
                           help="Maximum single-stock position weight"),
        OptimizerParamSpec("rebalance_frequency", "Rebalancing Frequency", "Portfolio Construction", "general",
                           "rebalance_frequency", ParamKind.CATEGORICAL, "quarterly",
                           allowed=("monthly", "quarterly", "semi_annual"),
                           help="Portfolio rebalancing frequency"),
    ]


_SPECS_BY_KEY: dict[str, OptimizerParamSpec] = {s.key: s for s in _default_specs()}


def get_specs() -> list[OptimizerParamSpec]:
    """Return all optimizer parameter specifications."""
    return list(_SPECS_BY_KEY.values())


def get_spec(key: str) -> OptimizerParamSpec | None:
    return _SPECS_BY_KEY.get(key)


def specs_for_keys(keys: list[str]) -> list[OptimizerParamSpec]:
    return [s for k, s in _SPECS_BY_KEY.items() if k in set(keys)]
