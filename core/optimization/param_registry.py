"""Generic, algorithm-agnostic parameter discovery framework for ARQM.

This module is the foundation of the Research Lab *Parameter Optimization*
engine. It deliberately knows **nothing** about individual ARQM modules such as
Momentum, Quality, Low Volatility, or Portfolio Construction. Instead, every
configurable block of :class:`~core.config.backtest_schema.BacktestParameters`
exposes a small, declarative *metadata* description of the parameters it is
willing to expose for optimization. The optimizer discovers parameters purely
from that metadata.

Design goals
------------
* **Zero hardcoding of strategy logic.** Adding a new ARQM module only requires
  registering its parameter metadata in :data:`PARAMETER_REGISTRY`. The
  optimization UI, search-space estimator, and (future) optimization algorithms
  automatically pick it up -- no engine changes needed.
* **Single source of truth.** The metadata mirrors the dataclass fields in
  ``backtest_schema.py`` so the current value always reflects a real, live
  configuration.
* **Extensible.** New parameter groups, new data types, and new validation
  rules are added by extending the declarative descriptors below.

The central abstraction is :class:`OptimizableParameter`, a self-describing
record of everything the optimizer (and UI) needs to know about one tunable
knob: its name, category, current/min/max/step values, data type, validation
rules, and the path into the underlying ``BacktestParameters`` object that owns
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from core.config.backtest_schema import BacktestParameters


class ParamType(str, Enum):
    """Supported data types for an optimizable parameter."""

    FLOAT = "float"
    INT = "int"
    CHOICE = "choice"
    BOOL = "bool"


@dataclass(frozen=True)
class ValidationRules:
    """Declarative validation constraints for a parameter.

    All fields are optional; a field left at its default means "no constraint".
    These rules are consumed both by the UI (to render sensible inputs) and by
    the future optimization engine (to validate candidate configurations).
    """

    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: tuple[str, ...] | None = None
    #: Whether this parameter is required to stay enabled for a valid strategy.
    required: bool = False
    #: Free-form human readable hint shown in the UI.
    help: str = ""


@dataclass(frozen=True)
class OptimizableParameter:
    """A self-describing tunable parameter owned by some ARQM config block.

    Attributes
    ----------
    name:
        Human-readable label (e.g. ``"Lookback Period"``).
    key:
        Stable machine key, unique within its category (e.g. ``"top_n"``).
    category:
        Grouping label used by the UI (e.g. ``"Momentum"``).
    block:
        The ``BacktestParameters`` sub-block that owns this field
        (e.g. ``"momentum"``).
    field_name:
        The exact attribute name on that sub-block dataclass.
    param_type:
        One of :class:`ParamType`.
    default_value:
        The schema default (used to reset / initialize the search space).
    current_value:
        The value from the selected base strategy configuration.
    validation:
        Declarative :class:`ValidationRules`.
    """

    name: str
    key: str
    category: str
    block: str
    field_name: str
    param_type: ParamType
    default_value: Any
    current_value: Any
    validation: ValidationRules = field(default_factory=ValidationRules)

    @property
    def min_value(self) -> float | None:
        return self.validation.min_value

    @property
    def max_value(self) -> float | None:
        return self.validation.max_value

    @property
    def step(self) -> float | None:
        return self.validation.step

    @property
    def choices(self) -> tuple[str, ...] | None:
        return self.validation.choices

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "key": self.key,
            "category": self.category,
            "block": self.block,
            "field_name": self.field_name,
            "param_type": self.param_type.value,
            "default_value": self.default_value,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "choices": list(self.choices) if self.choices else None,
            "validation": {
                "required": self.validation.required,
                "help": self.validation.help,
            },
        }


# ---------------------------------------------------------------------------
# Parameter metadata registry
# ---------------------------------------------------------------------------
# Each entry declares one optimizable parameter. The ``block``/``field_name``
# pair is the canonical path into BacktestParameters used to read the live
# current value and (later) to write candidate values back during optimization.
#
# To make a new ARQM module optimizable, simply append its parameter metadata
# here -- the discovery engine, UI, and search-space estimator pick it up
# automatically. Nothing about the optimizer itself changes.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ParamMeta:
    name: str
    key: str
    category: str
    block: str
    field_name: str
    param_type: ParamType
    default_value: Any
    validation: ValidationRules = field(default_factory=ValidationRules)


_PARAMETER_METADATA: list[_ParamMeta] = [
    # --- Market Timing ----------------------------------------------------
    _ParamMeta("Buy Threshold", "buy_trigger_pct", "Market Timing", "regime",
               "buy_trigger_pct", ParamType.FLOAT, 5.0,
               ValidationRules(min_value=-50.0, max_value=50.0, step=0.5,
                               help="Regime buy-trigger percentage")),
    _ParamMeta("Sell Threshold", "sell_trigger_pct", "Market Timing", "regime",
               "sell_trigger_pct", ParamType.FLOAT, -15.0,
               ValidationRules(min_value=-50.0, max_value=0.0, step=0.5,
                               help="Regime sell-trigger percentage")),

    # --- Portfolio Allocation (cap segmentation) -------------------------
    _ParamMeta("Large Cap %", "large_cap_weight", "Portfolio Allocation", "cap_segment",
               "large_cap_weight", ParamType.FLOAT, 0.60,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Large cap allocation fraction (with mid+small sums to 1.0)")),
    _ParamMeta("Mid Cap %", "mid_cap_weight", "Portfolio Allocation", "cap_segment",
               "mid_cap_weight", ParamType.FLOAT, 0.30,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Mid cap allocation fraction")),
    _ParamMeta("Small Cap %", "small_cap_weight", "Portfolio Allocation", "cap_segment",
               "small_cap_weight", ParamType.FLOAT, 0.10,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Small cap allocation fraction")),

    # --- Gate Parameters --------------------------------------------------
    _ParamMeta("Gate Order: Momentum", "mom_order", "Gate Parameters", "pipeline",
               "momentum_order", ParamType.INT, 1,
               ValidationRules(min_value=0, max_value=10, step=1,
                               help="Execution order of the momentum gate")),
    _ParamMeta("Gate Threshold: Momentum", "mom_top_pct", "Gate Parameters", "momentum",
               "top_pct", ParamType.FLOAT, 0.30,
               ValidationRules(min_value=0.05, max_value=1.0, step=0.05,
                               help="Momentum top-pct selection threshold")),
    _ParamMeta("Top N Selection: Momentum", "mom_top_n", "Gate Parameters", "momentum",
               "top_n", ParamType.INT, 50,
               ValidationRules(min_value=5, max_value=200, step=5,
                               help="Momentum top-N selection count")),
    _ParamMeta("Gate Threshold: Stability", "stab_top_pct", "Gate Parameters", "stability",
               "top_pct", ParamType.FLOAT, 0.50,
               ValidationRules(min_value=0.05, max_value=1.0, step=0.05,
                               help="Stability top-pct selection threshold")),
    _ParamMeta("Top N Selection: Stability", "stab_top_n", "Gate Parameters", "stability",
               "top_n", ParamType.INT, 50,
               ValidationRules(min_value=5, max_value=200, step=5,
                               help="Stability top-N selection count")),
    _ParamMeta("Min Quality Score", "min_quality_score", "Gate Parameters", "quality",
               "min_quality_score", ParamType.FLOAT, 0.0,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Minimum quality score threshold")),

    # --- Momentum ---------------------------------------------------------
    _ParamMeta("Lookback Period", "mom_lookback", "Momentum", "momentum",
               "horizon_months", ParamType.INT, 12,
               ValidationRules(min_value=1, max_value=36, step=1,
                               help="Momentum lookback horizon (months)")),
    _ParamMeta("Momentum Weight", "w_momentum", "Momentum", "scoring",
               "momentum_weight", ParamType.FLOAT, 0.40,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Final scoring momentum weight")),
    _ParamMeta("Beta Weight", "w_stability", "Momentum", "scoring",
               "stability_weight", ParamType.FLOAT, 0.20,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Stability (low-vol) weight in final scoring")),
    _ParamMeta("Semi-Deviation Weight", "w_stability2", "Momentum", "scoring",
               "stability_weight", ParamType.FLOAT, 0.20,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Stability weight in final scoring (semi-deviation pillar)")),

    # --- Quality ----------------------------------------------------------
    _ParamMeta("Profitability Weight", "w_profitability", "Quality", "quality",
               "profitability_pillar", ParamType.FLOAT, 0.30,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Quality profitability pillar weight")),
    _ParamMeta("Growth Weight", "w_growth", "Quality", "quality",
               "growth_pillar", ParamType.FLOAT, 0.30,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Quality growth pillar weight")),
    _ParamMeta("Financial Strength Weight", "w_fin_strength", "Quality", "quality",
               "fin_strength_pillar", ParamType.FLOAT, 0.15,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Quality financial strength pillar weight")),
    _ParamMeta("Cash Flow Quality Weight", "w_cashflow", "Quality", "quality",
               "cashflow_pillar", ParamType.FLOAT, 0.15,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Quality cash flow pillar weight")),
    _ParamMeta("Shareholder Quality Weight", "w_shareholder", "Quality", "quality",
               "shareholder_pillar", ParamType.FLOAT, 0.10,
               ValidationRules(min_value=0.0, max_value=1.0, step=0.05,
                               help="Quality shareholder return pillar weight")),

    # --- Portfolio Construction ------------------------------------------
    _ParamMeta("Number of Holdings", "total_size", "Portfolio Construction", "portfolio",
               "total_size", ParamType.INT, 50,
               ValidationRules(min_value=5, max_value=200, step=5,
                               help="Total portfolio size (number of holdings)")),
    _ParamMeta("Maximum Stock Weight", "max_position_pct", "Portfolio Construction", "portfolio",
               "max_position_pct", ParamType.FLOAT, 0.07,
               ValidationRules(min_value=0.01, max_value=1.0, step=0.01,
                               help="Maximum single-stock position weight")),
    _ParamMeta("Rebalancing Frequency", "rebalance_frequency", "Portfolio Construction", "general",
               "rebalance_frequency", ParamType.CHOICE, "quarterly",
               ValidationRules(choices=("monthly", "quarterly", "semi_annual"),
                               help="Portfolio rebalancing frequency")),
]


def _read_current_value(params: BacktestParameters, meta: _ParamMeta) -> Any:
    """Resolve the live current value of a metadata entry from a config object.

    Most parameters map directly to a dataclass field. A few (e.g. pipeline gate
    order) require special handling; those are resolved here without the
    optimizer needing to know the details.
    """
    block = getattr(params, meta.block)
    if meta.field_name.endswith("_order"):
        # Pipeline gate order is stored on the GateSpec tuple within PipelineConfig.
        gate_kind = meta.field_name.replace("_order", "")
        for gate in block.gates:
            if gate.kind == gate_kind:
                return gate.order
        return meta.default_value
    return getattr(block, meta.field_name, meta.default_value)


def discover_parameters(base_params: BacktestParameters | None = None) -> list[OptimizableParameter]:
    """Discover all optimizable parameters from the ARQM configuration.

    Parameters
    ----------
    base_params:
        The base strategy configuration whose current values seed the discovery.
        When ``None`` (e.g. no strategy selected yet), the schema defaults are
        used so the UI can still render every available parameter.

    Returns
    -------
    list[OptimizableParameter]
        Fully described, self-validating parameter records. Order mirrors
        :data:`_PARAMETER_METADATA` (i.e. grouped by category).
    """
    source = base_params if base_params is not None else BacktestParameters()
    discovered: list[OptimizableParameter] = []
    for meta in _PARAMETER_METADATA:
        current = _read_current_value(source, meta)
        discovered.append(
            OptimizableParameter(
                name=meta.name,
                key=meta.key,
                category=meta.category,
                block=meta.block,
                field_name=meta.field_name,
                param_type=meta.param_type,
                default_value=meta.default_value,
                current_value=current,
                validation=meta.validation,
            )
        )
    return discovered


def group_by_category(parameters: list[OptimizableParameter]) -> dict[str, list[OptimizableParameter]]:
    """Group parameters by their category, preserving first-seen order."""
    groups: dict[str, list[OptimizableParameter]] = {}
    for p in parameters:
        groups.setdefault(p.category, []).append(p)
    return groups


def get_parameter_metadata() -> list[_ParamMeta]:
    """Return the raw metadata registry (used by tests and introspection)."""
    return list(_PARAMETER_METADATA)
