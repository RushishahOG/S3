"""Backtest & simulation configuration schema (ARQM framework).

Every strategy parameter lives here as a typed, frozen dataclass so the
simulation engine never hardcodes a threshold, weight, or window. The UI builds
a plain ``dict`` of these settings; :class:`BacktestParameters` is the validated
container the engine consumes.

Design notes
------------
* All dataclasses are frozen + defaulted so a backtest is fully reproducible from
  a single ``BacktestParameters`` object (serialise to JSON/YAML for audit).
* Numeric weights that must sum to 1.0 are *validated* at construction via
  ``__post_init__`` guards -- the engine never re-checks sums.
* The schema is deliberately granular: each gate (Eligibility / Momentum /
  Stability / Quality / Portfolio) owns its own config block so a user can toggle
  and tune one stage without touching the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Literal

RebalanceFreq = Literal["monthly", "quarterly", "semi_annual"]
Normalization = Literal["zscore", "robust_zscore", "percentile", "minmax"]
BenchmarkName = Literal["NIFTY_500", "NIFTY_50"]
SelectionMode = Literal["top_pct", "top_n"]
CapacityMode = Literal["equal", "score", "hybrid"]


@dataclass(frozen=True)
class GeneralConfig:
    start_date: str = "2006-01-01"
    end_date: str = "2026-05-31"
    initial_capital: float = 10_000_000.0
    benchmark: BenchmarkName = "NIFTY_500"
    transaction_cost_pct: float = 0.05
    slippage_pct: float = 0.05
    rebalance_frequency: RebalanceFreq = "quarterly"


@dataclass(frozen=True)
class RegimeConfig:
    buy_trigger_pct: float = 5.0
    sell_trigger_pct: float = -15.0
    enable_swing_low: bool = True
    enable_peak_detection: bool = True
    swing_low_window: int = 20
    peak_window: int = 20
    reference: Literal["benchmark", "portfolio"] = "benchmark"


@dataclass(frozen=True)
class UniverseConfig:
    use_nifty500: bool = True
    min_trading_history_days: int = 252
    min_liquidity_avg_value: float = 0.0
    require_fundamental_data: bool = True
    require_quality_features: bool = True
    require_lowvol_features: bool = True
    require_momentum_data: bool = True
    min_avg_volume: float = 0.0


@dataclass(frozen=True)
class CapSegmentConfig:
    enabled: bool = True
    large_cap_weight: float = 0.60
    mid_cap_weight: float = 0.30
    small_cap_weight: float = 0.10

    def __post_init__(self) -> None:
        total = self.large_cap_weight + self.mid_cap_weight + self.small_cap_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Cap allocation must sum to 1.0 (got {total:.4f})")

    @property
    def weights(self) -> dict[str, float]:
        return {"large": self.large_cap_weight, "mid": self.mid_cap_weight, "small": self.small_cap_weight}


@dataclass(frozen=True)
class MomentumFactorConfig:
    name: str
    enabled: bool = True
    weight: float = 1.0
    horizon_months: int = 12
    lag_months: int = 1
    normalize: bool = True


@dataclass(frozen=True)
class MomentumConfig:
    factors: tuple[MomentumFactorConfig, ...] = field(
        default_factory=lambda: (
            MomentumFactorConfig(name="momentum_scaled", horizon_months=12, lag_months=1, weight=1.0),
            MomentumFactorConfig(name="momentum_unscaled", horizon_months=12, lag_months=1, weight=0.5),
        )
    )
    selection_mode: SelectionMode = "top_pct"
    top_pct: float = 0.30
    top_n: int = 50
    combine_method: Literal["weighted_score", "rank_average"] = "weighted_score"
    normalization: Normalization = "zscore"
    column_map: dict[str, str] = field(
        default_factory=lambda: {
            "momentum_scaled": "momentum_scaled",
            "momentum_unscaled": "momentum_unscaled",
        }
    )


@dataclass(frozen=True)
class StabilityFactorConfig:
    name: str
    enabled: bool = True
    weight: float = 1.0
    normalize: bool = True


@dataclass(frozen=True)
class StabilityConfig:
    factors: tuple[StabilityFactorConfig, ...] = field(
        default_factory=lambda: (
            StabilityFactorConfig(name="semi_deviation", weight=1.0),
            StabilityFactorConfig(name="beta", weight=1.0),
        )
    )
    selection_mode: SelectionMode = "top_pct"
    top_pct: float = 0.50
    top_n: int = 50
    combine_method: Literal["weighted_score", "rank_average"] = "weighted_score"
    normalization: Normalization = "zscore"
    column_map: dict[str, str] = field(
        default_factory=lambda: {
            "semi_deviation": "semi_deviation",
            "beta": "beta",
        }
    )


@dataclass(frozen=True)
class PersistenceConfig:
    enabled: bool = False
    required_periods: int = 2
    momentum_quantile: float = 0.50
    stability_quantile: float = 0.50


@dataclass(frozen=True)
class QualityFactorConfig:
    name: str
    pillar: Literal["profitability", "growth", "financial_strength", "cash_flow", "shareholder_return"]
    enabled: bool = True
    weight: float = 1.0
    min_threshold: float | None = None
    normalize: bool = True


@dataclass(frozen=True)
class QualityConfig:
    factors: tuple[QualityFactorConfig, ...] = field(
        default_factory=lambda: (
            # --- Profitability (30%) ---------------------------------------
            QualityFactorConfig("roe", "profitability", weight=1.0, min_threshold=0.12),
            QualityFactorConfig("roce", "profitability", weight=1.0, min_threshold=0.12),
            QualityFactorConfig("roa", "profitability", weight=0.8, min_threshold=0.08),
            QualityFactorConfig("cash_roce", "profitability", weight=0.8, min_threshold=0.10),
            # --- Growth (30%) : weighted-growth variants --------------------
            QualityFactorConfig("eps_growth_weighted", "growth", weight=1.0),
            QualityFactorConfig("revenue_growth_weighted", "growth", weight=1.0),
            QualityFactorConfig("roe_growth_weighted", "growth", weight=0.8),
            QualityFactorConfig("roce_growth_weighted", "growth", weight=0.8),
            QualityFactorConfig("dps_growth_weighted", "growth", weight=0.6),
            QualityFactorConfig("sustainable_growth_rate", "growth", weight=0.8),
            # --- Financial Strength (15%) ----------------------------------
            QualityFactorConfig("interest_coverage_ratio", "financial_strength", weight=1.0, min_threshold=1.5),
            QualityFactorConfig("equity_to_total_capital", "financial_strength", weight=1.0, min_threshold=0.40),
            # --- Cash Flow Quality (15%) -----------------------------------
            QualityFactorConfig("ocf_to_ebitda", "cash_flow", weight=1.0, min_threshold=0.10),
            # --- Shareholder Quality (10%) ---------------------------------
            QualityFactorConfig("dividend_payout_ratio", "shareholder_return", weight=0.6),
            QualityFactorConfig("dividend_payout_ratio_cumulative", "shareholder_return", weight=0.5),
        )
    )
    pillar_weights: dict[str, float] = field(
        default_factory=lambda: {
            "profitability": 0.30,
            "growth": 0.30,
            "financial_strength": 0.15,
            "cash_flow": 0.15,
            "shareholder_return": 0.10,
        }
    )
    normalization: Normalization = "zscore"
    min_quality_score: float = 0.0
    use_rollup: Literal["latest", "median", "weighted"] = "median"

    def __post_init__(self) -> None:
        total = sum(self.pillar_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Pillar weights must sum to 1.0 (got {total:.4f})")


@dataclass(frozen=True)
class ScoringConfig:
    momentum_weight: float = 0.40
    quality_weight: float = 0.40
    stability_weight: float = 0.20

    def __post_init__(self) -> None:
        total = self.momentum_weight + self.quality_weight + self.stability_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Overall score weights must sum to 1.0 (got {total:.4f})")

    @property
    def weights(self) -> dict[str, float]:
        return {"momentum": self.momentum_weight, "quality": self.quality_weight, "stability": self.stability_weight}


@dataclass(frozen=True)
class PortfolioConfig:
    total_size: int = 50
    large_size: int = 25
    mid_size: int = 15
    small_size: int = 10
    sizing_method: CapacityMode = "hybrid"
    max_position_pct: float = 0.07
    hybrid_momentum_weight: float = 0.40
    hybrid_quality_weight: float = 0.40
    hybrid_stability_weight: float = 0.20

    def __post_init__(self) -> None:
        total = self.hybrid_momentum_weight + self.hybrid_quality_weight + self.hybrid_stability_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Hybrid sizing weights must sum to 1.0 (got {total:.4f})")


@dataclass(frozen=True)
class ManagementConfig:
    quality_drop_quantile: float = 0.20
    momentum_drop_quantile: float = 0.20
    stability_drop_quantile: float = 0.20
    exit_on_sell_signal: bool = True


@dataclass(frozen=True)
class GateSpec:
    """A single stage in the ARQM pipeline.

    The pipeline engine executes every *enabled* gate in ascending ``order``.
    ``kind`` references a registered :class:`~core.backtesting.gate_registry.GateNode`
    (e.g. ``"momentum"``, ``"stability"``, ``"quality"``, ``"persistence"``,
    ``"eligibility"``). ``config_key`` points at the ``BacktestParameters`` block
    that supplies this gate's factor-specific parameters (``momentum`` ->
    ``MomentumConfig``, etc.). Future factors (Value / Size / ESG / ...) are added
    simply by registering a new node and appending a ``GateSpec`` -- no engine or
    schema changes required.
    """

    kind: str
    enabled: bool = True
    order: int = 0
    config_key: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Ordered, user-arrangeable list of pipeline gates.

    ``gates`` is sorted by ``order`` at runtime (see
    :func:`BacktestParameters.active_pipeline`). Disabled gates are skipped. The
    default replicates the classic ARQM order (Eligibility -> Momentum ->
    Stability -> Quality -> Persistence) but any permutation is supported.
    """

    gates: tuple[GateSpec, ...] = field(
        default_factory=lambda: (
            GateSpec(kind="eligibility", order=0, config_key="universe"),
            GateSpec(kind="momentum", order=1, config_key="momentum"),
            GateSpec(kind="stability", order=2, config_key="stability"),
            GateSpec(kind="quality", order=3, config_key="quality"),
            GateSpec(kind="persistence", order=4, config_key="persistence"),
        )
    )
    final_scoring: bool = True

    def __post_init__(self) -> None:
        orders = [g.order for g in self.gates]
        if len(set(orders)) != len(orders):
            raise ValueError(f"Pipeline gate orders must be unique (got {orders})")
        if not self.gates:
            raise ValueError("Pipeline must declare at least one gate")


def _dc(dc, data: dict):
    known = {f.name for f in fields(dc)}
    return dc(**{k: v for k, v in data.items() if k in known})


def _block_kwargs(dc, data: dict, factors=None):
    """Build kwargs for a block dataclass, injecting ``factors`` when provided."""
    known = {f.name for f in fields(dc)}
    kw = {k: v for k, v in data.items() if k in known}
    if factors is not None:
        kw["factors"] = factors
    return kw


def _factors(factory, items):
    if not items:
        return None
    flds = {f.name for f in fields(factory)}
    return tuple(factory(**{k: v for k, v in it.items() if k in flds}) for it in items)


@dataclass(frozen=True)
class BacktestParameters:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    cap_segment: CapSegmentConfig = field(default_factory=CapSegmentConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    stability: StabilityConfig = field(default_factory=StabilityConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def to_dict(self) -> dict:
        out = {}
        for f in fields(self):
            val = getattr(self, f.name)
            block = {}
            for sf in fields(val):
                v = getattr(val, sf.name)
                if isinstance(v, tuple):
                    block[sf.name] = [vars(x) for x in v]
                else:
                    block[sf.name] = v
            out[f.name] = block
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "BacktestParameters":
        mom = data.get("momentum", {})
        mom_factors = _factors(MomentumFactorConfig, mom.get("factors"))
        stab = data.get("stability", {})
        stab_factors = _factors(StabilityFactorConfig, stab.get("factors"))
        qual = data.get("quality", {})
        qual_factors = _factors(QualityFactorConfig, qual.get("factors"))
        pipe = data.get("pipeline", {})
        pipe_gates = _factors(GateSpec, pipe.get("gates"))
        return cls(
            general=_dc(GeneralConfig, data.get("general", {})),
            regime=_dc(RegimeConfig, data.get("regime", {})),
            universe=_dc(UniverseConfig, data.get("universe", {})),
            cap_segment=_dc(CapSegmentConfig, data.get("cap_segment", {})),
            momentum=MomentumConfig(**_block_kwargs(MomentumConfig, mom, factors=mom_factors)),
            stability=StabilityConfig(**_block_kwargs(StabilityConfig, stab, factors=stab_factors)),
            persistence=_dc(PersistenceConfig, data.get("persistence", {})),
            quality=QualityConfig(**_block_kwargs(QualityConfig, qual, factors=qual_factors)),
            scoring=_dc(ScoringConfig, data.get("scoring", {})),
            portfolio=_dc(PortfolioConfig, data.get("portfolio", {})),
            management=_dc(ManagementConfig, data.get("management", {})),
            pipeline=PipelineConfig(
                gates=tuple(pipe_gates) if pipe_gates is not None else pipe.get("gates", ()),
                final_scoring=pipe.get("final_scoring", True),
            ),
        )
