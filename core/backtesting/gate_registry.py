"""Flexible, registry-driven gate pipeline for the ARQM backtest engine.

The backtest is no longer a hardcoded sequence of calls. Instead every stage is a
:class:`GateNode` registered under a stable ``kind`` string (``momentum``,
``stability``, ``quality``, ``persistence``, ``eligibility``, ...). The engine
runs the gates listed in :class:`~core.config.backtest_schema.PipelineConfig` in
ascending ``order``, passing a shared :class:`PipelineContext` between them so
each gate consumes the *output* of the previous one (the incremental data-flow
requirement).

Adding a new factor (Value / Size / ESG / Growth / Dividend / Sentiment / ML) is a
pure extension: write a ``GateNode`` subclass, ``register`` it, and append a
:class:`GateSpec` to the pipeline. No engine or schema edits are required.

Standardised node interface
----------------------------
Every node exposes:

* ``kind`` / ``label`` -- identity and human-readable name.
* ``run(ctx) -> GateResult`` -- pure processing + ranking + selection for one
  rebalance cross-section. It reads what it needs from ``ctx`` (input universe,
  as-of factor frames, prior scores) and writes back the narrowed universe, the
  per-stock score series and the selected (passing) tickers.
* ``config_key`` -- which ``BacktestParameters`` block holds its parameters.

The engine wraps each ``run`` call to capture timing, input/output counts,
rejections and logs into a :class:`GateResult`, building the per-gate audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.config.backtest_schema import BacktestParameters
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Context & result contracts                                                   #
# --------------------------------------------------------------------------- #
@dataclass
class PipelineContext:
    """Mutable shared state threaded through every gate for one rebalance date.

    Gates read ``eligible`` (current input universe), the point-in-time factor
    frames (``market_features`` / ``quality``) and any prior scores, then narrow
    ``eligible`` and publish their score series / pillar scores back onto the
    context for downstream gates and the final scorer.
    """

    date: pd.Timestamp
    params: BacktestParameters
    prices: pd.DataFrame
    cap_tier: pd.Series
    market_features: pd.DataFrame  # as-of point-in-time daily market factors
    quality: pd.DataFrame  # as-of point-in-time yearly quality factors
    eligible: list[str]  # current input universe (mutated by gates)
    # --- published score outputs (filled by gates) -------------------------
    momentum_score: pd.Series = field(default_factory=pd.Series)
    stability_score: pd.Series = field(default_factory=pd.Series)
    quality_score: pd.Series = field(default_factory=pd.Series)
    pillar_scores: dict[str, pd.Series] = field(default_factory=dict)
    # --- extras used by special gates --------------------------------------
    momentum_history: list = field(default_factory=list)
    rejection_reasons: dict[str, str] = field(default_factory=dict)

    def narrowed_to(self, tickers: list[str]) -> None:
        """Restrict the working universe to ``tickers`` (a gate's pass-list)."""
        self.eligible = [t for t in self.eligible if t in set(tickers)]


@dataclass
class GateResult:
    """Persistent, inspectable record of one gate's execution at one rebalance."""

    kind: str
    label: str
    order: int
    enabled: bool
    status: str  # "completed" | "running" | "failed" | "skipped"
    input_universe: list[str] = field(default_factory=list)
    output_universe: list[str] = field(default_factory=list)
    selected: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    score: pd.Series = field(default_factory=pd.Series)  # ticker -> score (0..1)
    pillar_scores: dict[str, pd.Series] = field(default_factory=dict)
    execution_time_s: float = 0.0
    logs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def n_filtered(self) -> int:
        return len(self.input_universe) - len(self.output_universe)

    @property
    def n_retained(self) -> int:
        return len(self.output_universe)


# --------------------------------------------------------------------------- #
# Node base class                                                              #
# --------------------------------------------------------------------------- #
class GateNode:
    """Base class for every pipeline stage.

    Subclasses implement ``run`` and set ``kind`` / ``label`` / ``config_key``.
    """

    kind: str = "base"
    label: str = "Base Gate"
    config_key: str | None = None

    def run(self, ctx: PipelineContext) -> GateResult:
        raise NotImplementedError

    # --- small helpers shared by nodes -------------------------------------
    def _log(self, res: GateResult, msg: str) -> None:
        res.logs.append(msg)
        logger.debug("[%s] %s", self.kind, msg)

    def _warn(self, res: GateResult, msg: str) -> None:
        res.warnings.append(msg)
        logger.warning("[%s] %s", self.kind, msg)


# --------------------------------------------------------------------------- #
# Built-in gate nodes                                                          #
# --------------------------------------------------------------------------- #
class EligibilityGate(GateNode):
    kind = "eligibility"
    label = "Eligibility Filter"
    config_key = "universe"

    def run(self, ctx: PipelineContext) -> GateResult:
        from core.backtesting import gates as G

        res = GateResult(kind=self.kind, label=self.label, order=0, enabled=True, status="running")
        res.input_universe = list(ctx.eligible)
        eligible, reasons = G.eligibility_filter(
            ctx.prices, ctx.quality, ctx.market_features, ctx.cap_tier,
            ctx.eligible, ctx.date, ctx.params,
        )
        ctx.eligible = eligible
        ctx.rejection_reasons = reasons
        res.output_universe = list(eligible)
        res.rejected = [t for t in res.input_universe if t not in set(eligible)]
        self._log(res, f"{len(eligible)} eligible of {len(res.input_universe)}")
        res.status = "completed"
        return res


class MomentumGate(GateNode):
    kind = "momentum"
    label = "Momentum Discovery"
    config_key = "momentum"

    def run(self, ctx: PipelineContext) -> GateResult:
        from core.backtesting import gates as G

        res = GateResult(kind=self.kind, label=self.label, order=0, enabled=True, status="running")
        res.input_universe = list(ctx.eligible)
        score, selected = G.momentum_gate(ctx.market_features, ctx.eligible, ctx.params)
        ctx.momentum_score = score
        ctx.narrowed_to(selected)
        res.score = score
        res.selected = list(ctx.eligible)
        res.output_universe = list(ctx.eligible)
        res.rejected = [t for t in res.input_universe if t not in set(ctx.eligible)]
        self._log(res, f"top {len(ctx.eligible)} by momentum")
        res.status = "completed"
        return res


class StabilityGate(GateNode):
    kind = "stability"
    label = "Low Volatility / Stability"
    config_key = "stability"

    def run(self, ctx: PipelineContext) -> GateResult:
        from core.backtesting import gates as G

        res = GateResult(kind=self.kind, label=self.label, order=0, enabled=True, status="running")
        res.input_universe = list(ctx.eligible)
        score, selected = G.stability_gate(ctx.market_features, ctx.eligible, ctx.params)
        ctx.stability_score = score
        ctx.narrowed_to(selected)
        res.score = score
        res.selected = list(ctx.eligible)
        res.output_universe = list(ctx.eligible)
        res.rejected = [t for t in res.input_universe if t not in set(ctx.eligible)]
        self._log(res, f"top {len(ctx.eligible)} by stability")
        res.status = "completed"
        return res


class QualityGate(GateNode):
    kind = "quality"
    label = "Quality Validation"
    config_key = "quality"

    def run(self, ctx: PipelineContext) -> GateResult:
        from core.backtesting import gates as G

        res = GateResult(kind=self.kind, label=self.label, order=0, enabled=True, status="running")
        res.input_universe = list(ctx.eligible)
        pillar_scores, score, selected = G.quality_gate(ctx.quality, ctx.eligible, ctx.params)
        ctx.quality_score = score
        ctx.pillar_scores = pillar_scores
        ctx.narrowed_to(selected)
        res.score = score
        res.pillar_scores = pillar_scores
        res.selected = list(ctx.eligible)
        res.output_universe = list(ctx.eligible)
        res.rejected = [t for t in res.input_universe if t not in set(ctx.eligible)]
        self._log(res, f"top {len(ctx.eligible)} by quality")
        res.status = "completed"
        return res


class PersistenceGate(GateNode):
    kind = "persistence"
    label = "Persistence Filter"
    config_key = "persistence"

    def run(self, ctx: PipelineContext) -> GateResult:
        from core.backtesting import gates as G

        res = GateResult(kind=self.kind, label=self.label, order=0, enabled=True, status="running")
        res.input_universe = list(ctx.eligible)
        if not ctx.params.persistence.enabled:
            res.output_universe = list(ctx.eligible)
            res.selected = list(ctx.eligible)
            self._log(res, "disabled in config; passed through")
            res.status = "completed"
            return res
        passed = G.apply_persistence(
            ctx.momentum_score, ctx.stability_score, ctx.momentum_history, ctx.params
        )
        ctx.narrowed_to(passed)
        res.selected = list(ctx.eligible)
        res.output_universe = list(ctx.eligible)
        res.rejected = [t for t in res.input_universe if t not in set(ctx.eligible)]
        self._log(res, f"{len(ctx.eligible)} passed persistence")
        res.status = "completed"
        return res


# --------------------------------------------------------------------------- #
# Registry                                                                      #
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, type[GateNode]] = {}


def register(node_cls: type[GateNode]) -> type[GateNode]:
    """Register a gate node class under its ``kind`` string (idempotent)."""
    if not issubclass(node_cls, GateNode):
        raise TypeError(f"{node_cls} must subclass GateNode")
    _REGISTRY[node_cls.kind] = node_cls
    return node_cls


def get_gate(kind: str) -> GateNode:
    if kind not in _REGISTRY:
        raise KeyError(
            f"Unknown gate kind '{kind}'. Registered: {sorted(_REGISTRY)}. "
            f"Register a GateNode subclass via register()."
        )
    return _REGISTRY[kind]()


def list_gates() -> list[str]:
    return sorted(_REGISTRY)


def registry() -> dict[str, type[GateNode]]:
    return dict(_REGISTRY)


# Self-register the built-in gates (mirrors core/eligibility/registry.py idiom).
register(EligibilityGate)
register(MomentumGate)
register(StabilityGate)
register(QualityGate)
register(PersistenceGate)
