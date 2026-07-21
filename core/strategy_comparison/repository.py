"""Strategy repository and comparison engine.

This module provides a centralized repository for all completed strategies
(manual backtests, optimization results, Monte Carlo simulations) and
functions to compare them across configuration, performance, risk, and holdings.

All operations are read-only; no backtests are re-run.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import os
from typing import Any

import numpy as np
import pandas as pd

from app.pages.backtest.state import get_backtest_state, StrategyStatus


class StrategySource(str, Enum):
    MANUAL = "Manual Backtest"
    OPTIMIZATION = "Parameter Optimizer"
    MONTE_CARLO = "Monte Carlo Simulation"


@dataclass
class StrategyRecord:
    """Complete record of a strategy for comparison."""
    strategy_id: str
    name: str
    source: StrategySource
    created_at: datetime
    config: dict[str, Any]  # Full configuration as nested dict
    metrics: dict[str, float]
    equity: pd.Series  # NAV over time
    benchmark: pd.Series
    trades: pd.DataFrame
    regime: pd.DataFrame
    snapshots: dict  # holdings history
    factor_scores: dict
    monte_carlo_result: Any | None = None
    simulation_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
        }

    @property
    def cagr(self) -> float:
        return self.metrics.get("annual_return", 0.0)

    @property
    def sharpe(self) -> float:
        return self.metrics.get("sharpe", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self.metrics.get("max_drawdown", 0.0)

    @property
    def volatility(self) -> float:
        return self.metrics.get("annual_volatility", 0.0)

    @property
    def calmar(self) -> float:
        return self.metrics.get("calmar", 0.0)

    @property
    def sortino(self) -> float:
        return self.metrics.get("sortino", 0.0)

    @property
    def total_return(self) -> float:
        return self.metrics.get("total_return", 0.0)

    @property
    def win_rate(self) -> float:
        return self.metrics.get("win_rate", 0.0)

    @property
    def profit_factor(self) -> float:
        return self.metrics.get("profit_factor", 0.0)

    @property
    def turnover(self) -> float:
        return self.metrics.get("turnover", 0.0)

    @property
    def n_trades(self) -> int:
        return int(self.metrics.get("n_trades", 0))


class StrategyRepository:
    """Centralized repository for all strategy records."""

    _instance: StrategyRepository | None = None
    _lock = None

    def __new__(cls):
        import threading
        if cls._lock is None:
            cls._lock = threading.Lock()
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._cache: dict[str, StrategyRecord] = {}
        self._load_from_backtest_state()

    def _load_from_backtest_state(self) -> None:
        state = get_backtest_state()
        completed = state.get_completed()
        for exec_obj in completed:
            if exec_obj.status == StrategyStatus.COMPLETED and exec_obj.result:
                result = exec_obj.result
                params = exec_obj.config.params
                record = StrategyRecord(
                    strategy_id=exec_obj.strategy_id,
                    name=exec_obj.config.name,
                    source=StrategySource.MANUAL,
                    created_at=exec_obj.config.created_at,
                    config=self._extract_config(params),
                    metrics=dict(result.metrics) if result.metrics else {},
                    equity=result.nav if result.nav is not None else pd.Series(dtype=float),
                    benchmark=result.benchmark_nav if result.benchmark_nav is not None else pd.Series(dtype=float),
                    trades=result.trades if result.trades is not None else pd.DataFrame(),
                    regime=result.regime if result.regime is not None else pd.DataFrame(),
                    snapshots=result.snapshots if result.snapshots else {},
                    factor_scores=result.factor_scores if result.factor_scores else {},
                )
                self._cache[exec_obj.strategy_id] = record
        self._load_optimization_runs()

    def _load_optimization_runs(self) -> None:
        """Auto-ingest saved Parameter Optimizer runs as comparison candidates."""
        try:
            from core.utils.paths import PROJECT_ROOT
            from core.optimization.results import OptimizationRun
            from core.config.backtest_schema import BacktestParameters
            from core.optimization.spec import specs_for_keys
            from core.optimization.candidate import build_candidate
        except Exception:
            return
        out_dir = os.path.join(PROJECT_ROOT, "storage", "optimization_runs")
        if not os.path.isdir(out_dir):
            return
        for fn in sorted(os.listdir(out_dir)):
            if not fn.endswith(".json"):
                continue
            try:
                run = OptimizationRun.load(os.path.join(out_dir, fn))
            except Exception:
                continue
            if not run.base_config:
                continue
            try:
                base = BacktestParameters.from_dict(run.base_config)
            except Exception:
                continue
            for cand in run.results:
                if not getattr(cand, "valid", True):
                    continue
                sid = f"opt_{run.run_id}_{cand.candidate_id}"
                if sid in self._cache:
                    continue
                try:
                    specs = specs_for_keys(list(cand.params.keys()))
                    params = build_candidate(base, cand.params, specs)
                    cfg = self._extract_config(params)
                except Exception:
                    cfg = dict(cand.params)
                self._cache[sid] = StrategyRecord(
                    strategy_id=sid,
                    name=f"{run.algorithm}_{cand.candidate_id}",
                    source=StrategySource.OPTIMIZATION,
                    created_at=datetime.fromisoformat(run.timestamp) if run.timestamp else datetime.now(),
                    config=cfg,
                    metrics=dict(cand.metrics),
                    equity=pd.Series(dtype=float),
                    benchmark=pd.Series(dtype=float),
                    trades=pd.DataFrame(),
                    regime=pd.DataFrame(),
                    snapshots={},
                    factor_scores={},
                    simulation_id=run.run_id,
                )

    def add_monte_carlo_result(
        self, base_result, sim_result, strategy_id: str, name: str
    ) -> str:
        """Register a Monte Carlo simulation's base strategy for comparison."""
        sid = f"{strategy_id}_mc"
        record = StrategyRecord(
            strategy_id=sid,
            name=f"{name}_MC",
            source=StrategySource.MONTE_CARLO,
            created_at=datetime.now(),
            config=self._extract_config(getattr(base_result, "params", None)),
            metrics=dict(base_result.metrics) if base_result.metrics else {},
            equity=base_result.nav if base_result.nav is not None else pd.Series(dtype=float),
            benchmark=base_result.benchmark_nav if base_result.benchmark_nav is not None else pd.Series(dtype=float),
            trades=base_result.trades if base_result.trades is not None else pd.DataFrame(),
            regime=base_result.regime if base_result.regime is not None else pd.DataFrame(),
            snapshots=base_result.snapshots if base_result.snapshots else {},
            factor_scores=base_result.factor_scores if base_result.factor_scores else {},
            monte_carlo_result=sim_result,
            simulation_id=getattr(sim_result, "seed", None),
        )
        self._cache[sid] = record
        return sid

    @staticmethod
    def _safe_sub(obj, name):
        return getattr(obj, name, None)

    def _extract_config(self, params) -> dict[str, Any]:
        if params is None:
            return {}
        g = self._safe_sub(params, "general")
        cs = self._safe_sub(params, "cap_segment")
        pf = self._safe_sub(params, "portfolio")
        mo = self._safe_sub(params, "momentum")
        st = self._safe_sub(params, "stability")
        ql = self._safe_sub(params, "quality")
        rk = self._safe_sub(params, "risk")
        rg = self._safe_sub(params, "regime")
        sc = self._safe_sub(params, "scoring")
        mg = self._safe_sub(params, "management")
        pl = self._safe_sub(params, "pipeline")
        return {
            "general": {
                "initial_capital": getattr(g, "initial_capital", 0),
                "rebalance_frequency": getattr(g, "rebalance_frequency", ""),
                "start_date": getattr(g, "start_date", ""),
                "end_date": getattr(g, "end_date", ""),
                "benchmark": getattr(g, "benchmark", ""),
                "transaction_cost_pct": getattr(g, "transaction_cost_pct", 0),
                "slippage_pct": getattr(g, "slippage_pct", 0),
            },
            "cap_segment": {
                "enabled": getattr(cs, "enabled", True),
                "large_cap_weight": getattr(cs, "large_cap_weight", 0.6),
                "mid_cap_weight": getattr(cs, "mid_cap_weight", 0.3),
                "small_cap_weight": getattr(cs, "small_cap_weight", 0.1),
            },
            "portfolio": {
                "total_size": getattr(pf, "total_size", 50),
                "large_size": getattr(pf, "large_size", 25),
                "mid_size": getattr(pf, "mid_size", 15),
                "small_size": getattr(pf, "small_size", 10),
                "sizing_method": getattr(pf, "sizing_method", "hybrid"),
                "max_position_pct": getattr(pf, "max_position_pct", 0.07),
            },
            "momentum": {
                "selection_mode": getattr(mo, "selection_mode", "top_pct"),
                "top_pct": getattr(mo, "top_pct", 0.3),
                "top_n": getattr(mo, "top_n", 50),
                "combine_method": getattr(mo, "combine_method", "weighted_score"),
                "normalization": getattr(mo, "normalization", "zscore"),
            },
            "stability": {
                "selection_mode": getattr(st, "selection_mode", "top_pct"),
                "top_pct": getattr(st, "top_pct", 0.5),
                "top_n": getattr(st, "top_n", 50),
                "combine_method": getattr(st, "combine_method", "weighted_score"),
                "normalization": getattr(st, "normalization", "zscore"),
            },
            "quality": {
                "normalization": getattr(ql, "normalization", "zscore"),
                "use_rollup": getattr(ql, "use_rollup", "median"),
                "min_quality_score": getattr(ql, "min_quality_score", 0.0),
                "pillar_weights": getattr(ql, "pillar_weights", {}),
            },
            "risk": {
                "min_vol_target": getattr(rk, "min_vol_target", 0),
                "max_dd_target": getattr(rk, "max_dd_target", 0),
            },
            "regime": {
                "reference": getattr(rg, "reference", "benchmark"),
                "buy_trigger_pct": getattr(rg, "buy_trigger_pct", 5.0),
                "sell_trigger_pct": getattr(rg, "sell_trigger_pct", -15.0),
                "enable_swing_low": getattr(rg, "enable_swing_low", True),
                "enable_peak_detection": getattr(rg, "enable_peak_detection", True),
            },
            "scoring": {
                "momentum_weight": getattr(sc, "momentum_weight", 0.4),
                "quality_weight": getattr(sc, "quality_weight", 0.4),
                "stability_weight": getattr(sc, "stability_weight", 0.2),
            },
            "management": {
                "quality_drop_quantile": getattr(mg, "quality_drop_quantile", 0.2),
                "momentum_drop_quantile": getattr(mg, "momentum_drop_quantile", 0.2),
                "stability_drop_quantile": getattr(mg, "stability_drop_quantile", 0.2),
                "exit_on_sell_signal": getattr(mg, "exit_on_sell_signal", True),
            },
            "pipeline": {
                "gates": [{"kind": g.kind, "enabled": g.enabled, "order": g.order, "config_key": g.config_key} for g in getattr(pl, "gates", [])],
                "final_scoring": getattr(pl, "final_scoring", True),
            },
        }

    def list_all(self) -> list[StrategyRecord]:
        # Refresh cache from backtest state to pick up newly completed strategies
        self._load_from_backtest_state()
        return list(self._cache.values())

    def get(self, strategy_id: str) -> StrategyRecord | None:
        return self._cache.get(strategy_id)

    def clear(self) -> None:
        self._cache.clear()


def get_strategy_repository() -> StrategyRepository:
    return StrategyRepository()


__all__ = [
    "StrategySource",
    "StrategyRecord",
    "StrategyRepository",
    "get_strategy_repository",
]