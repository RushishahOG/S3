"""Parameter Optimization orchestration engine.

This is the generic, central optimization service for ARQM. It implements the
full pipeline described in the specification:

Parameter Selection -> Search Space -> Constraint Validation -> Algorithm ->
Backtest Execution -> Performance Evaluation -> Store -> Rank -> Return Best.

The engine is **module-agnostic**: it receives a base
:class:`~core.config.backtest_schema.BacktestParameters`, a set of selected
parameter keys (discovered from :mod:`core.optimization.spec`), an objective,
constraints and an algorithm, and returns a ranked
:class:`~core.optimization.results.OptimizationRun`.

Every candidate is built into an *isolated* configuration (the user's base is
never mutated) and backtested through the standard ARQM engine. Algorithm
implementations plug in via :mod:`core.optimization.algorithms`; objectives via
:mod:`core.optimization.objectives`; constraints via
:mod:`core.optimization.constraints`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from core.backtesting.engine import BacktestResult, run_backtest
from core.config.backtest_schema import BacktestParameters
from core.data.storage.storage_manager import StorageManager
from core.optimization.algorithms import build_algorithm, register_fitness_fn
from core.optimization.candidate import build_candidate
from core.optimization.constraints import evaluate_user_constraints, validate_structural
from core.optimization.objectives import get_objective
from core.optimization.results import CandidateResult, OptimizationRun
from core.optimization.spec import specs_for_keys

# Serialise storage-heavy backtests so the engine does not contend on the
# single DuckDB file (mirrors the backtest worker's I/O lock).
_BACKTEST_IO_LOCK = threading.Lock()


def _extended_metrics(result: BacktestResult) -> dict[str, float]:
    """Augment engine metrics with final value and turnover."""
    metrics = dict(result.metrics)
    nav = result.nav
    if not nav.empty:
        metrics["final_portfolio_value"] = float(nav.iloc[-1])
    else:
        metrics["final_portfolio_value"] = float("nan")
    trades = result.trades
    if trades is not None and not trades.empty and "weight" in trades.columns:
        metrics["turnover"] = float(trades["weight"].abs().sum() / 2.0)
    else:
        metrics["turnover"] = 0.0
    return metrics


def run_optimization(
    base: BacktestParameters,
    selected_keys: list[str],
    objective_key: str,
    algorithm_key: str,
    constraints: dict[str, Any],
    max_iterations: int,
    random_seed: int,
    storage_factory: Callable[[], StorageManager],
    max_runtime_seconds: float | None = None,
    top_n: int = 20,
    progress_callback: Callable[[dict], None] | None = None,
) -> OptimizationRun:
    """Run a full optimization and return a ranked :class:`OptimizationRun`.

    Parameters
    ----------
    base:
        Base strategy configuration (never mutated).
    selected_keys:
        Parameter keys to optimize (must be known to :mod:`core.optimization.spec`).
    objective_key:
        One of :data:`core.optimization.objectives.OBJECTIVES`.
    algorithm_key:
        One of the algorithm keys in :mod:`core.optimization.algorithms`.
    constraints:
        Optional user constraints (see :mod:`core.optimization.constraints`).
    max_iterations, random_seed:
        Search budget / reproducibility controls.
    storage_factory:
        Callable returning a storage handle; called lazily (and the heavy read is
        serialised on the I/O lock) so the engine can run in a background thread.
    max_runtime_seconds:
        Wall-clock budget; the loop stops after this (best-effort).
    top_n:
        How many ranked candidates to retain in the run.
    progress_callback:
        Optional; receives ``{"event": ..., ...}`` events for live UI updates.
    """
    specs = specs_for_keys(selected_keys)
    objective = get_objective(objective_key)
    algorithm = build_algorithm(algorithm_key, specs, max_iterations, random_seed)

    run = OptimizationRun(
        algorithm=algorithm_key,
        objective=objective_key,
        parameters_optimized=list(selected_keys),
        constraints=dict(constraints),
        total_iterations=0,
        random_seed=random_seed,
        base_config=base.to_dict(),
    )

    start = time.time()
    evaluated: list[CandidateResult] = []

    # Provide the fitness function to adaptive algorithms (SLSQP) without an
    # import cycle: it calls back into the engine's evaluation closure.
    pending_fitness: dict[tuple, float] = {}

    def _evaluate(candidate: dict[str, Any]) -> CandidateResult:
        res = CandidateResult(params=dict(candidate))
        # Structural validation first -- reject before any backtest.
        try:
            cfg = build_candidate(base, candidate, specs)
        except Exception as exc:
            res.valid = False
            res.rejection_reason = f"candidate build failed: {exc}"
            return res
        struct_viol = validate_structural(cfg, candidate, specs)
        if struct_viol:
            res.valid = False
            res.rejection_reason = "; ".join(struct_viol)
            return res
        # Backtest (serialised I/O).
        t0 = time.time()
        try:
            with _BACKTEST_IO_LOCK:
                storage = storage_factory()
                result = run_backtest(cfg, storage)
        except Exception as exc:
            res.valid = False
            res.rejection_reason = f"backtest failed: {exc}"
            res.runtime_seconds = time.time() - t0
            return res
        res.runtime_seconds = time.time() - t0
        metrics = _extended_metrics(result)
        res.metrics = metrics
        # User constraints evaluated against metrics.
        user_viol = evaluate_user_constraints(metrics, constraints)
        if user_viol:
            res.valid = False
            res.rejection_reason = "; ".join(user_viol)
            res.objective_score = objective.score(metrics)
            return res
        res.valid = True
        res.objective_score = objective.score(metrics)
        return res

    register_fitness_fn(lambda cand: pending_fitness.get(_hashable(cand), None))

    try:
        for raw_candidate in algorithm.candidates():
            if max_runtime_seconds is not None and (time.time() - start) > max_runtime_seconds:
                if progress_callback:
                    progress_callback({"event": "timeout"})
                break
            result = _evaluate(raw_candidate)
            evaluated.append(result)
            run.total_iterations += 1
            pending_fitness[_hashable(raw_candidate)] = -result.objective_score if result.valid else float("inf")
            if progress_callback:
                progress_callback({
                    "event": "candidate_done",
                    "iteration": run.total_iterations,
                    "valid": result.valid,
                    "score": result.objective_score,
                    "metrics": result.metrics,
                })
    finally:
        register_fitness_fn(lambda cand: None)

    # Rank valid candidates by objective score (descending).
    valid = [r for r in evaluated if r.valid]
    valid.sort(key=lambda r: r.objective_score, reverse=True)
    for i, r in enumerate(valid, start=1):
        r.rank = i
    run.results = valid[:top_n]
    run.runtime_seconds = time.time() - start

    if run.results:
        run.best_params = dict(run.results[0].params)
    if progress_callback:
        progress_callback({
            "event": "done",
            "run_id": run.run_id,
            "best_score": run.results[0].objective_score if run.results else None,
            "n_valid": len(valid),
            "n_total": len(evaluated),
            "runtime": run.runtime_seconds,
        })
    return run


def _hashable(cand: dict[str, Any]) -> tuple:
    return tuple(sorted((k, str(v)) for k, v in cand.items()))
