"""Search-space estimation for the Parameter Optimization engine.

Pure, side-effect-free helpers that turn a set of selected optimizable
parameters plus an algorithm choice into *estimates* of: the size of the search
space, the number of backtests that will be required, and the expected runtime.

These are deliberately **estimates only** -- no backtests are executed. The
numbers are used by the UI's *Search Space Summary* to give the user a sense of
scale before they commit to a run. The math is intentionally simple and
transparent so it is easy to reason about and extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable

from core.optimization.param_registry import OptimizableParameter, ParamType


# Rough per-backtest cost assumption used purely for runtime estimation.
# Tunable; in practice this gets calibrated once real backtests are timed.
_SECONDS_PER_BACKTEST = 2.0


@dataclass(frozen=True)
class SearchSpaceEstimate:
    """Estimated characteristics of an optimization search space."""

    parameter_count: int
    search_space_size: int
    estimated_backtests: int
    estimated_seconds: float

    @property
    def estimated_runtime_label(self) -> str:
        secs = self.estimated_seconds
        if secs < 60:
            return f"{int(secs)}s"
        minutes = int(secs // 60)
        if minutes < 60:
            return f"{minutes}m {int(secs) % 60}s"
        hours = minutes // 60
        if hours < 1000:
            return f"{hours}h {minutes % 60}m"
        # Beyond ~40 days the estimate is only illustrative; show compact form.
        return f"{hours:,}h"


def _grid_steps_for_param(param: OptimizableParameter) -> int:
    """Number of discrete values a single parameter contributes to a grid.

    For numeric parameters this is derived from the (min, max, step) range,
    falling back to a sensible default when a bound is missing. Choice and
    boolean parameters contribute one value per option.
    """
    if param.param_type == ParamType.CHOICE:
        return max(1, len(param.choices or []))
    if param.param_type == ParamType.BOOL:
        return 2

    lo, hi, step = param.min_value, param.max_value, param.step
    if lo is None or hi is None:
        # No explicit bounds: treat as a modest 5-step sweep around the current value.
        return 5
    if step is None or step <= 0:
        step = 1
    return max(1, int(round((hi - lo) / step)) + 1)


def estimate_grid_search(parameters: Iterable[OptimizableParameter]) -> SearchSpaceEstimate:
    """Estimate a full Cartesian grid over the selected parameters."""
    params = list(parameters)
    steps = [_grid_steps_for_param(p) for p in params]
    size = prod(steps) if steps else 1
    backtests = size
    return SearchSpaceEstimate(
        parameter_count=len(params),
        search_space_size=size,
        estimated_backtests=backtests,
        estimated_seconds=backtests * _SECONDS_PER_BACKTEST,
    )


def estimate_random_search(
    parameters: Iterable[OptimizableParameter], max_iterations: int = 200
) -> SearchSpaceEstimate:
    """Estimate a random / sample-based search bounded by ``max_iterations``."""
    params = list(parameters)
    steps = [_grid_steps_for_param(p) for p in params]
    full_size = prod(steps) if steps else 1
    backtests = min(max_iterations, max(1, full_size))
    return SearchSpaceEstimate(
        parameter_count=len(params),
        search_space_size=min(full_size, max_iterations),
        estimated_backtests=backtests,
        estimated_seconds=backtests * _SECONDS_PER_BACKTEST,
    )


# Mapping from algorithm id -> estimator so the UI never branches on algorithm
# names inline. Add new algorithms here as their estimation logic is defined.
_ALGORITHM_ESTIMATORS = {
    "grid_search": estimate_grid_search,
    "random_search": estimate_random_search,
    "bayesian_optimization": estimate_random_search,
    "genetic_algorithm": estimate_random_search,
    "particle_swarm_optimization": estimate_random_search,
    "simulated_annealing": estimate_random_search,
}


def estimate_search_space(
    algorithm: str,
    parameters: Iterable[OptimizableParameter],
    max_iterations: int = 200,
) -> SearchSpaceEstimate:
    """Estimate the search space for a given algorithm and selected parameters."""
    estimator = _ALGORITHM_ESTIMATORS.get(algorithm, estimate_random_search)
    if algorithm == "random_search":
        return estimator(parameters, max_iterations=max_iterations)
    return estimator(parameters)
