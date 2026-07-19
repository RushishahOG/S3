"""Optimization algorithms for the Parameter Optimization engine.

Each algorithm is a *candidate generator*: given the selected
:class:`~core.optimization.spec.OptimizerParamSpec` list and a budget (max
iterations / time), it yields candidate value-dicts ``{param_key: value}`` one
at a time. The orchestrating engine (see :mod:`core.optimization.engine`)
consumes these, builds the config, runs the backtest and feeds fitness back to
algorithms that are adaptive (SLSQP).

The strategy pattern keeps algorithms interchangeable and additive: a new
algorithm is a single class implementing :class:`OptimizationAlgorithm` plus one
registry entry -- no engine changes.
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterator

from core.optimization.spec import OptimizerParamSpec, ParamKind


def _grid_points(spec: OptimizerParamSpec) -> list[Any]:
    if spec.kind == ParamKind.CATEGORICAL:
        return list(spec.allowed or [])
    if spec.kind == ParamKind.BOOLEAN:
        return [True, False]
    if spec.kind == ParamKind.DISCRETE and spec.allowed:
        return list(spec.allowed)
    # Continuous / stepped numeric range.
    lo, hi, step = spec.min, spec.max, spec.step
    if lo is None or hi is None or step is None or step <= 0:
        return [spec.current]
    pts: list[Any] = []
    n = int(round((hi - lo) / step))
    for i in range(n + 1):
        v = lo + i * step
        val = round(v, 6)
        pts.append(int(val) if spec.kind == ParamKind.DISCRETE else val)
    return pts


class OptimizationAlgorithm(ABC):
    """Base interface for candidate generators."""

    key: str = "base"
    label: str = "Base"

    def __init__(self, specs: list[OptimizerParamSpec], max_iterations: int, seed: int = 42):
        self.specs = specs
        self.max_iterations = max(max_iterations, 1)
        self.seed = seed

    @abstractmethod
    def candidates(self) -> Iterator[dict[str, Any]]:
        """Yield candidate value-dicts."""
        raise NotImplementedError

    # --- adaptive feedback (optional) ---------------------------------------
    def update(self, candidate: dict[str, Any], fitness: float) -> None:
        """Called by the engine after evaluating a candidate (for adaptive algos)."""
        return None


class GridSearch(OptimizationAlgorithm):
    key = "grid_search"
    label = "Grid Search"

    def candidates(self) -> Iterator[dict[str, Any]]:
        axes = [_grid_points(s) for s in self.specs]
        if not axes:
            return
        product = list(itertools.product(*axes))
        if len(product) > self.max_iterations:
            step_idx = max(1, len(product) // self.max_iterations)
            product = product[::step_idx][: self.max_iterations]
        for combo in product[: self.max_iterations]:
            yield {s.key: v for s, v in zip(self.specs, combo)}


class RandomSearch(OptimizationAlgorithm):
    key = "random_search"
    label = "Random Search"

    def candidates(self) -> Iterator[dict[str, Any]]:
        rng = random.Random(self.seed)
        for _ in range(self.max_iterations):
            cand: dict[str, Any] = {}
            for s in self.specs:
                if s.kind == ParamKind.CATEGORICAL:
                    cand[s.key] = rng.choice(list(s.allowed or [s.current]))
                elif s.kind == ParamKind.BOOLEAN:
                    cand[s.key] = rng.choice([True, False])
                elif s.kind == ParamKind.DISCRETE and s.allowed:
                    cand[s.key] = rng.choice(list(s.allowed))
                else:
                    lo, hi, step = s.min, s.max, s.step
                    if lo is None or hi is None:
                        cand[s.key] = s.current
                        continue
                    if step and step > 0:
                        n = int(round((hi - lo) / step))
                        cand[s.key] = lo + rng.randint(0, n) * step
                    else:
                        cand[s.key] = lo + rng.random() * (hi - lo)
                    if s.kind == ParamKind.DISCRETE:
                        cand[s.key] = int(round(cand[s.key]))
                    else:
                        cand[s.key] = round(cand[s.key], 6)
            yield cand


class SLSQPOptimizer(OptimizationAlgorithm):
    """Sequential Least Squares Programming over continuous/stepped parameters.

    Discrete / categorical parameters are held fixed (midpoint of allowed set,
    or current value) for each multi-start, and the continuous subset is
    optimised with SciPy's SLSQP. The engine feeds fitness back via
    :meth:`update`. This is a pragmatic, generic SLSQP implementation; the
    architecture leaves room for full mixed-integer handling later.
    """

    key = "slsqp"
    label = "SLSQP"

    def __init__(self, specs: list[OptimizerParamSpec], max_iterations: int, seed: int = 42):
        super().__init__(specs, max_iterations, seed)
        self._continuous = [s for s in specs
                            if s.kind == ParamKind.CONTINUOUS
                            or (s.kind == ParamKind.DISCRETE and not s.allowed and s.step)]
        self._others = [s for s in specs if s not in self._continuous]
        self._best: dict[str, Any] | None = None
        self._best_fit = float("inf")

    def candidates(self) -> Iterator[dict[str, Any]]:
        from scipy.optimize import minimize

        rng = random.Random(self.seed)
        n_starts = max(1, min(8, self.max_iterations))
        evals_per_start = max(1, self.max_iterations // n_starts)

        for start in range(n_starts):
            fixed: dict[str, Any] = {}
            for s in self._others:
                if s.kind == ParamKind.CATEGORICAL and s.allowed:
                    fixed[s.key] = s.allowed[start % len(s.allowed)]
                elif s.kind == ParamKind.DISCRETE and s.allowed:
                    fixed[s.key] = s.allowed[len(s.allowed) // 2]
                elif s.kind == ParamKind.BOOLEAN:
                    fixed[s.key] = bool(start % 2)
                else:
                    fixed[s.key] = s.current

            if not self._continuous:
                yield fixed
                return

            bounds = [(s.min, s.max) for s in self._continuous]

            def _obj(x):
                cand = dict(fixed)
                for s, xi in zip(self._continuous, x):
                    cand[s.key] = int(round(xi)) if s.kind == ParamKind.DISCRETE else xi
                fit = _fitness_fn(cand)
                return fit if fit is not None else float("inf")

            x0 = [rng.uniform(b[0], b[1]) if (b[0] is not None and b[1] is not None) else s.current
                  for s, b in zip(self._continuous, bounds)]
            try:
                res = minimize(_obj, x0, method="SLSQP", bounds=bounds,
                               options={"maxiter": evals_per_start, "ftol": 1e-6})
                x = res.x
            except Exception:
                x = x0
            cand = dict(fixed)
            for s, xi in zip(self._continuous, x):
                cand[s.key] = int(round(xi)) if s.kind == ParamKind.DISCRETE else round(float(xi), 6)
            yield cand

    def update(self, candidate: dict[str, Any], fitness: float) -> None:
        if fitness < self._best_fit:
            self._best_fit = fitness
            self._best = dict(candidate)


# Injected by the engine so SLSQP can request fitness without importing the
# orchestrator (avoids an import cycle and keeps algorithms decoupled).
_fitness_fn: Callable[[dict[str, Any]], float | None] | None = None


def register_fitness_fn(fn: Callable[[dict[str, Any]], float | None]) -> None:
    global _fitness_fn
    _fitness_fn = fn


ALGORITHMS: dict[str, type[OptimizationAlgorithm]] = {
    GridSearch.key: GridSearch,
    RandomSearch.key: RandomSearch,
    SLSQPOptimizer.key: SLSQPOptimizer,
}

ALGORITHM_LABELS = {
    "grid_search": "Grid Search",
    "random_search": "Random Search",
    "slsqp": "SLSQP",
}
DEFAULT_ALGORITHM = "random_search"


def build_algorithm(key: str, specs: list[OptimizerParamSpec], max_iterations: int, seed: int) -> OptimizationAlgorithm:
    cls = ALGORITHMS.get(key, RandomSearch)
    return cls(specs, max_iterations, seed)
