"""Result data structures and persistence for optimization runs.

Defines the :class:`CandidateResult` (one evaluated configuration) and
:class:`OptimizationRun` (a full run with metadata, ranked results, and a
persistence helper that serialises to JSON so runs survive a Streamlit rerun).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from core.config.backtest_schema import BacktestParameters


@dataclass
class CandidateResult:
    rank: int = 0
    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    objective_score: float = float("-inf")
    runtime_seconds: float = 0.0
    valid: bool = True
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "params": self.params,
            "metrics": self.metrics,
            "objective_score": self.objective_score,
            "runtime_seconds": self.runtime_seconds,
            "valid": self.valid,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class OptimizationRun:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    algorithm: str = ""
    objective: str = ""
    parameters_optimized: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    results: list[CandidateResult] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    total_iterations: int = 0
    runtime_seconds: float = 0.0
    random_seed: int = 42
    base_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "algorithm": self.algorithm,
            "objective": self.objective,
            "parameters_optimized": self.parameters_optimized,
            "constraints": self.constraints,
            "results": [r.to_dict() for r in self.results],
            "best_params": self.best_params,
            "total_iterations": self.total_iterations,
            "runtime_seconds": self.runtime_seconds,
            "random_seed": self.random_seed,
            "base_config": self.base_config,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "OptimizationRun":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        run = cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            algorithm=data.get("algorithm", ""),
            objective=data.get("objective", ""),
            parameters_optimized=data.get("parameters_optimized", []),
            constraints=data.get("constraints", {}),
            results=[CandidateResult(**r) for r in data.get("results", [])],
            best_params=data.get("best_params", {}),
            total_iterations=data.get("total_iterations", 0),
            runtime_seconds=data.get("runtime_seconds", 0.0),
            random_seed=data.get("random_seed", 42),
            base_config=data.get("base_config", {}),
        )
        return run

    def best_config(self) -> BacktestParameters | None:
        """Reconstruct the best configuration as BacktestParameters (if stored)."""
        if not self.best_params or not self.base_config:
            return None
        try:
            base = BacktestParameters.from_dict(self.base_config)
            from core.optimization.spec import specs_for_keys
            from core.optimization.candidate import build_candidate
            specs = specs_for_keys(list(self.best_params.keys()))
            return build_candidate(base, self.best_params, specs)
        except Exception:
            return None
