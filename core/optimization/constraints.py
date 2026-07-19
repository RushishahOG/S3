"""Constraint validation engine for candidate configurations.

Before a candidate is backtested it is validated against two layers:

1. **Structural constraints** -- internal consistency required by the ARQM
   schema and the discovered parameter bounds (e.g. cap weights sum to 100%,
   scoring weights sum to 100%, quality pillars sum to 100%, top-N within
   bounds). These are derived automatically from the optimizer specs so the
   engine never hardcodes module logic.
2. **User constraints** -- optional thresholds supplied through the UI
   (maximum drawdown, minimum CAGR, cap allocation bounds, etc.). These map a
   constraint key to a *metric* it constrains and are evaluated after a backtest
   produces metrics, or pre-checked where possible.

A candidate that violates any structural constraint is rejected immediately
without running a backtest.
"""

from __future__ import annotations

from typing import Any

from core.config.backtest_schema import BacktestParameters
from core.optimization.spec import get_specs, OptimizerParamSpec

# Constraint keys -> metric name used for post-backtest evaluation.
_USER_CONSTRAINT_METRICS: dict[str, str] = {
    "max_drawdown_pct": "max_drawdown",
    "min_cagr_pct": "cagr",
    "max_volatility_pct": "annual_volatility",
    "min_sharpe": "sharpe",
    "min_information_ratio": "information_ratio",
}


def _near(val: float, target: float, tol: float = 1e-6) -> bool:
    return abs(val - target) <= tol


def validate_structural(
    params: BacktestParameters,
    values: dict[str, Any],
    specs: list[OptimizerParamSpec] | None = None,
) -> list[str]:
    """Validate structural / schema-level constraints. Returns violation strings."""
    violations: list[str] = []
    specs = specs or get_specs()
    spec_keys = {s.key for s in specs}

    # Bounds: only for parameters that were actually part of the candidate.
    for spec in specs:
        if spec.key not in values:
            continue
        v = values[spec.key]
        if spec.kind.value in ("continuous", "discrete"):
            if spec.min is not None and v < spec.min - 1e-9:
                violations.append(f"{spec.name} ({v}) below minimum {spec.min}.")
            if spec.max is not None and v > spec.max + 1e-9:
                violations.append(f"{spec.name} ({v}) above maximum {spec.max}.")
        if spec.kind.value == "categorical" and spec.allowed and v not in spec.allowed:
            violations.append(f"{spec.name} ({v}) not in allowed values {spec.allowed}.")

    # Sum-groups: check the resulting config blocks sum correctly (defensive).
    cap = params.cap_segment
    if any(k in values for k in ("large_cap_weight", "mid_cap_weight", "small_cap_weight")):
        cap_total = cap.large_cap_weight + cap.mid_cap_weight + cap.small_cap_weight
        if not _near(cap_total, 1.0):
            violations.append(f"Cap allocation sums to {cap_total:.4f} (must be 1.0).")

    sc = params.scoring
    if any(k in values for k in ("w_momentum", "w_stability")):
        sc_total = sc.momentum_weight + sc.quality_weight + sc.stability_weight
        if not _near(sc_total, 1.0):
            violations.append(f"Scoring weights sum to {sc_total:.4f} (must be 1.0).")

    qp = params.quality.pillar_weights
    if any(k in values for k in ("w_profitability", "w_growth", "w_fin_strength", "w_cashflow", "w_shareholder")):
        q_total = sum(qp.values())
        if not _near(q_total, 1.0):
            violations.append(f"Quality pillar weights sum to {q_total:.4f} (must be 1.0).")

    return violations


def evaluate_user_constraints(
    metrics: dict[str, float],
    constraints: dict[str, Any],
) -> list[str]:
    """Evaluate user constraints against backtest metrics. Returns violations."""
    violations: list[str] = []
    for key, threshold in constraints.items():
        metric = _USER_CONSTRAINT_METRICS.get(key)
        if metric is None:
            continue
        val = metrics.get(metric)
        if val is None or val != val:  # NaN
            continue
        # Convert percentage-based user thresholds. Constraints store pct as 0-100.
        if key.endswith("_pct"):
            val = val * 100.0
        if key.startswith("max") or key.startswith("min_drawdown"):
            # "max_*"/"min_drawdown" -> candidate must be <= threshold
            if val > threshold + 1e-9:
                violations.append(f"{key} = {val:.2f} exceeds limit {threshold:.2f}.")
        elif key.startswith("min"):
            if val < threshold - 1e-9:
                violations.append(f"{key} = {val:.2f} below minimum {threshold:.2f}.")
    return violations
