"""Parameter Optimization framework (generic, algorithm-agnostic).

Public API for discovering optimizable ARQM parameters and estimating search
spaces. The optimization UI and (future) algorithms consume these helpers
rather than hardcoding any strategy module.
"""

from __future__ import annotations

from core.optimization.param_registry import (
    ValidationRules,
    OptimizableParameter,
    ParamType,
    discover_parameters,
    get_parameter_metadata,
    group_by_category,
)
from core.optimization.search_space import (
    SearchSpaceEstimate,
    estimate_search_space,
)

__all__ = [
    "ValidationRules",
    "OptimizableParameter",
    "ParamType",
    "discover_parameters",
    "get_parameter_metadata",
    "group_by_category",
    "SearchSpaceEstimate",
    "estimate_search_space",
]
