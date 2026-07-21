"""Strategy comparison package for ARQM Research Lab."""

from core.strategy_comparison.repository import (
    StrategySource,
    StrategyRecord,
    StrategyRepository,
    get_strategy_repository,
)
from core.strategy_comparison.comparison import (
    ComparisonResult,
    compare_strategies,
)
from core.strategy_comparison import visualization as viz
from core.strategy_comparison import export as export

__all__ = [
    "StrategySource",
    "StrategyRecord",
    "StrategyRepository",
    "get_strategy_repository",
    "ComparisonResult",
    "compare_strategies",
    "viz",
    "export",
]