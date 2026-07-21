"""Efficient Frontier Portfolio Optimization Module.

This module provides institutional-quality portfolio optimization using
Modern Portfolio Theory (Markowitz) with practical constraints for
factor investing.
"""

from core.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    EfficientFrontier,
    PortfolioConstraints,
    SectorConstraints,
    MarketCapConstraints,
    CashConstraints,
    TurnoverConstraints,
    LiquidityConstraints,
)

from core.portfolio.risk_models import (
    expected_returns,
    covariance_matrix,
    risk_metrics,
    diversification_metrics,
    correlation_analysis,
)

from core.portfolio.visualization import (
    plot_efficient_frontier,
    plot_allocation_pie,
    plot_allocation_treemap,
    plot_risk_contribution,
    plot_correlation_heatmap,
    plot_capital_market_line,
)

__all__ = [
    "PortfolioOptimizer",
    "OptimizationResult",
    "EfficientFrontier",
    "PortfolioConstraints",
    "ExpectedReturnMethod",
    "CovarianceMethod",
    "OptimizationObjective",
    "OptimizationSolver",
    "UniverseSelection",
    "SectorConstraints",
    "MarketCapConstraints",
    "CashConstraints",
    "TurnoverConstraints",
    "LiquidityConstraints",
    "expected_returns",
    "covariance_matrix",
    "risk_metrics",
    "diversification_metrics",
    "correlation_analysis",
    "plot_efficient_frontier",
    "plot_allocation_pie",
    "plot_allocation_treemap",
    "plot_risk_contribution",
    "plot_correlation_heatmap",
    "plot_capital_market_line",
]