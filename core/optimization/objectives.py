"""Objective functions for the Parameter Optimization engine.

An objective maps a backtest result's metric bundle to a single scalar *score*
used for ranking. Each objective declares a direction (maximize / minimize) and
the metric it reads.

The engine computes the metric bundle in :mod:`core.optimization.engine`; the
objective layer is intentionally a thin, declarative mapping so new objectives
are pure data additions. Default objective is **Maximize Sharpe Ratio**.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable


class Direction(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True)
class Objective:
    key: str
    label: str
    metric: str
    direction: Direction
    #: Higher is always better when score() is applied (handles minimization).
    help: str = ""

    def score(self, metrics: dict[str, float]) -> float:
        """Return a rankable score (higher is better). NaN -> -inf for maximize."""
        val = metrics.get(self.metric)
        if val is None or (isinstance(val, float) and val != val):  # NaN
            return float("-inf")
        return -val if self.direction == Direction.MINIMIZE else val


# Metric name aliases used by the engine's result bundle.
_M_FINAL_VALUE = "final_portfolio_value"
_M_TURNOVER = "turnover"


OBJECTIVES: dict[str, Objective] = {
    "max_cagr": Objective("max_cagr", "Maximize CAGR", "annual_return", Direction.MAXIMIZE,
                          help="Compound annual growth rate."),
    "max_sharpe": Objective("max_sharpe", "Maximize Sharpe Ratio", "sharpe", Direction.MAXIMIZE,
                            help="Risk-adjusted return (default)."),
    "max_sortino": Objective("max_sortino", "Maximize Sortino Ratio", "sortino", Direction.MAXIMIZE,
                             help="Downside-risk-adjusted return."),
    "max_calmar": Objective("max_calmar", "Maximize Calmar Ratio", "calmar", Direction.MAXIMIZE,
                            help="Annual return / max drawdown."),
    "max_information_ratio": Objective("max_information_ratio", "Maximize Information Ratio",
                                       "information_ratio", Direction.MAXIMIZE,
                                       help="Excess return per unit of tracking error."),
    "min_max_drawdown": Objective("min_max_drawdown", "Minimize Maximum Drawdown",
                                  "max_drawdown", Direction.MINIMIZE,
                                  help="Largest peak-to-trough loss (minimized)."),
    "min_volatility": Objective("min_volatility", "Minimize Portfolio Volatility",
                               "annual_volatility", Direction.MINIMIZE,
                               help="Annualized standard deviation of returns."),
    "max_final_value": Objective("max_final_value", "Maximize Final Portfolio Value",
                                 _M_FINAL_VALUE, Direction.MAXIMIZE,
                                 help="Ending portfolio value in currency."),
}

DEFAULT_OBJECTIVE = "max_sharpe"


def get_objective(key: str) -> Objective:
    return OBJECTIVES.get(key, OBJECTIVES[DEFAULT_OBJECTIVE])


def available_objectives() -> list[Objective]:
    return list(OBJECTIVES.values())
