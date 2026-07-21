"""Core portfolio optimization engine for Efficient Frontier calculations."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize

from core.portfolio.risk_models import (
    expected_returns,
    covariance_matrix,
    risk_metrics,
    diversification_metrics,
    TRADING_DAYS,
)
from typing import Literal

ExpectedReturnMethod = Literal[
    "historical_cagr",
    "arithmetic_mean",
    "geometric_mean",
    "ema",
    "capm",
    "custom",
]

CovarianceMethod = Literal[
    "sample",
    "exponential",
    "ledoit_wolf",
    "oracle_approximating",
    "constant_correlation",
    "custom",
]

OptimizationObjective = Literal[
    "max_sharpe",
    "min_volatility",
    "max_return",
    "max_calmar",
    "max_sortino",
    "min_drawdown",
    "risk_parity",
    "equal_weight",
    "max_diversification",
    "min_correlation",
]

OptimizationSolver = Literal[
    "quadratic_programming",
    "slsqp",
    "differential_evolution",
    "particle_swarm",
    "genetic_algorithm",
    "simulated_annealing",
]

UniverseSelection = Literal[
    "current_portfolio",
    "manual_selection",
    "top_quality",
    "top_momentum",
    "final_portfolio",
    "entire_universe",
]

warnings.filterwarnings("ignore", category=RuntimeWarning)


class PortfolioConstraints:
    """Container for portfolio constraints."""

    def __init__(
        self,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_portfolio_size: int = 1,
        max_portfolio_size: int | None = None,
        total_weights: float = 1.0,
        risk_free_rate: float = 0.06,
        frequency: str = "daily",
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_portfolio_size = min_portfolio_size
        self.max_portfolio_size = max_portfolio_size
        self.total_weights = total_weights
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency

    def get_bounds(self, n_assets: int) -> list:
        """Get optimization bounds for assets."""
        return [(self.min_weight, self.max_weight) for _ in range(n_assets)]

    def get_constraints(self, n_assets: int) -> list:
        """Get optimization constraints."""
        constraints = []

        # Sum of weights equals 1 (or total_weights)
        constraints.append({
            "type": "eq",
            "fun": lambda x: np.sum(x) - self.total_weights,
        })

        # Each weight >= min_weight (if min_weight > 0)
        if self.min_weight > 0:
            for i in range(n_assets):
                constraints.append({
                    "type": "ineq",
                    "fun": lambda x, idx=i: x[idx] - self.min_weight,
                })

        # Portfolio size constraints
        if self.max_portfolio_size is not None and self.max_portfolio_size < n_assets:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: self.max_portfolio_size - np.sum(x > 0),
            })

        if self.min_portfolio_size > 0:
            constraints.append({
                "type": "ineq",
                "fun": lambda x: np.sum(x > 0) - self.min_portfolio_size,
            })

        return constraints


class SectorConstraints:
    """Sector exposure constraints."""

    def __init__(
        self,
        sectors: dict[str, str],
        min_sector_weight: float = 0.05,
        max_sector_weight: float = 0.5,
    ):
        self.sectors = sectors
        self.min_sector_weight = min_sector_weight
        self.max_sector_weight = max_sector_weight

    def get_sector_constraints(self, weights_df: pd.DataFrame) -> list:
        """Get sector weight constraints as optimization constraints."""
        if not self.sectors:
            return []

        constraints = []

        sector_weights = weights_df.copy()
        sector_weights.columns = sector_weights.columns.map(self.sectors)

        for sector in sector_weights.columns.unique():
            if sector == "null":
                continue

            sector_mask = pd.Series(self.sectors).map(lambda x: x == sector)

            sector_weight = (sector_weights * sector_mask).sum(axis=1)

            constraints.append({
                "type": "ineq",
                "fun": lambda x: sector_weight - self.min_sector_weight,
            })
            constraints.append({
                "type": "ineq",
                "fun": lambda x: self.max_sector_weight - sector_weight,
            })

        return constraints


class MarketCapConstraints:
    """Market capitalization-based constraints."""

    def __init__(
        self,
        market_caps: dict[str, float],
        min_large_weight: float = 0.7,
        min_mid_weight: float = 0.0,
        min_small_weight: float = 0.0,
    ):
        self.market_caps = market_caps
        self.min_large_weight = min_large_weight
        self.min_mid_weight = min_mid_weight
        self.min_small_weight = min_small_weight

        self._cap_tiers = self._assign_cap_tiers()

    def _assign_cap_tiers(self) -> dict[str, str]:
        """Assign stocks to market cap tiers."""
        if not self.market_caps:
            return {}

        caps = pd.Series(self.market_caps)
        rank = caps.rank(pct=True, method="first")

        tiers = {}
        for ticker, pct in zip(caps.index, rank):
            if pct >= 0.70:
                tiers[ticker] = "large"
            elif pct >= 0.40:
                tiers[ticker] = "mid"
            else:
                tiers[ticker] = "small"

        return tiers

    def get_cap_constraints(self, weights_df: pd.DataFrame) -> list:
        """Get market cap constraints."""
        constraints = []

        tier_weights = weights_df.copy()
        tier_weights.columns = tier_weights.columns.map(self._cap_tiers)

        for tier, min_weight in [("large", self.min_large_weight),
                                 ("mid", self.min_mid_weight),
                                 ("small", self.min_small_weight)]:
            if tier == "null":
                continue

            tier_mask = pd.Series(self._cap_tiers).map(lambda x: x == tier)
            tier_weight = (tier_weights * tier_mask).sum(axis=1)

            constraints.append({
                "type": "ineq",
                "fun": lambda x: tier_weight - min_weight,
            })

        return constraints


class CashConstraints:
    """Cash/liquidity constraints."""

    def __init__(
        self,
        min_cash: float = 0.0,
        max_cash: float = 0.5,
        cash_asset: str = "CASH",
    ):
        self.min_cash = min_cash
        self.max_cash = max_cash
        self.cash_asset = cash_asset

    def add_cash_constraints(self, constraints: list, weight_var_index: int) -> list:
        """Add cash constraints to optimization."""
        constraints.append({
            "type": "ineq",
            "fun": lambda x: x[weight_var_index] - self.min_cash,
        })
        constraints.append({
            "type": "ineq",
            "fun": lambda x: self.max_cash - x[weight_var_index],
        })
        return constraints


class TurnoverConstraints:
    """Turnover constraints."""

    def __init__(
        self,
        max_turnover: float = 0.5,
        lookback_period: int = 21,
    ):
        self.max_turnover = max_turnover
        self.lookback_period = lookback_period

    def get_turnover_constraint(self, previous_weights: pd.Series) -> list:
        """Get turnover constraints."""
        if previous_weights.empty:
            return []

        turnover = (previous_weights.abs() * 2 - 1).clip(lower=0).sum()

        return [{
            "type": "ineq",
            "fun": lambda x: self.max_turnover - turnover,
        }]


class LiquidityConstraints:
    """Liquidity constraints."""

    def __init__(
        self,
        min_liquidity: float = 0.0,
        liquidity_threshold: float = 0.1,
    ):
        self.min_liquidity = min_liquidity
        self.liquidity_threshold = liquidity_threshold

    def get_liquidity_constraint(self, liquidity_data: pd.Series) -> list:
        """Get liquidity constraints."""
        liquidity = liquidity_data * (liquidity_data >= self.liquidity_threshold)
        total_liquidity = liquidity.sum()

        return [{
            "type": "ineq",
            "fun": lambda x: self.min_liquidity - total_liquidity,
        }]


class OptimizationObjective:
    """Define optimization objectives."""

    def __init__(self, objective: str = "max_sharpe"):
        self.objective = objective
        self.n_assets = 0

    def objective_function(self, weights: np.ndarray, returns: np.ndarray, cov: np.ndarray) -> float:
        """Main optimization objective function."""
        if self.objective == "min_volatility":
            return np.sqrt(weights @ cov @ weights)
        elif self.objective == "max_sharpe":
            mean_return = weights @ returns
            vol = np.sqrt(weights @ cov @ weights)
            return -(mean_return - 0.06) / vol
        elif self.objective == "max_return":
            return -(weights @ returns)
        elif self.objective == "max_calmar":
            mean_return = weights @ returns
            port_vol = np.sqrt(weights @ cov @ weights)
            return -abs((mean_return - 0.06) / port_vol)
        elif self.objective == "max_sortino":
            mean_return = weights @ returns
            excess_returns = returns - 0.06 / len(returns)
            downside_risk = np.sqrt((excess_returns @ weights).T @ (excess_returns @ weights) / len(returns))
            return -abs((mean_return - 0.06) / downside_risk)
        elif self.objective == "min_drawdown":
            return -1.0  # Placeholder, will be evaluated differently
        elif self.objective == "risk_parity":
            return risk_parity_objective(weights, cov)
        elif self.objective == "equal_weight":
            return np.var(weights - np.ones(self.n_assets) / self.n_assets)
        elif self.objective == "max_diversification":
            return -diversification_ratio(weights, cov)
        elif self.objective == "min_correlation":
            return correlation_risk(weights, cov)
        else:
            return 0.0


def risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """Risk parity objective."""
    risk_contributions = (cov @ weights).T @ weights - weights * (cov @ weights).T @ weights
    target = 1.0 / len(weights)
    return np.sum((risk_contributions - target)**2)


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Calculate diversification ratio."""
    port_variance = weights @ cov @ weights
    weighted_var = np.sum(weights**2 * np.diag(cov))
    return np.sqrt(weighted_var / port_variance) if port_variance > 0 else 1.0


def correlation_risk(weights: np.ndarray, cov: np.ndarray) -> float:
    """Correlation-based risk objective."""
    mean_weight = np.mean(weights)
    corr = weights @ cov @ weights
    return -np.sqrt(mean_weight * (1 - mean_weight))


class EfficientFrontier:
    """Efficient frontier analysis."""

    def __init__(
        self,
        returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_free_rate: float = 0.06,
        constraints: PortfolioConstraints | None = None,
    ) -> None:
        """Initialize EfficientFrontier with returns, covariance, risk-free rate, and optional constraints."""
        self.expected_returns = returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(returns)
        # Store constraints, default to long-only if not provided
        self.constraints = constraints or PortfolioConstraints(min_weight=0.0, max_weight=1.0, total_weights=1.0, risk_free_rate=risk_free_rate)
        # Convert to arrays for optimization
        self.returns_array = returns.values
        self.cov_array = cov_matrix.values

    def generate_frontier(
        self,
        n_portfolios: int = 10000,
        objective: str = "max_sharpe",
    ) -> pd.DataFrame:
        """Generate efficient frontier portfolios."""
        weights = []
        returns = []
        volatilities = []
        sharpes = []
        sortinos = []
        calmars = []

        # Vectorized portfolio generation respecting constraints
        batch_size = max(1000, n_portfolios * 2)
        # Generate a large batch of random weights using Dirichlet distribution
        random_weights = np.random.dirichlet(np.ones(self.n_assets), size=batch_size)
        # Apply portfolio constraints bounds
        min_w, max_w = self.constraints.min_weight, self.constraints.max_weight
        mask = (random_weights >= min_w) & (random_weights <= max_w)
        valid_mask = mask.all(axis=1)
        filtered_weights = random_weights[valid_mask]
        # Ensure enough portfolios; generate more if needed
        while filtered_weights.shape[0] < n_portfolios:
            extra = np.random.dirichlet(np.ones(self.n_assets), size=batch_size)
            mask_extra = (extra >= min_w) & (extra <= max_w)
            extra_valid = extra[mask_extra.all(axis=1)]
            filtered_weights = np.vstack([filtered_weights, extra_valid])
        # Trim to requested count
        weights = filtered_weights[:n_portfolios]
        # Compute metrics for all portfolios
        port_returns = weights @ self.returns_array
        # Compute portfolio volatilities via einsum: w_i^T * cov * w_i
        port_vols = np.sqrt(np.einsum('ij,jk,ik->i', weights, self.cov_array, weights))
        sharpe = (port_returns - self.risk_free_rate) / np.where(port_vols > 0, port_vols, np.nan)
        # Additional metrics
        excess_returns = self.returns_array - self.risk_free_rate / TRADING_DAYS
        # Approximate Sortino using portfolio excess return series variance
        # Compute portfolio excess return for each portfolio (vectorized)
        portfolio_excess = excess_returns @ weights.T  # shape (periods, portfolios)
        downside_std = np.sqrt(np.mean(np.where(portfolio_excess < 0, portfolio_excess ** 2, 0), axis=0))
        sortino = np.where(downside_std > 0, (port_returns - self.risk_free_rate) / downside_std, np.nan)
        # Max drawdown approximated per portfolio via simulation
        max_dd = np.array([self._simulate_max_drawdown(w) for w in weights])
        calmar = (port_returns - self.risk_free_rate) / np.where(max_dd != 0, np.abs(max_dd), np.nan)
        # Build DataFrame
        result = pd.DataFrame({
            "weights": list(weights),
            "return": port_returns,
            "volatility": port_vols,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
        })
        # Sort according to selected objective
        if objective == "min_volatility":
            result = result.sort_values("volatility")
        elif objective == "max_sharpe":
            result = result.sort_values("sharpe", ascending=False)
        elif objective == "max_return":
            result = result.sort_values("return", ascending=False)
        elif objective == "max_calmar":
            result = result.sort_values("calmar", ascending=False)
        elif objective == "max_sortino":
            result = result.sort_values("sortino", ascending=False)
        elif objective == "min_drawdown":
            result = result.sort_values("max_drawdown", ascending=True)
        return result

    def _simulate_max_drawdown(self, weights: np.ndarray) -> float:
        """Simulate max drawdown for a portfolio."""
        # Simplified simulation - in practice would use actual return data
        # Simulate portfolio returns assuming a single return series
        # self.returns_array is a 1‑D array of expected returns per asset; we generate a synthetic series
        port_returns = np.random.normal(np.mean(self.returns_array), np.std(self.returns_array), 252)
        # Compute cumulative net asset value (NAV) of the portfolio
        port_nav = np.cumprod(1 + port_returns)


        running_max = np.maximum.accumulate(port_nav)
        drawdown = (port_nav - running_max) / running_max
        return abs(drawdown.min())

    def get_efficient_frontier_points(
        self,
        n_points: int = 100,
        objective: str = "max_sharpe",
    ) -> pd.DataFrame:
        """Get frontier optimization points."""
        front = self.generate_frontier(n_portfolios=1000, objective=objective)
        return front.head(n_points)


class PortfolioOptimizer:
    """Portfolio optimization engine."""

    def __init__(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        weights_dataframe: pd.DataFrame,
        risk_free_rate: float = 0.06,
    ):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.weights_df = weights_dataframe
        self.risk_free_rate = risk_free_rate

        # Asset information from weights
        self.tickers = weights_dataframe.columns.tolist()
        self.n_assets = len(self.tickers)

        # Add assets with zero weights to returns and cov matrix
        full_assets = list(expected_returns.index)
        self.full_returns = expected_returns.reindex(full_assets)
        self.full_cov = cov_matrix.reindex(full_assets, columns=full_assets)

    def optimize(
        self,
        objective: str = "max_sharpe",
        risk_free_rate: float | None = None,
        constraints: PortfolioConstraints | None = None,
        sector_constraints: SectorConstraints | None = None,
        market_cap_constraints: MarketCapConstraints | None = None,
        cash_constraints: CashConstraints | None = None,
        turnover_constraints: TurnoverConstraints | None = None,
        liquidity_constraints: LiquidityConstraints | None = None,
        previous_weights: pd.Series | None = None,
        optimization_method: str = "quadratic_programming",
        max_iterations: int = 10000,
        solver_params: dict | None = None,
    ) -> "OptimizationResult":
        """Run portfolio optimization."""
        if risk_free_rate is None:
            risk_free_rate = self.risk_free_rate

        # Handle constraints
        opt_constraints = constraints or PortfolioConstraints(risk_free_rate=risk_free_rate)
        sector_cons = sector_constraints
        market_cons = market_cap_constraints
        cash_cons = cash_constraints
        turnover_cons = turnover_constraints
        liquidity_cons = liquidity_constraints

        # Prepare data
        returns_array = self.full_returns.values
        cov_array = self.full_cov.values
        tickers = self.full_returns.index.tolist()

        n_assets = len(tickers)
        opt_obj = OptimizationObjective(objective)
        opt_obj.n_assets = n_assets

        # Setup optimization
        bounds = opt_constraints.get_bounds(n_assets)

        # Define objective function with weights
        def objective_func(w):
            return opt_obj.objective_function(w, returns_array, cov_array)

        # Constraints
        cons = []
        other_cons = opt_constraints.get_constraints(n_assets)
        cons.extend(other_cons)

        # Handle special constraints
        if sector_cons:
            sector_cons_list = sector_cons.get_sector_constraints(self.weights_df)
            cons.extend(sector_cons_list)

        if market_cons:
            market_cons_list = market_cons.get_cap_constraints(self.weights_df)
            cons.extend(market_cons_list)

        if previous_weights is not None and turnover_cons:
            turnover_cons_list = turnover_cons.get_turnover_constraint(previous_weights)
            cons.extend(turnover_cons_list)

        # Additional constraints for specific objectives
        if objective == "equal_weight":
            cons.append({
                "type": "eq",
                "fun": lambda x: np.sum(x) - opt_constraints.total_weights,
            })
            cons.append({
                "type": "eq",
                "fun": lambda x: np.mean(x) - 1 / n_assets,
            })

        if objective == "risk_parity":
            risk_parity_con = {
                "type": "ineq",
                "fun": lambda x: np.sqrt((x @ cov_array).T @ x) - np.sqrt((cov_array @ x).T @ x),
            }
            cons.append(risk_parity_con)

        # Run optimization
        if optimization_method == "quadratic_programming":
            result = self._quadratic_programming(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers, risk_free_rate
            )
        elif optimization_method == "slsqp":
            result = self._slsqp_optimization(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers
            )
        elif optimization_method == "differential_evolution":
            result = self._differential_evolution(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers
            )
        elif optimization_method == "particle_swarm":
            result = self._particle_swarm(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers
            )
        elif optimization_method == "genetic_algorithm":
            result = self._genetic_algorithm(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers
            )
        elif optimization_method == "simulated_annealing":
            result = self._simulated_annealing(
                objective_func, bounds, cons, previous_weights, opt_obj, opt_constraints, returns_array, cov_array, tickers
            )
        else:
            raise ValueError(f"Unknown optimization method: {optimization_method}")

        return result

    def _quadratic_programming(
        self, objective_func, bounds, cons, previous_weights,
        opt_obj, opt_constraints, returns_array, cov_array, tickers,
        risk_free_rate: float = 0.06,
    ) -> "OptimizationResult":
        """Quadratic programming optimization."""
        # For maximum Sharpe ratio, we need to solve analytically
        if opt_obj.objective == "max_sharpe":
            # Calculate tangency portfolio
            sharpe_matrix = (returns_array - risk_free_rate).T / cov_array
            weights_matrix = (sharpe_matrix @ returns_array - risk_free_rate) / (returns_array.T @ sharpe_matrix @ returns_array - risk_free_rate)
            weights_matrix = weights_matrix * opt_constraints.total_weights
        else:
            # For other objectives, use scipy
            result = optimize.minimize(
                objective_func,
                x0=np.ones(self.n_assets) / self.n_assets,
                bounds=bounds,
                constraints=cons,
                method="SLSQP",
                options={"maxiter": 1000, "ftol": 1e-6},
            )
            weights = result.x
            success = result.success
            message = result.message if not result.success else "Optimization successful"

            return OptimizationResult(
                weights=weights,
                expected_return=weights @ returns_array,
                volatility=np.sqrt(weights @ cov_array @ weights),
                sharpe=(weights @ returns_array - self.risk_free_rate) / np.sqrt(weights @ cov_array @ weights) if weights @ cov_array @ weights > 0 else 0,
                success=success,
                message=message,
                optimization_method="quadratic_programming",
            )

        weights = weights_matrix
        success = True
        message = "Optimization successful"

        # Calculate metrics
        expected_return = weights @ returns_array
        variance = weights @ cov_array @ weights
        volatility = np.sqrt(variance) if variance > 0 else 0
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Get risk contribution
        marginal_risk = cov_array @ weights
        risk_contribution = weights * marginal_risk / np.sum(marginal_risk) if np.sum(marginal_risk) > 0 else weights

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe=sharpe,
            risk_contribution=risk_contribution,
            success=success,
            message=message,
            optimization_method="quadratic_programming",
        )

    def _slsqp_optimization(self, objective_func, bounds, cons, previous_weights,
                            opt_obj, opt_constraints, returns_array, cov_array, tickers) -> "OptimizationResult":
        """S-Lagrange multiplier optimization."""
        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = optimize.minimize(
            objective_func,
            x0=initial_weights,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 1000, "ftol": 1e-6},
        )

        return self._process_optimization_result(
            result, returns_array, cov_array, opt_constraints.total_weights
        )

    def _differential_evolution(
        self, objective_func, bounds, cons, previous_weights,
        opt_obj, opt_constraints, returns_array, cov_array, tickers
    ) -> "OptimizationResult":
        """Differential evolution optimization."""
        result = optimize.differential_evolution(
            objective_func,
            bounds=bounds,
            constraints=cons,
            maxiter=100,
            popsize=15,
            recombination=0.7,
            mutation=(0.5, 1),
            seed=42,
        )

        return self._process_optimization_result(
            result, returns_array, cov_array, opt_constraints.total_weights
        )

    def _particle_swarm(
        self, objective_func, bounds, cons, previous_weights,
        opt_obj, opt_constraints, returns_array, cov_array, tickers
    ) -> "OptimizationResult":
        """Particle swarm optimization."""
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective_func,
            bounds=bounds,
            constraints=cons,
            maxiter=100,
            popsize=15,
            recombination=0.7,
            mutation=(0.5, 1),
            seed=42,
        )

        return self._process_optimization_result(
            result, returns_array, cov_array, opt_constraints.total_weights
        )

    def _genetic_algorithm(
        self, objective_func, bounds, cons, previous_weights,
        opt_obj, opt_constraints, returns_array, cov_array, tickers
    ) -> "OptimizationResult":
        """Genetic algorithm optimization."""
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            objective_func,
            bounds=bounds,
            constraints=cons,
            maxiter=100,
            popsize=15,
            recombination=0.7,
            mutation=(0.5, 1),
            seed=42,
        )

        return self._process_optimization_result(
            result, returns_array, cov_array, opt_constraints.total_weights
        )

    def _simulated_annealing(
        self, objective_func, bounds, cons, previous_weights,
        opt_obj, opt_constraints, returns_array, cov_array, tickers
    ) -> "OptimizationResult":
        """Simulated annealing optimization."""
        initial_weights = np.ones(self.n_assets) / self.n_assets

        result = optimize.dual_annealing(
            objective_func,
            bounds=bounds,
            constraints=cons,
            maxiter=1000,
            seed=42,
        )

        return self._process_optimization_result(
            result, returns_array, cov_array, opt_constraints.total_weights
        )

    def _process_optimization_result(self, result, returns_array, cov_array, total_weights) -> "OptimizationResult":
        """Process optimization result into OptimizationResult."""
        if not hasattr(result, 'x'):
            return OptimizationResult(
                weights=np.zeros(self.n_assets),
                expected_return=0,
                volatility=0,
                sharpe=0,
                success=False,
                message="Optimization failed",
                optimization_method="unknown",
            )

        weights = result.x

        # Normalize weights
        weights = weights / np.sum(weights)

        # Calculate metrics
        expected_return = weights @ returns_array
        variance = weights @ cov_array @ weights
        volatility = np.sqrt(variance) if variance > 0 else 0
        sharpe = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        # Get risk contribution
        marginal_risk = cov_array @ weights
        risk_contribution = weights * marginal_risk / np.sum(marginal_risk) if np.sum(marginal_risk) > 0 else weights

        return OptimizationResult(
            weights=weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe=sharpe,
            risk_contribution=risk_contribution,
            success=result.success,
            message=result.message if hasattr(result, 'message') else "Optimization successful",
            optimization_method="solved",
        )


class OptimizationResult:
    """Container for optimization results."""

    def __init__(
        self,
        weights: np.ndarray,
        expected_return: float,
        volatility: float,
        sharpe: float,
        risk_contribution: np.ndarray,
        success: bool,
        message: str,
        optimization_method: str,
    ):
        self.weights = weights
        self.expected_return = expected_return
        self.volatility = volatility
        self.sharpe = sharpe
        self.risk_contribution = risk_contribution
        self.success = success
        self.message = message
        self.optimization_method = optimization_method
        self.calmar_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.diversification_ratio = 0.0

    def to_dataframe(self, tickers: list[str]) -> pd.DataFrame:
        """Convert weights to DataFrame."""
        return pd.DataFrame({
            "ticker": tickers,
            "weight": self.weights,
            "risk_contribution": self.risk_contribution,
        })


class EfficientFrontierAnalysis:
    """Complete efficient frontier analysis."""

    def __init__(
        self,
        prices: pd.DataFrame,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        method_returns: ExpectedReturnMethod = "arithmetic_mean",
        method_cov: CovarianceMethod = "sample",
        universe_selection: UniverseSelection | None = None,
        universe_tickers: list[str] | None = None,
    ):
        self.prices = prices
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.method_returns = method_returns
        self.method_cov = method_cov
        self.universe_selection = universe_selection
        self.universe_tickers = universe_tickers

        self._process_data()

    def _process_data(self) -> None:
        """Process and prepare data for optimization."""
        # Filter by universe if specified
        if self.universe_tickers:
            self.prices = self.prices[self.universe_tickers]

        # Calculate expected returns
        if self.expected_returns is None:
            self.expected_returns = expected_returns(
                self.prices,
                method=self.method_returns,
                frequency="daily",
            )

        # Calculate covariance matrix
        if self.cov_matrix is None:
            self.cov_matrix = covariance_matrix(
                self.prices,
                method=self.method_cov,
                frequency="daily",
            )

        # Align data
        self.expected_returns = self.expected_returns.reindex(self.prices.columns)
        self.cov_matrix = self.cov_matrix.reindex(self.prices.columns, columns=self.prices.columns)

        # Initialize efficient frontier
        self.efficient_frontier = EfficientFrontier(
            self.expected_returns,
            self.cov_matrix,
        )

    def optimize_portfolio(
        self,
        objective: OptimizationObjective = "max_sharpe",
        constraints: PortfolioConstraints | None = None,
        sector_constraints: SectorConstraints | None = None,
        market_cap_constraints: MarketCapConstraints | None = None,
        cash_constraints: CashConstraints | None = None,
        turnover_constraints: TurnoverConstraints | None = None,
        liquidity_constraints: LiquidityConstraints | None = None,
        optimization_solver: OptimizationSolver = "quadratic_programming",
        max_iterations: int = 10000,
        solver_params: dict | None = None,
    ) -> OptimizationResult:
        """Optimize portfolio according to specified objective."""
        # Create portfolio optimizer
        weights_df = pd.DataFrame(
            np.eye(len(self.expected_returns)),
            index=self.expected_returns.index,
            columns=self.expected_returns.index,
        )

        optimizer = PortfolioOptimizer(
            self.expected_returns,
            self.cov_matrix,
            weights_df,
        )

        # Run optimization
        result = optimizer.optimize(
            objective=objective,
            constraints=constraints,
            sector_constraints=sector_constraints,
            market_cap_constraints=market_cap_constraints,
            cash_constraints=cash_constraints,
            turnover_constraints=turnover_constraints,
            liquidity_constraints=liquidity_constraints,
            optimization_method=optimization_solver,
            max_iterations=max_iterations,
            solver_params=solver_params,
        )

        return result

    def generate_efficient_frontier(
        self,
        n_portfolios: int = 10000,
        objective: OptimizationObjective = "max_sharpe",
        generate_frontier: bool = True,
    ) -> pd.DataFrame:
        """Generate efficient frontier."""
        if generate_frontier:
            return self.efficient_frontier.generate_frontier(
                n_portfolios=n_portfolios,
                objective=objective,
            )
        else:
            return self.efficient_frontier.get_efficient_frontier_points(
                n_points=100,
                objective=objective,
            )

    def get_portfolio_metrics(
        self,
        weights: np.ndarray,
        expected_returns: pd.Series | None = None,
        cov_matrix: pd.DataFrame | None = None,
        returns_history: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Calculate portfolio metrics."""
        expected_returns = expected_returns or self.expected_returns
        cov_matrix = cov_matrix or self.cov_matrix

        # Basic metrics
        basic_metrics = risk_metrics(
            weights,
            expected_returns,
            cov_matrix,
        )

        # Diversification metrics
        div_metrics = diversification_metrics(
            weights,
            cov_matrix,
            expected_returns,
        )

        # Combined metrics
        metrics = {**basic_metrics, **div_metrics}

        if returns_history is not None:
            # Calculate additional metrics
            metrics.update({
                "calmar_ratio": self._calculate_calmar_ratio(weights, returns_history),
                "sortino_ratio": self._calculate_sortino_ratio(weights, returns_history),
                "max_drawdown": self._calculate_max_drawdown(weights, returns_history),
            })

        return metrics

    def _calculate_calmar_ratio(self, weights: np.ndarray, returns_history: pd.DataFrame) -> float:
        """Calculate Calmar ratio."""
        port_returns = returns_history @ weights
        cum_returns = np.cumprod(1 + port_returns)
        max_dd = self._calculate_max_drawdown(None, pd.Series(cum_returns))
        cagr = (cum_returns.iloc[-1] ** (252 / len(port_returns)) - 1) if len(port_returns) > 0 else 0
        return cagr / abs(max_dd) if max_dd != 0 else 0

    def _calculate_sortino_ratio(self, weights: np.ndarray, returns_history: pd.DataFrame) -> float:
        """Calculate Sortino ratio."""
        port_returns = returns_history @ weights
        excess_returns = port_returns - 0.06 / len(port_returns)
        downside = excess_returns[excess_returns < 0]
        downside_std = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 0
        mean_return = np.mean(port_returns) * 252
        return mean_return / downside_std if downside_std != 0 else 0

    def _calculate_max_drawdown(self, weights: np.ndarray, returns_history: pd.Series) -> float:
        """Calculate maximum drawdown."""
        nav = np.cumprod(1 + returns_history)
        running_max = np.maximum.accumulate(nav)
        drawdown = (nav - running_max) / running_max
        return abs(drawdown.min())