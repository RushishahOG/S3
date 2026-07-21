"""Portfolio Risk Models: Expected Returns, Covariance, and Risk Metrics."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from scipy import linalg

warnings.filterwarnings("ignore", category=RuntimeWarning)

TRADING_DAYS = 252
MONTHS_PER_YEAR = 12

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


def expected_returns(
    prices: pd.DataFrame,
    method: ExpectedReturnMethod = "arithmetic_mean",
    frequency: str = "daily",
    risk_free_rate: float = 0.06,
    market_returns: pd.Series | None = None,
    custom_returns: pd.Series | None = None,
    ema_span: int = 60,
    lookback_days: int | None = None,
) -> pd.Series:
    """Calculate expected annual returns for each asset.

    Args:
        prices: DataFrame with dates as index and tickers as columns
        method: Method to estimate expected returns
        frequency: Data frequency ("daily", "weekly", "monthly")
        risk_free_rate: Risk-free rate for CAPM
        market_returns: Market returns for CAPM
        custom_returns: Custom expected returns (for "custom" method)
        ema_span: Span for EMA method
        lookback_days: Lookback period in days (None = full history)

    Returns:
        Series of expected annual returns per ticker
    """
    if prices.empty:
        return pd.Series(dtype=float)

    if lookback_days is not None:
        prices = prices.tail(lookback_days)

    returns = prices.pct_change().dropna()

    if frequency == "daily":
        periods_per_year = TRADING_DAYS
    elif frequency == "weekly":
        periods_per_year = 52
    elif frequency == "monthly":
        periods_per_year = MONTHS_PER_YEAR
    else:
        periods_per_year = TRADING_DAYS

    n_assets = len(prices.columns)
    result = pd.Series(index=prices.columns, dtype=float)

    if method == "historical_cagr":
        for ticker in prices.columns:
            p = prices[ticker].dropna()
            if len(p) < 2:
                result[ticker] = 0.0
                continue
            years = len(p) / periods_per_year
            cagr = (p.iloc[-1] / p.iloc[0]) ** (1 / years) - 1
            result[ticker] = cagr

    elif method == "arithmetic_mean":
        mean_ret = returns.mean() * periods_per_year
        result = mean_ret

    elif method == "geometric_mean":
        geom_mean = (1 + returns).prod() ** (periods_per_year / len(returns)) - 1
        result = geom_mean

    elif method == "ema":
        ema_returns = returns.ewm(span=ema_span).mean().iloc[-1] * periods_per_year
        result = ema_returns

    elif method == "capm":
        if market_returns is None:
            raise ValueError("market_returns required for CAPM method")
        market_ret = market_returns.dropna()
        aligned_returns, aligned_market = returns.align(market_ret, join="inner", axis=0)
        if len(aligned_market) < 30:
            raise ValueError("Insufficient market data for CAPM")
        market_var = aligned_market.var()
        if market_var == 0:
            raise ValueError("Market variance is zero")
        betas = aligned_returns.cov(aligned_market) / market_var
        market_premium = aligned_market.mean() * periods_per_year - risk_free_rate
        result = risk_free_rate + betas * market_premium
        result = result.reindex(prices.columns).fillna(risk_free_rate)

    elif method == "custom":
        if custom_returns is None:
            raise ValueError("custom_returns required for custom method")
        result = custom_returns.reindex(prices.columns).fillna(0.0)

    else:
        raise ValueError(f"Unknown method: {method}")

    return result.clip(-1.0, 5.0)


def covariance_matrix(
    prices: pd.DataFrame,
    method: CovarianceMethod = "sample",
    frequency: str = "daily",
    lookback_days: int | None = None,
    ema_span: int = 60,
    shrinkage_delta: float = 0.5,
    custom_matrix: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate covariance matrix of returns.

    Args:
        prices: DataFrame with dates as index and tickers as columns
        method: Covariance estimation method
        frequency: Data frequency
        lookback_days: Lookback period
        ema_span: Span for exponential covariance
        shrinkage_delta: Shrinkage intensity for Ledoit-Wolf
        custom_matrix: Custom covariance matrix

    Returns:
        Annualized covariance matrix
    """
    if prices.empty:
        return pd.DataFrame()

    if lookback_days is not None:
        prices = prices.tail(lookback_days)

    returns = prices.pct_change().dropna()

    if frequency == "daily":
        periods_per_year = TRADING_DAYS
    elif frequency == "weekly":
        periods_per_year = 52
    elif frequency == "monthly":
        periods_per_year = MONTHS_PER_YEAR
    else:
        periods_per_year = TRADING_DAYS

    n_assets = len(returns.columns)

    if method == "sample":
        cov = returns.cov() * periods_per_year

    elif method == "exponential":
        weights = np.exp(np.linspace(-1, 0, len(returns)))
        weights = weights / weights.sum()
        mean_ret = np.average(returns, axis=0, weights=weights)
        centered = returns - mean_ret
        cov = (centered.T * weights) @ centered * periods_per_year
        cov = pd.DataFrame(cov, index=returns.columns, columns=returns.columns)

    elif method == "ledoit_wolf":
        cov = _ledoit_wolf_shrinkage(returns, periods_per_year)

    elif method == "oracle_approximating":
        cov = _oracle_approximating_shrinkage(returns, periods_per_year)

    elif method == "constant_correlation":
        cov = _constant_correlation(returns, periods_per_year)

    elif method == "custom":
        if custom_matrix is None:
            raise ValueError("custom_matrix required for custom method")
        cov = custom_matrix.copy()
        if not cov.index.equals(returns.columns) or not cov.columns.equals(returns.columns):
            raise ValueError("Custom matrix columns must match tickers")
    else:
        raise ValueError(f"Unknown method: {method}")

    cov = _ensure_positive_definite(cov)
    return cov


def _ledoit_wolf_shrinkage(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage estimator."""
    n, p = returns.shape
    sample_cov = returns.cov().values * periods_per_year

    # Prior: diagonal matrix with sample variances
    prior = np.diag(np.diag(sample_cov))

    # Estimate shrinkage intensity
    returns_centered = returns - returns.mean()
    phi_mat = (returns_centered**2).T @ (returns_centered**2) / n - sample_cov**2
    phi = phi_mat.sum()

    theta_mat = ((returns_centered**2).T @ (returns_centered**2)) / n - sample_cov**2
    theta = np.sum(theta_mat)

    gamma = np.sum((sample_cov - prior) ** 2)

    if gamma == 0:
        delta = 0
    else:
        delta = min(max(phi / gamma, 0), 1)

    shrunk = delta * prior + (1 - delta) * sample_cov
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def _oracle_approximating_shrinkage(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """Oracle Approximating Shrinkage (OAS) estimator."""
    n, p = returns.shape
    sample_cov = returns.cov().values * periods_per_year

    # Prior: identity scaled by average variance
    avg_var = np.trace(sample_cov) / p
    prior = np.eye(p) * avg_var

    # OAS shrinkage intensity
    returns_centered = returns - returns.mean()
    phi_mat = (returns_centered**2).T @ (returns_centered**2) / n - sample_cov**2
    phi = phi_mat.sum()

    theta = np.trace(sample_cov @ sample_cov) / n - np.sum(sample_cov**2) / n

    gamma = np.sum((sample_cov - prior) ** 2)

    if gamma == 0:
        delta = 0
    else:
        delta = min(max(phi / gamma, 0), 1)

    shrunk = delta * prior + (1 - delta) * sample_cov
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def _constant_correlation(returns: pd.DataFrame, periods_per_year: int) -> pd.DataFrame:
    """Constant correlation model."""
    sample_cov = returns.cov().values * periods_per_year
    stds = np.sqrt(np.diag(sample_cov))
    corr = sample_cov / np.outer(stds, stds)
    np.fill_diagonal(corr, 1.0)
    avg_corr = (corr.sum() - p) / (p * (p - 1)) if (p := len(returns.columns)) > 1 else 1.0
    const_corr = np.full_like(corr, avg_corr)
    np.fill_diagonal(const_corr, 1.0)
    shrunk = const_corr * np.outer(stds, stds)
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)


def _ensure_positive_definite(cov: pd.DataFrame, min_eigenvalue: float = 1e-8) -> pd.DataFrame:
    """Ensure covariance matrix is positive definite via eigenvalue adjustment."""
    eigenvalues, eigenvectors = linalg.eigh(cov.values)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    fixed = (fixed + fixed.T) / 2
    return pd.DataFrame(fixed, index=cov.index, columns=cov.columns)


def risk_metrics(
    weights: np.ndarray,
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.06,
    returns: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Calculate portfolio risk metrics.

    Args:
        weights: Portfolio weights
        expected_returns: Expected annual returns
        cov_matrix: Annualized covariance matrix
        risk_free_rate: Risk-free rate
        returns: Historical returns for downside metrics

    Returns:
        Dictionary of risk metrics
    """
    w = np.asarray(weights)
    er = expected_returns.reindex(expected_returns.index).values
    cov = cov_matrix.reindex(expected_returns.index, columns=expected_returns.index).values

    port_return = float(w @ er)
    port_variance = float(w @ cov @ w)
    port_volatility = np.sqrt(max(port_variance, 0))

    sharpe = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0

    metrics = {
        "expected_return": port_return,
        "volatility": port_volatility,
        "variance": port_variance,
        "sharpe_ratio": sharpe,
    }

    if returns is not None:
        aligned_returns = returns.reindex(columns=expected_returns.index).dropna()
        if not aligned_returns.empty:
            port_returns = (aligned_returns.values @ w)
            sortino = _calculate_sortino(port_returns, risk_free_rate / TRADING_DAYS)
            calmar = _calculate_calmar(port_returns)
            max_dd = _calculate_max_drawdown(port_returns)
            metrics["sortino_ratio"] = sortino
            metrics["calmar_ratio"] = calmar
            metrics["max_drawdown"] = max_dd

    return metrics


def _calculate_sortino(returns: np.ndarray, risk_free: float) -> float:
    """Calculate Sortino ratio."""
    excess = returns - risk_free
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = np.std(downside) * np.sqrt(TRADING_DAYS)
    if downside_std == 0:
        return np.inf
    return np.mean(excess) * TRADING_DAYS / downside_std


def _calculate_calmar(returns: np.ndarray) -> float:
    """Calculate Calmar ratio."""
    cum_returns = np.cumprod(1 + returns)
    max_dd = _calculate_max_drawdown(returns)
    if max_dd == 0:
        return np.inf
    cagr = (cum_returns[-1] ** (TRADING_DAYS / len(returns)) - 1) if len(returns) > 0 else 0
    return cagr / abs(max_dd)


def _calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cum_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - running_max) / running_max
    return float(drawdown.min())


def diversification_metrics(
    weights: np.ndarray,
    cov_matrix: pd.DataFrame,
    expected_returns: pd.Series | None = None,
) -> dict[str, float]:
    """Calculate diversification metrics."""
    w = np.asarray(weights)
    cov = cov_matrix.values

    port_variance = float(w @ cov @ w)
    weighted_avg_var = float(np.sum(w**2 * np.diag(cov)))

    diversification_ratio = np.sqrt(weighted_avg_var / port_variance) if port_variance > 0 else 1.0

    herfindahl = float(np.sum(w**2))
    effective_n = 1 / herfindahl if herfindahl > 0 else len(w)
    weight_entropy = -float(np.sum(w[w > 0] * np.log(w[w > 0]))) if np.any(w > 0) else 0

    metrics = {
        "diversification_ratio": diversification_ratio,
        "effective_number_of_stocks": effective_n,
        "herfindahl_index": herfindahl,
        "weight_entropy": weight_entropy,
    }

    if expected_returns is not None:
        er = expected_returns.values
        contrib_return = w * er
        contrib_risk = w * (cov @ w) / port_variance if port_variance > 0 else np.zeros_like(w)
        metrics["return_contribution"] = contrib_return
        metrics["risk_contribution"] = contrib_risk
        metrics["risk_contribution_pct"] = contrib_risk / np.sum(contrib_risk) if np.sum(contrib_risk) > 0 else contrib_risk

    return metrics


def correlation_analysis(
    prices: pd.DataFrame,
    method: str = "pearson",
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Calculate correlation matrix with optional hierarchical clustering."""
    if prices.empty:
        return pd.DataFrame()

    if lookback_days is not None:
        prices = prices.tail(lookback_days)

    returns = prices.pct_change().dropna()
    corr = returns.corr(method=method)

    # Hierarchical clustering for ordering
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist = squareform(1 - corr.abs())
        link = linkage(dist, method="ward")
        order = leaves_list(link)
        corr = corr.iloc[order, order]
    except ImportError:
        pass

    return corr