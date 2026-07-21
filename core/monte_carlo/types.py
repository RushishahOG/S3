"""Monte Carlo simulation package for the ARQM Research Lab.

Vectorized, cached, regime-aware Monte Carlo simulations that resample the
*actual* historical trade / return sequence produced by a completed backtest
(rather than synthesising returns from a parametric distribution).
"""

from __future__ import annotations

from dataclasses import dataclass, field

TRADING_DAYS = 252

METHOD_RETURN_BOOTSTRAP = "return_bootstrap"
METHOD_TRADE_RANDOMIZATION = "trade_randomization"
METHOD_BLOCK_BOOTSTRAP = "block_bootstrap"
METHOD_REGIME_BOOTSTRAP = "regime_bootstrap"

METHOD_LABELS: dict[str, str] = {
    METHOD_RETURN_BOOTSTRAP: "Return Bootstrap",
    METHOD_TRADE_RANDOMIZATION: "Trade Sequence Randomization",
    METHOD_BLOCK_BOOTSTRAP: "Block Bootstrap",
    METHOD_REGIME_BOOTSTRAP: "Regime Bootstrap",
}

BLOCK_SIZES: list[int] = [5, 10, 20, 30]

N_SIMULATION_CHOICES: list[int] = [100, 250, 500, 1000, 2500, 5000, 10000]

METRIC_COLUMNS: list[str] = [
    "cagr",
    "total_return",
    "final_value",
    "annual_volatility",
    "sharpe",
    "sortino",
    "calmar",
    "max_drawdown",
    "ulcer_index",
    "avg_drawdown",
    "longest_dd_duration",
    "recovery_factor",
    "win_rate",
    "profit_factor",
    "expectancy",
    "n_trades",
    "exposure",
    "turnover",
]

MIN_DAILY_POINTS = 30


@dataclass
class SimulationConfig:
    """User-supplied configuration for a Monte Carlo run."""

    method: str
    n_simulations: int
    seed: int | None = None
    horizon: int | None = None
    block_size: int = 10
    parallel: bool = False
    parallel_threshold: int = 1000
    chunk_size: int = 500
    initial_capital: float = 1_000_000.0
    rf_annual: float = 0.0

    def effective_horizon(self, data_len: int) -> int:
        """Number of simulated days actually produced for the method."""
        if self.method in (METHOD_REGIME_BOOTSTRAP, METHOD_TRADE_RANDOMIZATION):
            return data_len
        return self.horizon or data_len

    def signature(self) -> str:
        """Stable string used for cache invalidation."""
        return (
            f"{self.method}|{self.n_simulations}|{self.seed}|{self.horizon}|"
            f"{self.block_size}|{self.parallel}|{self.initial_capital}|{self.rf_annual}"
        )


@dataclass
class MCInput:
    """Prepared, cached inputs extracted from a completed BacktestResult."""

    returns: "np.ndarray"
    dates: "pd.DatetimeIndex"
    equity: "np.ndarray"
    initial_capital: float
    trades: "pd.DataFrame"
    regime_states: "np.ndarray"
    original_metrics: dict
    n_trades: int
    exposure: float


@dataclass
class SimulationResult:
    """Full output of a Monte Carlo run."""

    config: SimulationConfig
    method: str
    n_simulations: int
    horizon_used: int
    sim_dates: "pd.DatetimeIndex"
    equity_curves: "np.ndarray"
    metrics_df: "pd.DataFrame"
    aggregate: "pd.DataFrame"
    probabilities: dict
    risk_summary: dict
    confidence_intervals: dict
    original_metrics: dict
    original_equity: "np.ndarray"
    seed: int | None
