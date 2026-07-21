"""Monte Carlo simulation engine.

Resamples the actual historical daily-return / trade sequence of a completed
backtest using four methodologies:

* ``return_bootstrap``     - i.i.d. sampling of daily returns with replacement
* ``trade_randomization``  - permutation of trade (holding-period) legs
* ``block_bootstrap``      - sampling of contiguous blocks (preserves dependence)
* ``regime_bootstrap``     - bootstrap *within* each regime, preserving regime order

All heavy loops are vectorised with NumPy. Long runs are processed in chunks so
the UI can show live progress and cancel mid-flight. Trade-sequence runs can be
parallelised with joblib when the simulation count exceeds a threshold.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from core.monte_carlo.statistics import (
    compute_aggregate,
    compute_probabilities,
    compute_risk_summary,
)
from core.monte_carlo.types import (
    MCInput,
    METHOD_BLOCK_BOOTSTRAP,
    METHOD_REGIME_BOOTSTRAP,
    METHOD_RETURN_BOOTSTRAP,
    METHOD_TRADE_RANDOMIZATION,
    SimulationConfig,
    SimulationResult,
    TRADING_DAYS,
)


class CancelSimulation(Exception):
    """Raised inside the engine when the user cancels a run."""


def build_mc_input(result: Any) -> MCInput:
    """Extract and prepare Monte Carlo inputs from a completed BacktestResult."""
    nav = result.nav
    if nav is None or getattr(nav, "empty", True) or len(nav) < 2:
        raise ValueError("Backtest produced no usable equity curve.")

    returns = nav.pct_change().fillna(0.0).to_numpy(dtype=float)
    equity = nav.to_numpy(dtype=float)
    dates = nav.index
    initial_capital = float(getattr(result.params.general, "initial_capital", equity[0]))

    regime_states = np.array(["flat"] * len(nav), dtype=object)
    if getattr(result, "regime", None) is not None and not result.regime.empty:
        aligned = result.regime["state"].reindex(dates).ffill().fillna("flat")
        regime_states = aligned.to_numpy()

    trades = result.trades if result.trades is not None else pd.DataFrame()
    n_trades = 0 if trades.empty else int(len(trades))

    invested = (regime_states == "invested").mean() if len(regime_states) else 0.0

    return MCInput(
        returns=returns,
        dates=dates,
        equity=equity,
        initial_capital=initial_capital,
        trades=trades,
        regime_states=regime_states,
        original_metrics=dict(result.metrics or {}),
        n_trades=n_trades,
        exposure=float(invested),
    )


def _regime_segments(inp: MCInput) -> list[np.ndarray]:
    states = inp.regime_states
    returns = inp.returns
    segments: list[np.ndarray] = []
    i, n = 0, len(returns)
    while i < n:
        j = i
        while j < n and states[j] == states[i]:
            j += 1
        seg = returns[i:j]
        if len(seg) > 0:
            segments.append(seg)
        i = j
    if not segments:
        segments.append(returns)
    return segments


def _trade_legs(inp: MCInput) -> tuple[list[np.ndarray], np.ndarray]:
    if inp.trades is None or inp.trades.empty:
        raise ValueError("Trade Sequence Randomization requires a non-empty trade log.")
    tdates = np.sort(pd.to_datetime(inp.trades["date"].unique()))
    pos = inp.dates.searchsorted(tdates)
    pos = np.clip(pos, 0, len(inp.returns))
    edges = np.concatenate([[0], pos, [len(inp.returns)]])
    edges = np.unique(edges)
    legs: list[np.ndarray] = []
    for k in range(len(edges) - 1):
        leg = inp.returns[edges[k]:edges[k + 1]]
        if len(leg) > 0:
            legs.append(leg)
    if not legs:
        raise ValueError("Could not derive trade legs from the trade log.")
    compounded = np.array([float(np.prod(1.0 + l) - 1.0) for l in legs])
    return legs, compounded


def _generate_chunk(
    config: SimulationConfig,
    inp: MCInput,
    n: int,
    rng: np.random.Generator,
    segments: list[np.ndarray],
    legs: list[np.ndarray],
    leg_compounded: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    method = config.method
    if method == METHOD_RETURN_BOOTSTRAP:
        n_total = len(inp.returns)
        h = config.effective_horizon(n_total)
        idx = rng.integers(0, n_total, size=(n, h))
        return inp.returns[idx], None

    if method == METHOD_BLOCK_BOOTSTRAP:
        n_total = len(inp.returns)
        h = config.effective_horizon(n_total)
        b = max(2, int(config.block_size))
        starts = np.arange(0, n_total - b + 1)
        if len(starts) == 0:
            h = n_total
            idx = rng.integers(0, n_total, size=(n, h))
            return inp.returns[idx], None
        block_matrix = inp.returns[starts[:, None] + np.arange(b)]
        n_blocks = int(np.ceil(h / b))
        chosen = rng.integers(0, len(starts), size=(n, n_blocks))
        sim = block_matrix[chosen].reshape(n, n_blocks * b)
        return sim[:, :h], None

    if method == METHOD_REGIME_BOOTSTRAP:
        h = sum(len(s) for s in segments)
        sim = np.empty((n, h), dtype=float)
        col = 0
        for seg in segments:
            length = len(seg)
            idx = rng.integers(0, length, size=(n, length))
            sim[:, col:col + length] = seg[idx]
            col += length
        return sim, None

    if method == METHOD_TRADE_RANDOMIZATION:
        h = sum(len(l) for l in legs)
        sim = np.empty((n, h), dtype=float)
        leg_mat = np.empty((n, len(legs)), dtype=float)
        for i in range(n):
            perm = rng.permutation(len(legs))
            seq = np.concatenate([legs[p] for p in perm])
            sim[i] = seq[:h]
            leg_mat[i] = leg_compounded[perm]
        return sim, leg_mat

    raise ValueError(f"Unknown simulation method: {method}")


def _gen_trade_chunk(
    legs: list[np.ndarray],
    leg_compounded: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h = sum(len(l) for l in legs)
    sim = np.empty((n, h), dtype=float)
    leg_mat = np.empty((n, len(legs)), dtype=float)
    for i in range(n):
        perm = rng.permutation(len(legs))
        seq = np.concatenate([legs[p] for p in perm])
        sim[i] = seq[:h]
        leg_mat[i] = leg_compounded[perm]
    return sim, leg_mat


def _compute_sim_metrics(
    sim_ret: np.ndarray,
    equity: np.ndarray,
    leg_rets: np.ndarray | None,
    inp: MCInput,
    config: SimulationConfig,
) -> pd.DataFrame:
    s, h = sim_ret.shape
    final = equity[:, -1]
    total_return = final / inp.initial_capital - 1.0
    years = h / TRADING_DAYS
    cagr = (
        np.power(final / inp.initial_capital, 1.0 / years) - 1.0
        if years > 0
        else np.full(s, np.nan)
    )

    mean_d = sim_ret.mean(axis=1)
    std_d = sim_ret.std(axis=1, ddof=1)
    ann_vol = std_d * np.sqrt(TRADING_DAYS)
    sharpe = np.where(std_d > 0, mean_d / std_d * np.sqrt(TRADING_DAYS), np.nan)
    downside = np.where(sim_ret < 0, sim_ret, 0.0)
    dstd = np.sqrt((downside ** 2).mean(axis=1))
    sortino = np.where(dstd > 0, mean_d / dstd * np.sqrt(TRADING_DAYS), np.nan)

    rollmax = np.maximum.accumulate(equity, axis=1)
    dd = equity / rollmax - 1.0
    mdd = dd.min(axis=1)
    calmar = np.where((mdd < 0) & ~np.isnan(cagr), cagr / np.abs(mdd), np.nan)
    ulcer = np.sqrt(((dd * 100.0) ** 2).mean(axis=1))
    avg_dd = dd.mean(axis=1)

    cur = np.zeros(s, dtype=int)
    running_peak = equity[:, 0].copy()
    longest = np.zeros(s, dtype=int)
    for t in range(1, h):
        new_high = equity[:, t] >= running_peak
        running_peak = np.maximum(running_peak, equity[:, t])
        cur = np.where(new_high, 0, cur + 1)
        longest = np.maximum(longest, cur)

    recovery = np.where(mdd < 0, total_return / np.abs(mdd), np.nan)

    if leg_rets is not None:
        win = (leg_rets > 0).mean(axis=1)
        gains = np.where(leg_rets > 0, leg_rets, 0.0).sum(axis=1)
        losses = np.where(leg_rets < 0, -leg_rets, 0.0).sum(axis=1)
        pf = np.where(losses > 0, gains / losses, np.where(gains > 0, np.inf, np.nan))
        exp = leg_rets.mean(axis=1)
        n_tr = leg_rets.shape[1]
    else:
        win = exp = pf = np.full(s, np.nan)
        n_tr = np.nan

    turnover = inp.original_metrics.get("turnover", np.nan)

    return pd.DataFrame({
        "cagr": cagr,
        "total_return": total_return,
        "final_value": final,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "ulcer_index": ulcer,
        "avg_drawdown": avg_dd,
        "longest_dd_duration": longest.astype(float),
        "recovery_factor": recovery,
        "win_rate": win,
        "profit_factor": pf,
        "expectancy": exp,
        "n_trades": n_tr,
        "exposure": float(inp.exposure),
        "turnover": float(turnover) if turnover == turnover else np.nan,
    })


def _make_sim_dates(inp: MCInput, h: int) -> pd.DatetimeIndex:
    if h <= len(inp.dates):
        return inp.dates[:h]
    last = inp.dates[-1]
    extra = pd.bdate_range(last + pd.Timedelta(days=1), periods=h - len(inp.dates))
    return inp.dates.append(extra)


def run_simulation(
    config: SimulationConfig,
    inp: MCInput,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_cb: Callable[[], bool] | None = None,
) -> SimulationResult:
    """Run the Monte Carlo simulation and return a fully aggregated result."""
    n_total = len(inp.returns)
    if n_total < 2:
        raise ValueError("Insufficient daily return data for simulation.")

    segments: list[np.ndarray] = []
    legs: list[np.ndarray] = []
    leg_compounded = np.ndarray((0,))
    if config.method == METHOD_REGIME_BOOTSTRAP:
        segments = _regime_segments(inp)
    elif config.method == METHOD_TRADE_RANDOMIZATION:
        legs, leg_compounded = _trade_legs(inp)

    s = config.n_simulations
    h = config.effective_horizon(n_total)

    use_parallel = (
        config.parallel
        and config.method == METHOD_TRADE_RANDOMIZATION
        and s > config.parallel_threshold
    )

    metrics_parts: list[pd.DataFrame] = []
    equity_parts: list[np.ndarray] = []

    def _maybe_cancel() -> None:
        if cancel_cb is not None and cancel_cb():
            raise CancelSimulation()

    if use_parallel:
        from joblib import Parallel, delayed

        n_chunks = max(1, s // config.chunk_size)
        chunk_sizes = [s // n_chunks + (1 if i < s % n_chunks else 0) for i in range(n_chunks)]
        seed_seq = np.random.SeedSequence(config.seed)
        child_seeds = seed_seq.spawn(n_chunks)
        chunks = Parallel(n_jobs=-1)(
            delayed(_gen_trade_chunk)(legs, leg_compounded, csize, cs)
            for csize, cs in zip(chunk_sizes, child_seeds)
        )
        done = 0
        for (sim_ret, leg_mat), csize in zip(chunks, chunk_sizes):
            equity = np.cumprod(1.0 + sim_ret, axis=1) * inp.initial_capital
            m = _compute_sim_metrics(sim_ret, equity, leg_mat, inp, config)
            metrics_parts.append(m)
            equity_parts.append(equity)
            done += csize
            if progress_cb:
                progress_cb(done, s)
            _maybe_cancel()
    else:
        rng = np.random.default_rng(config.seed)
        chunk = max(1, min(config.chunk_size, s))
        done = 0
        while done < s:
            n = min(chunk, s - done)
            sim_ret, leg_rets = _generate_chunk(
                config, inp, n, rng, segments, legs, leg_compounded
            )
            equity = np.cumprod(1.0 + sim_ret, axis=1) * inp.initial_capital
            m = _compute_sim_metrics(sim_ret, equity, leg_rets, inp, config)
            metrics_parts.append(m)
            equity_parts.append(equity)
            done += n
            if progress_cb:
                progress_cb(done, s)
            _maybe_cancel()

    metrics_df = pd.concat(metrics_parts, ignore_index=True)
    equity_curves = np.vstack(equity_parts)

    sim_dates = _make_sim_dates(inp, h)
    aggregate = compute_aggregate(metrics_df)
    probabilities = compute_probabilities(metrics_df)
    risk_summary, confidence_intervals = compute_risk_summary(metrics_df, config)

    return SimulationResult(
        config=config,
        method=config.method,
        n_simulations=s,
        horizon_used=h,
        sim_dates=sim_dates,
        equity_curves=equity_curves,
        metrics_df=metrics_df,
        aggregate=aggregate,
        probabilities=probabilities,
        risk_summary=risk_summary,
        confidence_intervals=confidence_intervals,
        original_metrics=inp.original_metrics,
        original_equity=inp.equity,
        seed=config.seed,
    )
