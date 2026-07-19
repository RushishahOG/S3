"""ARQM backtest engine.

Orchestrates the full simulation: load engineered data -> build rebalance
schedule -> at each rebalance date run the gates (eligibility / momentum /
stability / quality / persistence) -> segment by cap tier -> construct a target
book per bucket with configurable position sizing -> simulate daily NAV with
transaction costs, slippage and regime-driven entry/exit -> emit NAV, trades,
snapshots and factor scores.

The engine consumes ONLY engineered datasets (prices, low-vol features, quality
features, company metadata, universe). Momentum signals are derived from prices
point-in-time inside :mod:`core.backtesting.momentum`. Nothing here recalculates
raw fundamentals.

Determinism: given an identical ``BacktestParameters`` and DB state the engine
produces byte-identical outputs (no randomness, no network).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from core.backtesting import regime as regime_mod
from core.backtesting.data import BacktestData, load_backtest_data
from core.backtesting.gate_registry import (
    GateResult,
    PipelineContext,
    get_gate,
)
from core.backtesting.gates import final_scores
from core.backtesting.metrics import compute_all_metrics, daily_returns
from core.config.backtest_schema import BacktestParameters
from core.data.storage.storage_manager import StorageManager
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

_FREQUENCY_MONTHS = {"monthly": 1, "quarterly": 3, "semi_annual": 6}


@dataclass
class BacktestResult:
    nav: pd.Series
    benchmark_nav: pd.Series
    regime: pd.DataFrame
    trades: pd.DataFrame
    snapshots: dict[pd.Timestamp, pd.DataFrame]  # rebalance_date -> ranking table
    factor_scores: dict[pd.Timestamp, dict[str, pd.Series]]
    metrics: dict[str, float]
    params: BacktestParameters
    pipeline_audit: dict[pd.Timestamp, list[GateResult]] = field(default_factory=dict)


def _rebalance_dates(start: pd.Timestamp, end: pd.Timestamp, freq: str) -> list[pd.Timestamp]:
    months = _FREQUENCY_MONTHS[freq]
    dates: list[pd.Timestamp] = []
    cur = pd.Timestamp(start.year, start.month, 1) + pd.DateOffset(months=months - 1)
    # Align to end-of-quarter style months.
    while cur <= end:
        if cur >= start:
            dates.append(cur)
        cur = cur + pd.DateOffset(months=months)
    return dates


def run_backtest(
    params: BacktestParameters,
    storage: StorageManager,
    progress_callback: Callable[[dict], None] | None = None,
) -> BacktestResult:
    """Run the ARQM backtest.

    ``progress_callback`` (optional) is invoked after each gate completes on each
    rebalance date with a small event dict, and once at the end with
    ``{"event": "done", ...}``. The Streamlit UI uses this for progressive,
    non-blocking rendering of per-gate results.
    """
    def _load_progress(stage: str, status: str, duration: float | None, n_rows: int | None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({
                "event": "load_step",
                "stage": stage,
                "status": status,
                "duration_s": duration,
                "n_rows": n_rows,
            })
        except Exception:
            pass

    data = load_backtest_data(storage, params, progress=_load_progress)
    prices = data.prices
    if prices.empty or not data.universe_tickers:
        logger.warning(
            "No price data / universe for the configured range; returning empty "
            "result. requested_universe=%d, price_columns=%d, intersection=%d, "
            "price_rows=%d, range=[%s, %s]",
            len(data.universe_tickers) if data.universe_tickers else 0,
            len(prices.columns) if prices is not None else 0,
            len(set(data.universe_tickers) & set(prices.columns)) if prices is not None else 0,
            len(prices) if prices is not None else 0,
            data.start, data.end,
        )
        empty_nav = pd.Series(dtype=float)
        return BacktestResult(
            nav=empty_nav, benchmark_nav=empty_nav, regime=pd.DataFrame(),
            trades=pd.DataFrame(), snapshots={}, factor_scores={},
            metrics={}, params=params, pipeline_audit={},
        )

    # Benchmark returns (daily). Fall back to an equal-weight universe index
    # when the benchmark pseudo-ticker is not stored as a price series.
    bench = data.benchmark_prices.dropna()
    if bench.empty and not prices.empty:
        ew = prices.mean(axis=1).dropna()
        bench = ew
    bench_ret = bench.pct_change().fillna(0.0)
    if bench.empty:
        bench_nav = pd.Series(params.general.initial_capital, index=prices.index)
    else:
        bench_nav = (1.0 + bench_ret).cumprod()
        bench_nav = bench_nav / bench_nav.iloc[0] * params.general.initial_capital

    # Regime on the reference series. The portfolio-referenced regime is only
    # known after simulation, so we use the (resolved) benchmark series for
    # signal generation (documented behaviour).
    ref = bench if params.regime.reference == "benchmark" else None
    regime_df = regime_mod.detect_regime(ref if ref is not None else bench, params.regime)

    all_dates = prices.index
    start, end = data.start, data.end
    rbdates = _rebalance_dates(start, end, params.general.rebalance_frequency)
    # Warm-up is provided by the price buffer in load_backtest_data; all in-range
    # rebalance dates are eligible for book construction.

    # Announce the total rebalance count so the UI can show a determinate
    # progress bar during the (potentially long) streaming run.
    total_rebal = len([d for d in rbdates if start <= d <= end])
    if progress_callback is not None:
        try:
            progress_callback({"event": "pipeline_start", "total_rebalances": total_rebal})
        except Exception as exc:
            logger.warning("progress_callback(start) failed: %s", exc)

    # Cap tiers (snapshot proxy) used for segment sizing. Point-in-time factor
    # snapshots are built inside _construct_book per rebalance date.
    cap_tier = data.cap_tier

    # State.
    current_weights: pd.Series = pd.Series(dtype=float)  # ticker -> weight
    cash = 1.0
    nav_series: list[float] = []
    nav_index: list[pd.Timestamp] = []
    trades_rows: list[dict] = []
    snapshots: dict[pd.Timestamp, pd.DataFrame] = {}
    factor_scores: dict[pd.Timestamp, dict[str, pd.Series]] = {}
    pipeline_audit: dict[pd.Timestamp, list[GateResult]] = {}
    momentum_hist: list[tuple[pd.Series, pd.Series]] = []

    cost = (params.general.transaction_cost_pct + params.general.slippage_pct) / 100.0

    # Precompute time-series daily returns (axis=0) once, so each ticker's daily
    # return on date ``dt`` can be looked up directly.
    returns = prices.pct_change()

    prev_weights = pd.Series(dtype=float)
    held_since: dict[str, pd.Timestamp] = {}

    for dt in all_dates:
        if dt < start or dt > end:
            continue
        # Determine regime state at this date.
        state = "invested"
        if not regime_df.empty and dt in regime_df.index:
            state = regime_df.loc[dt, "state"]
        is_rebal = dt in rbdates
        sell_today = (not regime_df.empty and dt in regime_df.index and bool(regime_df.loc[dt, "sell_signal"]))
        buy_today = (not regime_df.empty and dt in regime_df.index and bool(regime_df.loc[dt, "buy_signal"]))

        target = pd.Series(dtype=float)
        if is_rebal and state == "invested":
            target, gate_results = _construct_book(
                data, dt, params, cap_tier, momentum_hist, factor_scores, snapshots,
                pipeline_audit, progress_callback,
            )
            if gate_results is not None:
                pipeline_audit[dt] = gate_results

        # Apply exits on sell signal (move entirely to cash).
        if params.management.exit_on_sell_signal and sell_today:
            target = pd.Series(dtype=float)

        # Entry gating: if flat and no buy signal yet, stay in cash.
        if current_weights.empty and target.empty and not (buy_today or state == "invested"):
            target = pd.Series(dtype=float)

        # Transaction costs on turnover.
        if not target.empty or not current_weights.empty:
            merged = pd.concat([current_weights.rename("old"), target.rename("new")], axis=1).fillna(0.0)
            turnover = float((merged["old"] - merged["new"]).abs().sum() / 2.0)
            # Apply cost to portfolio value at rebalance (handled in NAV below via cash drag).
            cost_drag = turnover * cost
        else:
            cost_drag = 0.0

        # Daily return of current book (drift).
        if not current_weights.empty:
            day_ret = returns.loc[dt] if dt in returns.index else pd.Series(dtype=float)
            w = current_weights.reindex(prices.columns).fillna(0.0)
            port_ret = float((w * day_ret.fillna(0.0)).sum()) if dt in returns.index else 0.0
        else:
            port_ret = 0.0

        # Update NAV.
        if not nav_series:
            nav_val = params.general.initial_capital
        else:
            nav_val = nav_series[-1] * (1.0 + port_ret)
        nav_val *= (1.0 - cost_drag)
        nav_series.append(nav_val)
        nav_index.append(dt)

        # Commit new book after the trading day.
        if not target.empty or sell_today:
            # Record trades.
            for t, w in target.items():
                if t not in current_weights.index or abs(current_weights.get(t, 0.0) - w) > 1e-9:
                    trades_rows.append({
                        "date": dt,
                        "ticker": t,
                        "action": "BUY" if w > 0 else "SELL",
                        "weight": w,
                        "bucket": cap_tier.get(t, "n/a"),
                        "reason": "rebalance_buy" if w > 0 else "rebalance_sell",
                    })
            for t in current_weights.index:
                if t not in target.index:
                    trades_rows.append({
                        "date": dt, "ticker": t, "action": "SELL",
                        "weight": 0.0, "bucket": cap_tier.get(t, "n/a"),
                        "reason": "sell_signal" if sell_today else "rebalance_sell",
                    })
            current_weights = target
            # track holding period
            for t in target.index:
                if t not in held_since:
                    held_since[t] = dt
            for t in list(held_since):
                if t not in target.index:
                    del held_since[t]

    nav = pd.Series(nav_series, index=nav_index, name="nav")
    # Align benchmark to same index.
    bench_nav_aligned = bench_nav.reindex(nav.index).ffill().bfill()
    bench_nav_aligned = bench_nav_aligned / bench_nav_aligned.iloc[0] * params.general.initial_capital

    metrics = compute_all_metrics(nav, bench_nav_aligned)

    trades = pd.DataFrame(trades_rows)
    result = BacktestResult(
        nav=nav, benchmark_nav=bench_nav_aligned, regime=regime_df,
        trades=trades, snapshots=snapshots, factor_scores=factor_scores,
        metrics=metrics, params=params, pipeline_audit=pipeline_audit,
    )
    if progress_callback is not None:
        try:
            progress_callback({"event": "done", "result": result})
        except Exception as exc:  # callback must never break the run
            logger.warning("progress_callback(done) failed: %s", exc)
    return result


def _construct_book(
    data: BacktestData,
    date: pd.Timestamp,
    params: BacktestParameters,
    cap_tier: pd.Series,
    momentum_hist: list,
    factor_scores: dict,
    snapshots: dict,
    pipeline_audit: dict,
    progress_callback: Callable[[dict], None] | None = None,
) -> tuple[pd.Series, list[GateResult] | None]:
    """Build the target weight vector for one rebalance date (per cap bucket).

    Runs the user-ordered, registry-driven pipeline of gates (each a
    :class:`~core.backtesting.gate_registry.GateNode`) over a shared
    :class:`PipelineContext`. Each gate narrows the universe and publishes its
    scores; the engine then combines the surviving scores into the final ARQM
    score, segments by cap tier and sizes positions. Every gate's execution is
    recorded into a :class:`GateResult` list (the per-rebalance audit trail) and,
    when a ``progress_callback`` is supplied, emitted immediately for progressive
    UI rendering.
    """
    # Point-in-time factor snapshots as-of the rebalance date. Eligibility then
    # requires a stock to actually *have* quality / momentum / low-vol data
    # available on this date, keeping all three windows aligned per rebalance.
    asof_market = _asof_market_features(data.market_features, list(data.universe_tickers), date)
    asof_quality = _asof_quality_features(data.quality_ts, list(data.universe_tickers), date)

    ctx = PipelineContext(
        date=date, params=params, prices=data.prices, cap_tier=cap_tier,
        market_features=asof_market, quality=asof_quality,
        eligible=list(data.universe_tickers), momentum_history=momentum_hist,
    )

    # Ordered, enabled gates from the user pipeline config.
    gate_specs = sorted(
        (g for g in params.pipeline.gates if g.enabled), key=lambda g: g.order
    )
    results: list[GateResult] = []

    for spec in gate_specs:
        node = get_gate(spec.kind)
        res = GateResult(
            kind=spec.kind, label=node.label, order=spec.order,
            enabled=True, status="running", input_universe=list(ctx.eligible),
        )
        if progress_callback is not None:
            try:
                progress_callback({"event": "gate_start", "date": date, "gate": res, "pipeline": results})
            except Exception as exc:
                logger.warning("progress_callback(gate_start) failed: %s", exc)
        t0 = time.perf_counter()
        try:
            out = node.run(ctx)
            res.score = out.score
            res.pillar_scores = out.pillar_scores
            res.selected = out.selected
            res.output_universe = out.output_universe
            res.rejected = out.rejected
            res.logs = out.logs
            res.warnings = out.warnings
            res.status = "completed"
        except Exception as exc:
            res.status = "failed"
            res.error = str(exc)
            logger.exception("Gate '%s' failed at %s", spec.kind, date)
        res.execution_time_s = time.perf_counter() - t0
        results.append(res)

        if progress_callback is not None:
            try:
                progress_callback({"event": "gate_done", "date": date, "gate": res, "pipeline": results})
            except Exception as exc:
                logger.warning("progress_callback(gate_done) failed: %s", exc)

        if res.status == "failed":
            snapshots[date] = pd.DataFrame()
            return pd.Series(dtype=float), results

    if not ctx.eligible:
        snapshots[date] = pd.DataFrame()
        return pd.Series(dtype=float), results

    # Final ARQM score over the surviving (passed-through) universe.
    overall = final_scores(ctx.momentum_score, ctx.stability_score, ctx.quality_score, ctx.eligible, params)
    overall = overall.dropna()
    if overall.empty:
        snapshots[date] = pd.DataFrame()
        return pd.Series(dtype=float), results

    factor_scores[date] = {
        "momentum": ctx.momentum_score.reindex(overall.index),
        "stability": ctx.stability_score.reindex(overall.index),
        "quality": ctx.quality_score.reindex(overall.index),
        "overall": overall,
    }
    momentum_hist.append((ctx.momentum_score, ctx.stability_score))

    # Cap segmentation. Renormalize bucket weights to the buckets that actually
    # contain stocks.
    weights = pd.Series(dtype=float)
    cap_weights = params.cap_segment.weights
    sizes = {
        "large": params.portfolio.large_size,
        "mid": params.portfolio.mid_size,
        "small": params.portfolio.small_size,
    }
    present = {
        b: [t for t in overall.index if cap_tier.get(t) == b]
        for b in cap_weights
    }
    present_buckets = {b: ts for b, ts in present.items() if ts}
    if not present_buckets:
        present_buckets = {"large": list(overall.index)}
    renorm = {b: w for b, w in cap_weights.items() if b in present_buckets}
    renorm_total = sum(renorm.values()) or 1.0
    for bucket, ts in present_buckets.items():
        bw = renorm[bucket] / renorm_total
        bucket_scores = overall.reindex(ts).sort_values(ascending=False)
        pick = bucket_scores.head(sizes[bucket])
        if pick.empty:
            continue
        w = _size_positions(pick, params, ctx.momentum_score, ctx.stability_score, ctx.quality_score)
        w = w * bw
        weights = pd.concat([weights, w])

    weights = weights[weights > 0]
    if weights.empty:
        snapshots[date] = pd.DataFrame()
        return pd.Series(dtype=float), results
    weights = weights.clip(upper=params.portfolio.max_position_pct)
    total = weights.sum()
    if total > 0:
        weights = weights / total

    snap = pd.DataFrame({
        "ticker": overall.index,
        "bucket": [cap_tier.get(t, "n/a") for t in overall.index],
        "momentum": ctx.momentum_score.reindex(overall.index).values,
        "stability": ctx.stability_score.reindex(overall.index).values,
        "quality": ctx.quality_score.reindex(overall.index).values,
        "overall": overall.values,
    }).sort_values("overall", ascending=False)
    snapshots[date] = snap
    return weights, results


def _size_positions(pick: pd.Series, params: BacktestParameters,
                    mscore: pd.Series, sscore: pd.Series, qscore: pd.Series) -> pd.Series:
    """Position sizing within a bucket: equal / score / hybrid."""
    method = params.portfolio.sizing_method
    if method == "equal":
        w = pd.Series(1.0 / len(pick), index=pick.index)
    elif method == "score":
        s = pick.copy()
        w = s / s.sum() if s.sum() > 0 else pd.Series(1.0 / len(pick), index=pick.index)
    else:  # hybrid
        hw = params.portfolio
        mat = pd.concat([
            mscore.reindex(pick.index),
            qscore.reindex(pick.index),
            sscore.reindex(pick.index),
        ], axis=1).fillna(0.0)
        mat.columns = ["m", "q", "s"]
        wts = pd.Series({
            "m": hw.hybrid_momentum_weight,
            "q": hw.hybrid_quality_weight,
            "s": hw.hybrid_stability_weight,
        })
        combo = (mat * wts).sum(axis=1)
        w = combo / combo.sum() if combo.sum() > 0 else pd.Series(1.0 / len(pick), index=pick.index)
    return w


def _asof_market_features(market_features: pd.DataFrame, tickers: list[str], date: pd.Timestamp) -> pd.DataFrame:
    """Point-in-time daily market factor snapshot (index=tickers, cols=factors).

    For every eligible ticker, take the latest stored feature_store row whose
    ``date`` is on/before the rebalance date. Each value is, by construction, the
    trailing 252-day rolling-window estimate (momentum / beta / semi_deviation) as
    of that day, so this realises the "rolling window per day" requirement without
    recomputing anything inside the backtest loop.
    """
    if market_features is None or market_features.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    # Defensive: the point-in-time snapshot requires a "ticker" key column. If the
    # stored feature panel is missing it (e.g. a partially-regenerated / legacy
    # feature store), degrade to an empty snapshot instead of raising
    # KeyError('ticker') inside the rebalance loop (which would crash the run).
    if "ticker" not in market_features.columns or "date" not in market_features.columns:
        logger.warning(
            "market_features missing required 'ticker'/'date' columns "
            "(found: %s); point-in-time market factors unavailable for this run.",
            list(market_features.columns),
        )
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    mf = market_features[market_features["date"] <= pd.Timestamp(date)]
    mf = mf[mf["ticker"].isin(tickers)]
    if mf.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    idx = mf.groupby("ticker")["date"].idxmax()
    out = mf.loc[idx].set_index("ticker")
    out = out[[c for c in out.columns if c != "date"]]
    return out.reindex(tickers)


def _asof_quality_features(quality_ts: pd.DataFrame, tickers: list[str], date: pd.Timestamp) -> pd.DataFrame:
    """Point-in-time yearly fundamental quality snapshot (index=tickers, cols=factors).

    For every eligible ticker, take the latest ``financial_year`` vintage whose
    publication can reasonably be assumed available as-of the rebalance date. The
    fundamental store does not carry audit/release dates, so we approximate
    vintage availability by treating a financial year ``FY`` as available from
    ``FY-09-30`` of the following calendar year (Indian corporates file annual
    results ~6 months after the March fiscal year-end). Stocks with no vintage
    available as-of ``date`` get NaN and are excluded by the quality gate.
    """
    if quality_ts is None or quality_ts.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    if "ticker" not in quality_ts.columns or "financial_year" not in quality_ts.columns:
        logger.warning(
            "quality_ts missing required 'ticker'/'financial_year' columns "
            "(found: %s); point-in-time quality factors unavailable for this run.",
            list(quality_ts.columns),
        )
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    q = quality_ts.copy()
    q["available_from"] = pd.to_datetime(q["financial_year"].astype("int64").astype("str") + "-09-30") + pd.DateOffset(years=1)
    q = q[q["available_from"] <= pd.Timestamp(date)]
    q = q[q["ticker"].isin(tickers)]
    if q.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    idx = q.groupby("ticker")["financial_year"].idxmax()
    out = q.loc[idx].set_index("ticker")
    out = out[[c for c in out.columns if c not in ("date", "financial_year", "available_from")]]
    return out.reindex(tickers)
