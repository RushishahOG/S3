"""Monte Carlo Simulation Research Lab module.

Professional Monte Carlo engine that resamples the *actual* historical return /
trade sequence of a completed ARQM backtest. The UI is split into four areas:

1. Configuration  - methodology, size, seed, horizon, block size, parallelism
2. Progress       - live progress bar, ETA, speed and cancellation
3. Results        - KPI dashboard, probabilities, risk stats, comparison
4. Visual Analytics + Export

Heavy computation runs in a background thread (see :mod:`core.monte_carlo.runner`)
so the UI never freezes and long runs can be cancelled.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

from app.layouts.base import section
from app.pages.backtest.state import StrategyStatus, get_backtest_state
from core.monte_carlo import (
    METHOD_BLOCK_BOOTSTRAP,
    METHOD_LABELS,
    METHOD_REGIME_BOOTSTRAP,
    METHOD_TRADE_RANDOMIZATION,
    MonteCarloRunner,
    SimulationConfig,
    build_mc_input,
    get_runner_bucket,
)
from core.monte_carlo import export as mc_export
from core.monte_carlo import plotting as mc_plot
from core.monte_carlo.types import (
    BLOCK_SIZES,
    MIN_DAILY_POINTS,
    N_SIMULATION_CHOICES,
)


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:,.{d}f}"


def _fmt_eta(s) -> str:
    if s is None:
        return "—"
    return f"{s:.0f}s"


def _validate(inp, method: str) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    if len(inp.returns) < MIN_DAILY_POINTS:
        msgs.append(
            f"Need at least {MIN_DAILY_POINTS} daily returns to simulate; "
            f"this backtest has {len(inp.returns)}."
        )
        ok = False
    if method == METHOD_TRADE_RANDOMIZATION and inp.n_trades == 0:
        msgs.append("Trade Sequence Randomization requires a non-empty trade log.")
        ok = False
    return ok, msgs


def render() -> None:
    """Render the Monte Carlo Simulation section."""
    section("Monte Carlo Simulation")
    st.caption(
        "Resamples the actual historical return & trade sequence of a completed "
        "backtest to assess robustness, risk and the distribution of outcomes."
    )

    state = get_backtest_state()
    completed = state.get_completed()
    successful = [e for e in completed if e.status == StrategyStatus.COMPLETED and e.result]

    if not successful:
        st.info(
            "Run a **successful backtest** first (Manual Testing → Portfolio Queue → "
            "Results). Monte Carlo simulations use the cached backtest outputs only."
        )
        return

    strategy_options = {
        f"{e.config.name} (ID: {e.strategy_id})": e for e in reversed(successful)
    }
    selected_label = st.selectbox(
        "Select backtest to simulate", list(strategy_options.keys()), key="mc_strategy"
    )
    exec_obj = strategy_options[selected_label]
    result = exec_obj.result

    try:
        inp = build_mc_input(result)
    except ValueError as exc:
        st.error(f"Cannot build simulation inputs: {exc}")
        return

    config = _render_config(inp)
    _handle_run(config, inp)

    _render_progress()

    if not st.session_state.get("mc_run_id"):
        res = st.session_state.get("mc_result")
        if res is not None:
            _render_results(res, result, exec_obj)


# --- Section 1: Configuration ------------------------------------------------


def _render_config(inp) -> SimulationConfig | None:
    section("1. Configuration")
    method = st.selectbox(
        "Simulation Methodology",
        options=list(METHOD_LABELS.keys()),
        format_func=lambda k: METHOD_LABELS[k],
        index=0,
        key="mc_method",
    )
    n_sim = st.selectbox(
        "Number of Simulations", N_SIMULATION_CHOICES, index=3, key="mc_nsim"
    )

    col1, col2 = st.columns(2)
    with col1:
        fix_seed = st.checkbox(
            "Fix random seed (deterministic)", value=True, key="mc_fixseed"
        )
    with col2:
        seed = st.number_input(
            "Seed", min_value=0, value=42, step=1,
            key="mc_seed", disabled=not fix_seed,
        )

    horizon_mode = st.radio(
        "Simulation Horizon",
        ["Original backtest length", "Custom"],
        index=0, key="mc_horizon_mode", horizontal=True,
    )
    horizon = None
    if horizon_mode == "Custom":
        horizon = st.number_input(
            "Horizon (trading days)", min_value=20, max_value=10000,
            value=len(inp.returns), step=1, key="mc_horizon",
        )

    block_size = 10
    if method == METHOD_BLOCK_BOOTSTRAP:
        block_size = st.selectbox(
            "Block Size (days)", BLOCK_SIZES, index=1, key="mc_block"
        )

    parallel = st.checkbox(
        "Parallel processing (recommended for > 1,000 sims, Trade method)",
        value=False, key="mc_parallel",
    )

    if method in (METHOD_REGIME_BOOTSTRAP, METHOD_TRADE_RANDOMIZATION) and horizon is not None:
        st.caption(
            "Note: Trade-Sequence and Regime methods preserve the original return "
            "sequence length, so a custom horizon is ignored."
        )

    ok, msgs = _validate(inp, method)
    for m in msgs:
        st.warning(m)

    config = SimulationConfig(
        method=method,
        n_simulations=int(n_sim),
        seed=(None if not fix_seed else int(seed)),
        horizon=horizon,
        block_size=int(block_size),
        parallel=bool(parallel),
        initial_capital=float(inp.initial_capital),
        rf_annual=0.0,
    )

    if not ok:
        st.error("Resolve the issues above before running the simulation.")
        return None
    return config


def _handle_run(config: SimulationConfig | None, inp) -> None:
    if st.session_state.get("mc_run_id"):
        return
    if st.button("Run Monte Carlo Simulation", key="mc_run", type="primary"):
        if config is None:
            st.error("Cannot run: configuration is invalid.")
            return
        runner = MonteCarloRunner(config, inp)
        run_id = runner.start()
        st.session_state["mc_run_id"] = run_id
        st.rerun()


# --- Section 2: Progress -----------------------------------------------------


def _render_progress() -> None:
    run_id = st.session_state.get("mc_run_id")
    if not run_id:
        return
    bucket = get_runner_bucket(run_id)
    if bucket is None:
        st.session_state["mc_run_id"] = None
        return

    if bucket.get("error"):
        st.error("Simulation failed:\n" + bucket["error"])
        from core.monte_carlo.runner import clear_runner
        clear_runner(run_id)
        st.session_state["mc_run_id"] = None
        return

    if not bucket.get("done"):
        section("2. Progress")
        completed = bucket.get("completed", 0)
        total = bucket.get("total", 1)
        pct = completed / max(1, total)
        st.progress(min(1.0, pct))
        st.caption(
            f"{completed:,} / {total:,} simulations · {pct * 100:.1f}% · "
            f"{bucket.get('speed', 0):.0f} sims/s · ETA {_fmt_eta(bucket.get('eta'))}"
        )
        if st.button("Cancel", key="mc_cancel"):
            bucket["cancel"] = True
            st.session_state["mc_run_id"] = None
            from core.monte_carlo.runner import clear_runner
            clear_runner(run_id)
            st.warning("Simulation cancelled.")
            time.sleep(0.3)
            st.rerun()
            return
        time.sleep(0.4)
        st.rerun()
        return

    result = bucket.get("result")
    from core.monte_carlo.runner import clear_runner
    clear_runner(run_id)
    st.session_state["mc_run_id"] = None
    if result is None:
        st.warning("Simulation was cancelled.")
        return
    st.session_state["mc_result"] = result
    st.success(
        f"Completed {result.n_simulations:,} simulations "
        f"({METHOD_LABELS.get(result.method, result.method)})."
    )
    st.rerun()


# --- Section 3: Results Dashboard --------------------------------------------


def _metric_grid(pairs: list[tuple[str, str]]) -> None:
    n = len(pairs)
    cols = st.columns(min(4, n))
    for i, (label, val) in enumerate(pairs):
        cols[i % 4].metric(label, val)


def _render_dashboard(res) -> None:
    section("2. Results Dashboard")
    r = res.risk_summary
    kpis = [
        ("P(Profit)", _fmt_pct(r["probability_of_profit"])),
        ("P(Loss)", _fmt_pct(r["probability_of_loss"])),
        ("Median CAGR", _fmt_pct(r["median_cagr"])),
        ("Expected CAGR", _fmt_pct(r["expected_cagr"])),
        ("Expected Final", _fmt_num(r["expected_final_portfolio"], 0)),
        ("Expected Sharpe", _fmt_num(r["expected_sharpe"])),
        ("Expected Sortino", _fmt_num(r["expected_sortino"])),
        ("Worst Drawdown", _fmt_pct(r["worst_drawdown"])),
        ("VaR 95%", _fmt_pct(r["var_95"])),
        ("CVaR 95%", _fmt_pct(r["cvar_95"])),
    ]
    _metric_grid(kpis)
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Probability Metrics")
        pdf = pd.DataFrame(
            list(res.probabilities.items()), columns=["Condition", "Probability"]
        )
        pdf["Probability"] = pdf["Probability"].map(_fmt_pct)
        st.dataframe(pdf, use_container_width=True, hide_index=True)
    with c2:
        st.subheader("Risk Statistics")
        rdf = pd.DataFrame(list(r.items()), columns=["Metric", "Value"])
        rdf["Value"] = rdf["Value"].map(
            lambda v: _fmt_num(v, 4) if isinstance(v, (int, float)) else str(v)
        )
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    st.subheader("Confidence Intervals")
    rows = []
    for m, v in res.confidence_intervals.items():
        rows.append([
            m,
            f"[{_fmt_num(v['ci95'][0])}, {_fmt_num(v['ci95'][1])}]",
            f"[{_fmt_num(v['ci99'][0])}, {_fmt_num(v['ci99'][1])}]",
        ])
    cidf = pd.DataFrame(rows, columns=["Metric", "95% CI", "99% CI"])
    st.dataframe(cidf, use_container_width=True, hide_index=True)


def _render_comparison(res) -> None:
    section("3. Simulation Comparison")
    orig = res.original_metrics
    orig_row = {
        "cagr": orig.get("annual_return"),
        "sharpe": orig.get("sharpe"),
        "sortino": orig.get("sortino"),
        "calmar": orig.get("calmar"),
        "max_drawdown": orig.get("max_drawdown"),
        "final_value": float(res.original_equity[-1]) if len(res.original_equity) else None,
    }
    agg = res.aggregate

    def g(metric: str, stat: str):
        return agg.loc[metric, stat] if metric in agg.index else np.nan

    rows = []
    for metric in ["cagr", "sharpe", "sortino", "calmar", "max_drawdown", "final_value"]:
        rows.append({
            "Metric": metric,
            "Original": orig_row.get(metric),
            "Mean": g(metric, "mean"),
            "Median": g(metric, "median"),
            "Worst 5%": g(metric, "p5"),
            "Best 5%": g(metric, "p95"),
        })
    comp = pd.DataFrame(rows)

    def fmt_row(r) -> dict:
        out = {}
        for k, v in r.items():
            if k == "Metric":
                out[k] = v
            elif k in ("cagr", "max_drawdown"):
                out[k] = _fmt_pct(v)
            else:
                out[k] = _fmt_num(v, 2)
        return out

    disp = comp.apply(fmt_row, axis=1, result_type="expand")
    st.dataframe(disp, use_container_width=True, hide_index=True)


# --- Section 4: Visual Analytics ---------------------------------------------


def _render_analytics(res) -> None:
    section("4. Visual Analytics")
    st.plotly_chart(mc_plot.fan_chart(res), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            mc_plot.distribution_fig(res.metrics_df["final_value"], "Final Portfolio Distribution", "Final Value"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            mc_plot.distribution_fig(res.metrics_df["cagr"], "CAGR Distribution", "CAGR"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            mc_plot.distribution_fig(
                res.metrics_df["max_drawdown"], "Max Drawdown Distribution", "Max Drawdown",
                risk_zones=[(-0.5, -0.2, "rgba(255,0,0,0.15)"), (-1.0, -0.5, "rgba(255,0,0,0.30)")],
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            mc_plot.distribution_fig(res.metrics_df["sharpe"], "Sharpe Distribution", "Sharpe"),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            mc_plot.scatter_fig(
                res.metrics_df["cagr"], res.metrics_df["max_drawdown"],
                "CAGR vs Max Drawdown", "CAGR", "Max Drawdown",
            ),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            mc_plot.scatter_fig(
                res.metrics_df["sharpe"], res.metrics_df["annual_volatility"],
                "Sharpe vs Volatility", "Sharpe", "Volatility",
            ),
            use_container_width=True,
        )

    st.plotly_chart(mc_plot.violin_fig(res), use_container_width=True)
    st.plotly_chart(mc_plot.box_fig(res), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(mc_plot.curve_sample_fig(res, "worst"), use_container_width=True)
    with c2:
        st.plotly_chart(mc_plot.curve_sample_fig(res, "best"), use_container_width=True)
    with c3:
        st.plotly_chart(mc_plot.curve_sample_fig(res, "random"), use_container_width=True)


# --- Section 5: Export -------------------------------------------------------


def _render_export(res) -> None:
    section("5. Export")
    base = f"mc_{res.method}_{res.n_simulations}"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Metrics (CSV)", mc_export.export_metrics_csv(res),
            f"{base}_metrics.csv", "text/csv", key="mc_dl_metrics",
        )
        st.download_button(
            "Equity Curves (CSV)", mc_export.export_equity_csv(res),
            f"{base}_equity.csv", "text/csv", key="mc_dl_equity",
        )
    with c2:
        st.download_button(
            "Summary (CSV)", mc_export.export_summary_csv(res),
            f"{base}_summary.csv", "text/csv", key="mc_dl_summary",
        )
        st.download_button(
            "Trade Stats (CSV)", mc_export.export_trade_stats_csv(res),
            f"{base}_tradestats.csv", "text/csv", key="mc_dl_tradestats",
        )
    with c3:
        st.download_button(
            "Full (JSON)", mc_export.export_json(res),
            f"{base}.json", "application/json", key="mc_dl_json",
        )
        st.download_button(
            "Full (Excel)", mc_export.export_excel(res),
            f"{base}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="mc_dl_excel",
        )
    with c4:
        st.download_button(
            "Report (PDF)", mc_export.export_pdf(res),
            f"{base}.pdf", "application/pdf", key="mc_dl_pdf",
        )
        if st.button("Clear Cached Result", key="mc_clear"):
            st.session_state.pop("mc_result", None)
            st.rerun()


def _render_results(res, base_result, exec_obj) -> None:
    r = res.config
    st.caption(
        f"Settings: {METHOD_LABELS.get(res.method, res.method)} · "
        f"{res.n_simulations:,} sims · horizon {res.horizon_used} days · "
        f"seed {res.seed}"
    )
    if st.button("Add base strategy to Strategy Comparison", key="mc_add_repo"):
        from core.strategy_comparison.repository import get_strategy_repository
        get_strategy_repository().add_monte_carlo_result(
            base_result, res, exec_obj.strategy_id, exec_obj.config.name,
        )
        st.success("Base strategy registered in the Strategy Comparison repository "
                   "(source: Monte Carlo Simulation).")
    _render_dashboard(res)
    st.divider()
    _render_comparison(res)
    st.divider()
    _render_analytics(res)
    st.divider()
    _render_export(res)


if __name__ == "__main__":
    render()
