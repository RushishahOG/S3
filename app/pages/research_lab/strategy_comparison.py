"""Strategy Comparison Research Lab module.

Professional multi-strategy decision-support dashboard. Consumes only cached
backtest / optimization / Monte Carlo results from the strategy repository
(nothing is re-run). See :mod:`core.strategy_comparison` for the analytics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.layouts.base import section
from core.backtesting.export import export_dataframe
from core.strategy_comparison import comparison as sc_engine
from core.strategy_comparison import export as sc_export
from core.strategy_comparison import visualization as sc_viz
from core.strategy_comparison.repository import (
    StrategySource,
    get_strategy_repository,
)

_DIRECTION = {
    "CAGR": "higher", "Annualized Return": "higher", "Volatility": "lower",
    "Sharpe": "higher", "Sortino": "higher", "Calmar": "higher",
    "Max DD": "lower", "Ulcer Index": "lower", "MAR": "higher",
    "Info Ratio": "higher", "Treynor": "higher", "Beta": None,
    "Alpha": "higher", "Win Rate": "higher", "Profit Factor": "higher",
    "Recovery Factor": "higher", "Avg Annual Return": "higher",
    "Median Annual Return": "higher", "Worst Year": "higher",
    "Best Year": "higher", "Std Dev": "lower", "Avg Holding Days": None,
    "Turnover": "lower", "Trades": None, "Final Value": "higher",
}

_PCT_COLS = {
    "CAGR", "Annualized Return", "Volatility", "Max DD", "Ulcer Index",
    "MAR", "Info Ratio", "Treynor", "Alpha", "Win Rate", "Avg Annual Return",
    "Median Annual Return", "Worst Year", "Best Year", "Std Dev",
    "Avg Gain", "Avg Loss", "Largest Winner", "Largest Loser", "Expectancy",
    "VaR 5%", "CVaR 5%", "Current DD", "Avg Drawdown",
    "Benchmark CAGR", "Excess CAGR", "Tracking Error", "Up Capture", "Down Capture",
}

_RANK_METRICS = ["CAGR", "Sharpe", "Max DD", "Calmar", "Sortino", "Volatility"]


def _fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_num(x, d: int = 2) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:,.{d}f}"


def _style_performance(df: pd.DataFrame):
    fmt = {}
    for c in df.columns:
        fmt[c] = "{:.2%}" if c in _PCT_COLS else "{:.2f}"
    styled = df.style.format(fmt)

    def _highlight(col):
        dirn = _DIRECTION.get(col.name)
        if dirn is None:
            return [""] * len(col)
        s = col.replace([np.inf, -np.inf], np.nan)
        if s.isna().all():
            return [""] * len(col)
        mx, mn = s.max(), s.min()
        out = []
        for v in s:
            if pd.isna(v):
                out.append("")
            elif dirn == "higher" and v == mx:
                out.append("background-color: #c6efce")
            elif dirn == "higher" and v == mn:
                out.append("background-color: #ffc7ce")
            elif dirn == "lower" and v == mn:
                out.append("background-color: #c6efce")
            elif dirn == "lower" and v == mx:
                out.append("background-color: #ffc7ce")
            else:
                out.append("")
        return out

    return styled.apply(_highlight, axis=0)


def _build_comparison(selected_records, weights) -> sc_engine.ComparisonResult:
    sig = tuple(sorted(r.strategy_id for r in selected_records))
    cache = st.session_state.setdefault("sc_cache", {})
    if sig in cache and weights is None:
        return cache[sig]
    result = sc_engine.compare_strategies(selected_records, ranking_weights=weights)
    cache[sig] = result
    return result


def render() -> None:
    """Render the Strategy Comparison section."""
    section("Strategy Comparison")
    st.caption(
        "Compare completed strategies (Manual Backtest, Parameter Optimizer, Monte Carlo) "
        "side-by-side. All analytics use cached results only — no backtests are re-run."
    )

    repo = get_strategy_repository()
    records = repo.list_all()
    if not records:
        st.info(
            "No strategies available yet. Run a **Manual Backtest**, complete a "
            "**Parameter Optimization**, or finish a **Monte Carlo** simulation — each "
            "is automatically added to the strategy repository."
        )
        return

    selected = _render_selection(records)
    if not selected:
        st.info("Select one or more strategies (up to 10) to begin the comparison.")
        return

    weights = st.session_state.get("sc_rank_weights")
    result = _build_comparison(selected, weights)

    tabs = st.tabs([
        "Configuration", "Performance", "Charts", "Risk Analytics",
        "Holdings", "Ranking", "Recommendations", "Export",
    ])
    with tabs[0]:
        _render_config(result)
    with tabs[1]:
        _render_performance(result)
    with tabs[2]:
        _render_charts(result)
    with tabs[3]:
        _render_risk(result)
    with tabs[4]:
        _render_holdings(result)
    with tabs[5]:
        _render_ranking(result, selected)
    with tabs[6]:
        _render_recommendations(result)
    with tabs[7]:
        _render_export(result)


# --- Section 1: Selection ---------------------------------------------------


def _render_selection(records) -> list:
    section("1. Strategy Selection")
    sort_by = st.selectbox(
        "Sort by", ["CAGR", "Sharpe", "Max Drawdown", "Calmar", "Date", "Source"],
        key="sc_sort",
    )
    search = st.text_input("Search strategy", key="sc_search")

    def _key(r):
        m = r.metrics
        return {
            "CAGR": m.get("annual_return", 0),
            "Sharpe": m.get("sharpe", 0),
            "Max Drawdown": m.get("max_drawdown", 0),
            "Calmar": m.get("calmar", 0),
            "Date": r.created_at,
            "Source": r.source.value,
        }[sort_by]

    filtered = [r for r in records if search.lower() in r.name.lower()]
    filtered.sort(key=_key, reverse=(sort_by != "Source"))

    options = {f"{r.name}  ·  {r.source.value}": r.strategy_id for r in filtered}
    opt_labels = list(options.keys())

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Select All (top 10)", key="sc_selall"):
            st.session_state["sc_multi"] = opt_labels[:10]
            st.rerun()
    with c2:
        if st.button("Clear All", key="sc_clear"):
            st.session_state["sc_multi"] = []
            st.rerun()

    chosen = st.multiselect(
        "Select strategies to compare (max 10)",
        opt_labels, max_selections=10, key="sc_multi",
        help="Check up to 10 strategies. Use search and sort to find them quickly.",
    )
    sel_ids = [options[l] for l in chosen]
    by_id = {r.strategy_id: r for r in records}
    selected = [by_id[i] for i in sel_ids if i in by_id]

    st.caption(f"{len(selected)} strategy(ies) selected.")
    return selected


# --- Section 2: Configuration ----------------------------------------------


def _render_config(result: sc_engine.ComparisonResult) -> None:
    section("2. Configuration Comparison")
    cfg = result.config_comparison
    if cfg is None or cfg.empty:
        st.info("No configuration data available for the selected strategies.")
        return

    styled = cfg.style.format(
        lambda v: f"{v:.2%}" if isinstance(v, float) and "Weight" in str(v) or "Cash" in str(v) else v
    )
    # Highlight cells that differ across strategies.
    def _diff(col):
        vals = col.dropna().unique()
        if len(vals) <= 1:
            return [""] * len(col)
        return ["background-color: #fff2cc" if not pd.isna(v) else "" for v in col]

    styled = styled.apply(_diff, axis=0)
    st.dataframe(styled, use_container_width=True, height=420)
    st.caption("Highlighted cells differ across the selected strategies.")


# --- Section 3: Performance -------------------------------------------------


def _render_performance(result: sc_engine.ComparisonResult) -> None:
    section("3. Performance Summary")
    perf = result.performance_table
    st.dataframe(_style_performance(perf), use_container_width=True, height=460)
    st.caption("Green = best, Red = worst per metric (direction-aware).")


# --- Section 4: Charts ------------------------------------------------------


def _render_charts(result: sc_engine.ComparisonResult) -> None:
    section("4. Visual Analytics")

    c1, c2 = st.columns(2)
    with c1:
        normalize = st.checkbox("Normalize equity (start = 100)", value=True, key="sc_norm")
    with c2:
        show_bench = st.checkbox("Overlay benchmark", value=False, key="sc_bench")

    fig = sc_viz.equity_curve_comparison(result, normalize=normalize)
    if show_bench:
        for s in result.strategies:
            if s.benchmark is not None and not s.benchmark.empty:
                b = s.benchmark
                if normalize and b.iloc[0] != 0:
                    b = b / b.iloc[0] * 100
                fig.add_trace(go.Scatter(x=b.index, y=b.values, name=f"{s.name} (Bench)",
                                        line=dict(width=1, dash="dot")))
    st.plotly_chart(fig, use_container_width=True)

    st.plotly_chart(sc_viz.drawdown_comparison(result), use_container_width=True)

    window = st.selectbox("Rolling window", [252, 756, 1260],
                          format_func=lambda w: {252: "1 Year", 756: "3 Year", 1260: "5 Year"}[w],
                          key="sc_roll")
    st.plotly_chart(_rolling_figure(result, window), use_container_width=True)

    st.plotly_chart(sc_viz.annual_return_heatmap(result), use_container_width=True)
    st.plotly_chart(sc_viz.monthly_return_heatmap(result), use_container_width=True)
    st.plotly_chart(sc_viz.risk_return_scatter(result), use_container_width=True)
    st.plotly_chart(sc_viz.efficient_frontier(result), use_container_width=True)
    st.plotly_chart(sc_viz.correlation_heatmap(result), use_container_width=True)
    st.plotly_chart(sc_viz.radar_chart(result), use_container_width=True)


def _rolling_figure(result: sc_engine.ComparisonResult, window: int):
    import plotly.graph_objects as go
    from core.strategy_comparison.comparison import TRADING_DAYS

    fig = go.Figure()
    for name in result.equity_curves.columns:
        eq = result.equity_curves[name].dropna()
        if len(eq) < window:
            continue
        ret = eq.pct_change().fillna(0)
        roll = ret.rolling(window).apply(
            lambda x: np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1 if len(x) else 0)
        roll = roll.dropna()
        if not roll.empty:
            fig.add_trace(go.Scatter(x=roll.index, y=roll.values, mode="lines", name=name))
    fig.update_layout(
        title=f"Rolling Annualized Returns ({window // 252}-Year Window)",
        xaxis_title="Date", yaxis_title="Rolling CAGR",
        template="plotly_white", hovermode="x unified",
    )
    return fig


# --- Section 5: Risk Analytics ----------------------------------------------


def _render_risk(result: sc_engine.ComparisonResult) -> None:
    section("5. Risk Analytics")
    st.dataframe(_style_performance(result.risk_table), use_container_width=True, height=360)
    st.plotly_chart(sc_viz.risk_analytics_chart(result), use_container_width=True)
    st.plotly_chart(sc_viz.benchmark_comparison_chart(result), use_container_width=True)
    if result.benchmark_df is not None and not result.benchmark_df.empty:
        st.subheader("Benchmark Comparison Table")
        st.dataframe(_style_performance(result.benchmark_df), use_container_width=True, height=240)
    st.subheader("Statistical Significance Tests")
    _render_stats_tests(result)


def _render_stats_tests(result: sc_engine.ComparisonResult) -> None:
    stats = result.stats_tests
    if not stats:
        st.info("Statistical tests require benchmark data for each strategy.")
        return
    rows = []
    for name, d in stats.items():
        ci = d.get("bootstrap_ci", (np.nan, np.nan))
        rows.append({
            "Strategy": name,
            "Excess CAGR": d.get("excess_cagr"),
            "Paired t": d.get("paired_t"),
            "p-value": d.get("paired_p"),
            "Bootstrap CI (lo)": ci[0],
            "Bootstrap CI (hi)": ci[1],
            "Outperf. Freq": d.get("outperformance_frequency"),
            "Rolling Outperf %": d.get("rolling_outperformance_pct"),
        })
    df = pd.DataFrame(rows).set_index("Strategy")
    styled = df.style.format("{:.2%}", subset=["Excess CAGR", "Outperf. Freq", "Rolling Outperf %"])
    styled = styled.format("{:.2f}", subset=["Paired t", "p-value", "Bootstrap CI (lo)", "Bootstrap CI (hi)"])
    st.dataframe(styled, use_container_width=True)


# --- Section 6: Holdings ----------------------------------------------------


def _render_holdings(result: sc_engine.ComparisonResult) -> None:
    section("6. Holdings Comparison")
    st.plotly_chart(sc_viz.holdings_overlap_heatmap(result), use_container_width=True)
    st.plotly_chart(sc_viz.holdings_comparison(result), use_container_width=True)
    if result.trade_df is not None and not result.trade_df.empty:
        st.subheader("Trade Analysis")
        st.dataframe(_style_performance(result.trade_df), use_container_width=True, height=320)
        st.plotly_chart(sc_viz.trade_analysis_chart(result), use_container_width=True)


# --- Section 7: Ranking -----------------------------------------------------


def _render_ranking(result: sc_engine.ComparisonResult, selected) -> None:
    section("7. Ranking Engine")
    st.markdown("Set weights for the composite score (normalized 0–1 per metric).")
    cols = st.columns(len(_RANK_METRICS))
    weights = {}
    current = st.session_state.get("sc_rank_weights") or {
        "CAGR": 0.25, "Sharpe": 0.20, "Max DD": 0.20, "Calmar": 0.15, "Sortino": 0.15, "Volatility": 0.05
    }
    for i, m in enumerate(_RANK_METRICS):
        with cols[i]:
            weights[m] = st.slider(m, 0.0, 1.0, float(current.get(m, 0.1)), 0.05, key=f"sc_w_{m}")
    total = sum(weights.values())
    if total <= 0:
        st.warning("Sum of weights must be > 0.")
        return
    if st.button("Apply Weights & Recompute", key="sc_apply"):
        norm = {k: v / total for k, v in weights.items()}
        st.session_state["sc_rank_weights"] = norm
        _build_comparison(selected, norm)
        st.rerun()

    st.dataframe(result.rankings.style.format("{:.3f}"), use_container_width=True, height=360)
    st.caption("Higher composite = better. Each sub-score is min-max normalized across strategies.")
    st.plotly_chart(sc_viz.radar_chart(result), use_container_width=True)
    st.plotly_chart(sc_viz.efficient_frontier(result), use_container_width=True)


# --- Section 8: Recommendations ---------------------------------------------


def _render_recommendations(result: sc_engine.ComparisonResult) -> None:
    section("8. Recommendation Engine")
    recs = result.recommendations
    if not recs:
        st.info("Not enough data to generate recommendations.")
        return
    cols = st.columns(2)
    for i, (k, v) in enumerate(recs.items()):
        with cols[i % 2]:
            title = k.replace("_", " ").title()
            if isinstance(v, dict) and "strategy" in v:
                strat = v.get("strategy")
                reason = v.get("reason", "")
                st.success(f"**{title}**\n\n{strat}\n\n_{reason}_")
            else:
                st.info(f"**{title}**: {v}")


# --- Section 9: Export ------------------------------------------------------


def _render_export(result: sc_engine.ComparisonResult) -> None:
    section("9. Export")
    base = "strategy_comparison"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Performance (CSV)", export_dataframe(result.performance_table, "csv"),
                           f"{base}_performance.csv", "text/csv", key="sc_dl_perf")
        st.download_button("Configuration (CSV)", export_dataframe(result.config_comparison, "csv"),
                           f"{base}_config.csv", "text/csv", key="sc_dl_cfg")
    with c2:
        st.download_button("Risk (CSV)", export_dataframe(result.risk_table, "csv"),
                           f"{base}_risk.csv", "text/csv", key="sc_dl_risk")
        st.download_button("Trade (CSV)", export_dataframe(result.trade_df, "csv"),
                           f"{base}_trade.csv", "text/csv", key="sc_dl_trade")
    with c3:
        st.download_button("Allocation (CSV)", export_dataframe(result.allocation_df, "csv"),
                           f"{base}_allocation.csv", "text/csv", key="sc_dl_alloc")
        st.download_button("Full (JSON)", sc_export.export_json(result),
                           f"{base}.json", "application/json", key="sc_dl_json")
    with c4:
        st.download_button("Full (Excel)", sc_export.export_excel(result),
                           f"{base}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="sc_dl_xlsx")
        st.download_button("Report (PDF)", sc_export.export_pdf(result),
                           f"{base}.pdf", "application/pdf", key="sc_dl_pdf")


if __name__ == "__main__":
    render()
