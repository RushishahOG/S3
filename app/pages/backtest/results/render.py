"""Results - Backtest Analysis Section.

This module displays completed backtest results with full analytical
reports including equity curves, metrics, trade logs, pipeline trace, etc.
"""

from __future__ import annotations

import io
import pandas as pd
import streamlit as st

from app.layouts.base import page_header, section
from app.pages.backtest.state import get_backtest_state, StrategyStatus
from app.components.logs import render_log_panel
from core.backtesting.export import export_dataframe


def render_results() -> None:
    """Render the Results section with completed backtest reports."""
    page_header("Results", "Completed Backtest Analysis & Reports")

    state = get_backtest_state()
    completed = state.get_completed()

    if not completed:
        st.info("No completed backtests yet. Submit strategies from **Manual Testing** and monitor in **Portfolio Queue**.")
        return

    # Filter only completed (successful) executions
    successful = [e for e in completed if e.status == StrategyStatus.COMPLETED and e.result]

    if not successful:
        st.info("No successful backtests to display.")
        return

    # Strategy selector
    st.subheader("Select Strategy to Analyze")
    strategy_options = {f"{e.config.name} (ID: {e.strategy_id})": e.strategy_id for e in reversed(successful)}
    selected_label = st.selectbox("Choose a completed backtest", list(strategy_options.keys()))
    selected_id = strategy_options[selected_label]

    exec_obj = state.get_completed_execution(selected_id)
    if not exec_obj or not exec_obj.result:
        st.error("Selected result not found")
        return

    result = exec_obj.result
    params = exec_obj.config.params

    # Render the full results report
    _render_full_report(result, params, exec_obj.config.name)

    render_log_panel()


def _render_full_report(result, params, strategy_name: str) -> None:
    """Render the complete backtest report with all tabs."""
    nav = result.nav
    bench = result.benchmark_nav

    tabs = st.tabs([
        "Overview", "Equity Curve", "Market Regime", "Performance Metrics",
        "Portfolio Allocation", "Holdings",
        "Trade Log", "Pipeline Trace", "Rebalance Snapshots",
        "Factor Attribution", "Explainability", "Risk Metrics", "Export Reports"
    ])

    with tabs[0]:
        _render_overview(result, nav, bench, strategy_name)
    with tabs[1]:
        _render_equity_curve(result, nav, bench)
    with tabs[2]:
        _render_market_regime(result)
    with tabs[3]:
        _render_performance_metrics(result)
    with tabs[4]:
        _render_allocation(result)
    with tabs[5]:
        _render_holdings(result)
    with tabs[6]:
        _render_trade_log(result)
    with tabs[7]:
        _render_pipeline_trace(result)
    with tabs[8]:
        _render_snapshots(result)
    with tabs[9]:
        _render_attribution(result)
    with tabs[10]:
        _render_explainability(result)
    with tabs[11]:
        _render_risk_metrics(result)
    with tabs[12]:
        _render_exports(result, params, strategy_name)


def _render_overview(result, nav, bench, strategy_name: str) -> None:
    """Render overview summary."""
    m = result.metrics

    section(f"Strategy: {strategy_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Period", f"{nav.index[0].date()} → {nav.index[-1].date()}")
    col2.metric("Total Rebalances", len(result.snapshots) if result.snapshots else 0)
    col3.metric("Total Trades", len(result.trades) if not result.trades.empty else 0)

    st.divider()

    # Key metrics
    section("Key Performance Metrics")
    cols = st.columns(4)
    metrics = [
        ("Total Return", f"{m.get('total_return', float('nan'))*100:.1f}%"),
        ("CAGR", f"{m.get('annual_return', float('nan'))*100:.1f}%"),
        ("Volatility", f"{m.get('annual_volatility', float('nan'))*100:.1f}%"),
        ("Sharpe", f"{m.get('sharpe', float('nan')):.2f}"),
        ("Sortino", f"{m.get('sortino', float('nan')):.2f}"),
        ("Calmar", f"{m.get('calmar', float('nan')):.2f}"),
        ("Max Drawdown", f"{m.get('max_drawdown', float('nan'))*100:.1f}%"),
        ("Beta", f"{m.get('beta', float('nan')):.2f}"),
        ("Alpha (ann)", f"{m.get('alpha_annual', float('nan'))*100:.1f}%"),
        ("Treynor", f"{m.get('treynor', float('nan')):.2f}"),
        ("Info Ratio", f"{m.get('information_ratio', float('nan')):.2f}"),
        ("Ulcer Index", f"{m.get('ulcer_index', float('nan')):.1f}"),
    ]
    for i, (k, v) in enumerate(metrics):
        cols[i % 4].metric(k, v)

    # Benchmark comparison
    section("Benchmark Comparison")
    cmp_df = pd.DataFrame({"Portfolio": nav, "Benchmark": bench}).dropna()
    st.line_chart(cmp_df)


def _render_equity_curve(result, nav, bench) -> None:
    """Render equity curve and drawdown."""
    section("Portfolio Equity Curve vs Benchmark")
    eq = pd.DataFrame({"Portfolio": nav, "Benchmark": bench})
    st.line_chart(eq)

    section("Drawdown Curve")
    dd = (eq["Portfolio"] / eq["Portfolio"].cummax() - 1.0) * 100.0
    st.area_chart(dd.rename("Drawdown %"))


def _render_market_regime(result) -> None:
    """Render market regime timeline."""
    section("Market Regime Timeline")
    reg = result.regime
    if not reg.empty:
        sig = reg.copy()
        sig["regime_state"] = (sig["state"] == "invested").astype(int)
        st.line_chart(sig[["close", "swing_low", "peak"]])
        st.line_chart(sig[["regime_state", "buy_signal", "sell_signal"]])
    else:
        st.info("No regime data available")


def _render_performance_metrics(result) -> None:
    """Render detailed performance metrics."""
    section("Detailed Performance Metrics")
    m = result.metrics

    # Group metrics
    return_metrics = {k: v for k, v in m.items() if 'return' in k.lower()}
    risk_metrics = {k: v for k, v in m.items() if any(x in k.lower() for x in ['vol', 'dd', 'drawdown', 'ulcer', 'var', 'es'])}
    ratio_metrics = {k: v for k, v in m.items() if any(x in k.lower() for x in ['sharpe', 'sortino', 'calmar', 'treynor', 'alpha', 'beta', 'info'])}
    other_metrics = {k: v for k, v in m.items() if k not in return_metrics and k not in risk_metrics and k not in ratio_metrics}

    for title, metrics_dict in [
        ("Return Metrics", return_metrics),
        ("Risk Metrics", risk_metrics),
        ("Ratio Metrics", ratio_metrics),
        ("Other Metrics", other_metrics),
    ]:
        if metrics_dict:
            st.markdown(f"**{title}**")
            cols = st.columns(4)
            for i, (k, v) in enumerate(metrics_dict.items()):
                if isinstance(v, float):
                    v_str = f"{v*100:.2f}%" if 'return' in k.lower() or 'drawdown' in k.lower() or 'alpha' in k.lower() else f"{v:.4f}"
                else:
                    v_str = str(v)
                cols[i % 4].metric(k.replace('_', ' ').title(), v_str)


def _render_allocation(result) -> None:
    """Render capital allocation by bucket."""
    section("Capital Allocation by Bucket (Last Rebalance)")
    if result.snapshots:
        last = next(reversed(result.snapshots.values()))
        if last is not None and not last.empty and "bucket" in last.columns:
            alloc = last.groupby("bucket")["overall"].count().rename("stocks")
            st.bar_chart(alloc)
        else:
            st.caption("No allocation (empty book at last rebalance).")


def _render_holdings(result) -> None:
    """Render current holdings."""
    section("Current Holdings (Last Rebalance)")
    if result.snapshots:
        last = next(reversed(result.snapshots.values()))
        if last is not None and not last.empty:
            cols = ["ticker", "bucket", "momentum", "stability", "quality", "overall"]
            display_cols = [c for c in cols if c in last.columns]
            st.dataframe(last[display_cols].sort_values("overall", ascending=False), use_container_width=True, hide_index=True)
        else:
            st.caption("No holdings data")


def _render_trade_log(result) -> None:
    """Render trade log."""
    section("Trade Log")
    if result.trades.empty:
        st.caption("No trades generated.")
        return
    trades = result.trades.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.date
    st.dataframe(trades, use_container_width=True, hide_index=True)
    st.caption(f"{len(trades)} trades")


def _render_pipeline_trace(result) -> None:
    """Render pipeline trace with per-gate audit."""
    section("Pipeline Trace - Stage-by-Stage Audit")
    audit = result.pipeline_audit
    if not audit:
        st.caption("No pipeline audit recorded.")
        return

    dates = list(audit.keys())
    pick = st.selectbox("Rebalance date", [d.date().isoformat() for d in dates], key="res_trace_date")
    chosen = next(d for d in dates if d.date().isoformat() == pick)
    gates = audit[chosen]
    st.caption(f"Rebalance {chosen.date()} - {len(gates)} gate stages")

    for g in gates:
        status_icon = "✅" if g.status == "completed" else ("❌" if g.status == "failed" else "⏳")
        title = (f"{status_icon} Gate {g.order} - {g.label} "
                 f"({g.status}) - in {len(g.input_universe)} to out {len(g.output_universe)} "
                 f"- filtered {g.n_filtered} - {g.execution_time_s*1000:.1f} ms")
        with st.expander(title, expanded=(g.order == 0)):
            c1, c2, c3 = st.columns(3)
            c1.metric("Stocks Entering", len(g.input_universe))
            c2.metric("Stocks Passing", len(g.output_universe))
            c3.metric("Stocks Rejected", g.n_filtered)
            if g.warnings:
                for w in g.warnings:
                    st.warning(w)
            if g.error:
                st.error(g.error)
            if g.score is not None and not g.score.empty:
                top = g.score.dropna().sort_values(ascending=False).head(15)
                st.bar_chart(top.rename("score"))
                if st.button(f"Export {g.kind} ranking CSV", key=f"res_exp_{g.kind}_{chosen:%Y%m%d}"):
                    df = g.score.rename("score").to_frame().sort_values("score", ascending=False)
                    st.download_button("Download", export_dataframe(df, "csv"),
                                       f"{g.kind}_{chosen:%Y%m%d}.csv", "text/csv",
                                       key=f"res_dl_{g.kind}_{chosen:%Y%m%d}")
            if g.logs:
                with st.expander("Logs", expanded=False):
                    for line in g.logs:
                        st.caption(line)

    # Full audit export
    rows = []
    for d, gs in audit.items():
        for g in gs:
            rows.append({
                "rebalance_date": d.date().isoformat(),
                "order": g.order, "gate": g.kind, "label": g.label,
                "status": g.status, "input_n": len(g.input_universe),
                "output_n": len(g.output_universe), "filtered": g.n_filtered,
                "exec_ms": round(g.execution_time_s * 1000, 2),
                "warnings": "; ".join(g.warnings),
                "error": g.error or "",
            })
    audit_df = pd.DataFrame(rows)
    st.download_button("Export full pipeline audit (CSV)",
                       export_dataframe(audit_df, "csv"), "pipeline_audit.csv", "text/csv")


def _render_snapshots(result) -> None:
    """Render rebalance snapshots."""
    section("Rebalance Snapshots")
    if not result.snapshots:
        st.caption("No snapshots.")
        return
    dates = list(result.snapshots.keys())
    pick = st.selectbox("Rebalance date", [d.date().isoformat() for d in dates], key="res_snap_date")
    chosen = next(d for d in dates if d.date().isoformat() == pick)
    snap = result.snapshots[chosen]
    if snap is None or snap.empty:
        st.caption("Empty book at this rebalance.")
        return
    st.dataframe(snap.sort_values("overall", ascending=False), use_container_width=True, hide_index=True)


def _render_attribution(result) -> None:
    """Render factor attribution."""
    section("Factor Attribution (Latest Rebalance)")
    if not result.factor_scores:
        st.caption("No factor scores.")
        return
    last = next(reversed(result.factor_scores.values()))
    if "overall" in last:
        contrib = last["overall"].rename("overall_score")
        st.bar_chart(contrib.sort_values(ascending=False).head(20))


def _jsonable(v):
    """Coerce a cell to a JSON-friendly value."""
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, bool):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def _render_explainability(result) -> None:
    """Render explainability / stock drill-down."""
    section("Explainability - Stock Drill-down (Per Gate)")
    if not result.snapshots:
        st.caption("No data.")
        return
    last = next(reversed(result.snapshots.values()))
    if last is None or last.empty:
        st.caption("Empty book.")
        return
    ticker = st.selectbox("Ticker", last["ticker"].tolist(), key="res_expl_ticker")
    row = last[last["ticker"] == ticker]
    if not row.empty:
        st.json({k: _jsonable(v) for k, v in row.iloc[0].to_dict().items()})

    st.caption("Per-rebalance score trace across gates:")
    trace_rows = []
    for d, gates in result.pipeline_audit.items():
        rec = {"rebalance": d.date().isoformat(), "ticker": ticker}
        for g in gates:
            sc = g.score.get(ticker, None) if g.score is not None else None
            rec[g.kind] = None if (sc is None or pd.isna(sc)) else round(float(sc), 4)
            rec[f"{g.kind}_passed"] = ticker in g.output_universe
        trace_rows.append(rec)
    if trace_rows:
        st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)


def _render_risk_metrics(result) -> None:
    """Render risk metrics."""
    section("Risk Metrics")
    m = result.metrics
    risk_items = {k: v for k, v in m.items() if any(x in k.lower() for x in ['var', 'es', 'drawdown', 'ulcer', 'vol', 'beta', 'downside', 'upside', 'tail'])}
    if risk_items:
        cols = st.columns(4)
        for i, (k, v) in enumerate(risk_items.items()):
            if isinstance(v, float):
                v_str = f"{v*100:.2f}%" if 'drawdown' in k.lower() or 'var' in k.lower() or 'es' in k.lower() else f"{v:.4f}"
            else:
                v_str = str(v)
            cols[i % 4].metric(k.replace('_', ' ').title(), v_str)
    else:
        st.caption("No specific risk metrics available")


def _render_exports(result, params, strategy_name: str) -> None:
    """Render export options."""
    section("Exports")
    nav = result.nav.rename("nav").to_frame()
    bench = result.benchmark_nav.rename("benchmark").to_frame()
    perf = pd.DataFrame([result.metrics]).T.rename(columns={0: "value"})
    trades = result.trades

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("NAV (CSV)", export_dataframe(nav, "csv"), f"{strategy_name}_nav.csv", "text/csv")
    with col2:
        st.download_button("Trades (CSV)", export_dataframe(trades, "csv"), f"{strategy_name}_trades.csv", "text/csv")
    with col3:
        st.download_button("Metrics (CSV)", export_dataframe(perf, "csv"), f"{strategy_name}_metrics.csv", "text/csv")

    try:
        st.download_button("NAV (Parquet)", export_dataframe(nav, "parquet"), f"{strategy_name}_nav.parquet", "application/octet-stream")
        st.download_button("Trades (Parquet)", export_dataframe(trades, "parquet"), f"{strategy_name}_trades.parquet", "application/octet-stream")
    except Exception:
        st.caption("Parquet export requires pyarrow.")

    try:
        from io import BytesIO
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as xw:
            nav.to_excel(xw, sheet_name="NAV")
            trades.to_excel(xw, sheet_name="Trades")
            perf.to_excel(xw, sheet_name="Metrics")
        st.download_button("Full Report (Excel)",
                           buf.getvalue(), f"{strategy_name}_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        st.caption(f"Excel export unavailable: {exc}")

    render_log_panel("results")


if __name__ == "__main__":
    render_results()