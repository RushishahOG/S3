"""Sensitivity Analysis Research Lab module.

Quantifies how robust the ARQM strategy is by systematically varying one or more
parameters and observing the impact on portfolio performance. It reuses the
engineered datasets and the standard ARQM backtest (via
:mod:`core.sensitivity.engine`) and only re-runs backtests for the parameter
combinations requested.

Sections
--------
1. Base Strategy Selection      2. Parameter Selection       3. Range Configuration
4. Analysis Mode                5. Performance Metric        6. Run Controls
7. Progress Monitor             8. Visualizations            9. Stability Analysis
10. Parameter Importance        11. Robustness Analysis      12. Recommendations
13. Export
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.layouts.base import section
from app.pages.backtest.state import StrategyStatus, get_backtest_state
from core.config.backtest_schema import BacktestParameters
from core.sensitivity import engine as sa_engine
from core.sensitivity import export as sa_export
from core.sensitivity.engine import (
    METRIC_LABELS,
    catalog_by_key,
)

_SA_PROGRESS: dict[str, dict] = {}
_SA_LOCK = threading.Lock()


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _current_value(spec, base: BacktestParameters) -> Any:
    try:
        return getattr(getattr(base, spec.block), spec.field, spec.current)
    except Exception:
        return spec.current


# ---------------------------------------------------------------------------
# Section 1 — Base Strategy Selection
# ---------------------------------------------------------------------------
def _render_base_strategy_selection() -> BacktestParameters | None:
    section("1. Base Strategy Selection")
    st.markdown("Choose the baseline strategy. Its **complete parameter configuration** "
                "is shown below and is held fixed for all non-varied parameters.")

    state = get_backtest_state()
    saved = state.list_strategies()
    completed = [e for e in state.get_completed()
                 if e.status == StrategyStatus.COMPLETED and e.result]

    sources: list[tuple[str, str, Any]] = []
    sources.append(("manual", "Manual Backtest (current config)", "manual"))
    sources.append(("optimization", "Parameter Optimizer Result", "optimization"))
    sources.append(("montecarlo", "Monte Carlo Result (completed backtest)", "montecarlo"))
    for s in saved:
        sources.append(("saved", f"Saved Strategy: {s.name} ({s.config_id})", s))
    for c in completed:
        sources.append(("completed", f"Completed Backtest: {c.config.name} ({c.strategy_id})", c))

    labels = [s[1] for s in sources]
    cur = st.session_state.get("sa_base_label")
    idx = labels.index(cur) if cur in labels else 0
    chosen = st.selectbox("Base Strategy Source", labels, index=idx, key="sa_base_source")
    st.session_state["sa_base_label"] = chosen
    source = next(s for s in sources if s[1] == chosen)
    kind, payload = source[0], source[2]

    base: BacktestParameters | None = None
    if kind == "manual":
        try:
            from app.pages.backtest.manual_testing.render import _build_config_from_session
            base = _build_config_from_session()
        except Exception:
            base = None
        st.caption("Using the live configuration from the Manual Testing page.")
    elif kind == "optimization":
        base = _render_base_from_optimization()
    elif kind == "montecarlo":
        base = _render_base_from_montecarlo(completed)
    elif kind == "saved":
        base = payload.params
        st.caption(f"Loaded saved strategy **{payload.name}**.")
    elif kind == "completed":
        base = payload.config.params
        st.caption(f"Loaded completed backtest **{payload.config.name}**.")

    if base is None:
        st.warning("No base strategy available — using schema defaults.")
        base = BacktestParameters()

    st.markdown("**Complete Parameter Configuration**")
    with st.expander("Show base configuration (JSON)", expanded=False):
        st.json(base.to_dict())
    st.divider()
    return base


def _render_base_from_optimization() -> BacktestParameters | None:
    from core.optimization.results import OptimizationRun
    try:
        last = st.session_state.get("po_last_run")
    except Exception:
        last = None
    options: list[tuple[str, BacktestParameters]] = []
    if isinstance(last, OptimizationRun) and last.results:
        base_cfg = BacktestParameters.from_dict(last.base_config)
        for r in last.results:
            cfg = _cfg_from_optimization(base_cfg, r.params)
            options.append((f"Rank {r.rank} (score {r.objective_score:.3f})", cfg))
        st.caption("Loaded from the last Parameter Optimization run in this session.")

    # Also offer saved optimization runs on disk.
    from core.utils.paths import PROJECT_ROOT
    run_dir = os.path.join(PROJECT_ROOT, "storage", "optimization_runs")
    if os.path.isdir(run_dir):
        for fn in sorted(os.listdir(run_dir), reverse=True)[:10]:
            if not fn.endswith(".json"):
                continue
            try:
                run = OptimizationRun.load(os.path.join(run_dir, fn))
                base_cfg = BacktestParameters.from_dict(run.base_config)
                if run.results:
                    cfg = _cfg_from_optimization(base_cfg, run.results[0].params)
                    options.append((f"Saved run {run.run_id} (rank 1)", cfg))
            except Exception:
                continue

    if not options:
        st.info("No Parameter Optimization result available. Run one first or pick another source.")
        return None
    label = st.selectbox("Select optimizer candidate", [o[0] for o in options], key="sa_opt_cand")
    return dict(options)[label]


def _cfg_from_optimization(base_cfg: BacktestParameters, params: dict) -> BacktestParameters:
    return sa_engine.build_sensitivity_candidate(base_cfg, params)


def _render_base_from_montecarlo(completed) -> BacktestParameters | None:
    if not completed:
        st.info("No completed backtest available as a Monte Carlo base. Run a backtest first.")
        return None
    opts = {f"{e.config.name} ({e.strategy_id})": e.config.params for e in reversed(completed)}
    label = st.selectbox("Select base backtest (Monte Carlo derives from this)",
                         list(opts.keys()), key="sa_mc_base")
    st.caption("Monte Carlo simulations resample a completed backtest — its configuration is the base.")
    return opts[label]


# ---------------------------------------------------------------------------
# Section 2 — Parameter Selection
# ---------------------------------------------------------------------------
def _render_parameter_selection(base: BacktestParameters) -> list[str]:
    section("2. Parameter Selection")
    st.markdown("Select one or more parameters to test for sensitivity. Parameters are "
                "grouped by category and reuse the live current values from the base strategy.")

    cat = catalog_by_key()
    groups: dict[str, list] = {}
    for s in cat.values():
        groups.setdefault(s.category, []).append(s)

    c_all, c_none = st.columns([1, 1])
    with c_all:
        if st.button("Select All", key="sa_sel_all", use_container_width=True):
            for k in cat:
                st.session_state[f"sa_en_{k}"] = True
            st.rerun()
    with c_none:
        if st.button("Clear All", key="sa_sel_none", use_container_width=True):
            for k in cat:
                st.session_state[f"sa_en_{k}"] = False
            st.rerun()

    selected: list[str] = []
    for category, params in groups.items():
        with st.expander(category, expanded=True):
            for s in params:
                cur = _current_value(s, base)
                enabled = st.checkbox(f"**{s.name}**", value=False, key=f"sa_en_{s.key}")
                st.caption(f"Current: `{_fmt(cur)}` · type: {s.kind.value} · {s.help or ''}")
                if enabled:
                    selected.append(s.key)
    if not selected:
        st.info("Select at least one parameter to analyse.")
    st.divider()
    return selected


# ---------------------------------------------------------------------------
# Section 3 — Range Configuration
# ---------------------------------------------------------------------------
def _render_range_configuration(selected: list[str], base: BacktestParameters) -> dict[str, dict]:
    section("3. Range Configuration")
    st.markdown("Specify the **Minimum**, **Maximum** and **Step** for every selected parameter. "
                "Categorical parameters use a multi-select of allowed values.")

    cat = catalog_by_key()
    ranges: dict[str, dict] = {}
    if not selected:
        st.info("No parameters selected yet.")
        st.divider()
        return ranges

    for key in selected:
        s = cat[key]
        cur = _current_value(s, base)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if s.kind.value == "categorical":
                choices = list(s.allowed or [])
                rng = st.multiselect(f"{s.name} — values", choices, default=list(choices),
                                     key=f"sa_rng_{key}")
                ranges[key] = {"choices": rng, "min": 0.0, "max": float(len(choices)),
                               "step": 1.0}
            else:
                dmin = s.min if s.min is not None else (float(cur) - 1)
                dmax = s.max if s.max is not None else (float(cur) + 1)
                dstep = s.step if s.step is not None else 0.05
                mn = st.number_input(f"{s.name} — Min", value=float(dmin),
                                     key=f"sa_min_{key}", format="%g")
                with col2:
                    mx = st.number_input(f"{s.name} — Max", value=float(dmax),
                                         key=f"sa_max_{key}", format="%g")
                with col3:
                    step = st.number_input(f"{s.name} — Step", value=float(dstep),
                                           key=f"sa_step_{key}", format="%g")
                ranges[key] = {"min": mn, "max": mx, "step": step}
        if s.kind.value != "categorical":
            st.caption("")
    st.divider()
    return ranges


# ---------------------------------------------------------------------------
# Section 4 — Analysis Mode
# ---------------------------------------------------------------------------
def _render_analysis_mode(selected: list[str]) -> tuple[str, int]:
    section("4. Analysis Mode")
    mode = st.radio("Mode", options=["One-Way", "Two-Way", "Multi-Parameter Grid"],
                    index=0, key="sa_mode", horizontal=True,
                    help="One-Way: vary one parameter. Two-Way: vary two (heatmaps/surfaces). "
                         "Multi: vary all selected (combinations capped).")
    mode_key = {"One-Way": "one_way", "Two-Way": "two_way",
                "Multi-Parameter Grid": "multi"}[mode]
    max_comb = 1000
    if mode_key == "multi":
        max_comb = st.number_input("Maximum Combinations", min_value=1, max_value=10000,
                                   value=1000, step=100, key="sa_maxcomb")
    if mode_key in ("two_way",) and len(selected) < 2:
        st.warning("Two-Way mode requires at least two selected parameters.")
    st.divider()
    return mode_key, int(max_comb)


# ---------------------------------------------------------------------------
# Section 5 — Performance Metric Selection
# ---------------------------------------------------------------------------
def _render_metric_selection() -> tuple[str, list[str]]:
    section("5. Performance Metric Selection")
    metric_opts = list(METRIC_LABELS.keys())
    primary = st.selectbox("Primary Metric (drives scores/ranking/axes)",
                           metric_opts, index=metric_opts.index("sharpe"),
                           format_func=lambda k: METRIC_LABELS[k], key="sa_primary")
    overlay = st.multiselect("Overlay Metrics (line charts / radar)",
                             metric_opts, default=[primary],
                             format_func=lambda k: METRIC_LABELS[k], key="sa_overlay")
    if primary not in overlay:
        overlay = [primary] + overlay
    st.divider()
    return primary, overlay


# ---------------------------------------------------------------------------
# Section 6 — Run Controls
# ---------------------------------------------------------------------------
def _estimate_combos(mode: str, selected: list[str], ranges: dict) -> int:
    cat = catalog_by_key()
    if not selected or not ranges:
        return 0
    if mode == "one_way":
        return len(sa_engine.generate_values(cat[selected[0]], ranges[selected[0]]))
    if mode == "two_way":
        a = len(sa_engine.generate_values(cat[selected[0]], ranges[selected[0]]))
        b = len(sa_engine.generate_values(cat[selected[1]], ranges[selected[1]]))
        return a * b
    # multi
    import itertools
    grids = [sa_engine.generate_values(cat[k], ranges[k]) for k in selected]
    return min(int(np.prod([len(g) for g in grids])), 10000)


def _render_run_controls(base, selected, ranges, mode, max_comb, primary) -> None:
    section("6. Run Controls")
    n_est = _estimate_combos(mode, selected, ranges)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        workers = st.number_input("Parallel Workers", min_value=1, max_value=8,
                                  value=1, step=1, key="sa_workers")
    with c2:
        st.metric("Estimated Combinations", n_est)
    with c3:
        cache_n = len(sa_engine._RESULT_CACHE)
        st.metric("Cached Runs (session)", cache_n)

    if n_est == 0:
        st.info("Configure a base strategy, select parameters and ranges to enable the run.")
        st.divider()
        return

    st.caption(
        "Each backtest reuses the cached engineered datasets (loaded once per run). "
        "Actual runtime depends on the ARQM simulation engine (≈10-40 s per "
        "combination on this machine). Results are cached across runs, so re-running "
        "with overlapping combinations is instant. Keep grids modest for interactive use."
    )

    if st.button("Run Sensitivity Analysis", type="primary", key="sa_run",
                 use_container_width=True):
        errs = _validate(selected, ranges, mode)
        if errs:
            for e in errs:
                st.error(e)
        else:
            run_id = _launch(base, selected, ranges, mode, max_comb, primary, int(workers))
            st.session_state["sa_run_id"] = run_id
            st.rerun()
    st.divider()


def _validate(selected, ranges, mode) -> list[str]:
    errs: list[str] = []
    if not selected:
        errs.append("Select at least one parameter (Section 2).")
    if mode == "two_way" and len(selected) < 2:
        errs.append("Two-Way mode needs at least two selected parameters.")
    for k in selected:
        r = ranges.get(k, {})
        if r.get("choices") is not None:
            if not r["choices"]:
                errs.append(f"{k}: select at least one categorical value.")
        else:
            if r.get("min", 0) > r.get("max", 0):
                errs.append(f"{k}: Minimum exceeds Maximum.")
            if r.get("step", 0) <= 0:
                errs.append(f"{k}: Step must be positive.")
    return errs


def _launch(base, selected, ranges, mode, max_comb, primary, workers) -> str:
    from app.services import get_storage
    run_id = f"sa_{int(time.time()*1000)}"
    with _SA_LOCK:
        _SA_PROGRESS[run_id] = {"running": True, "done": 0, "total": 0,
                               "last": None, "eta": None, "error": None}
    bucket = _SA_PROGRESS[run_id]

    def _progress(ev: dict) -> None:
        with _SA_LOCK:
            b = _SA_PROGRESS.get(run_id)
            if b is None:
                return
            if ev.get("event") == "start":
                b["total"] = ev.get("total", 0)
            elif ev.get("event") == "combo_done":
                b["done"] = ev.get("done", 0)
                b["total"] = ev.get("total", b["total"])
                b["last"] = ev.get("current")
                b["eta"] = ev.get("eta")
            elif ev.get("event") == "done":
                b["running"] = False
                b["result"] = ev.get("result")

    def _worker() -> None:
        try:
            result = sa_engine.run_sensitivity(
                base=base, selected_keys=selected, ranges=ranges, mode=mode,
                max_combinations=max_comb, primary_metric=primary,
                workers=workers, progress_callback=_progress,
                storage_factory=get_storage,
            )
            with _SA_LOCK:
                b = _SA_PROGRESS.get(run_id)
                if b is not None:
                    b["running"] = False
                    b["result"] = result
        except Exception as exc:
            import traceback as _tb
            with _SA_LOCK:
                b = _SA_PROGRESS.get(run_id)
                if b is not None:
                    b["error"] = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


# ---------------------------------------------------------------------------
# Section 7 — Progress Monitor
# ---------------------------------------------------------------------------
def _render_progress_monitor() -> None:
    run_id = st.session_state.get("sa_run_id")
    if not run_id:
        return
    with _SA_LOCK:
        b = _SA_PROGRESS.get(run_id)
    if b is None:
        return
    section("7. Progress Monitor")
    if b.get("error"):
        st.error("Sensitivity run failed:\n" + b["error"])
        with _SA_LOCK:
            _SA_PROGRESS.pop(run_id, None)
        st.session_state["sa_run_id"] = None
        return
    if b.get("running") or not b.get("result"):
        total = max(1, b.get("total", 1))
        done = b.get("done", 0)
        pct = min(1.0, done / total)
        st.progress(pct)
        remaining = total - done
        eta = b.get("eta")
        cols = st.columns(3)
        cols[0].metric("Completed", done)
        cols[1].metric("Remaining", remaining)
        cols[2].metric("ETA (s)", f"{eta:.0f}" if eta else "—")
        last = b.get("last")
        if last:
            st.caption("Current combination: " + ", ".join(
                f"{k}={_fmt(v)}" for k, v in last.items()))
        time.sleep(1.0)
        st.rerun()
        return
    # Done
    result = b.get("result")
    st.success(f"Sensitivity analysis complete: {len(result.records)} combinations evaluated.")
    st.session_state["sa_result"] = result
    st.session_state["sa_analytics"] = _compute_analytics(result, result.primary_metric)
    with _SA_LOCK:
        _SA_PROGRESS.pop(run_id, None)
    st.session_state["sa_run_id"] = None
    st.rerun()


# ---------------------------------------------------------------------------
# Analytics cache
# ---------------------------------------------------------------------------
def _compute_analytics(result, primary: str | None = None) -> dict:
    primary = primary or result.primary_metric
    return {
        "sensitivity": sa_engine.sensitivity_scores(result, primary),
        "stability": sa_engine.stability_analysis(result, primary),
        "importance": sa_engine.parameter_importance(result),
        "correlation": sa_engine.correlation_analysis(result),
        "interaction": sa_engine.interaction_analysis(result),
        "robustness": sa_engine.robustness_analysis(result, primary),
        "recommendations": sa_engine.recommendations(result, primary),
    }


# ---------------------------------------------------------------------------
# Section 8 — Visualizations
# ---------------------------------------------------------------------------
def _render_visualizations(result, analytics, overlay) -> None:
    section("8. Sensitivity Visualizations")
    if result.records:
        _render_line_charts(result, overlay)
        if result.mode == "two_way" and len(result.specs) >= 2:
            _render_heatmap(result)
            _render_surface(result)
        _render_spider(result, overlay)
    _render_tornado(analytics)
    st.divider()


def _render_line_charts(result, overlay) -> None:
    st.subheader("Line Charts — Parameter Sensitivity")
    cat = catalog_by_key()
    df = result.to_frame()
    for spec in result.specs:
        k = spec.key
        if k not in df.columns:
            continue
        sub = df.groupby(k)[list(overlay)].mean().reset_index()
        fig = px.line(sub, x=k, y=[METRIC_LABELS[m] for m in overlay],
                      markers=True, title=f"{spec.name} vs Performance")
        fig.update_layout(xaxis_title=spec.name, yaxis_title="Metric",
                          legend_title="Metric", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)


def _render_heatmap(result) -> None:
    st.subheader("Heatmap — Two-Parameter Surface")
    df = result.to_frame()
    k0, k1 = result.specs[0].key, result.specs[1].key
    metric = result.primary_metric
    try:
        grid = df.pivot_table(index=k0, columns=k1, values=metric)
        fig = px.imshow(grid.values, x=[_fmt(c) for c in grid.columns],
                        y=[_fmt(c) for c in grid.index], aspect="auto",
                        color_continuous_scale="Viridis",
                        title=f"{METRIC_LABELS.get(metric, metric)}: {result.specs[0].name} × {result.specs[1].name}")
        fig.update_layout(xaxis_title=result.specs[1].name,
                          yaxis_title=result.specs[0].name)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not build heatmap: {exc}")


def _render_surface(result) -> None:
    st.subheader("3D Surface Plot")
    df = result.to_frame()
    k0, k1 = result.specs[0].key, result.specs[1].key
    metric = result.primary_metric
    try:
        grid = df.pivot_table(index=k0, columns=k1, values=metric)
        x = np.arange(len(grid.columns))
        y = np.arange(len(grid.index))
        xx, yy = np.meshgrid(x, y)
        fig = go.Figure(data=[go.Surface(z=grid.values, x=xx, y=yy,
                                         colorscale="Viridis")])
        fig.update_layout(
            title=f"{METRIC_LABELS.get(metric, metric)} Surface",
            scene=dict(
                xaxis_title=result.specs[1].name,
                yaxis_title=result.specs[0].name,
                zaxis_title=METRIC_LABELS.get(metric, metric),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not build surface: {exc}")


def _render_spider(result, overlay) -> None:
    st.subheader("Spider Plot — Baseline vs Best Combination")
    df = result.to_frame()
    metrics = [m for m in overlay if m in df.columns]
    if not metrics:
        return
    base_row = {m: result.baseline_metrics.get(m, np.nan) for m in metrics}
    best_row = df.loc[df[result.primary_metric].idxmax()].to_dict()
    # Normalize each metric across the result distribution for comparability.
    norm_base, norm_best = [], []
    for m in metrics:
        col = df[m].dropna()
        lo, hi = col.min(), col.max()
        rng = (hi - lo) if hi > lo else 1.0
        bv = base_row.get(m, np.nan)
        gv = best_row.get(m, np.nan)
        norm_base.append((bv - lo) / rng if pd.notna(bv) else 0.0)
        norm_best.append((gv - lo) / rng if pd.notna(gv) else 0.0)
    labels = [METRIC_LABELS[m] for m in metrics]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm_base + [norm_base[0]], theta=labels + [labels[0]],
                                  fill="toself", name="Baseline"))
    fig.add_trace(go.Scatterpolar(r=norm_best + [norm_best[0]], theta=labels + [labels[0]],
                                  fill="toself", name="Best Combination"))
    fig.update_layout(title="Normalized Metric Profile (0=worst, 1=best in grid)",
                      polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    st.plotly_chart(fig, use_container_width=True)


def _render_tornado(analytics) -> None:
    st.subheader("Tornado Chart — Parameter Sensitivity Ranking")
    sens = analytics.get("sensitivity")
    if sens is None or sens.empty:
        st.info("No sensitivity data.")
        return
    fig = px.bar(sens.sort_values("sensitivity_score"),
                 x="sensitivity_score", y="parameter", orientation="h",
                 title="Sensitivity Score (0-100)", text="sensitivity_score")
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sections 9-12 — Analytics tables
# ---------------------------------------------------------------------------
def _render_stability(result, analytics) -> None:
    section("9. Stability Analysis")
    stab = analytics.get("stability")
    if stab is None or stab.empty:
        st.info("No data.")
        return
    st.dataframe(stab, use_container_width=True, hide_index=True)
    st.caption("Classification: CV < 5% Highly Stable · 5-15% Moderate · > 15% Highly Sensitive.")
    st.divider()


def _render_importance(analytics) -> None:
    section("10. Parameter Importance Ranking")
    imp = analytics.get("importance")
    if imp is None or imp.empty:
        st.info("No data.")
        return
    fig = px.bar(imp.sort_values("composite_impact"), x="composite_impact",
                 y="parameter", orientation="h", title="Composite Importance (0-100)")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(imp, use_container_width=True, hide_index=True)
    st.divider()


def _render_robustness(analytics) -> None:
    section("11. Robustness Analysis")
    rob = analytics.get("robustness")
    if rob is None or rob.empty:
        st.info("No data.")
        return
    st.dataframe(rob, use_container_width=True, hide_index=True)
    inter = analytics.get("interaction")
    if isinstance(inter, dict):
        st.subheader("Interaction Analysis")
        st.caption("Precise 2-D interaction" if inter.get("precise")
                   else "Proxy interaction (correlation of marginal impacts)")
        tbl = inter.get("table")
        if tbl is not None and not tbl.empty:
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.divider()


def _render_recommendations(analytics) -> None:
    section("12. Recommendations")
    rec = analytics.get("recommendations")
    if not rec:
        st.info("No data.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best single parameter combination**")
        if rec.get("best_single"):
            st.json({k: _fmt(v) for k, v in rec["best_single"].items()})
        st.markdown("**Parameters to keep fixed (insensitive)**")
        st.write(", ".join(rec.get("fix_params", [])) or "—")
        st.markdown("**Parameters to optimize (sensitive)**")
        st.write(", ".join(rec.get("optimize_params", [])) or "—")
    with c2:
        st.markdown("**Safest operating ranges**")
        for r in rec.get("safest_ranges", []):
            st.caption(r)
    st.markdown("**Explanations**")
    for e in rec.get("explanations", []):
        st.markdown(e)
    st.divider()


# ---------------------------------------------------------------------------
# Section 13 — Export
# ---------------------------------------------------------------------------
def _render_export(result, analytics) -> None:
    section("13. Export")
    base = f"sensitivity_{result.mode}_{result.primary_metric}"
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Results (CSV)", sa_export.export_csv(result),
                           f"{base}.csv", "text/csv", key="sa_dl_csv")
    with c2:
        st.download_button("Results (Excel)", sa_export.export_excel(result, analytics),
                           f"{base}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           key="sa_dl_xlsx")
    with c3:
        st.download_button("Results (JSON)", sa_export.export_json(result, analytics),
                           f"{base}.json", "application/json", key="sa_dl_json")
    with c4:
        st.download_button("Report (PDF)", sa_export.export_pdf(result, analytics),
                           f"{base}.pdf", "application/pdf", key="sa_dl_pdf")
    st.divider()


# ---------------------------------------------------------------------------
# Results table (sortable, filterable)
# ---------------------------------------------------------------------------
def _render_results_table(result) -> None:
    section("Results Table")
    df = result.to_frame()
    if df.empty:
        return
    metric_opts = [m for m in METRIC_LABELS if m in df.columns]
    sort_by = st.selectbox("Sort by", metric_opts,
                           index=metric_opts.index(result.primary_metric),
                           format_func=lambda k: METRIC_LABELS[k], key="sa_sort")
    asc = st.checkbox("Ascending", key="sa_asc")
    search = st.text_input("Filter (substring on any cell)", key="sa_search")
    disp = df.copy()
    if search:
        mask = disp.astype(str).apply(lambda r: r.str.contains(search, case=False).any(), axis=1)
        disp = disp[mask]
    disp = disp.sort_values(sort_by, ascending=asc)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(f"{len(disp)} rows · columns: " + ", ".join(df.columns))
    st.divider()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def render() -> None:
    section("Sensitivity Analysis")
    st.caption("Quantify ARQM strategy robustness by varying parameters while reusing "
               "engineered datasets and cached backtests.")

    base = _render_base_strategy_selection()
    selected = _render_parameter_selection(base)
    ranges = _render_range_configuration(selected, base)
    mode, max_comb = _render_analysis_mode(selected)
    primary, overlay = _render_metric_selection()
    _render_run_controls(base, selected, ranges, mode, max_comb, primary)

    _render_progress_monitor()

    result = st.session_state.get("sa_result")
    analytics = st.session_state.get("sa_analytics")
    if result is not None and analytics is not None:
        # Switching the primary metric should recompute analytics instantly
        # (no new backtests are required — all metrics are already stored).
        if result.primary_metric != primary:
            analytics = _compute_analytics(result, primary)
            st.session_state["sa_analytics"] = analytics
        _render_visualizations(result, analytics, overlay)
        _render_stability(result, analytics)
        _render_importance(analytics)
        _render_robustness(analytics)
        _render_recommendations(analytics)
        _render_results_table(result)
        _render_export(result, analytics)


if __name__ == "__main__":
    render()
