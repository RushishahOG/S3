"""ARQM Backtest & Simulation page.

Full configurable quantitative portfolio simulator built on a flexible, ordered,
registry-driven gate pipeline. Reads only engineered datasets (via
``core.backtesting``) and exposes every strategy parameter from the spec through
an in-page configuration panel.

Key UX features implemented here:

* **Pipeline editor** -- enable/disable and reorder gates (Eligibility, Momentum,
  Stability, Quality, Persistence, and any future registered gate) through a
  drag-free up/down + order UI. The engine runs them in that exact sequence.
* **Progressive / streaming execution** -- the backtest runs on a daemon thread; a
  ``progress_callback`` pushes per-gate events into ``st.session_state`` and the
  page re-renders collapsible per-gate cards the moment each gate completes, so
  the UI never blocks waiting for the whole pipeline.
* **Per-gate audit trail** -- every gate records input/output universe, stocks
  filtered/retained, ranking scores, execution time, logs and warnings, exposed
  in a dedicated "Pipeline Trace" tab with CSV export and a stock-level drill-down.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Any

import pandas as pd
import streamlit as st

from app.components.logs import render_log_panel
from app.components.sidebar import PAGES  # noqa: F401  (keeps import graph stable)
from app.services import get_storage
from app.layouts.base import page_header, section
from core.backtesting.engine import run_backtest
from core.backtesting.export import export_dataframe
from core.backtesting.gate_registry import GateResult, list_gates
from core.config.backtest_schema import (
    BacktestParameters,
    CapSegmentConfig,
    GateSpec,
    GeneralConfig,
    MomentumConfig,
    MomentumFactorConfig,
    PersistenceConfig,
    PipelineConfig,
    PortfolioConfig,
    QualityConfig,
    RegimeConfig,
    ScoringConfig,
    StabilityConfig,
    StabilityFactorConfig,
    UniverseConfig,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Cross-thread progress store                                                   #
# --------------------------------------------------------------------------- #
# The backtest runs on a plain worker thread that has NO Streamlit
# ScriptRunContext. Writing ``st.session_state`` from that thread is unsafe:
# either the write is dropped, or (if the main thread's context is attached)
# Streamlit raises ``StopException`` inside the worker on the next rerun and
# kills the backtest mid-run. So the worker never touches ``st.session_state``.
# Instead it writes into this module-level, lock-guarded dict keyed by run_id,
# and the main thread drains it into session_state on each poll.
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STORE: dict[str, dict] = {}


def _new_progress_bucket(run_id: str) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS_STORE[run_id] = {
            "load_steps": {},
            "gate_events": [],
            "total_rebal": 0,
            "phase": "loading",
            "result": None,
            "error": None,
            "params": None,
            "running": True,
        }


def _make_progress_handler(run_id: str):
    """Return a progress callback that writes to the thread-safe store."""

    def _handler(event: dict) -> None:
        try:
            ev = event.get("event")
            with _PROGRESS_LOCK:
                bucket = _PROGRESS_STORE.get(run_id)
                if bucket is None:
                    return
                if ev == "pipeline_start":
                    bucket["phase"] = "running"
                    bucket["total_rebal"] = event.get("total_rebalances", 0)
                elif ev == "load_step":
                    steps = bucket["load_steps"]
                    stage = event.get("stage")
                    if event.get("status") == "start":
                        steps[stage] = {"status": "running", "started": time.time()}
                    else:
                        prev = steps.get(stage, {})
                        steps[stage] = {
                            "status": "done",
                            "duration_s": event.get("duration_s"),
                            "n_rows": event.get("n_rows"),
                            "started": prev.get("started"),
                        }
                    bucket["phase"] = "loading"
                elif ev == "gate_done":
                    bucket["phase"] = "running"
                    pipeline = event["pipeline"]
                    bucket["gate_events"].append({
                        "date": event["date"],
                        "gates": [
                            {
                                "kind": g.kind, "label": g.label, "order": g.order,
                                "status": g.status, "enabled": g.enabled,
                                "input_n": len(g.input_universe), "output_n": len(g.output_universe),
                                "n_filtered": g.n_filtered, "exec_s": g.execution_time_s,
                                "warnings": list(g.warnings), "error": g.error,
                                "score": g.score,
                            }
                            for g in pipeline
                        ],
                    })
        except Exception:  # never let a callback break the worker
            pass

    return _handler


def _drain_progress(run_id: str) -> None:
    """Copy the worker's thread-safe progress into ``st.session_state``.

    Runs on the main thread (which owns the ScriptRunContext), so all writes are
    safe. Called on every poll while the backtest is running and once at the end.
    """
    with _PROGRESS_LOCK:
        bucket = _PROGRESS_STORE.get(run_id)
        if bucket is None:
            return
        snapshot = {
            "bt_load_steps": dict(bucket["load_steps"]),
            "bt_gate_events": list(bucket["gate_events"]),
            "bt_total_rebal": bucket["total_rebal"],
            "bt_phase": bucket["phase"],
            "bt_result": bucket["result"],
            "bt_error": bucket["error"],
            "bt_params": bucket["params"],
            "bt_running": bucket["running"],
        }
    if st.session_state.get("bt_run_id") == run_id:
        for k, v in snapshot.items():
            st.session_state[k] = v


# --------------------------------------------------------------------------- #
# Live progress state helpers                                                   #
# --------------------------------------------------------------------------- #
def _init_run_state() -> None:
    for key, default in {
        "bt_run_id": None,
        "bt_running": False,
        "bt_error": None,
        "bt_result": None,
        "bt_params": None,
        "bt_gate_events": [],   # list of pipeline lists (one per completed rebalance)
        "bt_total_rebal": 0,
        "bt_phase": "idle",     # idle | loading | running | done
        "bt_start": None,       # wall-clock start of the run
        "bt_load_steps": {},    # per-substep loading progress
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _run_backtest_threaded(cfg, storage) -> str:
    """Run ``run_backtest`` off the Streamlit main thread (keeps WebSocket alive).

    The worker opens its OWN read-only ``StorageManager`` connection *inside* the
    worker thread so the DuckDB connection is owned by that thread (DuckDB
    connections are not safe to share across threads -- reusing the app's main-
    thread connection deadlocks silently). The app's own connection is also
    read-only (see ``app.services.get_storage``), so the worker's read-only
    connection co-exists without file-lock contention.
    """
    from core.data.storage.storage_manager import StorageManager

    run_id = f"bt_{time.time()}"
    st.session_state["bt_run_id"] = run_id
    st.session_state["bt_running"] = True
    st.session_state["bt_error"] = None
    st.session_state["bt_result"] = None
    st.session_state["bt_gate_events"] = []
    st.session_state["bt_total_rebal"] = 0
    st.session_state["bt_phase"] = "loading"
    st.session_state["bt_load_steps"] = {}
    st.session_state["bt_start"] = time.time()

    _new_progress_bucket(run_id)
    handler = _make_progress_handler(run_id)
    # Emit an immediate "start" so the UI shows the first substep right away
    # (never a fully-pending list) and the watchdog can name the stuck substep.
    handler({"event": "load_step", "stage": "universe", "status": "start",
             "duration_s": None, "n_rows": None})

    def _worker() -> None:
        worker_storage = None
        try:
            from core.data.storage.provisioning import ensure_database

            ensure_database()
            worker_storage = StorageManager(read_only=True)
            result = run_backtest(cfg, worker_storage, progress_callback=handler)
            with _PROGRESS_LOCK:
                bucket = _PROGRESS_STORE.get(run_id)
                if bucket is not None:
                    bucket["result"] = result
                    bucket["params"] = cfg
                    bucket["phase"] = "done"
        except Exception as exc:
            with _PROGRESS_LOCK:
                bucket = _PROGRESS_STORE.get(run_id)
                if bucket is not None:
                    bucket["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if worker_storage is not None:
                try:
                    worker_storage.close()
                except Exception:
                    pass
            with _PROGRESS_LOCK:
                bucket = _PROGRESS_STORE.get(run_id)
                if bucket is not None:
                    bucket["running"] = False

    # The worker is a plain daemon thread with NO ScriptRunContext on purpose:
    # it communicates only through the lock-guarded ``_PROGRESS_STORE`` (drained
    # into session_state by the main thread), so Streamlit's rerun/StopException
    # machinery can never interrupt the running backtest.
    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return run_id


# --------------------------------------------------------------------------- #
# Render entrypoint                                                            #
# --------------------------------------------------------------------------- #
def render() -> None:
    page_header("ARQM Strategy Simulation & Backtesting", "Adaptive Regime-based Quality Momentum")
    storage = get_storage()
    _init_run_state()

    cfg = _config_panel()

    if st.button("Run Backtest", type="primary", key="run_bt"):
        _run_backtest_threaded(cfg, storage)
        st.rerun()

    # Drain the worker thread's progress into session_state (main thread owns
    # the ScriptRunContext, so writes here are always safe).
    run_id = st.session_state.get("bt_run_id")
    if run_id:
        _drain_progress(run_id)

    # Progressive rendering: while running, show live per-gate cards and poll.
    if st.session_state.get("bt_running"):
        _render_live_progress()
        # Poll without blocking the session (re-runs the script; the daemon
        # thread keeps computing). Keep the sleep short so progress feels live.
        time.sleep(0.5)
        st.rerun()

    if st.session_state.get("bt_error"):
        st.error(f"Backtest failed: {st.session_state['bt_error']}")
        st.session_state["bt_error"] = None

    result = st.session_state.get("bt_result")
    if result is None:
        st.info("Configure parameters and the pipeline below, then click **Run Backtest**.")
        render_log_panel()
        return

    if result.nav is None or result.nav.empty:
        st.warning(
            "Backtest returned no data. This usually means the price store has no "
            "rows for the selected universe/date range. Check the Data Extractor page."
        )
        render_log_panel()
        return

    _render_results(result, st.session_state.get("bt_params", cfg))
    render_log_panel()


# --------------------------------------------------------------------------- #
# Configuration panel                                                          #
# --------------------------------------------------------------------------- #
def _config_panel() -> BacktestParameters:
    st.subheader("Backtest Configuration")

    with st.expander("General", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.date_input("Start Date", pd.Timestamp("2006-01-01"))
            capital = st.number_input("Initial Capital", value=10_000_000.0, step=1_000_000.0)
        with col2:
            end = st.date_input("End Date", pd.Timestamp("2026-05-31"))
            benchmark = st.selectbox("Benchmark", ["NIFTY_500", "NIFTY_50"])
        with col3:
            txn = st.number_input("Transaction Cost %", value=0.05, step=0.01)
            slip = st.number_input("Slippage %", value=0.05, step=0.01)
        freq = st.selectbox("Rebalance Frequency", ["monthly", "quarterly", "semi_annual"])

    with st.expander("Market Regime"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            buy = st.number_input("Buy Trigger %", value=5.0, step=0.5)
        with col2:
            sell = st.number_input("Sell Trigger %", value=-15.0, step=0.5)
        with col3:
            swing = st.checkbox("Rolling swing-low detection", value=True)
        with col4:
            peak = st.checkbox("Rolling peak detection", value=True)

    with st.expander("Investment Universe"):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_hist = st.number_input("Min trading history (days)", value=252, step=21)
        with col2:
            req_q = st.checkbox("Require fundamental data", value=True)
        with col3:
            req_lv = st.checkbox("Require low-vol features", value=True)

    with st.expander("Cap Segmentation"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cap_on = st.checkbox("Enable cap segmentation", value=True)
        with col2:
            lc = st.number_input("Large Cap %", value=60, step=5, key="cap_lc") / 100.0
        with col3:
            mc = st.number_input("Mid Cap %", value=30, step=5, key="cap_mc") / 100.0
        with col4:
            sc = st.number_input("Small Cap %", value=10, step=5, key="cap_sc") / 100.0
        cap_total = lc + mc + sc
        if abs(cap_total - 1.0) > 1e-6:
            st.error(f"Cap allocation must sum to 100% (currently {cap_total*100:.1f}%).")

    with st.expander("Momentum Discovery"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mom_mode = st.selectbox("Selection", ["top_pct", "top_n"], key="mom_mode")
        with col2:
            mom_pct = st.number_input("Top %", value=30, step=5, key="mom_pct") / 100.0
        with col3:
            mom_n = st.number_input("Top N", value=50, step=5, key="mom_n")
        with col4:
            mom_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="mom_norm")

    with st.expander("Stability (Low Vol)"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stab_mode = st.selectbox("Selection", ["top_pct", "top_n"], key="stab_mode")
        with col2:
            stab_pct = st.number_input("Top %", value=50, step=5, key="stab_pct") / 100.0
        with col3:
            stab_n = st.number_input("Top N", value=50, step=5, key="stab_n")
        with col4:
            stab_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="stab_norm")

    with st.expander("Quality Validation"):
        col1, col2, col3 = st.columns(3)
        with col1:
            q_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="q_norm")
        with col2:
            q_roll = st.selectbox("Rollup", ["median", "latest", "weighted"], key="q_roll")
        with col3:
            q_min = st.number_input("Min quality score", value=0.0, step=0.05, key="q_min")

    with st.expander("Persistence Filter"):
        col1, col2 = st.columns(2)
        with col1:
            persist = st.checkbox("Enable persistence filter", value=False)
        with col2:
            persist_n = st.number_input("Required periods", value=2, step=1, key="p_n")

    with st.expander("Final Scoring & Portfolio"):
        col1, col2, col3 = st.columns(3)
        with col1:
            w_mom = st.number_input("Momentum weight", value=0.40, step=0.05, key="w_mom")
            w_q = st.number_input("Quality weight", value=0.40, step=0.05, key="w_q")
            w_stab = st.number_input("Stability weight", value=0.20, step=0.05, key="w_stab")
        with col2:
            psize = st.number_input("Total portfolio size", value=50, step=5, key="psize")
            sizing = st.selectbox("Position sizing", ["equal", "score", "hybrid"], key="sizing")
        with col3:
            maxpos = st.number_input("Max position %", value=7.0, step=1.0, key="maxpos") / 100.0
        total = w_mom + w_q + w_stab
        if abs(total - 1.0) > 1e-6:
            st.caption(f"Overall weights sum to {total:.2f} (should be 1.0)")
        total_n = int(psize)
        lc_n = max(0, round(lc * total_n))
        mc_n = max(0, round(mc * total_n))
        sc_n = max(0, total_n - lc_n - mc_n)
        st.caption("Cap counts auto-derived from Cap Segmentation % x Total portfolio size.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Large cap count", lc_n)
        m2.metric("Mid cap count", mc_n)
        m3.metric("Small cap count", sc_n)

    # --- Pipeline editor ---------------------------------------------------
    pipeline = _pipeline_editor()

    try:
        return BacktestParameters(
            general=GeneralConfig(
                start_date=str(start), end_date=str(end), initial_capital=capital,
                benchmark=benchmark, transaction_cost_pct=txn, slippage_pct=slip, rebalance_frequency=freq,
            ),
            regime=RegimeConfig(buy_trigger_pct=buy, sell_trigger_pct=sell,
                                enable_swing_low=swing, enable_peak_detection=peak),
            universe=UniverseConfig(min_trading_history_days=int(min_hist),
                                    require_fundamental_data=req_q, require_lowvol_features=req_lv),
            cap_segment=CapSegmentConfig(enabled=cap_on, large_cap_weight=lc, mid_cap_weight=mc, small_cap_weight=sc),
            momentum=MomentumConfig(selection_mode=mom_mode, top_pct=mom_pct, top_n=int(mom_n), normalization=mom_norm),
            stability=StabilityConfig(selection_mode=stab_mode, top_pct=stab_pct, top_n=int(stab_n), normalization=stab_norm),
            persistence=PersistenceConfig(enabled=persist, required_periods=int(persist_n)),
            quality=QualityConfig(normalization=q_norm, use_rollup=q_roll, min_quality_score=q_min),
            scoring=ScoringConfig(momentum_weight=w_mom, quality_weight=w_q, stability_weight=w_stab),
            portfolio=PortfolioConfig(total_size=int(psize), large_size=int(lc_n), mid_size=int(mc_n),
                                      small_size=int(sc_n), sizing_method=sizing, max_position_pct=maxpos),
            pipeline=pipeline,
        )
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()


def _pipeline_editor() -> PipelineConfig:
    """Enable/disable + reorder gates. Order is user-assigned via up/down."""
    st.subheader("Pipeline (Gate Order & Enable/Disable)")
    st.caption("Arrange gates in any order; disable any gate. The engine runs them "
               "top-to-bottom. Future factors (Value/Size/ESG/) appear here once registered.")

    registered = list_gates()
    # Default spec if session has none yet.
    if "bt_pipeline_draft" not in st.session_state:
        st.session_state["bt_pipeline_draft"] = [
            GateSpec(kind="eligibility", enabled=True, order=0, config_key="universe"),
            GateSpec(kind="momentum", enabled=True, order=1, config_key="momentum"),
            GateSpec(kind="stability", enabled=True, order=2, config_key="stability"),
            GateSpec(kind="quality", enabled=True, order=3, config_key="quality"),
            GateSpec(kind="persistence", enabled=True, order=4, config_key="persistence"),
        ]

    draft = st.session_state["bt_pipeline_draft"]
    # Ensure any newly registered gate is offered (added disabled by default).
    draft_kinds = {g.kind for g in draft}
    for k in registered:
        if k not in draft_kinds:
            draft.append(GateSpec(kind=k, enabled=False, order=len(draft), config_key=k))

    # Compact editor: each row shows enable toggle + kind + reorder buttons.
    new_draft = list(draft)
    for i, g in enumerate(new_draft):
        col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
        with col1:
            enabled = st.checkbox("On", value=g.enabled, key=f"pipe_en_{i}", label_visibility="collapsed")
            new_draft[i] = GateSpec(kind=g.kind, enabled=enabled, order=g.order, config_key=g.config_key)
        with col2:
            st.markdown(f"**{i+1}. {g.kind}**")
        with col3:
            if st.button("up", key=f"pipe_up_{i}", disabled=(i == 0)):
                new_draft[i - 1], new_draft[i] = new_draft[i], new_draft[i - 1]
        with col4:
            if st.button("down", key=f"pipe_dn_{i}", disabled=(i == len(new_draft) - 1)):
                new_draft[i + 1], new_draft[i] = new_draft[i], new_draft[i + 1]

    # Reassign order by position and persist.
    for i, g in enumerate(new_draft):
        new_draft[i] = GateSpec(kind=g.kind, enabled=g.enabled, order=i, config_key=g.config_key)
    st.session_state["bt_pipeline_draft"] = new_draft
    draft = new_draft

    active = [g.kind for g in draft if g.enabled]
    st.caption("Active sequence: " + " -> ".join(active))
    return PipelineConfig(gates=tuple(GateSpec(kind=g.kind, enabled=g.enabled, order=g.order, config_key=g.config_key) for g in draft))


# --------------------------------------------------------------------------- #
# Live progressive rendering                                                    #
# --------------------------------------------------------------------------- #
# Hard bound on the loading phase. A correct run loads in well under this; if we
# ever exceed it, the worker is stuck (e.g. a DB lock), so fail loudly instead of
# spinning the "Loading..." message forever.
_LOAD_TIMEOUT_S = 120.0

_LOAD_STEP_LABELS = {
    "universe": "Universe snapshot",
    "prices": "Price history (with warm-up buffer)",
    "benchmark_prices": "Benchmark price series",
    "quality_features": "Quality factor rollups",
    "lowvol_features": "Low-volatility features",
    "company_metadata": "Company metadata",
    "market_features": "Daily market features (point-in-time)",
    "quality_time_series": "Yearly quality time-series",
}

# Order in which substeps are expected to complete.
_LOAD_STEP_ORDER = list(_LOAD_STEP_LABELS.keys())


def _render_live_progress() -> None:
    section("Live Pipeline Progress")
    phase = st.session_state.get("bt_phase", "loading")
    total = st.session_state.get("bt_total_rebal", 0) or 1
    done = len(st.session_state.get("bt_gate_events", []))

    if phase == "loading":
        elapsed = int(time.time() - (st.session_state.get("bt_start") or time.time()))
        steps = st.session_state.get("bt_load_steps", {})

        # Watchdog: if loading runs suspiciously long with no movement, report it.
        if elapsed > _LOAD_TIMEOUT_S:
            st.session_state["bt_error"] = (
                f"Loading timed out after {elapsed}s with no progress. "
                f"Last substep: "
                f"{[s for s, v in steps.items() if v.get('status') == 'running'] or 'unknown'}. "
                f"This usually means a database lock/connection stall."
            )
            st.session_state["bt_running"] = False
            return

        st.info("Loading engineered datasets from storage (prices, market features, "
                f"fundamental quality)...  Elapsed: {elapsed}s")
        # Show each substep with its completion time so a stall is immediately
        # visible at the exact blocking substep.
        for stage in _LOAD_STEP_ORDER:
            label = _LOAD_STEP_LABELS.get(stage, stage)
            info = steps.get(stage)
            if info is None:
                st.caption(f"⧖ {label} -- pending")
            elif info.get("status") == "running":
                sub_el = int(time.time() - info.get("started", time.time()))
                st.caption(f"⏳ {label} -- running ({sub_el}s)")
            else:
                dur = info.get("duration_s")
                n = info.get("n_rows")
                st.caption(f"✓ {label} -- {dur:.1f}s{f' ({n} rows)' if n else ''}")
        return

    pct = min(1.0, done / total)
    st.progress(pct)
    st.caption(f"Processing rebalances: {done} / {total} completed "
               f"({pct*100:.0f}%)  -  each rebalance runs the full gate pipeline.")
    events = st.session_state.get("bt_gate_events", [])
    if not events:
        return
    latest = events[-1]
    st.caption(f"Latest rebalance: {latest['date'].date()}")
    for g in latest["gates"]:
        status_icon = "DONE" if g["status"] == "completed" else ("FAIL" if g["status"] == "failed" else "RUN")
        with st.container():
            st.markdown(
                f"{status_icon} **{g['label']}** -- in: {g['input_n']} ; "
                f"out: {g['output_n']} ; filtered: {g['n_filtered']} ; "
                f"{g['exec_s']*1000:.1f} ms"
            )
            if g["warnings"]:
                for w in g["warnings"]:
                    st.caption(f"warn {w}")
            if g["error"]:
                st.error(g["error"])


# --------------------------------------------------------------------------- #
# Results                                                                      #
# --------------------------------------------------------------------------- #
def _render_results(result, params: BacktestParameters) -> None:
    nav = result.nav
    bench = result.benchmark_nav

    tabs = st.tabs([
        "Performance", "Equity & Regime", "Allocation", "Distributions",
        "Trade Log", "Pipeline Trace", "Rebalance Snapshots", "Factor Attribution",
        "Explainability", "Exports",
    ])

    with tabs[0]:
        _render_performance(result, nav, bench)
    with tabs[1]:
        _render_equity_regime(result, nav, bench)
    with tabs[2]:
        _render_allocation(result)
    with tabs[3]:
        _render_distributions(result)
    with tabs[4]:
        _render_trade_log(result)
    with tabs[5]:
        _render_pipeline_trace(result)
    with tabs[6]:
        _render_snapshots(result)
    with tabs[7]:
        _render_attribution(result)
    with tabs[8]:
        _render_explainability(result)
    with tabs[9]:
        _render_exports(result, params)


def _render_performance(result, nav, bench) -> None:
    section("Performance Metrics")
    m = result.metrics
    cols = st.columns(4)
    labels = [
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
    for i, (k, v) in enumerate(labels):
        cols[i % 4].metric(k, v)
    section("Benchmark Comparison")
    cmp = pd.DataFrame({"Portfolio": nav, "Benchmark": bench}).dropna()
    st.line_chart(cmp)


def _render_equity_regime(result, nav, bench) -> None:
    section("Portfolio Equity Curve vs Benchmark")
    eq = pd.DataFrame({"Portfolio": nav, "Benchmark": bench})
    st.line_chart(eq)
    section("Drawdown Curve")
    dd = (eq["Portfolio"] / eq["Portfolio"].cummax() - 1.0) * 100.0
    st.area_chart(dd.rename("Drawdown %"))
    section("Market Regime Timeline")
    reg = result.regime
    if not reg.empty:
        sig = reg.copy()
        sig["regime_state"] = (sig["state"] == "invested").astype(int)
        st.line_chart(sig[["close", "swing_low", "peak"]])
        st.line_chart(sig[["regime_state", "buy_signal", "sell_signal"]])


def _render_allocation(result) -> None:
    section("Capital Allocation by Bucket (last rebalance)")
    if result.snapshots:
        last = next(reversed(result.snapshots.values()))
        if last is not None and not last.empty and "bucket" in last.columns:
            alloc = last.groupby("bucket")["overall"].count().rename("stocks")
            st.bar_chart(alloc)
        else:
            st.caption("No allocation (empty book at last rebalance).")


def _render_distributions(result) -> None:
    section("Quality / Momentum / Stability Distributions")
    if result.snapshots:
        last = next(reversed(result.snapshots.values()))
        if last is not None and not last.empty:
            st.bar_chart(last[["momentum", "stability", "quality"]].mean().rename("avg score"))
            st.dataframe(last[["ticker", "bucket", "momentum", "stability", "quality", "overall"]]
                         .sort_values("overall", ascending=False), use_container_width=True, hide_index=True)


def _render_trade_log(result) -> None:
    section("Trade Log")
    if result.trades.empty:
        st.caption("No trades generated.")
        return
    trades = result.trades.copy()
    trades["date"] = pd.to_datetime(trades["date"]).dt.date
    st.dataframe(trades, use_container_width=True, hide_index=True)
    st.caption(f"{len(trades)} trades - export from the Exports tab.")


def _render_pipeline_trace(result) -> None:
    """Per-rebalance, per-gate audit trail with collapsible cards + CSV export."""
    section("Pipeline Trace - Stage-by-Stage Audit")
    audit = result.pipeline_audit
    if not audit:
        st.caption("No pipeline audit recorded.")
        return
    dates = list(audit.keys())
    pick = st.selectbox("Rebalance date", [d.date().isoformat() for d in dates], key="trace_date")
    chosen = next(d for d in dates if d.date().isoformat() == pick)
    gates = audit[chosen]
    st.caption(f"Rebalance {chosen.date()} - {len(gates)} gate stages")

    for g in gates:
        status_icon = "DONE" if g.status == "completed" else ("FAIL" if g.status == "failed" else "RUN")
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
                if st.button(f"Export {g.kind} ranking CSV", key=f"exp_{g.kind}_{chosen:%Y%m%d}"):
                    df = g.score.rename("score").to_frame().sort_values("score", ascending=False)
                    st.download_button("Download", export_dataframe(df, "csv"),
                                       f"{g.kind}_{chosen:%Y%m%d}.csv", "text/csv",
                                       key=f"dl_{g.kind}_{chosen:%Y%m%d}")
            if g.logs:
                with st.expander("Logs", expanded=False):
                    for line in g.logs:
                        st.caption(line)

    # Full audit export: one row per (rebalance, gate).
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
    section("Rebalance Snapshots")
    if not result.snapshots:
        st.caption("No snapshots.")
        return
    dates = list(result.snapshots.keys())
    pick = st.selectbox("Rebalance date", [d.date().isoformat() for d in dates], key="snap_date")
    chosen = next(d for d in dates if d.date().isoformat() == pick)
    snap = result.snapshots[chosen]
    if snap is None or snap.empty:
        st.caption("Empty book at this rebalance.")
        return
    st.dataframe(snap.sort_values("overall", ascending=False), use_container_width=True, hide_index=True)


def _render_attribution(result) -> None:
    section("Factor Attribution (latest rebalance)")
    if not result.factor_scores:
        st.caption("No factor scores.")
        return
    last = next(reversed(result.factor_scores.values()))
    if "overall" in last:
        contrib = last["overall"].rename("overall_score")
        st.bar_chart(contrib.sort_values(ascending=False).head(20))


def _jsonable(v):
    """Coerce a cell to a JSON-friendly value.

    Numeric cells become ``float`` (or ``None`` if NaN); everything else
    (tickers, sector names, etc.) is passed through as its string form so a
    non-numeric column like ``ticker='NESTLEIND.NS'`` never breaks the panel.
    """
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
    section("Explainability - Stock Drill-down (per gate)")
    if not result.snapshots:
        st.caption("No data.")
        return
    last = next(reversed(result.snapshots.values()))
    if last is None or last.empty:
        st.caption("Empty book.")
        return
    ticker = st.selectbox("Ticker", last["ticker"].tolist(), key="expl_ticker")
    row = last[last["ticker"] == ticker]
    if not row.empty:
        st.json({k: _jsonable(v) for k, v in row.iloc[0].to_dict().items()})
    # Cross-rebalance trace for the selected ticker.
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


def _render_exports(result, params: BacktestParameters) -> None:
    section("Exports")
    nav = result.nav.rename("nav").to_frame()
    bench = result.benchmark_nav.rename("benchmark").to_frame()
    perf = pd.DataFrame([result.metrics]).T.rename(columns={0: "value"})
    trades = result.trades
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button("NAV (CSV)", export_dataframe(nav, "csv"), "nav.csv", "text/csv")
    with col2:
        st.download_button("Trades (CSV)", export_dataframe(trades, "csv"), "trades.csv", "text/csv")
    with col3:
        st.download_button("Metrics (CSV)", export_dataframe(perf, "csv"), "metrics.csv", "text/csv")
    try:
        st.download_button("NAV (Parquet)", export_dataframe(nav, "parquet"), "nav.parquet", "application/octet-stream")
        st.download_button("Trades (Parquet)", export_dataframe(trades, "parquet"), "trades.parquet", "application/octet-stream")
    except Exception:
        st.caption("Parquet export requires pyarrow.")
    try:
        st.download_button("Full Report (Excel)",
                           export_dataframe(_build_excel_bytes(nav, trades, perf), "excel"),
                           "arqm_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception as exc:
        st.caption(f"Excel export unavailable: {exc}")


def _build_excel_bytes(nav: pd.DataFrame, trades: pd.DataFrame, perf: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        nav.to_excel(xw, sheet_name="NAV")
        trades.to_excel(xw, sheet_name="Trades")
        perf.to_excel(xw, sheet_name="Metrics")
    return buf.getvalue()
