"""Manual Testing - Strategy Configuration Module.

This module contains the complete backtest configuration interface
for creating and submitting strategies to the execution queue.
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
from app.layouts.base import page_header, section
from app.pages.backtest.state import get_backtest_state
from app.services import get_storage
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


def render_manual_testing() -> None:
    """Render the Manual Testing section."""
    page_header("Manual Testing", "Configure and Submit ARQM Strategies")
    state = get_backtest_state()

    # Strategy Identity
    section("Strategy Identity")
    col1, col2 = st.columns([3, 1])
    with col1:
        strategy_name = st.text_input(
            "Strategy Name",
            value=st.session_state.get("bt_strategy_name", "ARQM_Strategy"),
            key="bt_strategy_name",
            help="Give your strategy a descriptive name for the queue and results"
        )
    with col2:
        if st.button("Save Configuration", type="secondary", use_container_width=True):
            cfg = _build_config_from_session()
            config = state.save_strategy(strategy_name, cfg)
            st.success(f"Configuration saved as '{config.name}' (ID: {config.config_id})")

    st.divider()

    # Configuration panel (renders form and returns config)
    cfg = _config_panel()

    st.divider()

    # Submit to queue
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Backtest \u2192 Add to Queue", type="primary", use_container_width=True):
            config = state.save_strategy(strategy_name, cfg)
            execution = state.submit_to_queue(config)
            # Auto-start the execution
            state.start_execution(execution.strategy_id)
            from app.services import get_storage
            storage = get_storage()
            state.run_backtest_async(execution.strategy_id, storage)
            state.set_active_section("portfolio_queue")
            st.success(f"Strategy '{config.name}' submitted and started (ID: {execution.strategy_id})")
            st.rerun()

    # Show saved configurations
    _render_saved_configs(state)

    render_log_panel("manual_testing")


def _build_config_from_session() -> BacktestParameters:
    """Build BacktestParameters from session state without re-rendering the form."""
    return BacktestParameters(
        general=GeneralConfig(
            start_date=str(st.session_state.get("mt_start", pd.Timestamp("2006-01-01"))),
            end_date=str(st.session_state.get("mt_end", pd.Timestamp("2026-05-31"))),
            initial_capital=st.session_state.get("mt_capital", 10_000_000.0),
            benchmark=st.session_state.get("mt_benchmark", "NIFTY_500"),
            transaction_cost_pct=st.session_state.get("mt_txn", 0.05),
            slippage_pct=st.session_state.get("mt_slip", 0.05),
            rebalance_frequency=st.session_state.get("mt_freq", "monthly"),
        ),
        regime=RegimeConfig(
            buy_trigger_pct=st.session_state.get("mt_buy", 5.0),
            sell_trigger_pct=st.session_state.get("mt_sell", -15.0),
            enable_swing_low=st.session_state.get("mt_swing", True),
            enable_peak_detection=st.session_state.get("mt_peak", True),
        ),
        universe=UniverseConfig(
            min_trading_history_days=int(st.session_state.get("mt_min_hist", 252)),
            require_fundamental_data=st.session_state.get("mt_req_q", True),
            require_lowvol_features=st.session_state.get("mt_req_lv", True),
        ),
        cap_segment=CapSegmentConfig(
            enabled=st.session_state.get("mt_cap_on", True),
            large_cap_weight=st.session_state.get("mt_cap_lc", 60) / 100.0,
            mid_cap_weight=st.session_state.get("mt_cap_mc", 30) / 100.0,
            small_cap_weight=st.session_state.get("mt_cap_sc", 10) / 100.0,
        ),
        momentum=MomentumConfig(
            selection_mode=st.session_state.get("mt_mom_mode", "top_pct"),
            top_pct=st.session_state.get("mt_mom_pct", 30) / 100.0,
            top_n=int(st.session_state.get("mt_mom_n", 50)),
            normalization=st.session_state.get("mt_mom_norm", "zscore"),
        ),
        stability=StabilityConfig(
            selection_mode=st.session_state.get("mt_stab_mode", "top_pct"),
            top_pct=st.session_state.get("mt_stab_pct", 50) / 100.0,
            top_n=int(st.session_state.get("mt_stab_n", 50)),
            normalization=st.session_state.get("mt_stab_norm", "zscore"),
        ),
        persistence=PersistenceConfig(
            enabled=st.session_state.get("mt_persist", False),
            required_periods=int(st.session_state.get("mt_persist_n", 2)),
        ),
        quality=QualityConfig(
            normalization=st.session_state.get("mt_q_norm", "zscore"),
            use_rollup=st.session_state.get("mt_q_roll", "median"),
            min_quality_score=st.session_state.get("mt_q_min", 0.0),
        ),
        scoring=ScoringConfig(
            momentum_weight=st.session_state.get("mt_w_mom", 0.40),
            quality_weight=st.session_state.get("mt_w_q", 0.40),
            stability_weight=st.session_state.get("mt_w_stab", 0.20),
        ),
        portfolio=PortfolioConfig(
            total_size=int(st.session_state.get("mt_psize", 50)),
            large_size=int(round(st.session_state.get("mt_cap_lc", 60) / 100.0 * int(st.session_state.get("mt_psize", 50)))),
            mid_size=int(round(st.session_state.get("mt_cap_mc", 30) / 100.0 * int(st.session_state.get("mt_psize", 50)))),
            small_size=int(round(st.session_state.get("mt_cap_sc", 10) / 100.0 * int(st.session_state.get("mt_psize", 50)))),
            sizing_method=st.session_state.get("mt_sizing", "equal"),
            max_position_pct=st.session_state.get("mt_maxpos", 7.0) / 100.0,
        ),
        pipeline=_get_pipeline_from_session(),
    )


def _get_pipeline_from_session() -> PipelineConfig:
    """Get pipeline config from session state."""
    draft = st.session_state.get("bt_pipeline_draft", [])
    return PipelineConfig(gates=tuple(GateSpec(kind=g.kind, enabled=g.enabled, order=g.order, config_key=g.config_key) for g in draft))


def _config_panel() -> BacktestParameters:
    """Render the complete backtest configuration panel."""
    st.subheader("Backtest Configuration")

    with st.expander("General", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            start = st.date_input("Start Date", pd.Timestamp("2006-01-01"), key="mt_start")
            capital = st.number_input("Initial Capital", value=10_000_000.0, step=1_000_000.0, key="mt_capital")
        with col2:
            end = st.date_input("End Date", pd.Timestamp("2026-05-31"), key="mt_end")
            benchmark = st.selectbox("Benchmark", ["NIFTY_500", "NIFTY_50"], key="mt_benchmark")
        with col3:
            txn = st.number_input("Transaction Cost %", value=0.05, step=0.01, key="mt_txn")
            slip = st.number_input("Slippage %", value=0.05, step=0.01, key="mt_slip")
        freq = st.selectbox("Rebalance Frequency", ["monthly", "quarterly", "semi_annual"], key="mt_freq")

    with st.expander("Market Regime"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            buy = st.number_input("Buy Trigger %", value=5.0, step=0.5, key="mt_buy")
        with col2:
            sell = st.number_input("Sell Trigger %", value=-15.0, step=0.5, key="mt_sell")
        with col3:
            swing = st.checkbox("Rolling swing-low detection", value=True, key="mt_swing")
        with col4:
            peak = st.checkbox("Rolling peak detection", value=True, key="mt_peak")

    with st.expander("Investment Universe"):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_hist = st.number_input("Min trading history (days)", value=252, step=21, key="mt_min_hist")
        with col2:
            req_q = st.checkbox("Require fundamental data", value=True, key="mt_req_q")
        with col3:
            req_lv = st.checkbox("Require low-vol features", value=True, key="mt_req_lv")

    with st.expander("Cap Segmentation"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            cap_on = st.checkbox("Enable cap segmentation", value=True, key="mt_cap_on")
        with col2:
            lc = st.number_input("Large Cap %", value=60, step=5, key="mt_cap_lc") / 100.0
        with col3:
            mc = st.number_input("Mid Cap %", value=30, step=5, key="mt_cap_mc") / 100.0
        with col4:
            sc = st.number_input("Small Cap %", value=10, step=5, key="mt_cap_sc") / 100.0
        cap_total = lc + mc + sc
        if abs(cap_total - 1.0) > 1e-6:
            st.error(f"Cap allocation must sum to 100% (currently {cap_total*100:.1f}%).")

    with st.expander("Momentum Discovery"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mom_mode = st.selectbox("Selection", ["top_pct", "top_n"], key="mt_mom_mode")
        with col2:
            mom_pct = st.number_input("Top %", value=30, step=5, key="mt_mom_pct") / 100.0
        with col3:
            mom_n = st.number_input("Top N", value=50, step=5, key="mt_mom_n")
        with col4:
            mom_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="mt_mom_norm")

    with st.expander("Stability (Low Vol)"):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            stab_mode = st.selectbox("Selection", ["top_pct", "top_n"], key="mt_stab_mode")
        with col2:
            stab_pct = st.number_input("Top %", value=50, step=5, key="mt_stab_pct") / 100.0
        with col3:
            stab_n = st.number_input("Top N", value=50, step=5, key="mt_stab_n")
        with col4:
            stab_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="mt_stab_norm")

    with st.expander("Quality Validation"):
        col1, col2, col3 = st.columns(3)
        with col1:
            q_norm = st.selectbox("Normalization", ["zscore", "robust_zscore", "percentile", "minmax"], key="mt_q_norm")
        with col2:
            q_roll = st.selectbox("Rollup", ["median", "latest", "weighted"], key="mt_q_roll")
        with col3:
            q_min = st.number_input("Min quality score", value=0.0, step=0.05, key="mt_q_min")

    with st.expander("Persistence Filter"):
        col1, col2 = st.columns(2)
        with col1:
            persist = st.checkbox("Enable persistence filter", value=False, key="mt_persist")
        with col2:
            persist_n = st.number_input("Required periods", value=2, step=1, key="mt_persist_n")

    with st.expander("Final Scoring & Portfolio"):
        col1, col2, col3 = st.columns(3)
        with col1:
            w_mom = st.number_input("Momentum weight", value=0.40, step=0.05, key="mt_w_mom")
            w_q = st.number_input("Quality weight", value=0.40, step=0.05, key="mt_w_q")
            w_stab = st.number_input("Stability weight", value=0.20, step=0.05, key="mt_w_stab")
        with col2:
            psize = st.number_input("Total portfolio size", value=50, step=5, key="mt_psize")
            sizing = st.selectbox("Position sizing", ["equal", "score", "hybrid"], key="mt_sizing")
        with col3:
            maxpos = st.number_input("Max position %", value=7.0, step=1.0, key="mt_maxpos") / 100.0
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
    if "bt_pipeline_draft" not in st.session_state:
        st.session_state["bt_pipeline_draft"] = [
            GateSpec(kind="eligibility", enabled=True, order=0, config_key="universe"),
            GateSpec(kind="momentum", enabled=True, order=1, config_key="momentum"),
            GateSpec(kind="stability", enabled=True, order=2, config_key="stability"),
            GateSpec(kind="quality", enabled=True, order=3, config_key="quality"),
            GateSpec(kind="persistence", enabled=True, order=4, config_key="persistence"),
        ]

    draft = st.session_state["bt_pipeline_draft"]
    draft_kinds = {g.kind for g in draft}
    for k in registered:
        if k not in draft_kinds:
            draft.append(GateSpec(kind=k, enabled=False, order=len(draft), config_key=k))

    new_draft = list(draft)
    for i, g in enumerate(new_draft):
        col1, col2, col3, col4 = st.columns([1, 3, 1, 1])
        with col1:
            enabled = st.checkbox("On", value=g.enabled, key=f"mt_pipe_en_{i}", label_visibility="collapsed")
            new_draft[i] = GateSpec(kind=g.kind, enabled=enabled, order=g.order, config_key=g.config_key)
        with col2:
            st.markdown(f"**{i+1}. {g.kind}**")
        with col3:
            if st.button("up", key=f"mt_pipe_up_{i}", disabled=(i == 0)):
                new_draft[i - 1], new_draft[i] = new_draft[i], new_draft[i - 1]
        with col4:
            if st.button("down", key=f"mt_pipe_dn_{i}", disabled=(i == len(new_draft) - 1)):
                new_draft[i + 1], new_draft[i] = new_draft[i], new_draft[i + 1]

    for i, g in enumerate(new_draft):
        new_draft[i] = GateSpec(kind=g.kind, enabled=g.enabled, order=i, config_key=g.config_key)
    st.session_state["bt_pipeline_draft"] = new_draft
    draft = new_draft

    active = [g.kind for g in draft if g.enabled]
    st.caption("Active sequence: " + " -> ".join(active))
    return PipelineConfig(gates=tuple(GateSpec(kind=g.kind, enabled=g.enabled, order=g.order, config_key=g.config_key) for g in draft))


def _render_saved_configs(state: BacktestStateManager) -> None:
    """Display saved strategy configurations."""
    strategies = state.list_strategies()

    if not strategies:
        return

    st.divider()
    section("Saved Configurations")

    for config in strategies:
        with st.expander(f"{config.name} (ID: {config.config_id})", expanded=False):
            st.caption(f"Created: {config.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Load", key=f"load_{config.config_id}"):
                    st.session_state["bt_strategy_name"] = config.name
                    st.info("Load functionality - populate form fields from saved config")
            with col2:
                if st.button("Delete", key=f"del_{config.config_id}"):
                    state.delete_strategy(config.config_id)
                    st.rerun()