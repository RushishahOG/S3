"""Portfolio Queue - Execution Dashboard.

This module provides the execution queue dashboard where all submitted
strategies are monitored from creation to completion.
"""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from app.layouts.base import page_header, section
from app.pages.backtest.state import get_backtest_state, BacktestStateManager, StrategyStatus
from app.components.logs import render_log_panel


def render_portfolio_queue() -> None:
    """Render the Portfolio Queue execution dashboard."""
    page_header("Portfolio Queue", "Execution Dashboard for Submitted Strategies")

    state = get_backtest_state()

    # Queue controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.subheader("Execution Queue")
    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col3:
        if st.button("⏹ Cancel All", type="secondary", use_container_width=True):
            _cancel_all_running(state)
            st.rerun()

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}  |  Auto-refresh every 5s when jobs running")

    # Get queue
    queue = state.get_queue()

    if not queue:
        st.info("No strategies in queue. Submit strategies from **Manual Testing**.")
        return

    # Separate by status
    running = [e for e in queue if e.status in (StrategyStatus.QUEUED, StrategyStatus.INITIALIZING, StrategyStatus.RUNNING, StrategyStatus.PROCESSING_GATE, StrategyStatus.OPTIMIZING)]
    completed = [e for e in queue if e.status == StrategyStatus.COMPLETED]
    failed = [e for e in queue if e.status == StrategyStatus.FAILED]
    cancelled = [e for e in queue if e.status == StrategyStatus.CANCELLED]

    # Active executions
    if running:
        section("🔄 Active & Queued Executions")
        _render_execution_table(state, running, show_actions=True)

    # Completed
    if completed:
        section("✅ Completed")
        _render_execution_table(state, completed, show_actions=False)

    # Failed
    if failed:
        section("❌ Failed")
        _render_execution_table(state, failed, show_actions=False)

    # Cancelled
    if cancelled:
        section("🚫 Cancelled")
        _render_execution_table(state, cancelled, show_actions=False)

    # Poll for progress updates
    if running:
        time.sleep(0.5)  # Faster polling for more responsive progress
        _poll_active_executions(state)
        st.rerun()

    render_log_panel("portfolio_queue")


def _render_execution_table(state: BacktestStateManager, executions: list, show_actions: bool) -> None:
    """Render a table of executions with progress bars and actions."""
    for exec_obj in executions:
        with st.container():
            col1, col2, col3, col4, col5 = st.columns([3, 1.5, 2, 2, 2])

            with col1:
                st.markdown(f"**{exec_obj.config.name}**")
                st.caption(f"ID: {exec_obj.strategy_id}  |  Submitted: {exec_obj.config.created_at.strftime('%H:%M:%S')}")

            with col2:
                _render_status_badge(exec_obj.status)
                if exec_obj.status == StrategyStatus.FAILED and exec_obj.error:
                    st.caption(exec_obj.error)

            with col3:
                st.markdown("**Progress**")
                progress = exec_obj.progress
                st.progress(progress, text=f"{int(progress * 100)}%")
                st.caption(f"Stage: {exec_obj.current_stage}")

            with col4:
                st.markdown("**Timing**")
                if exec_obj.started_at:
                    st.caption(f"Started: {exec_obj.started_at.strftime('%H:%M:%S')}")
                    st.caption(f"Elapsed: {exec_obj.duration_str()}")
                if exec_obj.estimated_remaining:
                    st.caption(f"Est. remaining: {exec_obj.estimated_remaining:.0f}s")

            with col5:
                if show_actions:
                    _render_execution_actions(state, exec_obj)
                else:
                    # View results link for completed
                    if exec_obj.status == StrategyStatus.COMPLETED:
                        if st.button("📊 View Results", key=f"view_{exec_obj.strategy_id}"):
                            state.select_result(exec_obj.strategy_id)
                            state.set_active_section("results")
                            st.rerun()

            # Expandable config view
            with st.expander("View Configuration", expanded=False):
                if exec_obj.status == StrategyStatus.FAILED and exec_obj.error:
                    st.error(f"**Error:** {exec_obj.error}")
                _render_config_summary(exec_obj.config)

            st.divider()


def _render_status_badge(status: StrategyStatus) -> None:
    """Render a colored status badge."""
    colors = {
        StrategyStatus.QUEUED: "🟡",
        StrategyStatus.INITIALIZING: "🔵",
        StrategyStatus.RUNNING: "🟢",
        StrategyStatus.PROCESSING_GATE: "🔵",
        StrategyStatus.OPTIMIZING: "🟣",
        StrategyStatus.COMPLETED: "✅",
        StrategyStatus.FAILED: "❌",
        StrategyStatus.CANCELLED: "⚪",
    }
    icons = {
        StrategyStatus.QUEUED: "Queued",
        StrategyStatus.INITIALIZING: "Initializing",
        StrategyStatus.RUNNING: "Running",
        StrategyStatus.PROCESSING_GATE: "Processing Gate",
        StrategyStatus.OPTIMIZING: "Optimizing",
        StrategyStatus.COMPLETED: "Completed",
        StrategyStatus.FAILED: "Failed",
        StrategyStatus.CANCELLED: "Cancelled",
    }
    st.markdown(f"{colors.get(status, '⚪')} **{icons.get(status, status.value)}**")


def _render_execution_actions(state: BacktestStateManager, exec_obj) -> None:
    """Render action buttons for an execution."""
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if exec_obj.status in (StrategyStatus.QUEUED, StrategyStatus.INITIALIZING):
            if st.button("▶ Start", key=f"start_{exec_obj.strategy_id}", use_container_width=True):
                _start_execution(state, exec_obj)
                st.rerun()

    with col_b:
        if exec_obj.status in (StrategyStatus.RUNNING, StrategyStatus.PROCESSING_GATE, StrategyStatus.INITIALIZING):
            if st.button("⏹ Cancel", key=f"cancel_{exec_obj.strategy_id}", use_container_width=True):
                state.cancel_execution(exec_obj.strategy_id)
                st.rerun()

    with col_c:
        if exec_obj.status == StrategyStatus.COMPLETED:
            if st.button("📊 Results", key=f"res_{exec_obj.strategy_id}", use_container_width=True):
                state.select_result(exec_obj.strategy_id)
                state.set_active_section("results")
                st.rerun()
        elif exec_obj.status == StrategyStatus.FAILED:
            if st.button("🔄 Re-run", key=f"rerun_{exec_obj.strategy_id}", use_container_width=True):
                _rerun_execution(state, exec_obj)
                st.rerun()


def _render_config_summary(config) -> None:
    """Render a summary of the strategy configuration."""
    st.markdown(f"**Strategy:** {config.name}")
    st.markdown(f"**Config ID:** {config.config_id}")
    st.markdown(f"**Created:** {config.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

    params = config.params
    if params:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**General**")
            st.caption(f"Period: {params.general.start_date} → {params.general.end_date}")
            st.caption(f"Capital: {params.general.initial_capital:,.0f}")
            st.caption(f"Benchmark: {params.general.benchmark}")
            st.caption(f"Rebalance: {params.general.rebalance_frequency}")

            st.markdown("**Regime**")
            st.caption(f"Buy: {params.regime.buy_trigger_pct}%  |  Sell: {params.regime.sell_trigger_pct}%")

        with col2:
            st.markdown("**Portfolio**")
            st.caption(f"Size: {params.portfolio.total_size}")
            st.caption(f"Sizing: {params.portfolio.sizing_method}")
            st.caption(f"Max Pos: {params.portfolio.max_position_pct*100:.1f}%")

            st.markdown("**Pipeline**")
            active_gates = [g.kind for g in params.pipeline.gates if g.enabled]
            st.caption(" → ".join(active_gates))


def _start_execution(state: BacktestStateManager, exec_obj) -> None:
    """Start the background execution for a queued strategy."""
    from app.services import get_storage
    storage = get_storage()
    # Use config_id (strategy_id) to start execution, run_id will be stored in exec_obj.run_id
    state.run_backtest_async(exec_obj.strategy_id, storage)
    state.update_execution_status(exec_obj.strategy_id, StrategyStatus.INITIALIZING, current_stage="Initializing...")


def _rerun_execution(state: BacktestStateManager, exec_obj) -> None:
    """Re-run a failed execution."""
    # Re-submit the same config
    new_exec = state.submit_to_queue(exec_obj.config)
    _start_execution(state, new_exec)


def _cancel_all_running(state: BacktestStateManager) -> None:
    """Cancel all running/queued executions."""
    queue = state.get_queue()
    for exec_obj in queue:
        if exec_obj.status in (StrategyStatus.QUEUED, StrategyStatus.INITIALIZING, StrategyStatus.RUNNING, StrategyStatus.PROCESSING_GATE):
            state.cancel_execution(exec_obj.strategy_id)


def _poll_active_executions(state: BacktestStateManager) -> None:
    """Poll background workers for progress updates."""
    queue = state.get_queue()
    for exec_obj in queue:
        if exec_obj.status in (StrategyStatus.RUNNING, StrategyStatus.INITIALIZING, StrategyStatus.PROCESSING_GATE):
            run_id = exec_obj.run_id
            if run_id and run_id.startswith("bt_"):
                progress = state.poll_progress(run_id)
                if progress:
                    _update_execution_from_progress(state, exec_obj.strategy_id, run_id, progress)


def _update_execution_from_progress(state: BacktestStateManager, strategy_id: str, run_id: str, progress: dict) -> None:
    """Update execution object from progress data."""
    gate_events = progress.get("gate_events", [])
    total_rebal = progress.get("total_rebal", 0)
    phase = progress.get("phase", "loading")
    result = progress.get("result")
    error = progress.get("error")
    running = progress.get("running", True)

    # Calculate progress
    done_rebals = len(gate_events)
    if total_rebal > 0:
        progress_pct = min(1.0, done_rebals / total_rebal)
    else:
        progress_pct = 0.1

    # Determine current stage
    if phase == "loading":
        stage = "Loading data..."
    elif gate_events:
        latest = gate_events[-1]
        if latest.get("gates"):
            last_gate = latest["gates"][-1]
            stage = f"Gate {last_gate['order']} - {last_gate['label']}"
        else:
            stage = "Processing..."
    else:
        stage = "Running..."

    state.update_execution_status(
        strategy_id,
        StrategyStatus.RUNNING if running else StrategyStatus.COMPLETED,
        progress=progress_pct,
        current_stage=stage,
        total_gates=total_rebal,
        completed_gates=done_rebals,
    )

    if result is not None:
        state.complete_execution(strategy_id, result)
    elif error is not None:
        state.fail_execution(strategy_id, error)


if __name__ == "__main__":
    render_portfolio_queue()