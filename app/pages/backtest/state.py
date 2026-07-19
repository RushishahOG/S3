"""Shared State Manager for ARQM Backtest Workflow.

This module provides a centralized state manager that coordinates between
the four sections: Manual Testing, Portfolio Queue, Research Lab, and Results.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import streamlit as st

from core.backtesting.engine import run_backtest
from core.backtesting.gate_registry import GateResult
from core.config.backtest_schema import BacktestParameters


class StrategyStatus(str, Enum):
    """Status values for a strategy in the queue."""
    QUEUED = "Queued"
    INITIALIZING = "Initializing"
    RUNNING = "Running"
    PROCESSING_GATE = "Processing Gate"
    OPTIMIZING = "Optimizing"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class StrategyConfig:
    """Complete configuration for a strategy."""
    name: str
    params: BacktestParameters
    created_at: datetime = field(default_factory=datetime.now)
    config_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": self.params,
            "created_at": self.created_at.isoformat(),
            "config_id": self.config_id,
        }


@dataclass
class StrategyExecution:
    """Runtime execution state for a strategy."""
    strategy_id: str
    config: StrategyConfig
    status: StrategyStatus = StrategyStatus.QUEUED
    progress: float = 0.0
    current_stage: str = "Queued"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining: float | None = None
    result: Any = None
    error: str | None = None
    gate_events: list[dict] = field(default_factory=list)
    total_gates: int = 0
    completed_gates: int = 0
    run_id: str | None = None  # Background worker run_id for polling

    def duration_str(self) -> str:
        if self.started_at is None:
            return "—"
        end = self.completed_at or datetime.now()
        delta = end - self.started_at
        total_seconds = int(delta.total_seconds())
        m, s = divmod(total_seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def progress_bar(self) -> str:
        filled = int(self.progress * 10)
        empty = 10 - filled
        return "█" * filled + "░" * empty + f" {int(self.progress * 100)}%"


# Thread-safe progress store for background workers
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_STORE: dict[str, dict] = {}

# On Windows, DuckDB allows either one writer OR many readers, but two heavy
# reader connections scanning the ~1.7M-row feature_store panel concurrently
# contend on the WAL/OS file handle and the load stalls (or trips a watchdog as
# a silent "Failed" at "Loading data"). Serialise the storage-heavy backtest
# execution so only one backtest reads the store at a time. Backtests still run
# on their own thread and report progress normally; they simply queue for the
# shared I/O instead of racing each other / the live app's connection.
_BACKTEST_IO_LOCK = threading.Lock()


def _new_progress_bucket(run_id: str) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS_STORE[run_id] = {
            "phase": "loading",
            "gate_events": [],
            "total_rebal": 0,
            "result": None,
            "error": None,
            "params": None,
            "running": True,
        }


def _make_progress_handler(run_id: str):
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
                    bucket["phase"] = "loading"
                elif ev == "gate_done":
                    bucket["phase"] = "running"
                    pipeline = event["pipeline"]
                    bucket["gate_events"].append({
                        "date": event["date"],
                        "gates": [
                            {
                                "kind": g.kind,
                                "label": g.label,
                                "order": g.order,
                                "status": g.status,
                                "enabled": g.enabled,
                                "input_n": len(g.input_universe),
                                "output_n": len(g.output_universe),
                                "n_filtered": g.n_filtered,
                                "exec_s": g.execution_time_s,
                                "warnings": list(g.warnings),
                                "error": g.error,
                                "score": g.score,
                            }
                            for g in pipeline
                        ],
                    })
        except Exception:
            pass
    return _handler


def _drain_progress(run_id: str) -> dict | None:
    with _PROGRESS_LOCK:
        bucket = _PROGRESS_STORE.get(run_id)
        if bucket is None:
            return None
        return {
            "gate_events": list(bucket["gate_events"]),
            "total_rebal": bucket["total_rebal"],
            "phase": bucket["phase"],
            "result": bucket["result"],
            "error": bucket["error"],
            "params": bucket["params"],
            "running": bucket["running"],
        }


class BacktestStateManager:
    """Centralized state manager for the backtest workflow."""

    _instance: BacktestStateManager | None = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._ensure_session_state()

    @staticmethod
    def _ensure_session_state() -> None:
        """Idempotently create the backtest workflow's session-state keys.

        Keys are created on every fresh Streamlit session (singleton state does
        not survive a rerun), so this must be callable regardless of the
        one-time ``_initialized`` flag. Without it, code reading
        ``st.session_state["bt_strategies"]`` can crash on a new session after
        the manager was already constructed in a prior run.
        """
        if "bt_strategies" not in st.session_state:
            st.session_state["bt_strategies"] = {}  # config_id -> StrategyConfig
        if "bt_queue" not in st.session_state:
            st.session_state["bt_queue"] = []  # list of strategy_ids in order
        if "bt_executions" not in st.session_state:
            st.session_state["bt_executions"] = {}  # strategy_id -> StrategyExecution
        if "bt_completed" not in st.session_state:
            st.session_state["bt_completed"] = {}  # strategy_id -> StrategyExecution (completed)
        if "bt_active_section" not in st.session_state:
            st.session_state["bt_active_section"] = "manual_testing"
        if "bt_selected_result" not in st.session_state:
            st.session_state["bt_selected_result"] = None

    # --- Strategy Configuration Management ---

    def save_strategy(self, name: str, params: BacktestParameters) -> StrategyConfig:
        """Save a new strategy configuration."""
        config = StrategyConfig(name=name, params=params)
        st.session_state["bt_strategies"][config.config_id] = config
        return config

    def get_strategy(self, config_id: str) -> StrategyConfig | None:
        return st.session_state["bt_strategies"].get(config_id)

    def list_strategies(self) -> list[StrategyConfig]:
        return list(st.session_state["bt_strategies"].values())

    def delete_strategy(self, config_id: str) -> bool:
        if config_id in st.session_state["bt_strategies"]:
            del st.session_state["bt_strategies"][config_id]
            return True
        return False

    # --- Queue Management ---

    def submit_to_queue(self, config: StrategyConfig) -> StrategyExecution:
        """Submit a strategy to the execution queue."""
        execution = StrategyExecution(
            strategy_id=config.config_id,
            config=config,
            status=StrategyStatus.QUEUED,
        )
        st.session_state["bt_executions"][execution.strategy_id] = execution
        st.session_state["bt_queue"].append(execution.strategy_id)
        return execution

    def get_queue(self) -> list[StrategyExecution]:
        """Get all queued/running executions in order."""
        executions = []
        for sid in st.session_state["bt_queue"]:
            exec_obj = st.session_state["bt_executions"].get(sid)
            if exec_obj:
                executions.append(exec_obj)
        return executions

    def get_execution(self, strategy_id: str) -> StrategyExecution | None:
        return st.session_state["bt_executions"].get(strategy_id)

    def update_execution_status(
        self,
        strategy_id: str,
        status: StrategyStatus,
        progress: float | None = None,
        current_stage: str | None = None,
        **kwargs
    ) -> None:
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if exec_obj:
            exec_obj.status = status
            if progress is not None:
                exec_obj.progress = progress
            if current_stage is not None:
                exec_obj.current_stage = current_stage
            for k, v in kwargs.items():
                if hasattr(exec_obj, k):
                    setattr(exec_obj, k, v)

    def start_execution(self, strategy_id: str) -> None:
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if exec_obj:
            exec_obj.status = StrategyStatus.INITIALIZING
            exec_obj.started_at = datetime.now()

    def complete_execution(self, strategy_id: str, result: Any) -> None:
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if exec_obj:
            exec_obj.status = StrategyStatus.COMPLETED
            exec_obj.progress = 1.0
            exec_obj.completed_at = datetime.now()
            exec_obj.result = result
            # Move to completed
            st.session_state["bt_completed"][strategy_id] = exec_obj

    def fail_execution(self, strategy_id: str, error: str) -> None:
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if exec_obj:
            exec_obj.status = StrategyStatus.FAILED
            exec_obj.error = error
            exec_obj.completed_at = datetime.now()
            st.session_state["bt_completed"][strategy_id] = exec_obj

    def cancel_execution(self, strategy_id: str) -> None:
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if exec_obj and exec_obj.status in (StrategyStatus.QUEUED, StrategyStatus.INITIALIZING, StrategyStatus.RUNNING):
            exec_obj.status = StrategyStatus.CANCELLED
            exec_obj.completed_at = datetime.now()

    def remove_from_queue(self, strategy_id: str) -> None:
        if strategy_id in st.session_state["bt_queue"]:
            st.session_state["bt_queue"].remove(strategy_id)

    # --- Completed Results ---

    def get_completed(self) -> list[StrategyExecution]:
        return list(st.session_state["bt_completed"].values())

    def get_completed_execution(self, strategy_id: str) -> StrategyExecution | None:
        return st.session_state["bt_completed"].get(strategy_id)

    def select_result(self, strategy_id: str) -> None:
        st.session_state["bt_selected_result"] = strategy_id

    def get_selected_result(self) -> StrategyExecution | None:
        sid = st.session_state.get("bt_selected_result")
        if sid:
            return st.session_state["bt_completed"].get(sid)
        return None

    # --- Background Execution ---

    def run_backtest_async(self, strategy_id: str, storage) -> str:
        """Run backtest in background thread."""
        exec_obj = st.session_state["bt_executions"].get(strategy_id)
        if not exec_obj:
            return ""

        run_id = f"bt_{time.time()}"
        exec_obj.run_id = run_id  # Store run_id for polling
        st.session_state["bt_run_id"] = run_id

        _new_progress_bucket(run_id)
        handler = _make_progress_handler(run_id)

        def _worker():
            try:
                from core.data.storage.provisioning import ensure_database
                from app.services import get_storage

                ensure_database()
                # Reuse the process-wide read-only connection (app.services.get_storage)
                # instead of opening a second DuckDB handle. On Windows a second handle
                # contends with the live app's connection on the WAL/OS file lock and
                # the data load either stalls or fails fast with a mute "Failed" at
                # "Loading data". A single shared handle + the I/O lock removes that
                # race entirely.
                worker_storage = get_storage()

                # NOTE: the worker runs in a background thread where st.session_state is
                # NOT populated, so it must never read/write st.session_state directly.
                # All status/progress is pushed to the thread-safe _PROGRESS_STORE and
                # applied to st.session_state by the main-thread poll loop
                # (_update_execution_from_progress). The "Loading data..." stage is set
                # there from the bucket's phase="loading".

                # Serialise storage-heavy execution to avoid DuckDB read contention
                # (see _BACKTEST_IO_LOCK). The GIL is released during the C-level
                # DuckDB scans, so other threads (including the UI poll loop) keep
                # running while this waits its turn.
                with _BACKTEST_IO_LOCK:
                    result = run_backtest(exec_obj.config.params, worker_storage, progress_callback=handler)

                with _PROGRESS_LOCK:
                    bucket = _PROGRESS_STORE.get(run_id)
                    if bucket is not None:
                        bucket["result"] = result
                        bucket["params"] = exec_obj.config.params
                        bucket["phase"] = "done"
            except Exception as exc:
                import traceback as _tb
                with _PROGRESS_LOCK:
                    bucket = _PROGRESS_STORE.get(run_id)
                    if bucket is not None:
                        bucket["error"] = f"{type(exc).__name__}: {exc}\n\n{_tb.format_exc()}"
            finally:
                with _PROGRESS_LOCK:
                    bucket = _PROGRESS_STORE.get(run_id)
                    if bucket is not None:
                        bucket["running"] = False

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        return run_id

    def poll_progress(self, run_id: str) -> dict | None:
        """Poll progress from background worker."""
        return _drain_progress(run_id)

    # --- Section Navigation ---

    def set_active_section(self, section: str) -> None:
        st.session_state["bt_active_section"] = section

    def get_active_section(self) -> str:
        return st.session_state.get("bt_active_section", "manual_testing")


def get_backtest_state() -> BacktestStateManager:
    """Get the singleton BacktestStateManager instance.

    Also guarantees the workflow's session-state keys exist for the *current*
    Streamlit session. The singleton is process-wide and may have been created
    in a prior run (where ``__init__`` already ran), so the one-time
    initialization cannot be relied upon to seed a brand-new session's state.
    """
    manager = BacktestStateManager()
    manager._ensure_session_state()
    return manager