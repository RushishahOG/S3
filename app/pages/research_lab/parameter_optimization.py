"""Parameter Optimization Research Lab module.

This is the central, generic optimization configuration page of the ARQM
Research Lab. It is built entirely on the algorithm-agnostic discovery
framework in :mod:`core.optimization`:

* it **discovers** every optimizable parameter from the selected base
  strategy via :func:`core.optimization.discover_parameters`;
* it renders a **dynamic configuration form** (no parameter is hardcoded into
  the UI);
* it computes a **live search-space estimate** as the user toggles parameters;
* it validates the entire configuration; and
* it exposes a single *Run Parameter Optimization* action that currently only
  validates (no backtests are executed yet).

Future optimization algorithms, objectives, and result-ranking logic plug in
without touching this page's architecture.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

import streamlit as st

from app.layouts.base import section
from app.pages.backtest.state import get_backtest_state
from core.config.backtest_schema import BacktestParameters
from core.optimization import estimate_search_space
from core.optimization.engine import run_optimization
from core.optimization.objectives import DEFAULT_OBJECTIVE, OBJECTIVES, available_objectives
from core.optimization.results import OptimizationRun
from core.optimization.search_space import SearchSpaceEstimate
from core.optimization.spec import get_spec, get_specs


# --- Section 1 --------------------------------------------------------------
def _render_overview() -> None:
    section("1. Optimization Overview")
    st.markdown(
        """
        **Parameter Optimization** automatically discovers the most robust
        combination of configurable ARQM strategy parameters.

        - The engine repeatedly executes historical backtests using different
          parameter combinations drawn from a search space you define.
        - Each candidate strategy is scored against your chosen objective.
        - The goal is to identify *robust* parameter sets that perform well
          across conditions — not merely to maximise a single metric.

        This page is the configuration front-end for that engine. No backtests
        are executed here yet; pressing **Run Parameter Optimization** validates
        your setup so it is ready for the optimization engine.
        """
    )
    st.divider()


# --- Section 2 --------------------------------------------------------------
_BASE_STRATEGY_KEY = "po_base_strategy"


def _render_base_strategy_selection() -> BacktestParameters | None:
    section("2. Base Strategy Selection")
    st.markdown("Choose the strategy that will act as the optimization template. "
                "Only one strategy may be selected.")

    state = get_backtest_state()
    saved = state.list_strategies()
    completed = state.get_completed()

    sources: list[tuple[str, str, Any]] = []
    sources.append(("current", "Current Manual Testing Configuration", "current"))
    for s in saved:
        sources.append(("saved", f"Saved Strategy: {s.name} ({s.config_id})", s))
    for c in completed:
        sources.append(("completed", f"Completed Backtest: {c.config.name} ({c.strategy_id})", c))
    sources.append(("imported", "Imported Configuration (JSON)", "imported"))

    labels = [s[1] for s in sources]
    current_label = st.session_state.get(_BASE_STRATEGY_KEY)
    idx = 0
    if current_label in labels:
        idx = labels.index(current_label)

    chosen = st.selectbox(
        "Base Strategy Source",
        options=labels,
        index=idx,
        key="po_base_source",
        help="The selected strategy seeds the current values of every optimizable parameter.",
    )
    st.session_state[_BASE_STRATEGY_KEY] = chosen

    source = next(s for s in sources if s[1] == chosen)
    kind, payload = source[0], source[2]

    base_params: BacktestParameters | None = None
    if kind == "current":
        try:
            from app.pages.backtest.manual_testing.render import _build_config_from_session
            base_params = _build_config_from_session()
        except Exception:
            base_params = None
        st.caption("Using the live configuration from the Manual Testing page.")
    elif kind == "saved":
        base_params = payload.params
        st.caption(f"Loaded saved strategy **{payload.name}**.")
    elif kind == "completed":
        base_params = payload.config.params
        st.caption(f"Loaded completed backtest **{payload.config.name}**.")
    elif kind == "imported":
        uploaded = st.file_uploader("Import configuration JSON", type=["json"], key="po_import")
        if uploaded is not None:
            import json
            try:
                data = json.loads(uploaded.getvalue().decode("utf-8"))
                base_params = BacktestParameters.from_dict(data)
                st.success("Configuration imported.")
            except Exception as exc:
                st.error(f"Failed to import configuration: {exc}")
                base_params = None
        else:
            st.info("Upload a JSON configuration file to use as the template.")

    if base_params is None:
        st.warning("No base strategy loaded — schema defaults will be used for parameter discovery.")
    st.divider()
    return base_params


# --- Section 3 --------------------------------------------------------------
def _current_value_from_base(base, block: str, field: str, fallback):
    """Resolve a parameter's live current value from the base configuration."""
    if base is None:
        return fallback
    try:
        return getattr(getattr(base, block), field, fallback)
    except Exception:
        return fallback


def _render_parameter_selection(base_params) -> dict[str, dict[str, Any]]:
    section("3. Parameter Selection")
    st.markdown("Enable optimization for any parameter below. The search space is built "
                "from your selections. Parameters are discovered automatically from the "
                "optimization engine's registered metadata and require no hardcoding.")

    from core.optimization.spec import get_specs
    specs = get_specs()
    groups: dict[str, list] = {}
    for s in specs:
        groups.setdefault(s.category, []).append(s)

    # Select-all / Clear-all controls for the whole parameter table.
    c_all, c_none, c_info = st.columns([1, 1, 4])
    with c_all:
        if st.button("Select All", key="po_sel_all", use_container_width=True):
            for s in specs:
                st.session_state[f"po_en_{s.key}"] = True
            st.rerun()
    with c_none:
        if st.button("Clear All", key="po_sel_none", use_container_width=True):
            for s in specs:
                st.session_state[f"po_en_{s.key}"] = False
            st.rerun()
    with c_info:
        st.caption(f"{len(specs)} optimizable parameters discovered across "
                   f"{len(groups)} categories.")

    selections: dict[str, dict[str, Any]] = {}

    for category, params in groups.items():
        with st.expander(f"{category}", expanded=True):
            for s in params:
                cur = _current_value_from_base(base_params, s.block, s.field, s.current)
                key_enable = f"po_en_{s.key}"
                enabled = st.checkbox(f"**{s.name}**", value=False, key=key_enable)

                col_cur, col_min, col_max, col_step, col_type = st.columns([2, 2, 2, 2, 2])
                with col_cur:
                    st.text_input("Current Value", value=_fmt(cur),
                                  key=f"po_cur_{s.key}", disabled=True)
                numeric = s.kind.value in ("continuous", "discrete")
                with col_min:
                    min_val = st.number_input("Minimum", value=_coerce_num(s, s.min, "min", cur),
                                              key=f"po_min_{s.key}", disabled=not enabled,
                                              format="%g")
                with col_max:
                    max_val = st.number_input("Maximum", value=_coerce_num(s, s.max, "max", cur),
                                              key=f"po_max_{s.key}", disabled=not enabled,
                                              format="%g")
                with col_step:
                    step_val = st.number_input("Step Size", value=_coerce_num(s, s.step, "step", cur),
                                               key=f"po_step_{s.key}", disabled=not enabled,
                                               format="%g")
                with col_type:
                    st.text_input("Parameter Type", value=s.kind.value,
                                  key=f"po_type_{s.key}", disabled=True)
                if s.kind.value == "categorical" and s.allowed:
                    st.caption("Allowed values: " + ", ".join(str(a) for a in s.allowed))
                if s.help:
                    st.caption(s.help)

                if enabled:
                    entry: dict[str, Any] = {
                        "enabled": True,
                        "min": min_val,
                        "max": max_val,
                        "step": step_val,
                    }
                    if s.kind.value == "categorical" and s.allowed:
                        entry["choices"] = list(s.allowed)
                    selections[s.key] = entry
                st.divider()

    st.divider()
    return selections


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _fmt_large(n: int) -> str:
    """Compact display for very large counts (millions+ shown as ~M/~B/~T)."""
    if n < 1_000_000:
        return f"{n:,}"
    for suffix, threshold in (("T", 1_000_000_000_000), ("B", 1_000_000_000), ("M", 1_000_000)):
        if n >= threshold:
            return f"~{n / threshold:.2f}{suffix}"
    return f"{n:,}"


def _coerce_num(s, v: Any, which: str, cur) -> float:
    """Coerce a min/max/step default for a numeric spec."""
    if s.kind.value not in ("continuous", "discrete") or v is None:
        return float(v) if v is not None else 0.0
    if which == "step":
        return float(v)
    # For min/max, if absent, derive from the current value.
    try:
        c = float(cur)
    except (TypeError, ValueError):
        c = 0.0
    if which == "min":
        return float(v) if v is not None else (c - 1.0)
    if which == "max":
        return float(v) if v is not None else (c + 1.0)
    return float(v)


# --- Section 4 --------------------------------------------------------------
def _render_search_space_summary(
    selections: dict[str, dict[str, Any]],
    algorithm: str,
    max_iterations: int,
) -> SearchSpaceEstimate:
    section("4. Search Space Summary")
    selected_specs = [s for s in get_specs() if s.key in selections]
    estimate = estimate_search_space(algorithm, selected_specs, selections=selections, max_iterations=max_iterations)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Optimized Parameters", estimate.parameter_count)
    c2.metric("Search Space Size", _fmt_large(estimate.search_space_size))
    c3.metric("Est. Backtests", _fmt_large(estimate.estimated_backtests))
    c4.metric("Est. Runtime", estimate.estimated_runtime_label)
    c5.metric("Active Algorithm", algorithm.replace("_", " ").title())

    if selected_specs:
        st.markdown("**Optimized Parameters:** " + ", ".join(s.name for s in selected_specs))
    else:
        st.info("No parameters selected yet. Enable parameters in Section 3 to build the search space.")
    st.divider()
    return estimate


# --- Section 5 --------------------------------------------------------------
# Map UI objective labels -> engine objective keys.
_OBJECTIVE_LABELS = {obj.label: obj.key for obj in available_objectives()}
_OBJECTIVE_ORDER = [obj.label for obj in available_objectives()]


def _render_objective() -> dict[str, Any]:
    section("5. Optimization Objective")
    mode = st.radio(
        "Optimization Mode",
        options=["Single Objective", "Target-Based", "Multi-Objective"],
        key="po_obj_mode",
        horizontal=True,
        help="The first implementation evaluates a single objective. "
             "Target-Based and Multi-Objective define the UI; ranking uses the "
             "single-objective engine for now.",
    )

    result: dict[str, Any] = {"mode": mode}

    if mode == "Single Objective":
        idx = _OBJECTIVE_ORDER.index(OBJECTIVES[DEFAULT_OBJECTIVE].label)
        obj_label = st.selectbox("Objective Metric", _OBJECTIVE_ORDER, index=idx, key="po_single_obj")
        result["objective"] = _OBJECTIVE_LABELS[obj_label]
        result["objective_label"] = obj_label

    elif mode == "Target-Based":
        st.markdown("Specify desired target values. Every field is optional — "
                    "leave a field blank to ignore it.")
        st.info("Example — Conservative: Expected CAGR ≥ 18%, Max Drawdown ≤ 10%, "
                "Max Volatility ≤ 14%.\n\nExample — Balanced: Expected CAGR ≥ 22%, "
                "Sharpe ≥ 1.50, Max Drawdown ≤ 15%.")
        targets: dict[str, Any] = {}
        t1, t2, t3 = st.columns(3)
        with t1:
            targets["expected_cagr"] = st.number_input("Expected CAGR (%)", key="po_t_cagr", value=0.0, step=0.5)
            targets["expected_sharpe"] = st.number_input("Expected Sharpe Ratio", key="po_t_sharpe", value=0.0, step=0.1)
        with t2:
            targets["expected_sortino"] = st.number_input("Expected Sortino Ratio", key="po_t_sortino", value=0.0, step=0.1)
            targets["expected_calmar"] = st.number_input("Expected Calmar Ratio", key="po_t_calmar", value=0.0, step=0.1)
        with t3:
            targets["max_drawdown"] = st.number_input("Maximum Acceptable Drawdown (%)", key="po_t_mdd", value=0.0, step=0.5)
            targets["max_volatility"] = st.number_input("Maximum Acceptable Volatility (%)", key="po_t_vol", value=0.0, step=0.5)
            targets["min_information_ratio"] = st.number_input("Minimum Information Ratio", key="po_t_ir", value=0.0, step=0.1)
        targets = {k: v for k, v in targets.items() if v != 0.0}
        result["targets"] = targets
        result["objective"] = DEFAULT_OBJECTIVE
        result["objective_label"] = OBJECTIVES[DEFAULT_OBJECTIVE].label
        if not targets:
            st.caption("No targets specified — all candidates will be considered equally valid.")

    else:  # Multi-Objective
        st.markdown("Select the metrics to optimize simultaneously. The engine will "
                    "later produce a Pareto-optimal solution set (UI scaffold).")
        metrics = ["CAGR", "Sharpe", "Sortino", "Calmar", "Drawdown", "Volatility", "Information Ratio"]
        chosen = []
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            if cols[i].checkbox(m, key=f"po_mo_{i}", value=(i < 3)):
                chosen.append(m)
        result["metrics"] = chosen
        result["objective"] = DEFAULT_OBJECTIVE
        result["objective_label"] = OBJECTIVES[DEFAULT_OBJECTIVE].label
        if not chosen:
            st.warning("Select at least one metric for multi-objective optimization.")

    st.divider()
    return result


# --- Section 6 --------------------------------------------------------------
def _render_constraints() -> dict[str, Any]:
    section("6. Optimization Constraints")
    st.markdown("Optional constraints applied before a candidate strategy is accepted.")
    c1, c2, c3 = st.columns(3)
    constraints: dict[str, Any] = {}
    with c1:
        max_dd = st.number_input("Maximum Drawdown (%)", key="po_c_mdd", value=0.0, step=1.0)
        min_cagr = st.number_input("Minimum CAGR (%)", key="po_c_cagr", value=0.0, step=1.0)
        max_vol = st.number_input("Maximum Volatility (%)", key="po_c_vol", value=0.0, step=1.0)
    with c2:
        min_sharpe = st.number_input("Minimum Sharpe", key="po_c_sharpe", value=0.0, step=0.1)
        min_trades = st.number_input("Minimum Number of Trades", key="po_c_trades", value=0, step=1)
        max_turnover = st.number_input("Maximum Portfolio Turnover", key="po_c_turn", value=0.0, step=0.1)
    with c3:
        max_w = st.number_input("Maximum Stock Weight", key="po_c_maxw", value=0.0, step=0.01)
        min_lc = st.number_input("Min Large Cap Allocation (%)", key="po_c_minlc", value=0.0, step=1.0)
        max_lc = st.number_input("Max Large Cap Allocation (%)", key="po_c_maxlc", value=0.0, step=1.0)
        min_mc = st.number_input("Min Mid Cap Allocation (%)", key="po_c_minmc", value=0.0, step=1.0)
        max_mc = st.number_input("Max Mid Cap Allocation (%)", key="po_c_maxmc", value=0.0, step=1.0)
        min_sc = st.number_input("Min Small Cap Allocation (%)", key="po_c_minsc", value=0.0, step=1.0)
        max_sc = st.number_input("Max Small Cap Allocation (%)", key="po_c_maxsc", value=0.0, step=1.0)

    mapping = {
        "max_drawdown_pct": max_dd, "min_cagr_pct": min_cagr, "max_volatility_pct": max_vol,
        "min_sharpe": min_sharpe, "min_trades": min_trades, "max_turnover": max_turnover,
        "max_stock_weight": max_w, "min_large_cap_pct": min_lc, "max_large_cap_pct": max_lc,
        "min_mid_cap_pct": min_mc, "max_mid_cap_pct": max_mc,
        "min_small_cap_pct": min_sc, "max_small_cap_pct": max_sc,
    }
    for k, v in mapping.items():
        if v not in (0, 0.0):
            constraints[k] = v
    st.divider()
    return constraints


# --- Section 7 --------------------------------------------------------------
# Algorithms implemented in the engine. The architecture allows adding Bayesian,
# Differential Evolution, Genetic Algorithm, PSO and Simulated Annealing later
# without changing the engine (see core/optimization/algorithms.py).
_ALGORITHM_CHOICES = {
    "Grid Search": "grid_search",
    "Random Search": "random_search",
    "SLSQP": "slsqp",
}


def _render_algorithm() -> str:
    section("7. Optimization Algorithm")
    st.markdown("Select the search algorithm. Grid Search, Random Search and SLSQP "
                "are implemented; the strategy pattern allows adding more (Bayesian, "
                "Genetic Algorithm, PSO, Simulated Annealing) without engine changes.")
    chosen = st.selectbox("Algorithm", list(_ALGORITHM_CHOICES.keys()),
                          key="po_algorithm", index=1)  # default: Random Search
    st.divider()
    return _ALGORITHM_CHOICES[chosen]


# --- Section 8 --------------------------------------------------------------
def _render_execution_config() -> dict[str, Any]:
    section("8. Execution Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_iter = st.number_input("Maximum Iterations", key="po_ex_iter", value=10, step=5, min_value=1)
        max_runtime = st.number_input("Maximum Runtime (min)", key="po_ex_rt", value=60, step=10, min_value=1)
    with c2:
        top_n = st.number_input("Top N Strategies", key="po_ex_topn", value=20, step=5, min_value=1)
        seed = st.number_input("Random Seed", key="po_ex_seed", value=42, step=1)
    with c3:
        workers = st.number_input("Parallel Workers", key="po_ex_workers", value=1, step=1, min_value=1)
        save_inter = st.checkbox("Save Results to Disk", key="po_ex_save", value=True)

    config = {
        "max_iterations": int(max_iter),
        "max_runtime_min": int(max_runtime),
        "top_n": int(top_n),
        "parallel_workers": int(workers),
        "random_seed": int(seed),
        "save_intermediate": save_inter,
    }
    st.divider()
    return config


# --- Section 9 --------------------------------------------------------------
def _render_summary(
    base_label: str,
    selections: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    algorithm: str,
    constraints: dict[str, Any],
    exec_config: dict[str, Any],
    estimate: SearchSpaceEstimate,
) -> None:
    section("9. Optimization Summary")
    selected_specs = [s for s in get_specs() if s.key in selections]

    obj_label = objective.get("objective_label") or objective.get("objective") or objective.get("mode")
    if objective.get("mode") == "Target-Based":
        obj_label = "Target-Based (engine uses " + OBJECTIVES[DEFAULT_OBJECTIVE].label + ")"
    elif objective.get("mode") == "Multi-Objective":
        obj_label = "Multi-Objective (engine uses " + OBJECTIVES[DEFAULT_OBJECTIVE].label + ")"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Selected Strategy:** {base_label}")
        st.markdown(f"**Optimization Mode:** {objective.get('mode')}")
        st.markdown(f"**Optimization Objective:** {obj_label}")
        st.markdown(f"**Selected Algorithm:** {algorithm.replace('_', ' ').title()}")
    with col2:
        st.markdown(f"**Parameters Selected:** {estimate.parameter_count}")
        st.markdown(f"**Estimated Search Space:** {_fmt_large(estimate.search_space_size)}")
        st.markdown(f"**Estimated Runtime:** {estimate.estimated_runtime_label}")
        st.markdown(f"**Constraints:** {len(constraints)} defined")

    if selected_specs:
        st.markdown("**Selected Parameters:** " + ", ".join(s.name for s in selected_specs))
    if constraints:
        st.markdown("**Active Constraints:** " + ", ".join(constraints.keys()))
    st.divider()


# --- Section 10 -------------------------------------------------------------
def _validate(
    selections: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    constraints: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    if not selections:
        errors.append("Select at least one parameter to optimize (Section 3).")
    for key, sel in selections.items():
        if sel.get("min") is not None and sel.get("max") is not None and sel["min"] > sel["max"]:
            errors.append(f"{key}: Minimum exceeds Maximum.")
        if sel.get("step") is not None and sel["step"] <= 0:
            errors.append(f"{key}: Step size must be positive.")
    if objective.get("mode") == "Multi-Objective" and not objective.get("metrics"):
        errors.append("Select at least one metric for multi-objective optimization (Section 5).")
    return errors


# --- Execution orchestration (background thread + progress store) ------------
_PO_PROGRESS: dict[str, dict] = {}
_PO_LOCK = threading.Lock()


def _persist_run(run: OptimizationRun) -> str | None:
    try:
        from core.utils.paths import PROJECT_ROOT
        out_dir = os.path.join(PROJECT_ROOT, "storage", "optimization_runs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"opt_run_{run.run_id}.json")
        run.save(path)
        return path
    except Exception:
        return None


def _launch_optimization(
    base_params: BacktestParameters,
    selected_keys: list[str],
    objective_key: str,
    algorithm_key: str,
    constraints: dict[str, Any],
    exec_config: dict[str, Any],
) -> str:
    """Launch the optimization in a background thread; return run_id."""
    from app.services import get_storage
    run_id = f"po_{int(time.time()*1000)}"
    with _PO_LOCK:
        _PO_PROGRESS[run_id] = {
            "running": True, "iteration": 0, "n_total": 0, "n_valid": 0,
            "best_score": None, "last_event": "", "done": False, "error": None,
        }

    def _progress(ev: dict) -> None:
        with _PO_LOCK:
            bucket = _PO_PROGRESS.get(run_id)
            if bucket is None:
                return
            bucket["last_event"] = ev.get("event", "")
            if ev.get("event") == "candidate_done":
                bucket["iteration"] = ev.get("iteration", 0)
            elif ev.get("event") == "done":
                bucket["done"] = True
                bucket["running"] = False
                bucket["n_valid"] = ev.get("n_valid", 0)
                bucket["n_total"] = ev.get("n_total", 0)
                bucket["best_score"] = ev.get("best_score")
                bucket["runtime"] = ev.get("runtime")

    def _worker() -> None:
        try:
            run = run_optimization(
                base=base_params,
                selected_keys=selected_keys,
                objective_key=objective_key,
                algorithm_key=algorithm_key,
                constraints=constraints,
                max_iterations=exec_config["max_iterations"],
                random_seed=exec_config["random_seed"],
                storage_factory=get_storage,
                max_runtime_seconds=exec_config["max_runtime_min"] * 60.0,
                top_n=exec_config["top_n"],
                progress_callback=_progress,
            )
            with _PO_LOCK:
                _PO_PROGRESS[run_id]["run"] = run
                if exec_config.get("save_intermediate"):
                    _persist_run(run)
        except Exception as exc:
            import traceback as _tb
            with _PO_LOCK:
                _PO_PROGRESS[run_id]["error"] = f"{type(exc).__name__}: {exc}\n{_tb.format_exc()}"

    threading.Thread(target=_worker, daemon=True).start()
    return run_id


def _render_run(
    base_params: BacktestParameters,
    selections: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    constraints: dict[str, Any],
    exec_config: dict[str, Any],
    algorithm: str,
) -> None:
    section("10. Run Parameter Optimization")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Parameter Optimization", type="primary", use_container_width=True, key="po_run"):
            errors = _validate(selections, objective, constraints)
            if errors:
                for e in errors:
                    st.error(e)
                st.warning("Configuration is invalid. Resolve the issues above before running.")
            else:
                selected_keys = list(selections.keys())
                run_id = _launch_optimization(
                    base_params, selected_keys, objective["objective"],
                    algorithm, constraints, exec_config,
                )
                st.session_state["po_run_id"] = run_id
                st.rerun()

    # Poll an in-flight run.
    run_id = st.session_state.get("po_run_id")
    if run_id:
        _render_run_progress(run_id)


def _render_run_progress(run_id: str) -> None:
    with _PO_LOCK:
        bucket = _PO_PROGRESS.get(run_id)
    if bucket is None:
        return
    if bucket.get("error"):
        st.error("Optimization failed:\n" + bucket["error"])
        return
    if not bucket.get("done"):
        st.info(f"Optimization running… iteration {bucket.get('iteration', 0)} "
                 f"(algorithm: {st.session_state.get('po_algorithm')})")
        st.progress(min(1.0, (bucket.get('iteration', 0) or 0) / max(1, st.session_state.get('po_ex_iter', 1))))
        st.caption("Backtests run sequentially under the shared storage lock. "
                   "This page auto-refreshes.")
        time.sleep(1.5)
        st.rerun()
        return
    # Done: store the run in session and render results.
    run: OptimizationRun | None = bucket.get("run")
    if run is None:
        return
    st.session_state["po_last_run"] = run
    st.success(f"Optimization complete. {bucket.get('n_valid', 0)} valid strategy(ies) "
               f"from {bucket.get('n_total', 0)} candidates in "
               f"{bucket.get('runtime', 0):.1f}s.")
    _render_results(run)
    # Clear the active run so reruns don't re-poll.
    with _PO_LOCK:
        _PO_PROGRESS.pop(run_id, None)
    st.session_state["po_run_id"] = None


# --- Section 11: Result Inspection -----------------------------------------
_METRIC_DISPLAY = [
    ("cagr", "annual_return"), ("sharpe", "sharpe"), ("sortino", "sortino"),
    ("calmar", "calmar"), ("max_drawdown", "max_drawdown"),
    ("volatility", "annual_volatility"), ("information_ratio", "information_ratio"),
    ("final_value", "final_portfolio_value"), ("turnover", "turnover"),
]


def _render_results(run: OptimizationRun) -> None:
    section("11. Optimization Results")
    if not run.results:
        st.warning("No valid candidate strategies were produced. Try widening the "
                   "search range or relaxing constraints.")
        return

    st.markdown(f"**Objective:** {OBJECTIVES.get(run.objective, OBJECTIVES[DEFAULT_OBJECTIVE]).label} "
                f"· **Algorithm:** {run.algorithm} · **Top {len(run.results)}**")

    rows = []
    for r in run.results:
        row = {"Rank": r.rank, "Score": round(r.objective_score, 4)}
        for label, key in _METRIC_DISPLAY:
            v = r.metrics.get(key)
            row[label] = round(v, 4) if isinstance(v, (int, float)) and v == v else None
        row["Runtime (s)"] = round(r.runtime_seconds, 1)
        rows.append(row)
    st.dataframe(rows, use_container_width=True)

    st.markdown("### Inspect & Apply Strategy")
    opts = {f"#{r.rank} — score {round(r.objective_score, 3)}": r for r in run.results}
    choice = st.selectbox("Select a strategy to inspect", list(opts.keys()), key="po_inspect")
    res = opts[choice]
    with st.expander(f"Strategy #{res.rank} details", expanded=True):
        st.markdown("**Parameter Values**")
        st.json({k: (_fmt(v) if isinstance(v, float) else v) for k, v in res.params.items()})
        st.markdown("**Backtest Metrics**")
        st.json({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.metrics.items()})

        # Build a full BacktestParameters for the best config to display
        # allocation / factor / quality composition.
        cfg = run.best_config() if res.rank == 1 else None
        if cfg is not None:
            st.markdown("**Portfolio Allocation (cap weights)**")
            st.json({k: round(v, 4) for k, v in cfg.cap_segment.weights.items()})
            st.markdown("**Factor Allocation (scoring weights)**")
            st.json({k: round(v, 4) for k, v in cfg.scoring.weights.items()})
            st.markdown("**Quality Pillar Weights**")
            st.json({k: round(v, 4) for k, v in cfg.quality.pillar_weights.items()})

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Apply to Manual Testing", key=f"po_apply_{res.candidate_id}"):
                _apply_to_manual_testing(run, res)
        with c2:
            _render_export_config(run, res)


def _apply_to_manual_testing(run: OptimizationRun, res) -> None:
    """Rebuild the config and save it into the backtest state as a strategy."""
    cfg = run.best_config() if res.rank == 1 else None
    if cfg is None:
        # Reconstruct from base + this candidate's params.
        try:
            base = BacktestParameters.from_dict(run.base_config)
            from core.optimization.candidate import build_candidate
            from core.optimization.spec import specs_for_keys
            specs = specs_for_keys(list(res.params.keys()))
            cfg = build_candidate(base, res.params, specs)
        except Exception as exc:
            st.error(f"Could not reconstruct configuration: {exc}")
            return
    state = get_backtest_state()
    name = f"Optimized_{run.algorithm}_{res.candidate_id}"
    config = state.save_strategy(name, cfg)
    st.success(f"Applied as '{config.name}' (ID: {config.config_id}). "
               f"Open Manual Testing to run or inspect it.")
    st.session_state["bt_active_section"] = "manual_testing"


def _render_export_config(run: OptimizationRun, res) -> None:
    """Export the candidate configuration as a downloadable JSON."""
    export = {
        "parameters_optimized": run.parameters_optimized,
        "objective": run.objective,
        "algorithm": run.algorithm,
        "params": res.params,
        "metrics": res.metrics,
    }
    st.download_button(
        "Export Configuration (JSON)",
        data=json.dumps(export, indent=2, default=str),
        file_name=f"opt_config_{res.candidate_id}.json",
        mime="application/json",
        key=f"po_export_{res.candidate_id}",
    )


# --- Section 12: Persistence -------------------------------------------------
def _render_persistence() -> None:
    section("12. Saved Optimization Runs")
    from core.utils.paths import PROJECT_ROOT
    out_dir = os.path.join(PROJECT_ROOT, "storage", "optimization_runs")
    if not os.path.isdir(out_dir):
        st.info("No saved optimization runs yet.")
        return
    files = sorted([f for f in os.listdir(out_dir) if f.endswith(".json")], reverse=True)
    if not files:
        st.info("No saved optimization runs yet.")
        return
    sel = st.selectbox("Load a saved run", files, key="po_load_run")
    if st.button("Load Run", key="po_load_btn"):
        try:
            run = OptimizationRun.load(os.path.join(out_dir, sel))
            st.session_state["po_last_run"] = run
            st.success(f"Loaded run {run.run_id} ({len(run.results)} results).")
            _render_results(run)
        except Exception as exc:
            st.error(f"Failed to load run: {exc}")


# --- Orchestration ----------------------------------------------------------
def render() -> None:
    """Render the complete Parameter Optimization configuration page."""
    section("Parameter Optimization")
    st.caption("Research Lab · Generic, algorithm-agnostic parameter optimization engine")

    # Show a previously loaded/completed run at the top if present.
    if st.session_state.get("po_last_run") is not None and not st.session_state.get("po_run_id"):
        _render_results(st.session_state["po_last_run"])
        st.divider()

    _render_overview()

    base_params = _render_base_strategy_selection()
    base_label = st.session_state.get(_BASE_STRATEGY_KEY, "None")

    selections = _render_parameter_selection(base_params)

    algorithm = _render_algorithm()
    exec_config = _render_execution_config()
    estimate = _render_search_space_summary(selections, algorithm, exec_config["max_iterations"])

    objective = _render_objective()
    constraints = _render_constraints()

    _render_summary(base_label, selections, objective, algorithm, constraints, exec_config, estimate)
    _render_run(base_params, selections, objective, constraints, exec_config, algorithm)

    _render_persistence()


if __name__ == "__main__":
    render()
