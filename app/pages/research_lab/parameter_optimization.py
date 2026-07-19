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

from typing import Any

import streamlit as st

from app.layouts.base import section
from app.pages.backtest.state import get_backtest_state
from core.config.backtest_schema import BacktestParameters
from core.optimization import (
    OptimizableParameter,
    ParamType,
    SearchSpaceEstimate,
    discover_parameters,
    estimate_search_space,
    group_by_category,
)


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
def _render_parameter_selection(parameters: list[OptimizableParameter]) -> dict[str, dict[str, Any]]:
    section("3. Parameter Selection")
    st.markdown("Enable optimization for any parameter below. The search space is built "
                "from your selections. Parameters are discovered automatically from the "
                "base strategy and require no hardcoding.")

    groups = group_by_category(parameters)
    selections: dict[str, dict[str, Any]] = {}

    for category, params in groups.items():
        with st.expander(f"{category}", expanded=True):
            for p in params:
                key_enable = f"po_en_{p.key}"
                enabled = st.checkbox(f"**{p.name}**", value=False, key=key_enable)

                col_cur, col_min, col_max, col_step, col_type = st.columns([2, 2, 2, 2, 2])
                with col_cur:
                    st.text_input("Current Value", value=_fmt(p.current_value),
                                  key=f"po_cur_{p.key}", disabled=True)
                with col_min:
                    min_val = st.number_input("Minimum", value=_coerce(p, p.min_value, "min"),
                                              key=f"po_min_{p.key}",
                                              disabled=not enabled,
                                              format=_fmt_format(p))
                with col_max:
                    max_val = st.number_input("Maximum", value=_coerce(p, p.max_value, "max"),
                                              key=f"po_max_{p.key}",
                                              disabled=not enabled,
                                              format=_fmt_format(p))
                with col_step:
                    step_val = st.number_input("Step Size", value=_coerce(p, p.step, "step"),
                                               key=f"po_step_{p.key}",
                                               disabled=not enabled,
                                               format=_fmt_format(p))
                with col_type:
                    st.text_input("Parameter Type", value=p.param_type.value,
                                  key=f"po_type_{p.key}", disabled=True)
                if p.validation.help:
                    st.caption(p.validation.help)

                if enabled:
                    entry: dict[str, Any] = {
                        "enabled": True,
                        "min": min_val,
                        "max": max_val,
                        "step": step_val,
                    }
                    if p.param_type == ParamType.CHOICE and p.choices:
                        entry["choices"] = list(p.choices)
                    selections[p.key] = entry
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


def _fmt_format(p: OptimizableParameter) -> str:
    return "%g" if p.param_type in (ParamType.FLOAT, ParamType.INT) else None  # type: ignore[return-value]


def _coerce(p: OptimizableParameter, v: Any, which: str) -> Any:
    # Choice / bool parameters have no meaningful numeric min/max/step.
    if p.param_type in (ParamType.CHOICE, ParamType.BOOL):
        return 0 if v is None else v
    if v is None:
        cur = float(p.current_value)
        if which == "min":
            return cur - 1.0
        if which == "max":
            return cur + 1.0
        if which == "step":
            return 0.1
    if p.param_type == ParamType.FLOAT and v is not None:
        return float(v)
    if p.param_type == ParamType.INT and v is not None:
        return int(v)
    return v


# --- Section 4 --------------------------------------------------------------
def _render_search_space_summary(
    parameters: list[OptimizableParameter],
    selections: dict[str, dict[str, Any]],
    algorithm: str,
    max_iterations: int,
) -> SearchSpaceEstimate:
    section("4. Search Space Summary")
    selected = [p for p in parameters if p.key in selections]
    estimate = estimate_search_space(algorithm, selected, max_iterations=max_iterations)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Optimized Parameters", estimate.parameter_count)
    c2.metric("Search Space Size", _fmt_large(estimate.search_space_size))
    c3.metric("Est. Backtests", _fmt_large(estimate.estimated_backtests))
    c4.metric("Est. Runtime", estimate.estimated_runtime_label)
    c5.metric("Active Algorithm", algorithm.replace("_", " ").title())

    if selected:
        st.markdown("**Optimized Parameters:** " + ", ".join(p.name for p in selected))
    else:
        st.info("No parameters selected yet. Enable parameters in Section 3 to build the search space.")
    st.divider()
    return estimate


# --- Section 5 --------------------------------------------------------------
_SINGLE_OBJECTIVES = [
    "Maximum CAGR",
    "Maximum Sharpe Ratio",
    "Maximum Sortino Ratio",
    "Maximum Calmar Ratio",
    "Minimum Maximum Drawdown",
    "Maximum Final Portfolio Value",
    "Minimum Portfolio Volatility",
    "Maximum Information Ratio",
]


def _render_objective() -> dict[str, Any]:
    section("5. Optimization Objective")
    mode = st.radio(
        "Optimization Mode",
        options=["Single Objective", "Target-Based", "Multi-Objective"],
        key="po_obj_mode",
        horizontal=True,
        help="Mode 3 (Multi-Objective) builds the UI only; ranking is future work.",
    )

    result: dict[str, Any] = {"mode": mode}

    if mode == "Single Objective":
        obj = st.selectbox("Objective Metric", _SINGLE_OBJECTIVES, key="po_single_obj")
        result["objective"] = obj

    elif mode == "Target-Based":
        st.markdown("Specify desired target values. Every field is optional — "
                    "leave a field blank to ignore it. The engine ranks candidates "
                    "by proximity to the provided targets.")
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
        # Drop zero-valued (left blank) targets.
        targets = {k: v for k, v in targets.items() if v != 0.0}
        result["targets"] = targets
        if not targets:
            st.caption("No targets specified — all candidates will be considered equally valid.")

    else:  # Multi-Objective
        st.markdown("Select the metrics to optimize simultaneously. The engine will "
                    "later produce a Pareto-optimal solution set (UI scaffold only).")
        metrics = ["CAGR", "Sharpe", "Sortino", "Calmar", "Drawdown", "Volatility", "Information Ratio"]
        chosen = []
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            if cols[i].checkbox(m, key=f"po_mo_{i}", value=(i < 3)):
                chosen.append(m)
        result["metrics"] = chosen
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
_ALGORITHMS = [
    "Grid Search",
    "Random Search",
    "Bayesian Optimization",
    "Genetic Algorithm",
    "Particle Swarm Optimization",
    "Simulated Annealing",
]
_ALGO_TO_KEY = {a: a.lower().replace(" ", "_") for a in _ALGORITHMS}


def _render_algorithm() -> str:
    section("7. Optimization Algorithm")
    st.markdown("Select the search algorithm. Algorithm implementations will be added later.")
    chosen = st.selectbox("Algorithm", _ALGORITHMS, key="po_algorithm", index=0)
    st.caption("Interface only — no optimization executes in the current scope.")
    st.divider()
    return _ALGO_TO_KEY[chosen]


# --- Section 8 --------------------------------------------------------------
def _render_execution_config() -> dict[str, Any]:
    section("8. Execution Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        max_iter = st.number_input("Maximum Iterations", key="po_ex_iter", value=200, step=50, min_value=1)
        max_runtime = st.number_input("Maximum Runtime (min)", key="po_ex_rt", value=60, step=10, min_value=1)
    with c2:
        workers = st.number_input("Parallel Workers", key="po_ex_workers", value=1, step=1, min_value=1)
        seed = st.number_input("Random Seed", key="po_ex_seed", value=42, step=1)
    with c3:
        early_stop = st.checkbox("Early Stopping", key="po_ex_early", value=False)
        save_inter = st.checkbox("Save Intermediate Results", key="po_ex_save", value=True)
        checkpoint = st.number_input("Checkpoint Frequency", key="po_ex_cp", value=10, step=5, min_value=1)

    config = {
        "max_iterations": int(max_iter),
        "max_runtime_min": int(max_runtime),
        "parallel_workers": int(workers),
        "random_seed": int(seed),
        "early_stopping": early_stop,
        "save_intermediate": save_inter,
        "checkpoint_frequency": int(checkpoint),
    }
    st.divider()
    return config


# --- Section 9 --------------------------------------------------------------
def _render_summary(
    base_label: str,
    parameters: list[OptimizableParameter],
    selections: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    algorithm: str,
    constraints: dict[str, Any],
    exec_config: dict[str, Any],
    estimate: SearchSpaceEstimate,
) -> None:
    section("9. Optimization Summary")
    selected = [p for p in parameters if p.key in selections]

    obj_label = objective.get("objective") or objective.get("mode")
    if objective.get("mode") == "Target-Based":
        obj_label = "Target-Based (" + ", ".join(objective.get("targets", {}).keys()) + ")"
    elif objective.get("mode") == "Multi-Objective":
        obj_label = "Multi-Objective (" + ", ".join(objective.get("metrics", [])) + ")"

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

    if selected:
        st.markdown("**Selected Parameters:** " + ", ".join(p.name for p in selected))
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

    if objective.get("mode") == "Single Objective" and not objective.get("objective"):
        errors.append("Select a single-objective metric (Section 5).")
    if objective.get("mode") == "Multi-Objective" and not objective.get("metrics"):
        errors.append("Select at least one metric for multi-objective optimization (Section 5).")
    return errors


def _render_run(
    selections: dict[str, dict[str, Any]],
    objective: dict[str, Any],
    constraints: dict[str, Any],
    exec_config: dict[str, Any],
    algorithm: str,
) -> None:
    section("10. Run Parameter Optimization")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Run Parameter Optimization", type="primary", use_container_width=True):
            errors = _validate(selections, objective, constraints)
            if errors:
                for e in errors:
                    st.error(e)
                st.warning("Configuration is invalid. Resolve the issues above before running.")
            else:
                st.success("Configuration validated successfully. "
                           "The optimization engine is ready to execute "
                           "(backend execution will be added in a future update).")
                with st.expander("Validated Configuration Snapshot", expanded=False):
                    st.json({
                        "algorithm": algorithm,
                        "parameters": selections,
                        "objective": objective,
                        "constraints": constraints,
                        "execution": exec_config,
                    })


# --- Orchestration ----------------------------------------------------------
def render() -> None:
    """Render the complete Parameter Optimization configuration page."""
    section("Parameter Optimization")
    st.caption("Research Lab · Generic, algorithm-agnostic parameter optimization engine")

    _render_overview()

    base_params = _render_base_strategy_selection()
    base_label = st.session_state.get(_BASE_STRATEGY_KEY, "None")

    parameters = discover_parameters(base_params)
    selections = _render_parameter_selection(parameters)

    algorithm = _render_algorithm()
    exec_config = _render_execution_config()
    estimate = _render_search_space_summary(parameters, selections, algorithm, exec_config["max_iterations"])

    objective = _render_objective()
    constraints = _render_constraints()

    _render_summary(base_label, parameters, selections, objective, algorithm, constraints, exec_config, estimate)
    _render_run(selections, objective, constraints, exec_config, algorithm)


if __name__ == "__main__":
    render()
