"""Efficient Frontier Research Lab module.

Interactive mean-variance efficient frontier construction, portfolio
optimization, and risk-return trade-off analysis using Modern Portfolio Theory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.layouts.base import section
from app.services import get_storage
from core.portfolio.optimizer import (
    EfficientFrontierAnalysis,
    PortfolioConstraints,
)
from core.portfolio.visualization import (
    plot_efficient_frontier,
    plot_allocation_pie,
    plot_allocation_treemap,
    plot_risk_contribution,
)

_OBJECTIVE_LABELS = {
    "max_sharpe": "Maximum Sharpe Ratio",
    "min_volatility": "Minimum Volatility",
    "max_return": "Maximum Return",
    "max_calmar": "Maximum Calmar Ratio",
    "max_sortino": "Maximum Sortino Ratio",
    "min_drawdown": "Minimum Drawdown",
    "risk_parity": "Risk Parity",
    "equal_weight": "Equal Weight",
    "max_diversification": "Maximum Diversification",
    "min_correlation": "Minimum Correlation",
}

_RETURN_LABELS = {
    "arithmetic_mean": "Arithmetic Mean",
    "historical_cagr": "Historical CAGR",
    "geometric_mean": "Geometric Mean",
    "ema": "EMA (60-day)",
    "capm": "CAPM",
}

_COV_LABELS = {
    "sample": "Sample Covariance",
    "exponential": "Exponential Weighted",
    "ledoit_wolf": "Ledoit-Wolf Shrinkage",
    "oracle_approximating": "Oracle Approximating",
    "constant_correlation": "Constant Correlation",
}

_SOLVER_LABELS = {
    "slsqp": "SLSQP (recommended)",
    "differential_evolution": "Differential Evolution",
    "simulated_annealing": "Simulated Annealing",
    "quadratic_programming": "Quadratic Programming",
}


def render() -> None:
    """Render the Efficient Frontier section."""
    section("Efficient Frontier")
    st.caption(
        "Construct mean-variance efficient frontiers, optimize portfolios "
        "using Modern Portfolio Theory, and visualize the risk-return trade-off."
    )

    storage = get_storage()
    available_tickers = storage.stored_tickers()
    if not available_tickers:
        st.info(
            "No market data available yet. Go to **Data Extractor** → "
            "Market Data Downloader to download price data first."
        )
        return

    _render_data_selection(available_tickers)

    selected_tickers = st.session_state.get("ef_tickers", [])
    if len(selected_tickers) < 2:
        st.info("Select at least 2 tickers to construct the efficient frontier.")
        return

    start = st.session_state.get("ef_start_date")
    end = st.session_state.get("ef_end_date")
    if start is None or end is None:
        return

    config = _render_config()

    if not st.button("Compute Efficient Frontier", type="primary", key="ef_compute"):
        return

    with st.spinner("Loading price data and computing efficient frontier..."):
        prices = storage.get_adjusted_price_panel(
            tickers=selected_tickers,
            start=start,
            end=end,
        )

        if prices.empty or prices.shape[1] < 2:
            st.error("Insufficient price data. Try a different date range or ticker set.")
            return

        prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.5))
        if prices.shape[1] < 2:
            st.error("Too many tickers have missing data. Try a wider date range.")
            return

        prices = prices.ffill().bfill()
        tickers = list(prices.columns)

        analysis = EfficientFrontierAnalysis(
            prices=prices,
            method_returns=config["return_method"],
            method_cov=config["cov_method"],
        )

        with st.status("Generating efficient frontier...") as status:
            frontier_df = analysis.generate_efficient_frontier(
                n_portfolios=config["n_portfolios"],
                objective=config["objective"],
            )

            constraints = PortfolioConstraints(
                min_weight=config["min_weight"],
                max_weight=config["max_weight"],
                total_weights=1.0,
                risk_free_rate=config["risk_free_rate"],
            )

            status.update(label="Optimizing portfolios...")

            min_var_result = analysis.optimize_portfolio(
                objective="min_volatility",
                constraints=constraints,
                optimization_solver=config["solver"],
            )

            max_sharpe_result = analysis.optimize_portfolio(
                objective="max_sharpe",
                constraints=constraints,
                optimization_solver=config["solver"],
            )

            optimal_result = analysis.optimize_portfolio(
                objective=config["objective"],
                constraints=constraints,
                optimization_solver=config["solver"],
            )

            status.update(label="Computing portfolio metrics...")

            port_returns = prices.pct_change().dropna()
            metrics = analysis.get_portfolio_metrics(
                weights=optimal_result.weights,
                returns_history=port_returns,
            )

            min_var_metrics = analysis.get_portfolio_metrics(
                weights=min_var_result.weights,
                returns_history=port_returns,
            )

            max_sharpe_metrics = analysis.get_portfolio_metrics(
                weights=max_sharpe_result.weights,
                returns_history=port_returns,
            )

            corr_matrix = analysis.cov_matrix.corr()
            status.update(label="Done.", state="complete")

    _render_results(
        config, frontier_df, analysis, tickers,
        optimal_result, min_var_result, max_sharpe_result,
        metrics, min_var_metrics, max_sharpe_metrics,
        corr_matrix, prices, port_returns,
    )


# ---- Data Selection -------------------------------------------------------


def _render_data_selection(available_tickers: list[str]) -> None:
    section("1. Data Selection")

    search = st.text_input("Search tickers", key="ef_search", help="Filter the ticker list by name or symbol.")
    filtered = [t for t in available_tickers if search.upper() in t.upper()] if search else available_tickers

    selected = st.multiselect(
        "Select tickers (2–50)",
        options=filtered,
        default=st.session_state.get("ef_tickers", []),
        max_selections=50,
        key="ef_tickers",
        help="Choose the assets to include in the efficient frontier analysis.",
    )
    st.caption(f"{len(selected)} ticker(s) selected.")

    c1, c2 = st.columns(2)
    with c1:
        st.date_input(
            "Start date",
            value=st.session_state.get("ef_start_date", pd.Timestamp("2015-01-01")),
            key="ef_start_date",
        )
    with c2:
        st.date_input(
            "End date",
            value=st.session_state.get("ef_end_date", pd.Timestamp.today()),
            key="ef_end_date",
        )


# ---- Configuration --------------------------------------------------------


def _render_config() -> dict:
    section("2. Optimization Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        objective = st.selectbox(
            "Optimization objective",
            options=list(_OBJECTIVE_LABELS.keys()),
            format_func=lambda k: _OBJECTIVE_LABELS[k],
            index=0,
            key="ef_objective",
        )
    with c2:
        return_method = st.selectbox(
            "Expected return method",
            options=list(_RETURN_LABELS.keys()),
            format_func=lambda k: _RETURN_LABELS[k],
            index=0,
            key="ef_return_method",
        )
    with c3:
        cov_method = st.selectbox(
            "Covariance method",
            options=list(_COV_LABELS.keys()),
            format_func=lambda k: _COV_LABELS[k],
            index=0,
            key="ef_cov_method",
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        solver = st.selectbox(
            "Solver",
            options=list(_SOLVER_LABELS.keys()),
            format_func=lambda k: _SOLVER_LABELS[k],
            index=0,
            key="ef_solver",
        )
    with c2:
        n_portfolios = st.slider(
            "Number of random portfolios",
            min_value=1000, max_value=100000, value=10000, step=1000,
            key="ef_n_portfolios",
            help="More portfolios produce a smoother frontier but take longer.",
        )
    with c3:
        risk_free_rate = st.number_input(
            "Risk-free rate (annual)",
            min_value=0.0, max_value=0.20, value=0.06, step=0.005, format="%.3f",
            key="ef_rfr",
        )

    st.caption("Weight constraints")
    c1, c2 = st.columns(2)
    with c1:
        min_weight = st.number_input(
            "Min weight per asset",
            min_value=-1.0, max_value=1.0, value=0.0, step=0.05,
            key="ef_min_w",
        )
    with c2:
        max_weight = st.number_input(
            "Max weight per asset",
            min_value=0.0, max_value=1.0, value=1.0, step=0.05,
            key="ef_max_w",
        )

    return {
        "objective": objective,
        "return_method": return_method,
        "cov_method": cov_method,
        "solver": solver,
        "n_portfolios": n_portfolios,
        "risk_free_rate": risk_free_rate,
        "min_weight": min_weight,
        "max_weight": max_weight,
    }


# ---- Results --------------------------------------------------------------


def _render_results(
    config: dict,
    frontier_df: pd.DataFrame,
    analysis: EfficientFrontierAnalysis,
    tickers: list[str],
    optimal_result,
    min_var_result,
    max_sharpe_result,
    metrics: dict,
    min_var_metrics: dict,
    max_sharpe_metrics: dict,
    corr_matrix: pd.DataFrame,
    prices: pd.DataFrame,
    port_returns: pd.DataFrame,
) -> None:
    section("3. Results")

    tabs = st.tabs([
        "Efficient Frontier", "Optimal Portfolio", "Allocation",
        "Risk Analysis", "Metrics",
    ])

    with tabs[0]:
        _render_frontier_tab(
            frontier_df, optimal_result, min_var_result, max_sharpe_result, config,
        )

    with tabs[1]:
        _render_optimal_portfolio_tab(
            optimal_result, tickers, config["objective"], metrics,
        )

    with tabs[2]:
        _render_allocation_tab(tickers, optimal_result, min_var_result, max_sharpe_result)

    with tabs[3]:
        _render_risk_tab(analysis, optimal_result.weights, tickers, corr_matrix)

    with tabs[4]:
        _render_metrics_tab(
            metrics, min_var_metrics, max_sharpe_metrics,
            optimal_result, min_var_result, max_sharpe_result,
        )


def _render_frontier_tab(
    frontier_df: pd.DataFrame,
    optimal_result,
    min_var_result,
    max_sharpe_result,
    config: dict,
) -> None:
    fig = plot_efficient_frontier(
        frontier_df=frontier_df,
        risk_free_rate=config["risk_free_rate"],
        min_variance_point=(min_var_result.expected_return, min_var_result.volatility),
        max_sharpe_point=(max_sharpe_result.expected_return, max_sharpe_result.volatility),
    )

    fig.add_trace(go.Scatter(
        x=[optimal_result.volatility],
        y=[optimal_result.expected_return],
        mode="markers",
        name=f"{_OBJECTIVE_LABELS[config['objective']]}",
        marker=dict(color="#d62728", size=14, symbol="cross"),
        hovertemplate="<br>%{text}<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>",
        text=[f"{_OBJECTIVE_LABELS[config['objective']]}"],
        showlegend=True,
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Showing {len(frontier_df)} simulated portfolios. "
        f"Star = min volatility, diamond = max Sharpe, "
        f"red cross = optimal portfolio ({_OBJECTIVE_LABELS[config['objective']]})."
    )


def _render_optimal_portfolio_tab(
    result,
    tickers: list[str],
    objective: str,
    metrics: dict,
) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Expected Return", f"{result.expected_return:.2%}")
    with c2:
        st.metric("Volatility", f"{result.volatility:.2%}")
    with c3:
        st.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
    with c4:
        st.metric("Success", "Yes" if result.success else "No",
                  delta_color="off" if result.success else "inverse")

    if result.message and not result.success:
        st.warning(result.message)

    weights_df = result.to_dataframe(tickers)
    weights_df["abs_weight"] = weights_df["weight"].abs()
    weights_df = weights_df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    weights_df["weight_pct"] = weights_df["weight"].apply(lambda x: f"{x:.2%}")
    weights_df["risk_contribution_pct"] = weights_df["risk_contribution"].apply(lambda x: f"{x:.2%}")

    st.subheader("Portfolio Weights")
    st.dataframe(
        weights_df[["ticker", "weight_pct", "risk_contribution_pct"]].rename(
            columns={"ticker": "Ticker", "weight_pct": "Weight", "risk_contribution_pct": "Risk Contribution"}
        ),
        use_container_width=True,
        height=min(35 * len(weights_df) + 38, 600),
    )

    if metrics:
        st.subheader("Portfolio Metrics")
        mcols = st.columns(3)
        rows = [
            ("Sharpe Ratio", metrics.get("sharpe_ratio"), "{:.2f}"),
            ("Sortino Ratio", metrics.get("sortino_ratio"), "{:.2f}"),
            ("Calmar Ratio", metrics.get("calmar_ratio"), "{:.2f}"),
            ("Max Drawdown", metrics.get("max_drawdown"), "{:.2%}"),
            ("Diversification Ratio", metrics.get("diversification_ratio"), "{:.3f}"),
            ("Effective N", metrics.get("effective_number_of_stocks", metrics.get("effective_n_assets")), "{:.1f}"),
        ]
        for i, (label, val, fmt) in enumerate(rows):
            with mcols[i % 3]:
                display = fmt.format(val) if val is not None and val == val else "—"
                st.metric(label, display)


def _render_allocation_tab(
    tickers: list[str],
    optimal_result,
    min_var_result,
    max_sharpe_result,
) -> None:
    sel = st.radio(
        "Show allocation for:",
        ["Optimal Portfolio", "Min Volatility", "Max Sharpe"],
        horizontal=True,
        key="ef_alloc_sel",
    )

    if sel == "Optimal Portfolio":
        weights, label = optimal_result.weights, "Optimal Portfolio"
    elif sel == "Min Volatility":
        weights, label = min_var_result.weights, "Min Volatility"
    else:
        weights, label = max_sharpe_result.weights, "Max Sharpe"

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_allocation_pie(weights, tickers, title=f"{label} — Allocation"), use_container_width=True)
    with c2:
        st.plotly_chart(plot_allocation_treemap(weights, tickers, title=f"{label} — Treemap"), use_container_width=True)


def _render_risk_tab(
    analysis: EfficientFrontierAnalysis,
    weights: np.ndarray,
    tickers: list[str],
    corr_matrix: pd.DataFrame,
) -> None:
    st.subheader("Risk Contribution")
    risk_contrib = analysis.efficient_frontier.cov_array @ weights
    risk_contrib = weights * risk_contrib / np.sum(risk_contrib)
    st.plotly_chart(
        plot_risk_contribution(weights, tickers, risk_contribution=risk_contrib),
        use_container_width=True,
    )

    st.subheader("Asset Correlation Matrix")
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 8},
    ))
    fig_corr.update_layout(
        height=600,
        margin=dict(l=50, r=50, t=30, b=50),
    )
    st.plotly_chart(fig_corr, use_container_width=True)


def _render_metrics_tab(
    metrics: dict,
    min_var_metrics: dict,
    max_sharpe_metrics: dict,
    optimal_result,
    min_var_result,
    max_sharpe_result,
) -> None:
    metric_names = [
        ("expected_return", "Expected Return", "{:.2%}"),
        ("volatility", "Volatility", "{:.2%}"),
        ("sharpe_ratio", "Sharpe Ratio", "{:.2f}"),
        ("sortino_ratio", "Sortino Ratio", "{:.2f}"),
        ("calmar_ratio", "Calmar Ratio", "{:.2f}"),
        ("max_drawdown", "Max Drawdown", "{:.2%}"),
        ("diversification_ratio", "Diversification Ratio", "{:.3f}"),
        ("effective_number_of_stocks", "Effective N Assets", "{:.1f}"),
        ("herfindahl_index", "Herfindahl Index", "{:.4f}"),
        ("weight_entropy", "Weight Entropy", "{:.3f}"),
    ]

    rows = []
    for key, label, fmt in metric_names:
        rows.append({
            "Metric": label,
            "Optimal": fmt.format(metrics.get(key, 0)) if metrics.get(key) is not None else "—",
            "Min Vol": fmt.format(min_var_metrics.get(key, 0)) if min_var_metrics.get(key) is not None else "—",
            "Max Sharpe": fmt.format(max_sharpe_metrics.get(key, 0)) if max_sharpe_metrics.get(key) is not None else "—",
        })

    st.dataframe(
        pd.DataFrame(rows).set_index("Metric"),
        use_container_width=True,
    )

    st.subheader("Optimization Details")
    st.json({
        "optimal_success": optimal_result.success,
        "optimal_message": optimal_result.message,
        "min_var_success": min_var_result.success,
        "max_sharpe_success": max_sharpe_result.success,
        "method": optimal_result.optimization_method,
    })
