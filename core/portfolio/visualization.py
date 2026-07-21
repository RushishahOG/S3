"""Portfolio visualization utilities for the Efficient Frontier module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

import streamlit as st

from typing import Any, Optional


def plot_efficient_frontier(
    frontier_df: pd.DataFrame,
    min_variance_port: Optional[pd.Series] = None,
    max_sharpe_port: Optional[pd.Series] = None,
    current_strategy: Optional[pd.Series] = None,
    risk_free_rate: float = 0.06,
    min_variance_point: Optional[tuple[float, float]] = None,
    max_sharpe_point: Optional[tuple[float, float]] = None,
    current_strategy_point: Optional[tuple[float, float]] = None,
) -> go.Figure:
    """
    Create an interactive Efficient Frontier plot.

    Args:
        frontier_df: DataFrame with columns ['return', 'volatility', 'sharpe', 'sortino', 'calmar']
        min_variance_port: Series with weights for minimum variance portfolio
        max_sharpe_port: Series with weights for maximum Sharpe portfolio
        current_strategy: Series with weights for current strategy portfolio
        risk_free_rate: Risk-free rate for Capital Market Line
        min_variance_point: (return, volatility) tuple for min variance portfolio (takes precedence over weights)
        max_sharpe_point: (return, volatility) tuple for max Sharpe portfolio (takes precedence over weights)
        current_strategy_point: (return, volatility) tuple for current strategy (takes precedence over weights)

    Returns:
        Plotly figure object
    """
    # Sort by volatility for smooth frontier curve
    sorted_frontier = frontier_df.sort_values('volatility').reset_index(drop=True)

    fig = go.Figure()

    # Efficient frontier curve
    fig.add_trace(go.Scatter(
        x=sorted_frontier['volatility'],
        y=sorted_frontier['return'],
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='#1f77b4', width=3),
        hovertemplate='<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe: %{customdata[0]:.2f}<extra></extra>',
        customdata=sorted_frontier[['sharpe']].values
    ))

    # Color mapping for portfolios
    colors = {'min_var': '#ff7f0e', 'max_sharpe': '#2ca02c', 'current': '#d62728'}

    # Minimum variance portfolio (leftmost point of frontier)
    if min_variance_point is not None:
        min_var_return, min_var_vol = min_variance_point
        fig.add_trace(go.Scatter(
            x=[min_var_vol],
            y=[min_var_return],
            mode='markers',
            name='Min Volatility Portfolio',
            marker=dict(color=colors['min_var'], size=12, symbol='star'),
            hovertemplate='<br>Min Volatility Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))
    elif min_variance_port is not None and not min_variance_port.empty:
        prev_min_var_return = np.dot(min_variance_port.values, sorted_frontier['return'].values)
        prev_min_var_vol = np.dot(min_variance_port.values, sorted_frontier['volatility'].values)
        fig.add_trace(go.Scatter(
            x=[prev_min_var_vol],
            y=[prev_min_var_return],
            mode='markers',
            name='Min Volatility Portfolio',
            marker=dict(color=colors['min_var'], size=12, symbol='star'),
            hovertemplate='<br>Min Volatility Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))

    # Maximum Sharpe portfolio
    if max_sharpe_point is not None:
        max_sharpe_return, max_sharpe_vol = max_sharpe_point
        fig.add_trace(go.Scatter(
            x=[max_sharpe_vol],
            y=[max_sharpe_return],
            mode='markers',
            name='Max Sharpe Portfolio',
            marker=dict(color=colors['max_sharpe'], size=12, symbol='diamond'),
            hovertemplate='<br>Max Sharpe Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))
    elif max_sharpe_port is not None and not max_sharpe_port.empty:
        prev_max_sharpe_return = np.dot(max_sharpe_port.values, sorted_frontier['return'].values)
        prev_max_sharpe_vol = np.dot(max_sharpe_port.values, sorted_frontier['volatility'].values)
        fig.add_trace(go.Scatter(
            x=[prev_max_sharpe_vol],
            y=[prev_max_sharpe_return],
            mode='markers',
            name='Max Sharpe Portfolio',
            marker=dict(color=colors['max_sharpe'], size=12, symbol='diamond'),
            hovertemplate='<br>Max Sharpe Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))

    # Current strategy portfolio
    if current_strategy_point is not None:
        current_return, current_vol = current_strategy_point
        fig.add_trace(go.Scatter(
            x=[current_vol],
            y=[current_return],
            mode='markers',
            name='Current Strategy',
            marker=dict(color=colors['current'], size=12, symbol='circle', line=dict(width=2, color='black')),
            hovertemplate='<br>Current Strategy<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))
    elif current_strategy is not None and not current_strategy.empty:
        prev_current_return = np.dot(current_strategy.values, sorted_frontier['return'].values)
        prev_current_vol = np.dot(current_strategy.values, sorted_frontier['volatility'].values)
        fig.add_trace(go.Scatter(
            x=[prev_current_vol],
            y=[prev_current_return],
            mode='markers',
            name='Current Strategy',
            marker=dict(color=colors['current'], size=12, symbol='circle', line=dict(width=2, color='black')),
            hovertemplate='<br>Current Strategy<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            showlegend=True
        ))

    # Capital Market Line
    if len(sorted_frontier) > 0:
        max_sharpe_point = sorted_frontier.iloc[sorted_frontier['sharpe'].idxmax()]
        cml_start = (risk_free_rate, risk_free_rate)
        cml_end = (max_sharpe_point['volatility'], max_sharpe_point['return'])
        cml_slope = (max_sharpe_point['return'] - risk_free_rate) / max_sharpe_point['volatility'] if max_sharpe_point['volatility'] > 0 else 0

        fig.add_trace(go.Scatter(
            x=[cml_start[0], cml_end[0]],
            y=[cml_start[1], cml_end[1]],
            mode='lines',
            name='Capital Market Line',
            line=dict(color='#9467bd', width=2, dash='dot'),
            hovertemplate='<br>Capital Market Line<br>Slope: %{customdata:.2f}<extra></extra>',
            customdata=[cml_slope],
            showlegend=True
        ))

    fig.update_layout(
        title='Efficient Frontier Analysis',
        xaxis_title='Annualized Volatility (Std Dev)',
        yaxis_title='Expected Annual Return',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)'),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        height=600,
    )

    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)

    return fig


def plot_allocation_pie(
    weights: np.ndarray,
    tickers: list[str],
    title: str = "Portfolio Allocation",
) -> go.Figure:
    """
    Create a pie chart of portfolio allocation.

    Args:
        weights: Portfolio weights
        tickers: List of ticker names
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Filter out zero weights for better visualization
    non_zero_mask = np.abs(weights) > 0.01
    display_weights = weights[non_zero_mask]
    display_tickers = np.array(tickers)[non_zero_mask]

    fig = go.Figure(data=[go.Pie(
        labels=display_tickers,
        values=display_weights,
        hole=0.3,
        textinfo='percent+label',
        marker=dict(colors=px.colors.qualitative.Set3),
        hovertemplate='<br>%{label}<br>Weight: %{value:.2%}<extra></extra>'
    )])

    fig.update_layout(
        title=title,
        margin=dict(l=50, r=50, t=50, b=50),
        height=500,
    )

    return fig


def plot_allocation_treemap(
    weights: np.ndarray,
    tickers: list[str],
    sectors: Optional[dict[str, str]] = None,
    title: str = "Portfolio Allocation Treemap",
) -> go.Figure:
    """
    Create a treemap visualization of portfolio allocation.

    Args:
        weights: Portfolio weights
        tickers: List of ticker names
        sectors: Dictionary mapping tickers to sectors
        title: Chart title

    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        'ticker': tickers,
        'weight': weights,
        'sector': [sectors.get(ticker, 'Unknown') if sectors else 'Unknown' for ticker in tickers]
    })

    # Filter out zero weights
    df = df[df['weight'] > 0.01]

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No holdings to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = px.treemap(
        df, path=['sector'], values='weight',
        title=title
    )

    fig.update_layout(
        margin=dict(l=50, r=50, t=50, b=50),
        height=600,
        treemapcolorway=px.colors.qualitative.Set3,
    )

    fig.update_traces(textinfo='label+percent+value')

    return fig


def plot_risk_contribution(
    weights: np.ndarray,
    tickers: list[str],
    sector_weights: Optional[dict[str, float]] = None,
    return_contribution: Optional[np.ndarray] = None,
    risk_contribution: Optional[np.ndarray] = None,
    title: str = "Risk Contribution Analysis",
) -> go.Figure:
    """
    Create an interactive risk contribution visualization.

    Args:
        weights: Portfolio weights
        tickers: List of ticker names
        sector_weights: Dictionary mapping sectors to weights
        return_contribution: Array of return contribution per asset
        risk_contribution: Array of risk contribution per asset
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Filter out zero weights
    non_zero_mask = np.abs(weights) > 0.01
    display_tickers = np.array(tickers)[non_zero_mask]
    display_weights = np.array(weights)[non_zero_mask]

    if len(display_tickers) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No holdings to display",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    # Create subplots for different contribution types
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Allocation',
            'Return Contribution',
            'Risk Contribution',
            'Percent Contribution'
        ),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'pie'}, {'type': 'bar'}]]
    )

    # Pie chart: Portfolio allocation
    fig.add_trace(
        go.Pie(
            labels=display_tickers,
            values=display_weights,
            name="Allocation",
            hole=0.3,
            textinfo='percent',
            marker=dict(colors=px.colors.qualitative.Set3),
        ),
        row=1, col=1
    )

    # Bar chart: Return contribution
    if return_contribution is not None:
        disp_return_contrib = return_contribution[non_zero_mask]
        fig.add_trace(
            go.Bar(
                x=display_tickers,
                y=disp_return_contrib,
                name="Return Contribution",
                marker_color=px.colors.qualitative.Set3[:len(display_tickers)],
                hovertemplate='<br>%{x}<br>Return Contribution: %{y:.2%}<extra></extra>',
            ),
            row=1, col=2
        )

    # Pie chart: Sector allocation (if available)
    if sector_weights is not None:
        sector_df = pd.DataFrame(list(sector_weights.items()), columns=['sector', 'weight'])
        fig.add_trace(
            go.Pie(
                labels=sector_df['sector'],
                values=sector_df['weight'],
                name="Sectors",
                hole=0.3,
                textinfo='percent',
                marker=dict(colors=px.colors.qualitative.Set2),
            ),
            row=2, col=1
        )

    # Bar chart: Risk contribution
    if risk_contribution is not None:
        disp_risk_contrib = risk_contribution[non_zero_mask]
        total_risk = np.sum(disp_risk_contrib)
        disp_risk_contrib_pct = disp_risk_contrib / total_risk if total_risk > 0 else disp_risk_contrib

        fig.add_trace(
            go.Bar(
                x=display_tickers,
                y=disp_risk_contrib_pct,
                name="Risk Contribution (%)",
                marker_color=px.colors.qualitative.Set3[:len(display_tickers)],
                hovertemplate='<br>%{x}<br>Risk Contribution: %{y:.2%}<extra></extra>',
            ),
            row=2, col=2
        )

    fig.update_layout(
        title=title,
        height=700,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Heatmap",
    clustering: bool = True,
    color_scale: str = "RdBu",
) -> go.Figure:
    """
    Create an interactive correlation heatmap with hierarchical clustering.

    Args:
        correlation_matrix: DataFrame with correlation values
        title: Chart title
        clustering: Whether to perform hierarchical clustering for better visualization
        color_scale: Color scale for the heatmap

    Returns:
        Plotly figure object
    """
    # Apply hierarchical clustering if requested
    if clustering and len(correlation_matrix) > 1:
        try:
            from scipy.cluster.hierarchy import linkage, leaves_list
            from scipy.spatial.distance import squareform

            # Convert correlation to distance
            dist = squareform(1 - np.abs(correlation_matrix.values))
            link = linkage(dist, method='ward')
            order = leaves_list(link)

            # Reorder the matrix
            correlation_matrix = correlation_matrix.iloc[order, order]
        except ImportError:
            pass

    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        colorscale=color_scale,
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='<br>%{y} × %{x}<br>Correlation: %{z:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Assets',
        yaxis_title='Assets',
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    return fig


def plot_capital_market_line(
    efficient_frontier_df: pd.DataFrame,
    min_variance_port: Optional[pd.Series] = None,
    max_sharpe_port: Optional[pd.Series] = None,
    risk_free_rate: float = 0.06,
    title: str = "Capital Market Line",
) -> go.Figure:
    """
    Create a Capital Market Line plot.

    Args:
        efficient_frontier_df: DataFrame with frontier data
        min_variance_port: Series with weights for minimum variance portfolio
        max_sharpe_port: Series with weights for maximum Sharpe portfolio
        risk_free_rate: Risk-free rate
        title: Chart title

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Find maximum Sharpe portfolio point
    if len(efficient_frontier_df) > 0:
        max_sharpe_idx = efficient_frontier_df['sharpe'].idxmax()
        max_sharpe_point = efficient_frontier_df.loc[max_sharpe_idx]

        # Capital Market Line from risk-free rate to max Sharpe point
        x_vals = [risk_free_rate, max_sharpe_point['volatility']]
        y_vals = [risk_free_rate, max_sharpe_point['return']]

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='lines',
            name='Capital Market Line',
            line=dict(color='#9467bd', width=3, dash='dot'),
            hovertemplate='<br>Slope: %{customdata:.2f}<br> Sharpe Ratio: %{customdata}<extra></extra>',
            customdata=[max_sharpe_point['sharpe']],
        ))

        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_frontier_df['volatility'],
            y=efficient_frontier_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
        ))

        # Plot key points
        if min_variance_port is not None and not min_variance_port.empty:
            min_var_return = np.dot(min_variance_port.values, efficient_frontier_df['return'].values)
            min_var_vol = np.dot(min_variance_port.values, efficient_frontier_df['volatility'].values)
            fig.add_trace(go.Scatter(
                x=[min_var_vol],
                y=[min_var_return],
                mode='markers',
                name='Min Volatility Portfolio',
                marker=dict(color='#ff7f0e', size=12, symbol='star'),
                hovertemplate='<br>Min Volatility Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            ))

        if max_sharpe_port is not None and not max_sharpe_port.empty:
            max_sharpe_return = np.dot(max_sharpe_port.values, efficient_frontier_df['return'].values)
            max_sharpe_vol = np.dot(max_sharpe_port.values, efficient_frontier_df['volatility'].values)
            fig.add_trace(go.Scatter(
                x=[max_sharpe_vol],
                y=[max_sharpe_return],
                mode='markers',
                name='Max Sharpe Portfolio',
                marker=dict(color='#2ca02c', size=12, symbol='diamond'),
                hovertemplate='<br>Max Sharpe Portfolio<br>Return: %{y:.2%}<br>Volatility: %{x:.2%}<extra></extra>',
            ))

    fig.update_layout(
        title=title,
        xaxis_title='Annualized Volatility (Std Dev)',
        yaxis_title='Expected Annual Return',
        hovermode='closest',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)'),
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
        height=600,
    )

    fig.update_xaxes(gridcolor='lightgray', gridwidth=0.5)
    fig.update_yaxes(gridcolor='lightgray', gridwidth=0.5)

    return fig


def create_risk_dashboard(
    optimization_result: Any,
    expected_returns: pd.Series,
    weights: np.ndarray,
    risk_metrics: dict,
    diversification_metrics: dict,
    correlation_matrix: pd.DataFrame,
    tickers: list[str],
) -> go.Figure:
    """
    Create a comprehensive risk dashboard with multiple visualizations.

    Args:
        optimization_result: Optimization result object
        expected_returns: Expected returns series
        weights: Portfolio weights
        risk_metrics: Risk metrics dictionary
        diversification_metrics: Diversification metrics dictionary
        correlation_matrix: Correlation matrix
        tickers: List of ticker names

    Returns:
        Plotly figure object
    """
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Efficient Frontier with Optimal Portfolio',
            'Portfolio Allocation',
            'Risk Contribution',
            'Diversification Metrics',
            'Correlation Heatmap',
            'Performance Metrics'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'heatmap'}, {'type': 'table'}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )

    # Get optimal portfolio from optimization result
    opt_weights = optimization_result.weights if hasattr(optimization_result, 'weights') else weights

    # 1. Efficient Frontier with optimal portfolio
    try:
        # Create synthetic frontier data for visualization
        frontier_data = pd.DataFrame({
            'volatility': np.sqrt(np.diag(optimization_result.cov_matrix.values) if hasattr(optimization_result, 'cov_matrix') else np.diag(np.cov(expected_returns.values.T))),
            'return': expected_returns.values,
        })
        # Add random portfolios to create frontier
        n_random = 50
        for _ in range(n_random):
            w = np.random.dirichlet(np.ones(len(expected_returns)))
            vol = np.sqrt(w @ (optimization_result.cov_matrix.values if hasattr(optimization_result, 'cov_matrix') else np.cov(expected_returns.values.T)) @ w)
            ret = np.dot(w, expected_returns.values)
            frontier_data = pd.concat([frontier_data, pd.DataFrame({
                'volatility': [vol],
                'return': [ret]
            })])

        # Plot efficient frontier
        fig.add_trace(
            go.Scatter(
                x=frontier_data['volatility'],
                y=frontier_data['return'],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='#1f77b4', width=2),
                showlegend=False,
                opacity=0.5,
            ),
            row=1, col=1
        )

        # Plot optimal portfolio
        opt_vol = np.sqrt(opt_weights @ (optimization_result.cov_matrix.values if hasattr(optimization_result, 'cov_matrix') else np.cov(expected_returns.values.T)) @ opt_weights)
        opt_ret = np.dot(opt_weights, expected_returns.values)

        fig.add_trace(
            go.Scatter(
                x=[opt_vol],
                y=[opt_ret],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(color='#d62728', size=15, symbol='star'),
                showlegend=False,
            ),
            row=1, col=1
        )
    except Exception as e:
        # Fallback: just plot markers
        pass

    # 2. Portfolio Allocation
    non_zero_mask = np.abs(opt_weights) > 0.01
    alloc_tickers = np.array(tickers)[non_zero_mask]
    alloc_weights = opt_weights[non_zero_mask]

    fig.add_trace(
        go.Pie(
            labels=alloc_tickers,
            values=alloc_weights,
            name='Portfolio Allocation',
            hole=0.3,
            textinfo='percent',
            marker=dict(colors=px.colors.qualitative.Set3),
        ),
        row=1, col=2
    )

    # 3. Risk Contribution
    try:
        if hasattr(optimization_result, 'risk_contribution'):
            risk_contrib = optimization_result.risk_contribution
        else:
            marginal_risk = (optimization_result.cov_matrix.values if hasattr(optimization_result, 'cov_matrix') else np.cov(expected_returns.values.T)) @ opt_weights
            risk_contrib = opt_weights * marginal_risk / np.sum(marginal_risk) if np.sum(marginal_risk) > 0 else opt_weights

        disp_tickers = alloc_tickers
        disp_risk_contrib = risk_contrib[non_zero_mask]

        fig.add_trace(
            go.Bar(
                x=disp_tickers,
                y=disp_risk_contrib,
                name='Risk Contribution',
                marker_color=px.colors.qualitative.Set3[:len(disp_tickers)],
            ),
            row=2, col=1
        )
    except Exception as e:
        pass

    # 4. Diversification Metrics
    div_metrics_names = ['Diversification Ratio', 'Effective Number', 'Herfindahl Index', 'Entropy']
    div_metrics_values = [
        diversification_metrics.get('diversification_ratio', 0),
        diversification_metrics.get('effective_number_of_stocks', 0),
        diversification_metrics.get('herfindahl_index', 0),
        diversification_metrics.get('weight_entropy', 0),
    ]

    fig.add_trace(
        go.Bar(
            x=div_metrics_names,
            y=div_metrics_values,
            name='Diversification',
            marker_color=px.colors.qualitative.Set2,
        ),
        row=2, col=2
    )

    # 5. Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ),
        row=3, col=1
    )

    # 6. Performance Metrics Table
    performance_metrics = [
        ['Metric', 'Value'],
        ['Expected Return', f"{risk_metrics.get('expected_return', 0):.2%}"],
        ['Volatility', f"{risk_metrics.get('volatility', 0):.2%}"],
        ['Sharpe Ratio', f"{risk_metrics.get('sharpe_ratio', 0):.2f}"],
        ['Sortino Ratio', f"{risk_metrics.get('sortino_ratio', 0):.2f}"],
        ['Calmar Ratio', f"{risk_metrics.get('calmar_ratio', 0):.2f}"],
        ['Max Drawdown', f"{risk_metrics.get('max_drawdown', 0):.2%}"],
        ['Diversification Ratio', f"{diversification_metrics.get('diversification_ratio', 0):.3f}"],
        ['Effective Number of Stocks', f"{diversification_metrics.get('effective_number_of_stocks', 0):.1f}"],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=['Metric', 'Value'], fill_color='#E6E6E6', align='left'),
            cells=dict(values=list(zip(*performance_metrics)), align='left'),
        ),
        row=3, col=2
    )

    fig.update_layout(
        title='Risk Dashboard - Portfolio Optimization Results',
        height=1000,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(title_text='', showticklabels=False)
    fig.update_yaxes(title_text='', showticklabels=False)

    return fig


def create_streamlit_dashboard(
    optimization_result: Any,
    efficient_frontier_df: pd.DataFrame,
    expected_returns: pd.Series,
    weights: np.ndarray,
    tickers: list[str],
    risk_metrics: dict,
    diversification_metrics: dict,
    correlation_matrix: pd.DataFrame,
    sectors: Optional[dict[str, str]] = None,
) -> None:
    """
    Create a Streamlit dashboard for the Efficient Frontier module.

    Args:
        optimization_result: Optimization result object
        efficient_frontier_df: DataFrame with frontier data
        expected_returns: Expected returns series
        weights: Portfolio weights
        tickers: List of ticker names
        risk_metrics: Risk metrics dictionary
        diversification_metrics: Diversification metrics dictionary
        correlation_matrix: Correlation matrix
        sectors: Dictionary mapping tickers to sectors
    """
    st.title("Efficient Frontier Optimization Dashboard")

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Expected Return", f"{risk_metrics.get('expected_return', 0):.2%}")

    with col2:
        st.metric("Volatility", f"{risk_metrics.get('volatility', 0):.2%}")

    with col3:
        st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")

    with col4:
        st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")

    # Charts section
    st.header("Efficient Frontier Analysis")

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Efficient Frontier',
            'Portfolio Allocation',
            'Risk Contribution',
            'Correlation Heatmap'
        ),
        specs=[[{'type': 'scatter'}, {'type': 'pie'}],
               [{'type': 'bar'}, {'type': 'heatmap'}]]
    )

    # Add efficient frontier
    fig.add_trace(
        go.Scatter(
            x=efficient_frontier_df['volatility'],
            y=efficient_frontier_df['return'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#1f77b4', width=2),
            showlegend=False,
        ),
        row=1, col=1
    )

    # Add optimal portfolio
    opt_vol = np.sqrt(weights @ (optimization_result.cov_matrix.values if hasattr(optimization_result, 'cov_matrix') else np.cov(expected_returns.values.T)) @ weights)
    opt_ret = np.dot(weights, expected_returns.values)

    fig.add_trace(
        go.Scatter(
            x=[opt_vol],
            y=[opt_ret],
            mode='markers',
            name='Optimal Portfolio',
            marker=dict(color='#d62728', size=15, symbol='star'),
            showlegend=False,
        ),
        row=1, col=1
    )

    # Add portfolio allocation pie chart
    non_zero_mask = np.abs(weights) > 0.01
    alloc_tickers = np.array(tickers)[non_zero_mask]
    alloc_weights = weights[non_zero_mask]

    fig.add_trace(
        go.Pie(
            labels=alloc_tickers,
            values=alloc_weights,
            name='Portfolio Allocation',
            hole=0.3,
            textinfo='percent',
        ),
        row=1, col=2
    )

    # Add risk contribution
    try:
        if hasattr(optimization_result, 'risk_contribution'):
            risk_contrib = optimization_result.risk_contribution
        else:
            marginal_risk = (optimization_result.cov_matrix.values if hasattr(optimization_result, 'cov_matrix') else np.cov(expected_returns.values.T)) @ weights
            risk_contrib = weights * marginal_risk / np.sum(marginal_risk) if np.sum(marginal_risk) > 0 else weights

        disp_risk_contrib = risk_contrib[non_zero_mask]

        fig.add_trace(
            go.Bar(
                x=alloc_tickers,
                y=disp_risk_contrib,
                name='Risk Contribution',
                marker_color=px.colors.qualitative.Set3[:len(alloc_tickers)],
            ),
            row=2, col=1
        )
    except Exception as e:
        pass

    # Add correlation heatmap
    fig.add_trace(
        go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            showscale=False,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50),
    )

    fig.update_xaxes(title_text='Volatility', row=1, col=1)
    fig.update_yaxes(title_text='Return', row=1, col=1)

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Additional metrics section
    st.header("Portfolio Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Risk Metrics")
        st.write(f"**Sharpe Ratio:** {risk_metrics.get('sharpe_ratio', 0):.2f}")
        st.write(f"**Sortino Ratio:** {risk_metrics.get('sortino_ratio', 0):.2f}")
        st.write(f"**Calmar Ratio:** {risk_metrics.get('calmar_ratio', 0):.2f}")
        st.write(f"**Max Drawdown:** {risk_metrics.get('max_drawdown', 0):.2%}")

    with col2:
        st.subheader("Diversification")
        st.write(f"**Diversification Ratio:** {diversification_metrics.get('diversification_ratio', 0):.3f}")
        st.write(f"**Effective Number:** {diversification_metrics.get('effective_number_of_stocks', 0):.1f}")
        st.write(f"**Herfindahl Index:** {diversification_metrics.get('herfindahl_index', 0):.3f}")
        st.write(f"**Weight Entropy:** {diversification_metrics.get('weight_entropy', 0):.3f}")

    with col3:
        st.subheader("Portfolio Statistics")
        st.write(f"**Number of Holdings:** {len(alloc_tickers)}")
        st.write(f"**Largest Weight:** {np.max(np.abs(alloc_weights)):.2%}")
        st.write(f"**Average Weight:** {np.mean(np.abs(alloc_weights)):.2%}")
        st.write(f"**Concentration:** {np.sum(np.abs(alloc_weights) ** 2):.2f}")

    # Export section
    st.header("Export Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Create downloadable data
        export_data = pd.DataFrame({
            'ticker': alloc_tickers,
            'weight': alloc_weights,
            'sector': [sectors.get(ticker, 'Unknown') if sectors else 'Unknown' for ticker in alloc_tickers],
        })

        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download Portfolio Allocation (CSV)",
            data=csv,
            file_name="portfolio_allocation.csv",
            mime="text/csv",
        )

    with col2:
        # Create optimization summary
        summary_data = {
            'metric': ['Expected Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Diversification Ratio', 'Effective Number', 'Herfindahl Index', 'Weight Entropy'],
            'value': [
                risk_metrics.get('expected_return', 0),
                risk_metrics.get('volatility', 0),
                risk_metrics.get('sharpe_ratio', 0),
                risk_metrics.get('sortino_ratio', 0),
                risk_metrics.get('calmar_ratio', 0),
                risk_metrics.get('max_drawdown', 0),
                diversification_metrics.get('diversification_ratio', 0),
                diversification_metrics.get('effective_number_of_stocks', 0),
                diversification_metrics.get('herfindahl_index', 0),
                diversification_metrics.get('weight_entropy', 0),
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_csv = summary_df.to_csv(index=False)

        st.download_button(
            label="Download Summary (CSV)",
            data=summary_csv,
            file_name="optimization_summary.csv",
            mime="text/csv",
        )

    with col3:
        st.info("All metrics are annualized and calculated using standard institutional formulas.")


if __name__ == "__main__":
    st.set_page_config(layout="wide")

    # Sample data for demonstration
    np.random.seed(42)
    n_assets = 10
    tickers = [f"Asset_{i}" for i in range(n_assets)]

    # Generate sample expected returns
    expected_returns = pd.Series(
        np.random.normal(0.12, 0.15, n_assets),
        index=tickers
    )

    # Generate sample covariance matrix
    cov_matrix = pd.DataFrame(
        np.random.normal(0.01, 0.03, (n_assets, n_assets)),
        index=tickers,
        columns=tickers
    )
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    np.fill_diagonal(cov_matrix.values, np.random.normal(0.15, 0.05, n_assets))

    # Generate sample weights
    weights = np.random.dirichlet(np.ones(n_assets))

    # Create sample risk metrics
    risk_metrics = {
        'expected_return': np.dot(weights, expected_returns.values),
        'volatility': np.sqrt(weights @ cov_matrix.values @ weights),
        'sharpe_ratio': (np.dot(weights, expected_returns.values) - 0.06) / np.sqrt(weights @ cov_matrix.values @ weights) if weights @ cov_matrix.values @ weights > 0 else 0,
        'sortino_ratio': 0.5,
        'calmar_ratio': 0.3,
        'max_drawdown': 0.15,
    }

    # Create sample diversification metrics
    diversification_metrics = {
        'diversification_ratio': 0.7,
        'effective_number_of_stocks': 6.5,
        'herfindahl_index': 0.15,
        'weight_entropy': 1.8,
    }

    # Create sample correlation matrix
    correlation_matrix = pd.DataFrame(
        np.random.uniform(-0.5, 0.5, (n_assets, n_assets)),
        index=tickers,
        columns=tickers
    )
    np.fill_diagonal(correlation_matrix.values, 1.0)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2

    # Create sample efficient frontier
    n_portfolios = 100
    frontier_data = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = np.dot(w, expected_returns.values)
        vol = np.sqrt(w @ cov_matrix.values @ w)
        sharpe = (ret - 0.06) / vol if vol > 0 else 0
        frontier_data.append({
            'return': ret,
            'volatility': vol,
            'sharpe': sharpe,
            'sortino': 0.5,
            'calmar': 0.3,
        })

    efficient_frontier_df = pd.DataFrame(frontier_data)

    # Create sample optimization result
    class SampleOptimizationResult:
        def __init__(self):
            self.weights = weights
            self.expected_return = risk_metrics['expected_return']
            self.volatility = risk_metrics['volatility']
            self.sharpe = risk_metrics['sharpe_ratio']
            self.risk_contribution = weights * 0.01
            self.cov_matrix = cov_matrix

    optimization_result = SampleOptimizationResult()

    # Create sample sectors
    sectors = {ticker: f"Sector_{i % 3}" for i, ticker in enumerate(tickers)}

    # Create the dashboard
    create_streamlit_dashboard(
        optimization_result=optimization_result,
        efficient_frontier_df=efficient_frontier_df,
        expected_returns=expected_returns,
        weights=weights,
        tickers=tickers,
        risk_metrics=risk_metrics,
        diversification_metrics=diversification_metrics,
        correlation_matrix=correlation_matrix,
        sectors=sectors,
    )