"""Plotly visualizations for strategy comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.strategy_comparison.comparison import ComparisonResult


def equity_curve_comparison(result: ComparisonResult, normalize: bool = True) -> go.Figure:
    """Plot equity curves of all selected strategies."""
    fig = go.Figure()
    for name in result.equity_curves.columns:
        curve = result.equity_curves[name]
        if normalize and curve.iloc[0] != 0:
            curve = curve / curve.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=curve.index, y=curve.values, mode="lines", name=name,
            line=dict(width=2),
        ))
    fig.update_layout(
        title="Equity Curve Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value" + (" (Normalized)" if normalize else ""),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def drawdown_comparison(result: ComparisonResult) -> go.Figure:
    """Plot drawdown curves of all selected strategies."""
    fig = go.Figure()
    for name in result.drawdown_curves.columns:
        dd = result.drawdown_curves[name] * 100
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values, mode="lines", name=name,
            line=dict(width=2),
        ))
    fig.update_layout(
        title="Drawdown Comparison",
        xaxis_title="Date",
        yaxis_title="Drawdown %",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def risk_return_scatter(result: ComparisonResult) -> go.Figure:
    """Scatter plot: Volatility vs CAGR, bubble size = Sharpe."""
    fig = go.Figure()
    perf = result.performance_table
    for name in perf.index:
        fig.add_trace(go.Scatter(
            x=[perf.loc[name, "Volatility"]],
            y=[perf.loc[name, "CAGR"]],
            mode="markers",
            name=name,
            marker=dict(
                size=perf.loc[name, "Sharpe"] * 20 + 10,
                color=perf.loc[name, "Sharpe"],
                colorscale="Viridis",
                showscale=True,
                line=dict(width=1, color="black"),
            ),
            text=name,
            hovertemplate=f"<b>{name}</b><br>Vol: %{{x:.2%}}<br>CAGR: %{{y:.2%}}<br>Sharpe: {perf.loc[name, 'Sharpe']:.2f}<extra></extra>",
        ))
    fig.update_layout(
        title="Risk-Return Scatter",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="CAGR",
        template="plotly_white",
        hovermode="closest",
    )
    return fig


def correlation_heatmap(result: ComparisonResult) -> go.Figure:
    """Correlation matrix heatmap of strategy returns."""
    if result.correlation_matrix.empty:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=result.correlation_matrix.values,
        x=result.correlation_matrix.columns,
        y=result.correlation_matrix.index,
        colorscale="RdBu",
        zmin=-1, zmax=1,
        colorbar=dict(title="Correlation"),
    ))
    fig.update_layout(title="Return Correlation Matrix", template="plotly_white")
    return fig


def annual_return_heatmap(result: ComparisonResult) -> go.Figure:
    """Annual returns heatmap: years vs strategies."""
    if result.annual_returns.empty:
        return go.Figure()
    data = result.annual_returns * 100
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=[d.year for d in data.index],
        colorscale="RdYlGn",
        zmin=-50, zmax=50,
        colorbar=dict(title="Annual Return %"),
        text=data.round(1).values,
        texttemplate="%{text:.1f}%",
        textfont={"size": 10},
    ))
    fig.update_layout(title="Annual Returns Heatmap", template="plotly_white")
    return fig


def monthly_return_heatmap(result: ComparisonResult) -> go.Figure:
    """Monthly returns heatmap: months vs years."""
    if result.monthly_returns.empty:
        return go.Figure()
    data = result.monthly_returns * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    year_month = data.index.strftime("%Y-%m")
    unique_years = sorted(set(d[:4] for d in year_month))
    fig = go.Figure(data=go.Heatmap(
        z=data.values,
        x=data.columns,
        y=year_month,
        colorscale="RdYlGn",
        zmin=-20, zmax=20,
        colorbar=dict(title="Monthly Return %"),
        text=data.round(1).values,
        texttemplate="%{text:.1f}%",
    ))
    fig.update_layout(title="Monthly Returns Heatmap", template="plotly_white")
    return fig


def radar_chart(result: ComparisonResult) -> go.Figure:
    """Radar chart of normalized metrics."""
    metrics = ["CAGR", "Sharpe", "Sortino", "Calmar", "Max DD"]
    fig = go.Figure()
    for name in result.rankings.index:
        values = []
        for m in metrics:
            v = result.performance_table.loc[name, m] if m in result.performance_table.index else 0
            if m == "Max DD":
                v = -v if v < 0 else 0
            values.append(v)
        values.append(values[0])
        angles = metrics + [metrics[0]]
        fig.add_trace(go.Scatterpolar(
            r=values, theta=angles, fill="toself", name=name,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Strategy Radar Chart",
        showlegend=True,
        template="plotly_white",
    )
    return fig


def performance_heatmap(result: ComparisonResult) -> go.Figure:
    """Heatmap of all performance metrics."""
    perf = result.performance_table * 100
    fig = go.Figure(data=go.Heatmap(
        z=perf.values,
        x=perf.columns,
        y=perf.index,
        colorscale="RdBu",
        zmin=-100, zmax=100,
        colorbar=dict(title="Metric %"),
    ))
    fig.update_layout(title="Performance Metrics Heatmap", template="plotly_white")
    return fig


def holdings_comparison(result: ComparisonResult) -> go.Figure:
    """Bar chart comparing top holdings across strategies."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, s in enumerate(result.strategies):
        if s.snapshots:
            last = next(reversed(s.snapshots.values()))
            if last is not None and not last.empty and "ticker" in last.columns:
                top = last.sort_values("overall", ascending=False).head(5)
                fig.add_trace(go.Bar(
                    name=s.name, x=top["ticker"], y=top["overall"],
                    marker_color=colors[i % len(colors)],
                ))
    fig.update_layout(
        barmode="group", title="Top Holdings Comparison",
        xaxis_title="Ticker", yaxis_title="Score",
        template="plotly_white",
    )
    return fig


def rolling_performance_comparison(result: ComparisonResult) -> go.Figure:
    """Overlay rolling returns of all strategies."""
    fig = go.Figure()
    for name in result.rolling_returns.columns:
        rolling = result.rolling_returns[name].dropna()
        if not rolling.empty:
            fig.add_trace(go.Scatter(
                x=rolling.index, y=rolling.values, mode="lines", name=name,
                line=dict(width=2),
            ))
    fig.update_layout(
        title="Rolling Annualized Returns (252-day)",
        xaxis_title="Date", yaxis_title="Rolling CAGR",
        template="plotly_white",
    )
    return fig


def efficient_frontier(result: ComparisonResult) -> go.Figure:
    """Risk-return frontier with annotated standout strategies."""
    perf = result.performance_table
    if perf.empty:
        return go.Figure()
    fig = go.Figure()
    for name in perf.index:
        fig.add_trace(go.Scatter(
            x=[perf.loc[name, "Volatility"]], y=[perf.loc[name, "CAGR"]],
            mode="markers", name=name, text=name,
            marker=dict(
                size=14,
                color=perf.loc[name, "Sharpe"],
                colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe"),
                line=dict(width=1, color="black"),
            ),
            hovertemplate=f"<b>{name}</b><br>Vol: %{{x:.2%}}<br>CAGR: %{{y:.2%}}<extra></extra>",
        ))
    ann = []
    try:
        ann.append(("Highest Sharpe", perf["Sharpe"].idxmax()))
        ann.append(("Highest CAGR", perf["CAGR"].idxmax()))
        ann.append(("Lowest DD", perf["Max DD"].idxmin()))
        ann.append(("Highest Calmar", perf["Calmar"].idxmax()))
        ann.append(("Best Sortino", perf["Sortino"].idxmax()))
    except Exception:
        ann = []
    for label, name in ann:
        fig.add_annotation(
            x=[perf.loc[name, "Volatility"]], y=[perf.loc[name, "CAGR"]],
            text=label, showarrow=True, arrowhead=1, ax=20, ay=-30,
            font=dict(size=9),
        )
    fig.update_layout(
        title="Efficient Frontier (Risk-Return)",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="CAGR",
        template="plotly_white", hovermode="closest",
    )
    return fig


def holdings_overlap_heatmap(result: ComparisonResult) -> go.Figure:
    """Jaccard similarity heatmap of holdings overlap."""
    mat = result.holdings_overlap_df
    if mat is None or mat.empty:
        return go.Figure()
    fig = go.Figure(data=go.Heatmap(
        z=mat.values, x=mat.columns, y=mat.index,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="Jaccard"),
        text=mat.round(2).values, texttemplate="%{text:.2f}",
        textfont={"size": 10},
    ))
    fig.update_layout(title="Holdings Overlap (Jaccard Similarity)", template="plotly_white")
    return fig


def trade_analysis_chart(result: ComparisonResult) -> go.Figure:
    """Win rate (bars) and profit factor (line) per strategy."""
    df = result.trade_df
    if df is None or df.empty:
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.index, y=df["Win Rate"], name="Win Rate",
        marker_color="#2ca02c", yaxis="y",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Profit Factor"], name="Profit Factor",
        mode="lines+markers", line=dict(color="#d62728", width=2), yaxis="y2",
    ))
    fig.update_layout(
        title="Trade Analysis: Win Rate & Profit Factor",
        xaxis_title="Strategy", yaxis=dict(title="Win Rate"),
        yaxis2=dict(title="Profit Factor", overlaying="y", side="right"),
        template="plotly_white", legend=dict(orientation="h"),
    )
    return fig


def risk_analytics_chart(result: ComparisonResult) -> go.Figure:
    """Grouped bar of key risk metrics per strategy."""
    df = result.risk_table
    if df is None or df.empty:
        return go.Figure()
    metrics = ["Volatility", "Downside Vol", "VaR 5%", "CVaR 5%", "Ulcer Index", "Max DD"]
    metrics = [m for m in metrics if m in df.columns]
    fig = go.Figure()
    for m in metrics:
        fig.add_trace(go.Bar(name=m, x=df.index, y=df[m]))
    fig.update_layout(
        barmode="group", title="Risk Analytics Comparison",
        xaxis_title="Strategy", yaxis_title="Value",
        template="plotly_white", legend=dict(orientation="h"),
    )
    return fig


def benchmark_comparison_chart(result: ComparisonResult) -> go.Figure:
    """Excess CAGR and information ratio vs benchmark."""
    df = result.benchmark_df
    if df is None or df.empty:
        return go.Figure()
    fig = go.Figure()
    if "Excess CAGR" in df.columns:
        fig.add_trace(go.Bar(name="Excess CAGR", x=df.index, y=df["Excess CAGR"], marker_color="#1f77b4"))
    if "Info Ratio" in df.columns:
        fig.add_trace(go.Bar(name="Info Ratio", x=df.index, y=df["Info Ratio"], marker_color="#ff7f0e"))
    fig.update_layout(
        barmode="group", title="Benchmark Comparison",
        xaxis_title="Strategy", yaxis_title="Value",
        template="plotly_white", legend=dict(orientation="h"),
    )
    return fig


__all__ = [
    "equity_curve_comparison",
    "drawdown_comparison",
    "risk_return_scatter",
    "correlation_heatmap",
    "annual_return_heatmap",
    "monthly_return_heatmap",
    "radar_chart",
    "performance_heatmap",
    "holdings_comparison",
    "rolling_performance_comparison",
    "efficient_frontier",
    "holdings_overlap_heatmap",
    "trade_analysis_chart",
    "risk_analytics_chart",
    "benchmark_comparison_chart",
]