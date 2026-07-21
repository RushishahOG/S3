"""Plotting engine for Monte Carlo results.

Builds interactive Plotly figures for the UI and Matplotlib PNGs for the PDF
report. All functions are pure (no Streamlit calls) so they can be unit-tested.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.monte_carlo.types import SimulationResult

_PALETTE = {
    "median": "#1f77b4",
    "band": "rgba(31,119,184,0.18)",
    "band_inner": "rgba(31,119,184,0.30)",
    "original": "#d62728",
    "accent": "#2ca02c",
}


def _percentile_curves(equity: np.ndarray, dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    return {
        "p5": np.percentile(equity, 5, axis=0),
        "p25": np.percentile(equity, 25, axis=0),
        "p50": np.percentile(equity, 50, axis=0),
        "p75": np.percentile(equity, 75, axis=0),
        "p95": np.percentile(equity, 95, axis=0),
    }


def fan_chart(result: SimulationResult) -> go.Figure:
    eq = result.equity_curves
    dates = result.sim_dates
    p = _percentile_curves(eq, dates)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=p["p95"], line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p["p5"], line=dict(width=0), fill="tonexty",
        fillcolor=_PALETTE["band"], name="5%–95%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p["p75"], line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p["p25"], line=dict(width=0), fill="tonexty",
        fillcolor=_PALETTE["band_inner"], name="25%–75%", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=p["p50"], mode="lines", name="Median",
        line=dict(color=_PALETTE["median"], width=2),
    ))

    orig_len = min(len(result.original_equity), len(dates))
    if orig_len > 1:
        fig.add_trace(go.Scatter(
            x=dates[:orig_len], y=result.original_equity[:orig_len], mode="lines",
            name="Original Backtest", line=dict(color=_PALETTE["original"], width=2, dash="dash"),
        ))

    fig.update_layout(
        title="Equity Curve Fan Chart",
        xaxis_title="Date", yaxis_title="Portfolio Value",
        hovermode="x unified", template="plotly_white", height=460,
    )
    return fig


def _kde_series(values: np.ndarray, n: int = 200):
    from scipy.stats import gaussian_kde

    clean = values[np.isfinite(values)]
    if len(clean) < 2:
        return np.array([]), np.array([])
    sample = clean if len(clean) <= 4000 else np.random.choice(clean, 4000, replace=False)
    kde = gaussian_kde(sample)
    lo, hi = clean.min(), clean.max()
    pad = (hi - lo) * 0.05 or 1.0
    xs = np.linspace(lo - pad, hi + pad, n)
    return xs, kde(xs)


def distribution_fig(
    values: np.ndarray,
    title: str,
    xlabel: str,
    kde: bool = True,
    risk_zones: list[tuple[float, float, str]] | None = None,
) -> go.Figure:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    fig = go.Figure()
    if risk_zones:
        for lo, hi, color in risk_zones:
            fig.add_vrect(x0=lo, x1=hi, fillcolor=color, opacity=0.25, line_width=0,
                          annotation_text="Risk Zone", annotation_position="top left")
    if len(values) == 0:
        fig.update_layout(title=title, xaxis_title=xlabel, template="plotly_white", height=420)
        return fig
    fig.add_trace(go.Histogram(
        x=values, histnorm="probability density", name="Density",
        marker_color=_PALETTE["median"], opacity=0.75, nbinsx=50,
    ))
    if kde:
        xs, ys = _kde_series(values)
        if len(xs):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                     name="KDE", line=dict(color=_PALETTE["original"], width=2)))
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title="Density",
        template="plotly_white", height=420, bargap=0.01,
    )
    return fig


def scatter_fig(
    x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str,
    color: np.ndarray | None = None,
) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    color = color[mask] if color is not None else None
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Simulations",
        marker=dict(size=4, color=color, colorscale="Viridis", opacity=0.55,
                    showscale=color is not None),
    ))
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel,
        template="plotly_white", height=420,
    )
    return fig


def violin_fig(result: SimulationResult) -> go.Figure:
    df = result.metrics_df
    specs = [
        ("cagr", "CAGR", True),
        ("sharpe", "Sharpe", False),
        ("max_drawdown", "Max Drawdown", False),
        ("sortino", "Sortino", False),
    ]
    fig = go.Figure()
    for col, label, pct in specs:
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        fig.add_trace(go.Violin(
            y=vals, name=label, box_visible=True, meanline_visible=True,
            points=False, line=dict(width=1),
        ))
    fig.update_layout(
        title="Distribution of Key Metrics (Violin)",
        yaxis_title="Value", template="plotly_white", height=460,
        showlegend=False,
    )
    return fig


def box_fig(result: SimulationResult, cols: list[str] | None = None) -> go.Figure:
    cols = cols or ["cagr", "sharpe", "sortino", "calmar", "max_drawdown", "annual_volatility"]
    fig = go.Figure()
    for col in cols:
        vals = result.metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if vals.empty:
            continue
        fig.add_trace(go.Box(y=vals, name=col, boxpoints="outliers"))
    fig.update_layout(
        title="Simulation Metrics (Box Plot)",
        yaxis_title="Value", template="plotly_white", height=460, showlegend=False,
    )
    return fig


def curve_sample_fig(
    result: SimulationResult, kind: str, n: int = 10, random_n: int = 100
) -> go.Figure:
    eq = result.equity_curves
    dates = result.sim_dates
    final = eq[:, -1]
    fig = go.Figure()
    if kind == "worst":
        idx = np.argsort(final)[:n]
        name = f"Worst {n}"
    elif kind == "best":
        idx = np.argsort(final)[-n:]
        name = f"Best {n}"
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(eq.shape[0], size=min(random_n, eq.shape[0]), replace=False)
        name = f"Random {len(idx)}"
    for i in idx:
        fig.add_trace(go.Scatter(x=dates, y=eq[i], mode="lines", name=f"Sim {i}",
                                 line=dict(width=1), showlegend=False, opacity=0.7))
    fig.update_layout(
        title=f"{name} Equity Curves", xaxis_title="Date", yaxis_title="Portfolio Value",
        template="plotly_white", height=460,
    )
    return fig


# --- Matplotlib figures for the PDF report ---------------------------------


def _pdf_figures(result: SimulationResult) -> list[tuple[str, bytes]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: list[tuple[str, bytes]] = []

    def _save(fig: plt.Figure) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    eq = result.equity_curves
    dates = result.sim_dates
    p = _percentile_curves(eq, dates)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.fill_between(dates, p["p5"], p["p95"], color="#1f77b4", alpha=0.18, label="5–95%")
    ax.fill_between(dates, p["p25"], p["p75"], color="#1f77b4", alpha=0.30, label="25–75%")
    ax.plot(dates, p["p50"], color="#1f77b4", lw=1.6, label="Median")
    orig_len = min(len(result.original_equity), len(dates))
    if orig_len > 1:
        ax.plot(dates[:orig_len], result.original_equity[:orig_len], color="#d62728",
                lw=1.4, ls="--", label="Original")
    ax.set_title("Equity Curve Fan Chart")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    out.append(("fan", _save(fig)))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.hist(result.metrics_df["cagr"].replace([np.inf, -np.inf], np.nan).dropna(),
            bins=50, density=True, color="#1f77b4", alpha=0.75)
    ax.set_title("CAGR Distribution")
    ax.set_xlabel("CAGR")
    out.append(("cagr", _save(fig)))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.hist(result.metrics_df["max_drawdown"].replace([np.inf, -np.inf], np.nan).dropna(),
            bins=50, density=True, color="#d62728", alpha=0.75)
    ax.axvline(-0.20, color="k", ls=":", lw=1)
    ax.axvline(-0.30, color="k", ls=":", lw=1)
    ax.set_title("Maximum Drawdown Distribution")
    ax.set_xlabel("Max Drawdown")
    out.append(("mdd", _save(fig)))

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    cagr_c = result.metrics_df["cagr"].replace([np.inf, -np.inf], np.nan)
    mdd_c = result.metrics_df["max_drawdown"].replace([np.inf, -np.inf], np.nan)
    ax.scatter(cagr_c, mdd_c, s=5, alpha=0.4, color="#2ca02c")
    ax.set_title("CAGR vs Max Drawdown")
    ax.set_xlabel("CAGR")
    ax.set_ylabel("Max Drawdown")
    out.append(("scatter", _save(fig)))

    return out
