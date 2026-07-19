"""ARQM Backtest & Research Platform - Main Orchestrator.

This is the main entry point for the refactored backtest module.
It provides four integrated sections:
1. Manual Testing - Configure and submit strategies
2. Portfolio Queue - Execution dashboard
3. Results - Completed backtest analysis
4. Research Lab - Research tools (Parameter Optimization, Monte Carlo, etc.)
"""

from __future__ import annotations

import streamlit as st

from app.layouts.base import page_header, section
from app.pages.backtest.manual_testing.render import render_manual_testing
from app.pages.backtest.portfolio_queue.render import render_portfolio_queue
from app.pages.backtest.results.render import render_results
from app.pages.backtest.state import get_backtest_state
from app.pages.research_lab.research_lab import render_research_lab


def render() -> None:
    """Main entry point for the ARQM Backtest & Research Platform."""
    page_header("ARQM Strategy Simulation & Backtesting", "Adaptive Regime-based Quality Momentum")

    state = get_backtest_state()

    # Main navigation tabs
    tabs = st.tabs([
        "📝 Manual Testing",
        "📦 Portfolio Queue",
        "📊 Results",
        "🔬 Research Lab",
    ])

    # Render each section
    with tabs[0]:
        render_manual_testing()

    with tabs[1]:
        render_portfolio_queue()

    with tabs[2]:
        render_results()

    with tabs[3]:
        render_research_lab()


if __name__ == "__main__":
    render()