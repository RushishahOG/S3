"""Research Lab - Main component for ARQM Backtest page.

This module provides the horizontal tab navigation and dynamically renders
the selected research module. Each tab is an independent module that can
be implemented independently in the future.
"""

from __future__ import annotations

import streamlit as st

from app.pages.research_lab import (
    parameter_optimization,
    monte_carlo,
    efficient_frontier,
    sensitivity_analysis,
    strategy_comparison,
)


RESEARCH_TABS = [
    ("parameter_optimization", "Parameter Optimization", parameter_optimization),
    ("monte_carlo", "Monte Carlo Simulation", monte_carlo),
    ("efficient_frontier", "Efficient Frontier", efficient_frontier),
    ("sensitivity_analysis", "Sensitivity Analysis", sensitivity_analysis),
    ("strategy_comparison", "Strategy Comparison", strategy_comparison),
]


def render_research_lab() -> None:
    """Render the Research Lab section with horizontal tab navigation.

    This function creates a horizontal tab navigation bar and renders the
    selected research module. The selected tab is preserved in session state
    across page interactions.
    """
    # Initialize session state for selected tab
    if "research_lab_tab" not in st.session_state:
        st.session_state["research_lab_tab"] = "parameter_optimization"

    # Build tab labels and keys
    tab_keys = [t[0] for t in RESEARCH_TABS]
    tab_labels = [t[1] for t in RESEARCH_TABS]
    tab_modules = {t[0]: t[2] for t in RESEARCH_TABS}

    # Create horizontal tabs using Streamlit's native tabs
    tabs = st.tabs(tab_labels)

    # Get current tab index from session state
    current_tab = st.session_state.get("research_lab_tab", "parameter_optimization")
    current_idx = tab_keys.index(current_tab) if current_tab in tab_keys else 0

    # Render each tab's content
    for idx, (tab, key) in enumerate(zip(tabs, tab_keys)):
        with tab:
            # Update session state when tab is selected
            if idx == current_idx:
                st.session_state["research_lab_tab"] = key
            # Render the module content
            tab_modules[key].render()


def render_research_lab_navigation() -> None:
    """Render just the horizontal navigation bar (tabs) without content.

    Use this if you need to place the tabs separately from the content.
    """
    tab_keys = [t[0] for t in RESEARCH_TABS]
    tab_labels = [t[1] for t in RESEARCH_TABS]

    # Use st.tabs for native horizontal scrolling and active tab highlighting
    tabs = st.tabs(tab_labels)

    current_tab = st.session_state.get("research_lab_tab", "parameter_optimization")
    current_idx = tab_keys.index(current_tab) if current_tab in tab_keys else 0

    for idx, (tab, key) in enumerate(zip(tabs, tab_keys)):
        with tab:
            if idx == current_idx:
                st.session_state["research_lab_tab"] = key
            # Content is rendered separately via render_research_lab_content()


def render_research_lab_content() -> None:
    """Render the content for the currently selected research tab.

    Call this after render_research_lab_navigation() to display the
    active module's content.
    """
    tab_modules = {t[0]: t[2] for t in RESEARCH_TABS}
    current_tab = st.session_state.get("research_lab_tab", "parameter_optimization")
    tab_modules[current_tab].render()


if __name__ == "__main__":
    render_research_lab()