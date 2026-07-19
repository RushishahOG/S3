"""Strategy Comparison Research Lab module.

Future implementation: Side-by-side comparison of multiple strategies,
factor models, and portfolio configurations with statistical testing.
"""

import streamlit as st

from app.layouts.base import section


def render() -> None:
    """Render the Strategy Comparison section."""
    section("Strategy Comparison")
    st.markdown(
        """
        **Side-by-side comparison of multiple strategies, factor models,
        and portfolio configurations with statistical significance testing.**

        This module will provide:
        - Multi-strategy backtest comparison dashboard
        - Performance metric comparison tables
        - Statistical significance testing (t-test, bootstrap, Diebold-Mariano)
        - Correlation analysis between strategies
        - Overlap and diversification analysis
        - Regime-conditional performance comparison
        - Risk-adjusted metric comparison (Sharpe, Sortino, Calmar, etc.)
        - Turnover and transaction cost comparison
        - Visual comparison: equity curves, drawdowns, rolling metrics
        """
    )
    st.divider()
    st.info("Implementation will be added in a future update.")


if __name__ == "__main__":
    render()