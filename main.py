"""Application entry point.

Configures Streamlit, renders the navigation sidebar, and dispatches to the
selected page. All business logic lives in ``core.*``; this file is purely
presentation wiring.
"""

from __future__ import annotations

import streamlit as st

from app.components.sidebar import render_sidebar
from app.pages import (
    dashboard,
    data_extractor,
    dataset_explorer,
    feature_engineering,
    mongo_cloud,
    backtesting,
)
from core.utils.logging_config import configure_logging

PAGE_RENDERERS = {
    "dashboard": dashboard.render,
    "data_extractor": data_extractor.render,
    "dataset_explorer": dataset_explorer.render,
    "feature_engineering": feature_engineering.render,
    "backtesting": backtesting.render,
    "mongo_cloud": mongo_cloud.render,
}


def main() -> None:
    configure_logging()
    st.set_page_config(
        page_title="Smart Beta Research Platform",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    choice = render_sidebar()
    renderer = PAGE_RENDERERS.get(choice, dashboard.render)
    renderer()


if __name__ == "__main__":
    main()
