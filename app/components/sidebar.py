"""Sidebar navigation + global status for the presentation layer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.services import get_storage
from core.config.settings import settings
from core.data.providers.registry import available_providers
from core.data.storage.provisioning import ensure_database, local_status, is_configured

PAGES = [
    ("dashboard", "Dashboard", "Project overview, storage & system diagnostics"),
    ("data_extractor", "Data Extractor", "Download / update market & fundamental data"),
    ("dataset_explorer", "Dataset Explorer", "Inspect & validate stored data"),
    ("feature_engineering", "Feature Engineering", "Generate / regenerate engineered features"),
    ("backtesting", "ARQM Backtest & Research", "Strategy simulation, queue, results & research lab"),
    ("mongo_cloud", "Mongo Cloud Controls", "Manage the DuckDB store in MongoDB GridFS"),
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### {settings.app.name}")
        st.caption(f"v{settings.app.version} · {settings.app.environment}")

        st.divider()
        choice = st.radio(
            "Navigation",
            options=[p[0] for p in PAGES],
            format_func=lambda k: next(p[1] for p in PAGES if p[0] == k),
            key="nav",
        )

        st.divider()
        _render_db_status()
        _render_status()

    st.divider()
    st.caption("Default provider: " + settings.providers.default)
    st.caption("Benchmark: " + settings.universe.benchmark)

    return choice


def _render_db_status() -> None:
    st.markdown("**Database Status**")
    local = local_status()
    configured = is_configured()

    if local["exists"]:
        st.success(f"Local DB: {local['size_mb']:.0f} MB")
        st.caption(local["path"])
    else:
        st.warning("Local DB: Not downloaded")
        if configured:
            if st.button("📥 Download from MongoDB", key="sidebar_db_download"):
                with st.spinner("Downloading ~775 MB from MongoDB GridFS..."):
                    try:
                        ensure_database()
                        st.success("Download complete! Reloading...")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Download failed: {exc}")
        else:
            st.caption("Set MONGO_URI in secrets to enable download")

    st.divider()


def _render_status() -> None:
    try:
        storage = get_storage()
        stats = storage.storage_statistics()
        st.metric("Stored tickers", stats["stored_tickers"])
        st.metric("Price rows", f"{stats['price_rows']:,}")
        st.metric("Feature rows", f"{stats['feature_rows']:,}")
    except Exception:  # pragma: no cover - storage may not be ready
        st.caption("Storage not initialised")
