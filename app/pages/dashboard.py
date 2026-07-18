"""Dashboard page: project overview, dataset summary, latest update, cache
status, and system diagnostics (storage, cache, freshness, providers).

System Information was merged into this page so the platform exposes a single
landing view; the standalone System Information entry was removed from the
navigation.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from app.components.logs import render_log_panel
from app.layouts.base import page_header, section
from app.services import get_market_data_manager, get_storage
from app.pages import eligibility_analyzer, universe_explorer
from core.config import settings
from core.data.providers.registry import available_providers
from core.utils.dates import MAX_BACKTEST_DATE
from core.utils.dates import date_range_business
from core.utils.paths import PROJECT_ROOT


def render() -> None:
    page_header("Dashboard", "Smart Beta Quantitative Research Platform - Version 1")

    tab_dash, tab_elig, tab_uni = st.tabs([
        "Dashboard",
        "Eligibility Analyzer",
        "Universe Explorer",
    ])
    with tab_dash:
        _render_dashboard()
    with tab_elig:
        eligibility_analyzer._render_body()
    with tab_uni:
        universe_explorer._render_body()

    render_log_panel()


def _render_dashboard() -> None:
    storage = get_storage()
    mdm = get_market_data_manager()

    stats = storage.storage_statistics()
    feat_lo, feat_hi = storage.feature_date_range()

    # --- Project overview ---
    section("Project Overview")
    st.markdown(
        """
        A modular, layered **quantitative research platform** (not a trading app).
        Version 1 delivers:
        - Provider-abstracted market data ingestion (Yahoo Finance)
        - Local DuckDB storage with incremental caching
        - Universe management (NIFTY 500)
        - Feature engineering pipeline & reusable feature store
        - **Momentum** and **Low Volatility** factor frameworks
        - Interactive factor exploration dashboard
        """
    )

    # --- Dataset summary ---
    section("Dataset Summary")
    cols = st.columns(4)
    with cols[0]:
        st.metric("Stored securities", stats["stored_tickers"])
    with cols[1]:
        st.metric("Price rows", f"{stats['price_rows']:,}")
    with cols[2]:
        st.metric("Feature columns", stats["feature_columns"])
    with cols[3]:
        st.metric("Feature rows", f"{stats['feature_rows']:,}")

    if feat_lo and feat_hi:
        st.info(
            f"Feature store covers **{feat_lo.date()}** to **{feat_hi.date()}** "
            f"across **{stats['stored_tickers']}** securities."
        )
    else:
        st.warning("No features generated yet. Visit **Feature Engineering** after downloading data.")

    # --- Latest update ---
    section("Latest Update")
    latest = storage.latest_date_per_ticker()
    if latest:
        most_recent = max(latest.values())
        # "Up to date" means data reaches the fixed competition end date.
        up_to_date = sum(1 for d in latest.values() if d >= MAX_BACKTEST_DATE)
        st.metric("Most recent stored date", str(most_recent))
        st.caption(
            f"{up_to_date} / {len(latest)} securities reach the competition end date "
            f"({MAX_BACKTEST_DATE.date()})."
        )
    else:
        st.info("No data downloaded yet. Use the **Data Extractor** page.")

    # --- Cache status ---
    section("Cache Status")
    if latest:
        stale = [t for t, d in latest.items() if d < MAX_BACKTEST_DATE]
        if stale:
            st.warning(
                f"{len(stale)} securities do not yet reach {MAX_BACKTEST_DATE.date()} "
                "and may need a refresh."
            )
            with st.expander("Show securities needing update"):
                st.write(stale)
        else:
            st.success("All stored securities reach the competition end date.")
    else:
        st.info("Nothing cached yet.")

    # --- System information (merged) ---
    _render_system_information(storage, mdm)


def _render_system_information(storage, mdm) -> None:
    """Storage, cache, freshness and provider diagnostics."""
    section("System Information")

    # --- Storage usage ---
    section("Storage Usage")
    s = storage.storage_statistics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DB size (MB)", f"{s['db_size_bytes'] / 1e6:.2f}")
    c2.metric("Price rows", f"{s['price_rows']:,}")
    c3.metric("Feature rows", f"{s['feature_rows']:,}")
    c4.metric("Feature columns", s["feature_columns"])
    st.caption(f"Database: `{s['db_path']}`")

    # --- Cache size ---
    section("Cache Size")
    cache_bytes = _dir_size(os.path.join(PROJECT_ROOT, "storage"))
    logs_bytes = _dir_size(os.path.join(PROJECT_ROOT, "storage", "logs"))
    c1, c2 = st.columns(2)
    c1.metric("Total storage / cache (MB)", f"{cache_bytes / 1e6:.2f}")
    c2.metric("Log size (MB)", f"{logs_bytes / 1e6:.2f}")

    if st.button("Export feature store to Parquet"):
        out = os.path.join(PROJECT_ROOT, "storage", "parquet", "feature_store.parquet")
        storage.export_parquet(settings.storage.feature_store_table, out)
        st.success(f"Exported to {out}")

    # --- Download statistics ---
    section("Download Statistics")
    latest = storage.latest_date_per_ticker()
    c1, c2 = st.columns(2)
    c1.metric("Securities downloaded", len(latest))
    c2.metric("Providers configured", len(available_providers()))

    # --- Data freshness ---
    section("Data Freshness")
    if latest:
        rows = [
            {"ticker": t, "last_date": d.date(), "days_to_competition_end": (MAX_BACKTEST_DATE - d).days}
            for t, d in sorted(latest.items(), key=lambda kv: kv[1])
        ]
        fresh = pd.DataFrame(rows)
        st.dataframe(fresh, use_container_width=True)
    else:
        st.info("No downloaded data yet.")

    # --- Installed providers ---
    section("Installed Providers")
    for p in available_providers():
        try:
            prov = mdm.provider if p == mdm.provider_key else None
            available = prov.is_available() if prov else True
        except Exception:
            available = False
        st.write(f"- **{p}** · {'available' if available else 'unavailable'}")


def _dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total
