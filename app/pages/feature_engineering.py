"""Feature Engineering page.

Regenerates the engineered feature store using the current risk engine, which
emits the four daily features ``beta``, ``momentum_unscaled``,
``momentum_scaled`` and ``semi_deviation``. Obsolete columns left behind by
previous code versions (the old std/semi/beta monthly+weekly matrix) are pruned.
This is what populates the "Engineered Features" tab in the Dataset Explorer.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.logs import render_log_panel
from app.layouts.base import page_header, section
from app.services import get_storage
from core.config.settings import settings
from core.data.storage.storage_manager import StorageManager
from core.feature_engineering.return_engine import (
    compute_all_returns,
    merge_returns_into_panel,
    prepare_price_panel,
)
from core.feature_engineering.risk_engine import compute_all_risk


def _valid_risk_columns() -> set[str]:
    """Daily risk columns the current engine produces."""
    return {"beta", "momentum_unscaled", "momentum_scaled", "semi_deviation"}


def render() -> None:
    page_header("Feature Engineering", "Generate / regenerate the engineered feature store")

    storage = get_storage()

    tab_fund, tab_mkt = st.tabs(["Fundamentals Feature Engineering", "Market Data Feature Engineering"])
    with tab_fund:
        _render_fundamentals_feature_engineering(storage)
    with tab_mkt:
        _render_market_feature_engineering(storage)

    render_log_panel()


def _render_fundamentals_feature_engineering(storage) -> None:
    # --- Quality Factor (Screener, 16 factors) -----------------------------
    section("Quality Factor — Screener Engine (16 factors)")
    st.caption(
        "Computes all 16 supported Quality factors (Profitability, Solvency, "
        "Dividend, Cash Flow, Growth) from the normalised Screener tables into "
        "**fundamental_quality_features**. Growth factors expose Median + "
        "Weighted-Average roll-ups (recency-weighted). Precomputed and cached — "
        "never calculated on dashboard load. Requires a Screener ingest."
    )
    from core.factors.fundamental import (
        FundamentalQualityEngine,
        QUALITY_FACTOR_FUNCTIONS,
    )

    qf_options = sorted(QUALITY_FACTOR_FUNCTIONS.keys())
    qf_labels = {k: k.replace("_", " ").title() for k in qf_options}
    selected_qf = st.multiselect(
        "Quality factors to engineer",
        options=qf_options,
        default=qf_options,
        format_func=lambda k: qf_labels[k],
        key="qf_subset",
    )
    if st.button("Engineer Quality Factors (Screener)", key="eng_qf_screener", type="primary"):
        if not selected_qf:
            st.warning("Select at least one factor.")
        else:
            with st.spinner("Engineering Quality factors..."):
                fqe = FundamentalQualityEngine(storage)
                qfeat = fqe.compute(store=True, features=selected_qf)
            if qfeat.empty:
                st.warning("No screener data found. Run the **Data Extractor** (Fundamental Data Downloader) first.")
            else:
                st.success(
                    f"Engineered {len(qfeat):,} rows across "
                    f"{qfeat['ticker'].nunique():,} tickers "
                    f"({len(selected_qf)} factor(s))."
                )
                cov = qfeat[[c for c in qfeat.columns if c not in ("ticker", "financial_year")]].notna().mean().sort_values(ascending=False)
                st.dataframe(cov.rename("coverage").reset_index().rename(columns={"index": "factor"}), height=320)


def _render_market_feature_engineering(storage) -> None:
    stats = storage.storage_statistics()
    if stats["price_rows"] == 0:
        st.info("No market data stored yet. Download data on the **Data Extractor** (Market Data Downloader) page first.")
        return

    st.caption(
        "Daily market features are computed: **beta**, **momentum_unscaled**, "
        "**momentum_scaled** and **semi_deviation** (12-month / 12-1 windows). "
        "Regenerating also removes obsolete columns from earlier versions."
    )

    if st.button("Generate / Regenerate Features", type="primary"):
        with st.spinner("Building return panel..."):
            prices = storage.get_prices(fields=["ticker", "date", "adj_close"])
            prices = prices.rename(
                columns={"ticker": "Ticker", "date": "Date", "adj_close": "Adj Close"}
            )
            panel_prep = prepare_price_panel(prices, "Adj Close")
            returns_dict = compute_all_returns(panel_prep, "Adj Close")
            panel = merge_returns_into_panel(panel_prep, returns_dict)

        with st.spinner("Computing benchmark returns..."):
            bench_ret = _build_benchmark_returns(prices, returns_dict)

        with st.spinner("Computing risk features (this can take a minute)..."):
            risk = compute_all_risk(panel, bench_ret)
            risk = risk.rename(columns={"Ticker": "ticker", "Date": "date"})
            # Writes need a read-write connection; the shared app connection is
            # read-only so it can coexist with the read-only backtest worker.
            with StorageManager() as rw_storage:
                n = rw_storage.upsert_features(risk)

        # Prune obsolete risk columns produced by previous code versions
        # (e.g. old 3/6/9/12m std/semi/beta daily+weekly matrix).
        with st.spinner("Pruning obsolete columns..."):
            valid = _valid_risk_columns()
            stale = [
                c
                for c in storage.feature_columns()
                if (
                    c.startswith("std_")
                    or c.startswith("semi_dev_")
                    or c.startswith("beta_")
                    or c.startswith("STDDEV_")
                    or c.startswith("SEMI_")
                    or c.startswith("BETA_")
                    or c.startswith("MOM_")
                )
                and c not in valid
            ]
            dropped = storage.drop_feature_columns(stale) if stale else 0

        st.success(
            f"Upserted {n:,} feature rows. "
            f"Removed {dropped} obsolete risk column(s): "
            f"{', '.join(stale) if stale else 'none'}."
        )

    # --- Current feature inventory -----------------------------------------
    section("Current Feature Store")
    cols = storage.feature_columns()
    if not cols:
        st.info("Feature store is empty. Click **Generate / Regenerate Features** above.")
        return

    valid = _valid_risk_columns()
    risk_cols = sorted([c for c in cols if c in valid])
    other = sorted([c for c in cols if c not in valid])

    c1, c2 = st.columns(2)
    c1.metric("Risk/momentum columns", len(risk_cols))
    c2.metric("Other columns", len(other))

    st.caption(
        "**Daily features present:** " + (", ".join(risk_cols) if risk_cols else "—")
    )
    if other:
        with st.expander(f"Other columns ({len(other)})"):
            st.write(other)

    stale = [
        c
        for c in cols
        if (
            c.startswith("std_")
            or c.startswith("semi_dev_")
            or c.startswith("beta_")
            or c.startswith("STDDEV_")
            or c.startswith("SEMI_")
            or c.startswith("BETA_")
            or c.startswith("MOM_")
        )
        and c not in valid
    ]
    if stale:
        st.warning(
            f"Stale risk columns still present: {', '.join(stale)}. "
            "Regenerate features to remove them."
        )


def _build_benchmark_returns(prices: pd.DataFrame, returns_dict: dict) -> pd.DataFrame:
    """Build a benchmark return series for beta calculation.

    Prefers the configured benchmark ticker (NIFTY 500); falls back to the
    cross-sectional mean daily return as a market proxy.
    """
    bench_ticker = settings.universe.benchmark
    prices = prices.copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    if bench_ticker in set(prices["Ticker"]):
        bp = (
            prices[prices["Ticker"] == bench_ticker][["Date", "Adj Close"]]
            .sort_values("Date")
            .set_index("Date")["Adj Close"]
            .pct_change()
            .dropna()
            .reset_index()
        )
        bp.columns = ["Date", "benchmark_return"]
        return bp
    daily = returns_dict["daily_return"]
    mean_ret = daily.groupby("Date")["daily_return"].mean().reset_index()
    mean_ret.columns = ["Date", "benchmark_return"]
    return mean_ret.dropna()
