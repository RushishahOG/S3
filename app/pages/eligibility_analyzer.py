"""Eligibility Analyzer page.

Derives the earliest valid backtest start date from *actual data availability*
(not a fixed constant) and presents a complete, interactive eligibility
timeline across the full competition window (01-01-2006 → 31-05-2026). All
controls (lookback, rebalance frequency, coverage threshold) are fully
configurable and recompute the analysis automatically on change.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.layouts.base import page_header, section
from app.services import get_storage, get_universe_manager
from core.eligibility import (
    REBALANCE_FREQUENCIES,
    EligibilityAnalyzer,
    list_factors,
    max_lookback_for,
)
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE

# Known companies with continuous data from 2006, used for the validation panel.
KNOWN_2006_TICKERS = [
    "RELIANCE.NS",
    "INFY.NS",
    "TCS.NS",
    "ITC.NS",
    "SBIN.NS",
    "ICICIBANK.NS",
    "HDFCBANK.NS",
    "ACC.NS",
    "ABB.NS",
]


def render() -> None:
    page_header(
        "Eligibility Analyzer",
        "Earliest valid backtest start date derived from real data availability",
    )
    _render_body()


def _render_body() -> None:
    storage = get_storage()
    universe = get_universe_manager().default_universe()
    universe_tickers = universe.tickers

    # --- Factor selection (auto-discovered from the registry) ---------------
    factors = list_factors()
    factor_names = sorted(factors.keys())
    selected = st.multiselect(
        "Factors to include",
        options=factor_names,
        default=["Beta", "Momentum"],
        help="The required lookback is the maximum horizon across the selected factors.",
    )

    # --- Configuration controls (all recompute automatically) ---------------
    computed_lookback = max_lookback_for(selected) or 12
    default_lookback = min(max(computed_lookback, 1), 120)

    max_lookback = st.slider(
        "Maximum Lookback (Months)",
        min_value=1,
        max_value=120,
        value=default_lookback,
        step=1,
        help="Warm-up window a stock needs before it becomes eligible. Defaults to "
        "the max required by the selected factors; override with any value 1-120.",
    )

    freq_label = st.selectbox(
        "Rebalance Frequency",
        options=list(REBALANCE_FREQUENCIES.keys()),
        index=list(REBALANCE_FREQUENCIES.keys()).index("Monthly"),
        help="Cadence of candidate rebalance dates. New frequencies can be added "
        "to REBALANCE_FREQUENCIES in core/eligibility/analyzer.py.",
    )
    rebalance_freq = REBALANCE_FREQUENCIES[freq_label]

    threshold = st.slider(
        "Minimum Universe Coverage %",
        min_value=0,
        max_value=100,
        value=80,
        step=1,
        help="Coverage threshold the recommended start date must achieve.",
    )
    threshold_frac = threshold / 100.0

    # --- Run analysis (cheap; recomputes automatically on any change) -------
    with st.spinner("Computing eligibility timeline..."):
        earliest = storage.earliest_date_per_ticker(universe_tickers)
        latest = storage.latest_date_per_ticker(universe_tickers)
        analyzer = EligibilityAnalyzer(universe_tickers, earliest, latest)
        result = analyzer.analyze(max_lookback, threshold_frac, rebalance_freq)

    _render_results(result, universe, selected, max_lookback, threshold)


def _render_results(result, universe, selected_factors, max_lookback, threshold) -> None:
    summary = result.summary()
    rec = result.recommended_start

    # --- Headline metric cards ----------------------------------------------
    section("Recommended Backtest Configuration")
    c1, c2, c3 = st.columns(3)
    with c1:
        if rec is not None:
            st.metric(
                "Recommended Backtest Start",
                rec.date().isoformat(),
                help="First rebalance date where coverage >= threshold.",
            )
        else:
            st.metric("Recommended Backtest Start", "Not achievable")
    with c2:
        st.metric(
            "Eligible Stocks",
            f"{summary['eligible_at_start']:,}",
            help="Stocks eligible at the recommended start date.",
        )
    with c3:
        st.metric(
            "Universe Coverage %",
            f"{summary['coverage_at_start']:.1f}%",
            help="Coverage at the recommended start date.",
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        st.metric("Universe Size", f"{summary['universe_size']:,}")
    with c5:
        st.metric("Downloaded Stocks", f"{summary['data_universe_size']:,}")
    with c6:
        st.metric("Required Lookback", f"{max_lookback} months")

    if rec is None:
        st.warning(
            f"No rebalance date reaches {threshold}% coverage of the {summary['universe_size']:,}-"
            f"stock universe (max achieved: {summary['coverage_at_end']:.1f}% at "
            f"{MAX_BACKTEST_DATE.date()}). Download more history or lower the "
            "threshold / lookback."
        )

    # --- Eligibility timeline chart -----------------------------------------
    section("Eligibility Timeline")
    _render_timeline_chart(result, threshold)

    # --- Debug / full timeline table ---------------------------------------
    section("Debug Panel — Full Eligibility Timeline")
    st.caption(
        "Every rebalance date from 2006-01-01 to 2026-05-31, with the complete "
        "eligibility breakdown."
    )
    _render_debug_table(result)

    # --- Validation panel ---------------------------------------------------
    _render_validation(result)

    # --- Per-stock eligibility table ----------------------------------------
    section("Per-Stock Eligibility")
    st.caption(
        "First trading date, warm-up-adjusted first eligible date, and last "
        "available date for each constituent."
    )
    _render_per_stock_table(result)


def _render_timeline_chart(result, threshold: int) -> None:
    tl = result.timeline.copy()
    tl["date"] = pd.to_datetime(tl["date"])

    base = (
        alt.Chart(tl)
        .mark_line(color="#1f77b4", point=False)
        .encode(
            x=alt.X("date:T", title="Rebalance Date", scale=alt.Scale(domain=["2006-01-01", "2026-05-31"])),
            y=alt.Y("coverage_pct:Q", title="Universe Coverage %", scale=alt.Scale(0, 100)),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("coverage_pct:Q", title="Coverage %", format=".1f"),
                alt.Tooltip("eligible_count:Q", title="Eligible"),
                alt.Tooltip("stocks_with_data:Q", title="With Data"),
            ],
        )
    )

    thr_rule = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(color="red", strokeDash=[6, 4])
        .encode(y="y:Q")
    )

    layers = [base, thr_rule]

    if result.recommended_start is not None:
        rec_df = pd.DataFrame({"x": [pd.Timestamp(result.recommended_start)]})
        rec_rule = (
            alt.Chart(rec_df)
            .mark_rule(color="green", strokeWidth=2)
            .encode(x="x:T")
        )
        layers.append(rec_rule)

    chart = (
        alt.layer(*layers)
        .properties(height=380, title="Universe Coverage Over Time (2006-01 → 2026-05)")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _render_debug_table(result) -> None:
    dbg = result.timeline.copy()
    dbg["date"] = pd.to_datetime(dbg["date"]).dt.date
    dbg = dbg.rename(
        columns={
            "date": "Date",
            "universe_size": "Universe Size",
            "stocks_with_data": "Stocks With Data",
            "eligible_count": "Eligible Stocks",
            "coverage_pct": "Coverage %",
            "excluded_missing": "Excluded (Missing Data)",
            "excluded_insufficient": "Excluded (Insufficient Lookback)",
        }
    )[
        [
            "Date",
            "Universe Size",
            "Stocks With Data",
            "Eligible Stocks",
            "Coverage %",
            "Excluded (Missing Data)",
            "Excluded (Insufficient Lookback)",
        ]
    ]
    st.dataframe(dbg, use_container_width=True, height=420)
    csv = dbg.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export full timeline as CSV",
        data=csv,
        file_name="eligibility_timeline.csv",
        mime="text/csv",
        key="export_timeline",
    )


def _render_validation(result) -> None:
    section("Validation — 2006-Era Stocks")
    ps = result.per_stock
    present = ps[ps["ticker"].isin(KNOWN_2006_TICKERS)]
    if present.empty:
        st.info(
            "None of the known 2006-era tickers (Reliance, Infosys, TCS, ITC, SBI, "
            "ICICI Bank, HDFC Bank, ACC, ABB, ...) are present in the current data. "
            "Download their history to validate eligibility."
        )
        return
    rows = []
    for _, r in present.iterrows():
        if not r["has_data"]:
            status = "No data"
        elif pd.isna(r["first_eligible_date"]):
            status = "No data"
        else:
            status = f"Eligible from {pd.Timestamp(r['first_eligible_date']).date()}"
        rows.append({"Ticker": r["ticker"], "First Trading": pd.Timestamp(r["first_trading_date"]).date(), "Status": status})
    st.table(pd.DataFrame(rows))


def _render_per_stock_table(result) -> None:
    ps = result.per_stock.copy()
    ps["first_trading_date"] = pd.to_datetime(ps["first_trading_date"]).dt.date
    ps["first_eligible_date"] = pd.to_datetime(ps["first_eligible_date"]).dt.date
    ps["latest_date"] = pd.to_datetime(ps["latest_date"]).dt.date
    ps = ps.rename(
        columns={
            "ticker": "Ticker",
            "has_data": "Has Data",
            "first_trading_date": "First Trading",
            "first_eligible_date": "First Eligible",
            "latest_date": "Latest Data",
        }
    )
    ps = ps.sort_values("Ticker").reset_index(drop=True)

    search = st.text_input("Search by ticker", "", key="elig_search")
    if search:
        ps = ps[ps["Ticker"].str.contains(search, case=False, na=False)]

    st.dataframe(ps, use_container_width=True, height=420)
