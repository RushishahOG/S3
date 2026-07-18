"""NIFTY 500 Historical Constituents Explorer.

Visualises how the NIFTY 500 universe evolved across the full competition
window (2006-01-01 → 2026-05-31). Built on the modular
:mod:`core.universe_explorer` package, so any future index (NIFTY 50, Midcap
150, custom or international) plugs in via its own provider with no UI changes.

Sections
--------
1. Historical Constituents by Period (two-year windows)
2. Longest Continuous Members
3. Membership Timeline (interactive Gantt for all 500 stocks)
4. Annual Universe Summary (additions / removals / sector mix + charts)
5. Universe at Backtest Start Date (integration with the Eligibility Analyzer)
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from app.layouts.base import page_header, section
from app.services import get_storage, get_universe_manager
from core.eligibility import EligibilityAnalyzer, REBALANCE_FREQUENCIES
from core.universe_explorer import CurrentSnapshotProvider, UniverseExplorer
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE

DATA_NOTE = (
    "The current NIFTY 500 constituent file (ind_nifty500list_*.csv) is a "
    "**point-in-time snapshot** — it contains a *Company Name*, *Industry*, "
    "*Symbol*, *Series* and *ISIN* but **no Sector, entry or exit dates**. "
    "True index-inclusion history is therefore unavailable from this file. "
    "Entry dates shown here are derived as a **proxy** from the earliest "
    "available price history for each stock; all stocks are treated as "
    "**Active through the last trading day**. Upload historical constituent "
    "files (via `CsvMembershipProvider`) to replace these proxies with exact "
    "inclusion/exit dates."
)


@st.cache_data(show_spinner="Loading NIFTY 500 membership...")
def load_memberships():
    storage = get_storage()
    mgr = get_universe_manager()
    universe = mgr.default_universe()
    constituents_path = mgr.get_provider(universe.name).constituents_path
    provider = CurrentSnapshotProvider(
        universe=universe, constituents_path=constituents_path, storage=storage
    )
    return provider.get_memberships()


def render() -> None:
    page_header(
        "NIFTY 500 Historical Constituents Explorer",
        "How the NIFTY 500 universe evolved from 2006-01-01 to 2026-05-31",
    )
    _render_body()


def _render_body() -> None:
    with st.expander("Data availability & methodology", expanded=False):
        st.caption(DATA_NOTE)

    memberships = load_memberships()
    explorer = UniverseExplorer(memberships)

    st.metric("Constituents tracked", f"{len(memberships):,}",
              help="Size of the current NIFTY 500 snapshot.")

    tabs = st.tabs([
        "Historical Constituents by Period",
        "Longest Continuous Members",
        "Membership Timeline (Gantt)",
        "Annual Universe Summary",
    ])

    with tabs[0]:
        _render_period_explorer(explorer)
    with tabs[1]:
        _render_longest(explorer)
    with tabs[2]:
        _render_timeline(explorer)
    with tabs[3]:
        _render_annual(explorer)

    st.divider()
    _render_integration(explorer)


# --------------------------------------------------------------------------- #
# 1. Historical constituents by period
# --------------------------------------------------------------------------- #
def _render_period_explorer(explorer: UniverseExplorer) -> None:
    section("NIFTY 500 Universe — Constituents Present in Each Period")
    st.caption(
        "Two-year windows across the competition period. 'Present' = a stock "
        "available at any point in the window. 'Present Throughout' = available "
        "for the entire window."
    )
    for label, start, end in explorer.periods():
        members = explorer.constituents_in_period(start, end)
        throughout = [m for m, s in members if s == "Present Throughout"]
        with st.expander(f"{label}: {len(members):,} constituents ({len(throughout):,} present throughout)"):
            rows = []
            for m, s in members:
                rows.append({
                    "Company Name": m.company_name,
                    "Ticker": m.ticker,
                    "Entry (proxy)": m.entry_date.date() if m.entry_date else None,
                    "Status": m.status,
                    "Period Status": s,
                    "Sector": m.sector or "N/A",
                })
            df = pd.DataFrame(rows).sort_values(["Period Status", "Company Name"])
            st.dataframe(df, use_container_width=True, height=400)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"Export {label} as CSV", data=csv,
                file_name=f"nifty500_{label}.csv", mime="text/csv",
                key=f"exp_{label}",
            )


# --------------------------------------------------------------------------- #
# 2. Longest continuous members
# --------------------------------------------------------------------------- #
def _render_longest(explorer: UniverseExplorer) -> None:
    section("Longest Continuous Members of NIFTY 500")
    st.caption(
        "Ranked by years in the index. 'Present Throughout' flags stocks that "
        "appear to have been constituents for the entire competition window."
    )
    df = explorer.longest_continuous_df()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Median tenure (yrs)", f"{df['Years in Index'].median():.1f}")
    with c2:
        st.metric("Max tenure (yrs)", f"{df['Years in Index'].max():.1f}")
    with c3:
        st.metric("Present throughout", f"{int((df['Present Throughout'] == 'Yes').sum()):,}")

    top_n = st.slider("Show top N", 10, min(200, len(df)), 50, key="longest_n")
    st.dataframe(df.head(top_n), use_container_width=True, height=520)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Export full ranking as CSV", data=csv,
                       file_name="nifty500_longest_members.csv", mime="text/csv",
                       key="exp_longest")


# --------------------------------------------------------------------------- #
# 3. Membership timeline (Gantt)
# --------------------------------------------------------------------------- #
def _render_timeline(explorer: UniverseExplorer) -> None:
    section("Membership Timeline — All 500 Stocks")
    st.caption(
        "Each bar spans a stock's membership from its (proxy) entry date to "
        "exit (or the last trading day). Use the filters then zoom/pan the chart."
    )
    tl = explorer.timeline_df().copy()
    tl["Entry"] = pd.to_datetime(tl["Entry"])
    tl["Exit"] = pd.to_datetime(tl["Exit"])

    col1, col2, col3 = st.columns(3)
    with col1:
        search = st.text_input("Search ticker / company", "", key="tl_search")
    with col2:
        sectors = sorted(tl["Sector"].unique().tolist())
        sector_filter = st.multiselect("Filter by Industry", options=sectors, key="tl_sector")
    with col3:
        view = st.radio("View", ["All", "Active only", "Removed only", "Present throughout only"],
                        key="tl_view", horizontal=True)

    sub = tl
    if search:
        sub = sub[sub["Ticker"].str.contains(search, case=False, na=False) |
                  sub["Company"].str.contains(search, case=False, na=False)]
    if sector_filter:
        sub = sub[sub["Sector"].isin(sector_filter)]
    if view == "Active only":
        sub = sub[sub["Status"] == "Active"]
    elif view == "Removed only":
        sub = sub[sub["Status"] == "Removed"]
    elif view == "Present throughout only":
        sub = sub[sub["Present Throughout"] == "Yes"]

    if sub.empty:
        st.info("No stocks match the current filters.")
        return

    height = min(1800, max(500, 16 * len(sub)))
    chart = (
        alt.Chart(sub)
        .mark_bar(size=9)
        .encode(
            x=alt.X("Entry:T", title="Membership Start",
                    scale=alt.Scale(domain=[explorer.min_date, explorer.max_date])),
            x2="Exit:T",
            y=alt.Y("Company:N", title="Stock",
                    sort=alt.EncodingSortField(field="Entry", op="min", order="ascending"),
                    axis=alt.Axis(labels=False, ticks=False) if len(sub) > 80 else alt.Axis()),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Active", "Removed"],
                                                        range=["#2ca02c", "#d62728"])),
            tooltip=[
                alt.Tooltip("Company:N", title="Company"),
                alt.Tooltip("Ticker:N", title="Ticker"),
                alt.Tooltip("Sector:N", title="Industry"),
                alt.Tooltip("Entry:T", title="Entry"),
                alt.Tooltip("Exit Label:N", title="Exit"),
                alt.Tooltip("Years in Index:Q", title="Years", format=".1f"),
            ],
        )
        .properties(height=height, title=f"NIFTY 500 membership ({len(sub)} stocks)")
        .configure_view(strokeWidth=0)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Scroll/pan and zoom the chart to inspect individual stocks. "
               "Axis labels are hidden when many stocks are shown — use search to isolate.")


# --------------------------------------------------------------------------- #
# 4. Annual universe summary
# --------------------------------------------------------------------------- #
def _render_annual(explorer: UniverseExplorer) -> None:
    section("Annual Universe Summary")
    st.caption("Constituents active, new additions, removals and corporate actions "
               "per calendar year across the competition window.")
    ys = explorer.year_summary_df()

    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(
            alt.Chart(ys).mark_bar(color="#1f77b4").encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Total Constituents:Q", title="Constituents active"),
                tooltip=["Year", "Total Constituents", "New Additions", "Removals"],
            ).properties(height=300, title="Constituents active per year"),
            use_container_width=True,
        )
    with c2:
        add_rem = ys.melt(id_vars=["Year"],
                          value_vars=["New Additions", "Removals", "Delistings", "Corporate Actions"],
                          var_name="Category", value_name="Count")
        st.altair_chart(
            alt.Chart(add_rem).mark_bar().encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Category:N"),
            ).properties(height=300, title="Additions / removals / delistings / corp-actions"),
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        # Cumulative active (running total of constituents active at year end)
        cum = ys.copy()
        st.altair_chart(
            alt.Chart(cum).mark_line(color="#ff7f0e", point=True).encode(
                x=alt.X("Year:O", title="Year"),
                y=alt.Y("Total Constituents:Q", title="Cumulative active constituents"),
            ).properties(height=280, title="Cumulative Active Constituents"),
            use_container_width=True,
        )
    with col2:
        # Histogram of years in index
        lc = explorer.longest_continuous_df()
        bins = pd.cut(lc["Years in Index"],
                      bins=[0, 5, 10, 15, 20, 25, 100],
                      labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25+"])
        hist = bins.value_counts().reindex(["0-5", "5-10", "10-15", "15-20", "20-25", "25+"]).fillna(0)
        hist_df = hist.reset_index()
        hist_df.columns = ["Tenure (yrs)", "Stocks"]
        st.altair_chart(
            alt.Chart(hist_df).mark_bar(color="#9467bd").encode(
                x=alt.X("Tenure (yrs):O", title="Years in NIFTY 500"),
                y=alt.Y("Stocks:Q", title="Number of stocks"),
            ).properties(height=280, title="Distribution of tenure"),
            use_container_width=True,
        )

    section("Sector / Industry Mix by Year")
    year = st.selectbox("Select year", options=ys["Year"].tolist(),
                        index=len(ys) - 1, key="annual_year")
    sec = explorer.sector_distribution(int(year))
    if sec.empty:
        st.info(f"No constituents tracked for {year}.")
    else:
        st.altair_chart(
            alt.Chart(sec).mark_bar(color="#17becf").encode(
                x=alt.X("Constituents:Q", title="Constituents"),
                y=alt.Y("Sector:N", title="Industry", sort="-x"),
                tooltip=["Sector", "Constituents"],
            ).properties(height=max(300, 22 * len(sec)), title=f"Industry mix in {year}"),
            use_container_width=True,
        )
        st.dataframe(sec, use_container_width=True, height=400)

    st.divider()
    st.subheader("Full annual table")
    st.dataframe(ys, use_container_width=True, height=420)
    csv = ys.to_csv(index=False).encode("utf-8")
    st.download_button("Export annual summary as CSV", data=csv,
                       file_name="nifty500_annual_summary.csv", mime="text/csv",
                       key="exp_annual")


# --------------------------------------------------------------------------- #
# 5. Integration with Eligibility Analyzer
# --------------------------------------------------------------------------- #
def _render_integration(explorer: UniverseExplorer) -> None:
    section("Universe at Backtest Start Date — Eligibility Analyzer Integration")
    st.caption(
        "Pick a backtest start date to see the NIFTY 500 universe as it stood "
        "then, the stocks eligible for backtesting, and the churn (newly added "
        "vs removed) relative to the current date."
    )
    start = st.date_input(
        "Backtest start date",
        value=MIN_BACKTEST_DATE.date(),
        min_value=MIN_BACKTEST_DATE.date(),
        max_value=MAX_BACKTEST_DATE.date(),
        key="uni_start",
    )
    snap = explorer.universe_at_date(pd.Timestamp(start))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Universe at date", f"{snap['universe_size']:,}")
    with c2:
        st.metric("Historical Constituents", f"{len(snap['historical_constituents']):,}")
    with c3:
        st.metric("Newly Added (prior 12m)", f"{len(snap['newly_added']):,}")
    with c4:
        st.metric("Removed by date", f"{len(snap['removed']):,}")

    # Eligible stocks via the Eligibility Analyzer (reused, not reimplemented).
    with st.spinner("Computing eligible stocks..."):
        eligible = _eligible_at(start.isoformat())
    if eligible is not None:
        st.metric("Eligible Stocks (with data + lookback)", f"{len(eligible):,}",
                  help="Stocks with data and sufficient warm-up at the selected date.")
        with st.expander("View eligible tickers"):
            st.write(eligible)

    with st.expander(f"Newly Added ({len(snap['newly_added'])})"):
        if snap["newly_added"]:
            st.dataframe(
                pd.DataFrame([{"Ticker": m.ticker, "Company": m.company_name,
                               "Entry (proxy)": m.entry_date.date() if m.entry_date else None}
                              for m in snap["newly_added"]]),
                use_container_width=True, height=300,
            )
        else:
            st.info("No newly added stocks in the trailing 12 months from this date.")

    with st.expander(f"Removed ({len(snap['removed'])})"):
        if snap["removed"]:
            st.dataframe(
                pd.DataFrame([{"Ticker": m.ticker, "Company": m.company_name,
                               "Exit": m.exit_date.date() if m.exit_date else None,
                               "Status": m.status}
                              for m in snap["removed"]]),
                use_container_width=True, height=300,
            )
        else:
            st.info("No removed stocks as of this date.")


@st.cache_data(show_spinner=False)
def _eligible_at(start_iso: str) -> list[str] | None:
    try:
        start_ts = pd.Timestamp(start_iso)
        storage = get_storage()
        universe = get_universe_manager().default_universe()
        tickers = universe.tickers
        earliest = storage.earliest_date_per_ticker(tickers)
        latest = storage.latest_date_per_ticker(tickers)
        analyzer = EligibilityAnalyzer(tickers, earliest, latest)
        # Monthly cadence keeps this cheap; eligible_at is date-independent.
        result = analyzer.analyze(12, 0.0, REBALANCE_FREQUENCIES["Monthly"])
        return result.eligible_at(start_ts)
    except Exception:  # pragma: no cover - storage may be locked / unavailable
        return None
