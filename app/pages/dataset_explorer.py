"""Dataset Explorer page.

A read-only inspection and validation layer over the local analytical store,
organised into a horizontal navigation of four viewers:

* **Fundamental Data Viewer** — raw Screener financial statements (income,
  balance sheet, cash flow, dividends, ratios, quality) for a selected stock.
* **Market Data Viewer** — OHLCV detail, statistics, candlestick chart and
  export for a selected security.
* **Fundamental Feature Engineering Viewer** — traceability dashboard: raw
  financials -> intermediate calculations -> final engineered Quality factors.
* **Market Data Feature Engineering Viewer** — engineered market features
  (return / risk engines) for a selected security.

All data is read directly from storage - no API calls.
"""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from app.components.logs import render_log_panel
from app.explorer import DATASET_SOURCES, get_dataset_source
from app.explorer.base import Severity
from app.explorer.charts import candlestick_chart
from app.layouts.base import page_header, section
from app.services import get_storage
from core.data.providers.base_provider import PriceColumns
from core.utils.dates import MAX_BACKTEST_DATE

# Status-filter presets surfaced in the UI.
FILTER_SUCCESS = "Successfully Downloaded"
FILTER_FAILED = "Failed Downloads"
FILTER_MISSING = "Missing Data"
FILTER_COMPLETE = "Complete Datasets"
FILTER_OPTIONS = [FILTER_SUCCESS, FILTER_FAILED, FILTER_MISSING, FILTER_COMPLETE]


def render() -> None:
    page_header("Dataset Explorer", "Inspect, validate and explore all locally stored data")

    tab_fund, tab_mkt, tab_fund_fe, tab_mkt_fe = st.tabs([
        "Fundamental Data Viewer",
        "Market Data Viewer",
        "Fundamental Feature Engineering Viewer",
        "Market Data Feature Engineering Viewer",
    ])
    with tab_fund:
        _render_fundamental_data_viewer()
    with tab_mkt:
        _render_market_data_viewer()
    with tab_fund_fe:
        _render_fundamental_feature_viewer()
    with tab_mkt_fe:
        _render_market_feature_viewer()

    render_log_panel()


# --------------------------------------------------------------------------- #
# Storage statistics (shared)                                                  #
# --------------------------------------------------------------------------- #
def _storage_stats_block(source_key: str = "market_data") -> None:
    source = get_dataset_source(source_key, storage=get_storage())
    section("Storage Statistics")
    stats = source.storage_statistics()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Securities", stats["total_securities"])
    c2.metric("Total Rows Stored", f"{stats['total_rows']:,}")
    c3.metric("Database Size", _fmt_bytes(stats["db_size_bytes"]))
    c4.metric("Cache Size", _fmt_bytes(stats["cache_size_bytes"]))
    c5.metric(
        "Last Download",
        stats["last_download_time"].strftime("%Y-%m-%d %H:%M")
        if stats["last_download_time"] else "—",
    )


# --------------------------------------------------------------------------- #
# 1. Fundamental Data Viewer                                                   #
# --------------------------------------------------------------------------- #
def _render_fundamental_data_viewer() -> None:
    section("Fundamental Data Viewer")
    st.caption("Raw Screener financial statements for a single stock (no feature engineering).")
    storage = get_storage()
    tickers = sorted(storage.tickers_with_screener_data())
    if not tickers:
        st.info(
            "No screener data yet. Run the **Data Extractor** (Fundamental Data Downloader, "
            "Screener API) to populate the fundamentals tables."
        )
        return
    chosen = st.selectbox("Select a stock", options=tickers, key="fdv_ticker")
    _render_fundamental_detail(storage, chosen)


def _render_fundamental_detail(storage, ticker: str) -> None:
    """Render the raw fundamental tables for a single security."""
    income = storage.get_fundamentals_income_annual([ticker])
    dividends = storage.get_fundamentals_dividends([ticker])
    ratios = storage.get_fundamentals_ratios([ticker])

    st.subheader("Raw Fundamental Data")
    _render_fundamental_table(income, f"Income Statements — {ticker}", f"inc_{ticker}")
    _render_fundamental_table(dividends, f"Dividend History — {ticker}", f"div_{ticker}")
    _render_fundamental_table(ratios, f"Ratio Snapshot — {ticker}", f"rat_{ticker}")

    st.subheader("Screener (single source of truth)")
    sc_company = storage.get_fundamentals_company([ticker])
    sc_income = storage.get_fundamentals_income_annual([ticker])
    sc_qtr = storage.get_fundamentals_income_quarterly([ticker])
    sc_bal = storage.get_fundamentals_balance_sheet([ticker])
    sc_cf = storage.get_fundamentals_cashflow([ticker])
    sc_div = storage.get_fundamentals_dividends([ticker])
    sc_ratios = storage.get_fundamentals_ratios([ticker])
    sc_quality = storage.get_fundamental_quality_features([ticker])

    if sc_company.empty and sc_income.empty and sc_quality.empty:
        st.info("No Screener data ingested for this ticker yet.")
    else:
        _render_fundamental_table(sc_company, f"Company — {ticker}", f"sc_co_{ticker}")
        _render_fundamental_table(sc_income, f"Annual Income — {ticker}", f"sc_ia_{ticker}")
        _render_fundamental_table(sc_qtr, f"Quarterly Income — {ticker}", f"sc_iq_{ticker}")
        _render_fundamental_table(sc_bal, f"Balance Sheet — {ticker}", f"sc_bs_{ticker}")
        _render_fundamental_table(sc_cf, f"Cash Flow — {ticker}", f"sc_cf_{ticker}")
        _render_fundamental_table(sc_div, f"Dividends — {ticker}", f"sc_div_{ticker}")
        _render_fundamental_table(sc_ratios, f"Ratios — {ticker}", f"sc_rat_{ticker}")
        # ROE is stored on the engineered quality features (anchor years); derive
        # it from the quality frame for display.
        sc_roe = pd.DataFrame()
        if not sc_quality.empty and "roe" in sc_quality.columns:
            sc_roe = sc_quality[["ticker", "financial_year", "roe"]].copy()
        if not sc_roe.empty:
            _render_fundamental_table(sc_roe, f"Return on Equity (ROE) — {ticker}", f"sc_roe_{ticker}")
        sq_cols = [c for c in sc_quality.columns if c not in ("ticker", "financial_year", "created_at")]
        if not sc_quality.empty and sq_cols:
            picked_q = st.multiselect(
                "Screener Quality factor columns",
                options=sq_cols, default=sq_cols, key=f"scq_cols_{ticker}",
            )
            if picked_q:
                _render_fundamental_table(
                    sc_quality[["ticker", "financial_year", *picked_q]].drop(columns=["financial_year"], errors="ignore"),
                    f"Screener Quality Features — {ticker}", f"scq_{ticker}",
                )


def _render_fundamental_table(df: pd.DataFrame, title: str, search_key: str) -> None:
    """Render one fundamental table with search + CSV export (sortable via UI)."""
    section(title)
    if df is None or df.empty:
        st.info("No records stored.")
        return
    search = st.text_input("Search", "", key=f"search_{search_key}")
    view = df.copy()
    if search:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search, case=False, na=False)
        ).any(axis=1)
        view = view[mask]
    st.dataframe(view, use_container_width=True, height=380)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export as CSV", data=csv,
        file_name=f"{search_key}.csv", mime="text/csv",
        key=f"export_{search_key}",
    )


# --------------------------------------------------------------------------- #
# 2. Market Data Viewer                                                        #
# --------------------------------------------------------------------------- #
def _render_market_data_viewer() -> None:
    section("Market Data Viewer")
    st.caption("OHLCV detail, statistics and price chart for a single security.")
    source = get_dataset_source("market_data", storage=get_storage())
    summary = source.security_summary()
    if summary.empty:
        st.info("No securities stored yet. Use the **Data Extractor** page to download data.")
        return

    search = st.text_input("Search by company name or ticker", "", key="mdv_search")
    df = summary.copy()
    if search:
        mask = df["ticker"].str.contains(search, case=False, na=False) | \
               df["company_name"].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]
    df = df.reset_index(drop=True)
    st.caption(f"{len(df)} of {len(summary)} securities shown.")

    if df.empty:
        st.info("No securities match the current search.")
        return

    options = [f"{r.ticker} — {r.company_name}" for r in df.itertuples()]
    chosen = st.selectbox("Select a security to inspect", options, key="mdv_select")
    ticker = chosen.split(" — ")[0]
    _render_market_detail(source, ticker)


def _render_market_detail(source, ticker: str) -> None:
    df = source.fetch_dataset(ticker)
    if df.empty:
        st.warning(f"No records stored for {ticker}.")
        return

    if PriceColumns.ADJ_CLOSE not in df.columns or df[PriceColumns.ADJ_CLOSE].isna().all():
        st.error(
            f"⚠️ **Adjusted Close missing** for {ticker}. Momentum and all "
            "return-based factors will be unreliable for this security."
        )

    stats = source.dataset_statistics(df, ticker)
    section(f"Summary — {ticker}")
    cols = st.columns(4)
    cols[0].metric("Total Trading Days", f"{stats['total_trading_days']:,}")
    cols[1].metric("First Trading Date", str(stats["first_trading_date"]))
    cols[2].metric("Last Trading Date", str(stats["last_trading_date"]))
    cols[3].metric("Avg Volume", f"{stats['avg_volume']:,}")

    cols2 = st.columns(4)
    cols2[0].metric("Min Price", f"{stats['min_price']:.2f}")
    cols2[1].metric("Max Price", f"{stats['max_price']:.2f}")
    cols2[2].metric("Missing Values", stats["missing_values"])
    cols2[3].metric("Duplicate Rows", stats["duplicate_rows"])

    section("Price Chart")
    st.altair_chart(candlestick_chart(df, ticker), use_container_width=True)

    section("Complete Dataset")
    table_cols = [c for c in source.display_columns() if c in df.columns]
    view = df[table_cols].copy()
    view[PriceColumns.DATE] = pd.to_datetime(view[PriceColumns.DATE]).dt.date
    st.dataframe(view, use_container_width=True, height=400)

    csv = source.export_csv(ticker).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export displayed dataset as CSV",
        data=csv,
        file_name=f"{ticker}_dataset.csv",
        mime="text/csv",
        key=f"export_{ticker}",
    )


# --------------------------------------------------------------------------- #
# 3. Fundamental Feature Engineering Viewer                                    #
# --------------------------------------------------------------------------- #
def _render_fundamental_feature_viewer() -> None:
    section("Fundamental Feature Engineering Viewer")
    st.caption("Traceability: raw financials -> intermediate calcs -> engineered Quality factors.")
    storage = get_storage()
    tickers = sorted(storage.tickers_with_screener_data())
    if not tickers:
        st.info(
            "No screener data yet. Run the **Data Extractor** (Fundamental Data Downloader, "
            "Screener API) to populate the fundamentals tables."
        )
        return

    chosen = st.selectbox("Select a stock", options=tickers, key="ffev_ticker")
    _render_fundamental_feature_stock(storage, chosen)


def _render_fundamental_feature_stock(storage, ticker: str) -> None:
    from core.factors.fundamental import (
        FundamentalQualityEngine,
        compute_cash_roce,
        compute_dividend_payout_ratio,
        compute_dividend_payout_ratio_cumulative,
        compute_equity_to_total_capital,
        compute_interest_coverage_ratio,
        compute_ocf_to_ebitda,
        compute_roa,
        compute_roce,
        compute_roe,
        compute_sustainable_growth_rate,
    )

    company = storage.get_fundamentals_company([ticker])
    income = storage.get_fundamentals_income_annual([ticker])
    qtr = storage.get_fundamentals_income_quarterly([ticker])
    balance = storage.get_fundamentals_balance_sheet([ticker])
    cf = storage.get_fundamentals_cashflow([ticker])
    div = storage.get_fundamentals_dividends([ticker])
    ratios = storage.get_fundamentals_ratios([ticker])
    features = storage.get_fundamental_quality_features([ticker])

    # --- Company information ----------------------------------------------
    section("Company Information")
    if not company.empty:
        c = company.iloc[0]
        cols = st.columns(5)
        cols[0].metric("Company", c.get("company_name", ticker))
        cols[1].metric("Sector", c.get("sector", "—"))
        cols[2].metric("Industry", c.get("industry", "—"))
        cols[3].metric("Market Cap", f"{_fmt(c.get('market_cap'))}")
        cols[4].metric("Current Price", f"{_fmt(c.get('current_price'))}")
        cols2 = st.columns(3)
        cols2[0].metric("P/E", f"{_fmt(c.get('pe'))}")
        cols2[1].metric("P/B", f"{_fmt(c.get('pb'))}")
        cols2[2].metric("Div Yield %", f"{_fmt(c.get('dividend_yield'))}")
    else:
        st.caption("No company record.")

    # --- Raw statements ---------------------------------------------------
    _table(income, "Annual Income Statement")
    _table(qtr, "Quarterly Income Statement")
    _table(balance, "Balance Sheet")
    _table(cf, "Cash Flow Statement")
    _table(div, "Dividend History")
    _table(ratios, "Raw Ratios")

    # --- Intermediate calculations ----------------------------------------
    section("Intermediate Calculations")
    st.caption("Per-year inputs behind each engineered factor (before aggregation).")
    inter = _fundamental_intermediate(
        storage, ticker, income, balance, cf, div, ratios,
        compute_cash_roce, compute_dividend_payout_ratio,
        compute_dividend_payout_ratio_cumulative, compute_equity_to_total_capital,
        compute_interest_coverage_ratio, compute_ocf_to_ebitda, compute_roa,
        compute_roce, compute_roe, compute_sustainable_growth_rate,
    )
    _table(inter, "Intermediate Calculations")

    # --- Final engineered factor values -----------------------------------
    section("Final Engineered Quality Factors")
    if not features.empty:
        disp = features.copy()
        front = ["ticker", "financial_year", "roe", "roce", "roa",
                 "interest_coverage_ratio", "equity_to_total_capital",
                 "dividend_payout_ratio", "ocf_to_ebitda", "cash_roce",
                 "sustainable_growth_rate"]
        rest = [c for c in disp.columns if c not in front]
        disp = disp[[c for c in front if c in disp.columns] + rest]
        _table(disp, "Engineered Quality Features")
    else:
        st.warning("No engineered features. Run the Quality engine.")


def _fundamental_intermediate(storage, ticker, income, balance, cf, div, ratios,
                              compute_cash_roce, compute_dividend_payout_ratio,
                              compute_dividend_payout_ratio_cumulative,
                              compute_equity_to_total_capital,
                              compute_interest_coverage_ratio, compute_ocf_to_ebitda,
                              compute_roa, compute_roce, compute_roe,
                              compute_sustainable_growth_rate) -> pd.DataFrame:
    if income is None or income.empty:
        return pd.DataFrame()
    years = pd.to_numeric(income["financial_year"], errors="coerce").dropna().astype("int").tolist()
    base = pd.DataFrame({"financial_year": years})
    parts = {
        "roe": compute_roe(income, balance),
        "roce": compute_roce(ratios, years),
        "roa_net_profit": compute_roa(income, balance),
        "icr_ebit": compute_interest_coverage_ratio(income),
        "equity_to_capital": compute_equity_to_total_capital(balance),
        "payout_ratio": compute_dividend_payout_ratio(income, div),
        "payout_cumulative": compute_dividend_payout_ratio_cumulative(income, div),
        "ocf_to_ebitda": compute_ocf_to_ebitda(income, cf),
        "cash_roce": compute_cash_roce(balance, cf),
        "sgr": compute_sustainable_growth_rate(income, balance, div, ratios, years),
    }
    out = base
    for name, s in parts.items():
        if s is None or s.empty:
            continue
        s = s.rename(name) if s.name != name else s
        out = out.join(s, how="left")
    return out


def _table(df: pd.DataFrame, title: str) -> None:
    section(title)
    if df is None or df.empty:
        st.caption("No records stored.")
        return
    view = df.copy()
    st.dataframe(view, use_container_width=True, hide_index=True)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"Export {title} CSV", data=csv,
        file_name=f"{title.replace(' ', '_').lower()}.csv",
        mime="text/csv", key=f"export_{title}_{abs(hash(title))}",
    )


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


# --------------------------------------------------------------------------- #
# 4. Market Data Feature Engineering Viewer                                    #
# --------------------------------------------------------------------------- #
def _render_market_feature_viewer() -> None:
    section("Market Data Feature Engineering Viewer")
    st.caption(
        "Daily market features from the risk / momentum engine: "
        "`beta`, `momentum_unscaled`, `momentum_scaled`, `semi_deviation`."
    )
    source = get_dataset_source("market_data", storage=get_storage())
    summary = source.security_summary()
    if summary.empty:
        st.info("No securities stored yet. Download market data on the **Data Extractor** page first.")
        return

    search = st.text_input("Search by company name or ticker", "", key="mfv_search")
    df = summary.copy()
    if search:
        mask = df["ticker"].str.contains(search, case=False, na=False) | \
               df["company_name"].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]
    df = df.reset_index(drop=True)
    st.caption(f"{len(df)} of {len(summary)} securities shown.")
    if df.empty:
        st.info("No securities match the current search.")
        return

    options = [f"{r.ticker} — {r.company_name}" for r in df.itertuples()]
    chosen = st.selectbox("Select a security", options, key="mfv_select")
    ticker = chosen.split(" — ")[0]
    _render_market_feature_detail(ticker)


# Daily feature columns produced by the current market engine, with labels.
_MARKET_DAILY_FEATURES = {
    "beta": "Beta (12M vs benchmark)",
    "momentum_unscaled": "Momentum (12-1, unscaled)",
    "momentum_scaled": "Momentum (risk-adjusted)",
    "semi_deviation": "Semi-deviation (12M, annualised)",
}


def _render_market_feature_detail(ticker: str) -> None:
    """Show the engineered daily market features for a single security."""
    feats = _load_features(ticker)
    if feats.empty:
        st.info(
            f"No engineered features stored for {ticker} yet. Generate them on the "
            "**Feature Engineering** page; they will appear here automatically."
        )
        return

    feat_cols = [c for c in feats.columns if c not in ("date", "ticker")]
    avail = [c for c in _MARKET_DAILY_FEATURES if c in feat_cols]
    stale = [c for c in feat_cols if c not in _MARKET_DAILY_FEATURES]

    st.caption(
        f"**{len(feat_cols)}** columns · **{len(feats):,}** daily rows · "
        f"**{len(avail)}** current daily feature(s)"
    )

    # Coverage / freshness summary for the current daily features.
    if avail:
        cov_rows = []
        for c in avail:
            s = feats[c].dropna()
            cov_rows.append({
                "Feature": _MARKET_DAILY_FEATURES[c],
                "Coverage %": f"{100 * feats[c].notna().mean():.1f}",
                "Mean": round(float(s.mean()), 4) if not s.empty else None,
                "Min": round(float(s.min()), 4) if not s.empty else None,
                "Max": round(float(s.max()), 4) if not s.empty else None,
            })
        st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

    # Feature chart selector.
    chartable = avail if avail else feat_cols
    if chartable:
        pick = st.multiselect(
            "Plot features",
            options=chartable,
            default=chartable[: min(len(chartable), 2)],
            format_func=lambda c: _MARKET_DAILY_FEATURES.get(c, c),
            key=f"mfv_plot_{ticker}",
        )
        if pick:
            chart_df = feats[["date"] + pick].copy()
            chart_df["date"] = pd.to_datetime(chart_df["date"])
            chart_df = chart_df.set_index("date").sort_index()
            st.line_chart(chart_df)

    search = st.text_input("Search features by name", "", key=f"feat_search_{ticker}")
    view_cols = feat_cols
    if search:
        view_cols = [c for c in feat_cols if search.lower() in c.lower()]

    display_cols = ["date"] + sorted(view_cols)
    view = feats[display_cols].copy()
    view["date"] = pd.to_datetime(view["date"])

    # Trim to features from 2015-01-01 onwards.
    start = pd.Timestamp("2015-01-01")
    view = view[view["date"] >= start].copy()
    if view.empty:
        st.info("No features available from 2015-01-01 for this security.")
        return
    view["date"] = view["date"].dt.date

    # Rolling features need a full 12-month window before they produce values,
    # so the earliest rows are blank. Hide that warm-up by default so the table
    # matches the chart (which skips NaN points); offer a toggle to show all.
    warmup_hidden = False
    if avail:
        all_na_mask = view[avail].isna().all(axis=1)
        if all_na_mask.any() and not all_na_mask.all():
            warmup_hidden = not st.checkbox(
                "Show warm-up rows (blank until 12M of history)",
                value=False,
                key=f"mfv_warmup_{ticker}",
            )
            if warmup_hidden:
                view = view[~all_na_mask].reset_index(drop=True)
                st.caption(
                    f"Hiding {int(all_na_mask.sum())} warm-up row(s) with no values yet "
                    f"(first valid date: {view['date'].iloc[0]})."
                )

    sort_col = st.selectbox(
        "Sort by",
        options=display_cols,
        index=0,
        key=f"feat_sort_col_{ticker}",
    )
    sort_asc = st.checkbox("Ascending", value=True, key=f"feat_sort_asc_{ticker}")
    view = view.sort_values(sort_col, ascending=sort_asc)

    st.dataframe(view, use_container_width=True, height=400)

    if stale:
        with st.expander(f"{len(stale)} obsolete column(s) still present"):
            st.write(stale)
            st.caption(
                "Regenerate features on the **Feature Engineering** page to prune these."
            )

    col_csv, col_pq = st.columns(2)
    csv_data = view.to_csv(index=False).encode("utf-8")
    col_csv.download_button(
        "Export as CSV",
        data=csv_data,
        file_name=f"{ticker}_features.csv",
        mime="text/csv",
        key=f"feat_csv_{ticker}",
    )

    buf = io.BytesIO()
    view.to_parquet(buf, index=False)
    col_pq.download_button(
        "Export as Parquet",
        data=buf.getvalue(),
        file_name=f"{ticker}_features.parquet",
        mime="application/octet-stream",
        key=f"feat_pq_{ticker}",
    )


def _load_features(ticker: str) -> pd.DataFrame:
    """Load features for a ticker, trying FeatureStore then StorageManager."""
    from core.feature_engineering.feature_store import FeatureStore

    try:
        store = FeatureStore()
        feats = store.get_features(tickers=[ticker])
        if not feats.empty:
            return feats
    except Exception:
        pass

    try:
        return get_storage().get_features(tickers=[ticker])
    except Exception:
        return pd.DataFrame()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _fmt_bytes(n: int) -> str:
    if n is None:
        return "—"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


if __name__ == "__main__":
    render()
