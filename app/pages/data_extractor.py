"""Data Extractor page: combined market & fundamental data ingestion.

This page merges the former *Data Manager* (market data) and *Fundamental
Downloader* (Screener API) pages behind a single navigation entry, exposing a
horizontal ``st.tabs`` switch between the two extraction flows.

Tabs
----
* **Fundamental Data Downloader** — Screener API ingestion of NIFTY 500
  fundamentals / Quality data.
* **Market Data Downloader** — provider / universe selection, download,
  validation and storage statistics for NIFTY 500 market (price) data.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from app.components.logs import render_log_panel
from app.layouts.base import page_header, section
from app.services import get_market_data_manager, get_storage, get_universe_manager
from core.config.settings import settings
from core.config.providers_config import is_provider_enabled, providers_config
from core.data.ingestion.screener_downloader import ScreenerDownloader, ScreenerJob
from core.data.providers.apify_screener_provider import ApifyScreenerProvider
from core.data.providers.registry import available_providers
from core.factors.fundamental import FundamentalQualityEngine
from core.utils.dates import (
    DateRangeError,
    DEFAULT_BACKTEST_START,
    MAX_BACKTEST_DATE,
    MIN_BACKTEST_DATE,
    validate_backtest_range,
)
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

_SCREENER_CSV = "nifty_500_constituents/nifty500_screener_urls.csv"


def render() -> None:
    page_header("Data Extractor", "Ingest market & fundamental data for the NIFTY 500")

    tab_fund, tab_mkt = st.tabs(["Fundamental Data Downloader", "Market Data Downloader"])
    with tab_fund:
        _render_fundamental()
    with tab_mkt:
        _render_market()

    render_log_panel()


# --------------------------------------------------------------------------- #
# Fundamental Data Downloader tab                                              #
# --------------------------------------------------------------------------- #
def _load_screener_jobs(csv_path: str) -> list[ScreenerJob]:
    """Build ScreenerJob list from the uploaded CSV (Company Name,Symbol,URL)."""
    df = pd.read_csv(csv_path)
    jobs: list[ScreenerJob] = []
    for _, row in df.iterrows():
        url = str(row.get("Screener URL", "")).strip()
        if not url or url.lower() == "nan":
            continue
        sym = str(row.get("Symbol", "")).strip().upper()
        ticker = f"{sym}.NS" if sym and not sym.endswith(".NS") else sym
        name = str(row.get("Company Name", ticker)).strip()
        jobs.append(ScreenerJob(url=url, ticker=ticker, company_name=name))
    return jobs


def _render_fundamental() -> None:
    section("Fundamental Data Downloader")
    st.caption("Ingest fundamentals via the Screener API (single source of truth)")

    storage = get_storage()
    cfg = providers_config.get("apify", {})
    batch_size = int(cfg.get("batch_size", 25))

    # Process a batch if a run is active (at top => live incremental updates).
    _maybe_process_batch(storage, batch_size)

    # --- Live stats -------------------------------------------------------
    total = 0
    downloaded = len(storage.tickers_with_screener_data())
    meta = storage.get_fundamental_download_metadata()
    failed = int(meta["status"].isin(["failed", "no_data", "partial"]).sum()) if not meta.empty else 0
    try:
        jobs_all = _load_screener_jobs(_SCREENER_CSV)
        total = len(jobs_all)
    except Exception as exc:
        st.warning(f"Could not load screener URL CSV at {_SCREENER_CSV}: {exc}")

    if not is_provider_enabled("apify") or not cfg.get("api_token"):
        st.warning(
            "Apify provider is not enabled or APIFY_API_TOKEN is missing. "
            "Set the token in the project `.env` file. The pipeline will fail "
            "until the token is provided."
        )

    # --- Metrics ----------------------------------------------------------
    section("Download Status")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Screener URLs", total)
    c2.metric("Downloaded", downloaded)
    c3.metric("Remaining", max(total - downloaded, 0) if not st.session_state.get("fund_running") else len(st.session_state.get("fund_remaining", [])))
    c4.metric("Failed", failed)
    c5.metric("Retry Queue", failed)

    # --- Progress + ETA ---------------------------------------------------
    if st.session_state.get("fund_running"):
        done = st.session_state.fund_done
        remaining_now = len(st.session_state.get("fund_remaining", []))
        total_planned = done + remaining_now
        pct = (done / total_planned) if total_planned else 0.0
        st.progress(min(pct, 1.0))
        elapsed = time.time() - (st.session_state.get("fund_start") or time.time())
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (remaining_now / rate) if rate > 0 else 0.0
        sc, ec = st.columns([1, 4])
        with sc:
            if st.button("Stop", key="fund_stop"):
                st.session_state.fund_running = False
                st.rerun()
        st.caption(
            f"Mode: {st.session_state.get('fund_mode')} · Batch: {batch_size} · "
            f"Processed {done}/{total_planned} · Elapsed {elapsed:.0f}s · ETA ~{eta:.0f}s"
        )

    # --- Dry run (selected stock) ----------------------------------------
    section("Dry Run (Single Stock)")
    st.caption(
        "Validate the Screener actor + parsing + storage + feature engineering on "
        "the Screener URL of a single selected stock before a full bulk run."
    )
    dry_options = sorted({j.ticker for j in jobs_all}) if total else []
    if dry_options:
        dry_ticker = st.selectbox(
            "Stock to dry run", options=dry_options, index=0, key="dry_ticker"
        )
        if st.button("Run Dry Run", key="dry_btn"):
            _run_dry_run(storage, dry_ticker)
    else:
        st.info("No screener URLs loaded; cannot run a dry run.")

    # --- Bulk Controls ----------------------------------------------------
    section("Bulk Controls")
    st.caption(
        "Iterate over every Screener URL from the CSV. The downloader resumes "
        "automatically; failed URLs can be retried."
    )
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Download All", type="primary", key="dl_all"):
            _begin_run(storage, "all")
    with b2:
        if st.button("Resume Missing", key="dl_resume"):
            _begin_run(storage, "resume")
    with b3:
        if st.button("Retry Failed", key="dl_retry"):
            _begin_run(storage, "retry")

    b4, b5, b6 = st.columns(3)
    with b4:
        if st.button("Download Selected…", key="dl_sel"):
            st.session_state["show_sel"] = True
    with b5:
        if st.button("Download Missing Only", key="dl_missing"):
            _begin_run(storage, "missing")
    with b6:
        if st.button("Refresh Existing Records", key="dl_refresh"):
            _begin_run(storage, "refresh")

    if st.session_state.get("show_sel"):
        sel = st.multiselect(
            "Select stocks (by ticker)",
            options=sorted({j.ticker for j in jobs_all}) if total else [],
            key="dl_sel_tickers",
        )
        if st.button("Start Selected", key="dl_sel_go"):
            _begin_run(storage, "selected", selected=sel)

    # --- Live log ---------------------------------------------------------
    if st.session_state.get("fund_log"):
        section("Recent Activity")
        for line in st.session_state.fund_log[-15:]:
            st.caption(line)


def _run_dry_run(storage, ticker: str) -> None:
    try:
        jobs = _load_screener_jobs(_SCREENER_CSV)
    except Exception as exc:
        st.error(f"Cannot load screener CSV: {exc}")
        return
    job = next((j for j in jobs if j.ticker == ticker), None)
    if job is None:
        st.error(f"No screener URL found for ticker {ticker}.")
        return
    sample = [job]
    provider = ApifyScreenerProvider()
    if not provider.is_available():
        st.error("Screener provider not configured (check APIFY_API_TOKEN / actor id).")
        return

    summary_rows = []
    raw_items: dict[str, list] = {}
    with st.spinner(f"Dry-running {ticker} screener URL ..."):
        for job in sample:
            rec = {"ticker": job.ticker, "url": job.url}
            try:
                result, items = provider.fetch_with_raw(job.url)
                raw_items[job.ticker] = items or []
                rec["success"] = bool(result.company or result.income_annual)
                rec["missing_sections"] = _missing_sections(result)
                rec["invalid_url"] = False
                rec["error"] = ""
                # Persist + engineer to validate the full chain.
                dl = ScreenerDownloader(storage, [job], provider=provider)
                dl.run_batch([job])
                eng = FundamentalQualityEngine(storage)
                eng.compute(tickers=[job.ticker], store=True)
            except Exception as exc:  # noqa: BLE001
                rec["success"] = False
                rec["missing_sections"] = "—"
                rec["invalid_url"] = True
                rec["error"] = f"{type(exc).__name__}: {exc}"
            summary_rows.append(rec)

    df = pd.DataFrame(summary_rows)
    st.success(f"Dry run complete: {int(df['success'].sum())}/{len(df)} succeeded")
    st.dataframe(df, use_container_width=True, hide_index=True)

    with st.expander("Dry Run Summary", expanded=True):
        ok = df[df["success"]]
        bad = df[~df["success"]]
        st.markdown(f"**Success:** {len(ok)}  **Failure / Invalid:** {len(bad)}")
        if not bad.empty:
            st.markdown("**Failures / Missing Sections:**")
            for _, r in bad.iterrows():
                st.caption(f"{r['ticker']}: {r['error'] or r['missing_sections']}")

    # --- Raw API responses (for schema validation) -------------------------
    section("Raw API Responses")
    st.caption(
        "Complete unparsed payloads returned by the Screener actor for each "
        "dry-run URL. Use these to verify the extraction mapping matches the "
        "live schema."
    )
    for job in sample:
        items = raw_items.get(job.ticker, [])
        with st.expander(f"{job.ticker} — raw items ({len(items)})", expanded=False):
            st.json(items if items else "[] (no response)")


def _missing_sections(result) -> str:
    missing = []
    if not result.company:
        missing.append("company")
    if not result.income_annual:
        missing.append("income_annual")
    if not result.balance_sheet:
        missing.append("balance_sheet")
    if not result.cashflow:
        missing.append("cashflow")
    if not result.dividends:
        missing.append("dividends")
    if not result.ratios:
        missing.append("ratios")
    return ", ".join(missing) or "none"


def _begin_run(storage, mode: str, selected: list[str] | None = None) -> None:
    try:
        jobs_all = _load_screener_jobs(_SCREENER_CSV)
    except Exception as exc:
        st.error(f"Cannot load screener CSV: {exc}")
        return

    if mode == "resume":
        dl = ScreenerDownloader(storage, jobs_all)
        plan = dl.plan(force_refresh=False)
    elif mode == "missing":
        dl = ScreenerDownloader(storage, jobs_all)
        plan = dl.plan(force_refresh=False)
    elif mode == "retry":
        dl = ScreenerDownloader(storage, jobs_all)
        plan = dl.failed_jobs()
    elif mode == "refresh":
        plan = list(jobs_all)
    elif mode == "selected":
        sel = set(selected or [])
        plan = [j for j in jobs_all if j.ticker in sel]
    else:  # all
        plan = list(jobs_all)

    if not plan:
        st.info("Nothing to download for this mode.")
        return

    st.session_state["fund_running"] = True
    st.session_state["fund_mode"] = mode
    st.session_state["fund_remaining"] = list(plan)
    st.session_state["fund_done"] = 0
    st.session_state["fund_api_calls"] = 0
    st.session_state["fund_start"] = time.time()
    st.session_state["fund_log"] = [f"Started {mode} run with {len(plan)} URLs"]
    st.rerun()


def _process_batch(storage, batch_size: int) -> None:
    jobs_all = _load_screener_jobs(_SCREENER_CSV)
    jobs_map = {j.ticker: j for j in jobs_all}
    remaining = st.session_state.fund_remaining
    batch = remaining[:batch_size]
    # Resolve each remaining job to its full ScreenerJob (url + name).
    batch_jobs = [jobs_map.get(j.ticker, j) if isinstance(j, ScreenerJob) else j for j in batch]
    dl = ScreenerDownloader(
        storage, batch_jobs,
        config={"batch_size": batch_size, "max_concurrency": int(providers_config.get("apify", {}).get("max_concurrency", 3))},
    )
    recs = dl.run_batch(batch_jobs)
    st.session_state.fund_remaining = remaining[len(batch):]
    st.session_state.fund_done += len(batch)
    for r in recs:
        st.session_state.fund_log.append(f"{r['ticker']}: {r['status']}")
    if not st.session_state.fund_remaining:
        st.session_state.fund_running = False
        eng = FundamentalQualityEngine(storage)
        eng.compute(store=True)
        st.session_state.fund_log.append("Download finished; quality features engineered.")
    st.rerun()


def _maybe_process_batch(storage, batch_size: int) -> None:
    if st.session_state.get("fund_running") and st.session_state.get("fund_remaining"):
        _process_batch(storage, batch_size)
        st.rerun()


# --------------------------------------------------------------------------- #
# Market Data Downloader tab                                                   #
# --------------------------------------------------------------------------- #
def _render_market() -> None:
    section("Market Data Downloader")
    st.caption("Ingest and validate NIFTY 500 market data (Version 1 ingestion layer)")

    mdm = get_market_data_manager()
    um = get_universe_manager()
    storage = get_storage()

    # --- Provider selection ---
    section("Select Data Provider")
    provider = st.selectbox("Provider", available_providers(), index=0)

    # --- Universe (Version 1: NIFTY 500 only) ---
    section("Investment Universe")
    universe = um.default_universe()
    st.info(f"**{universe.name.upper()}** · {len(universe)} constituents")
    st.caption(universe.description)

    # --- Backtest window (fixed end date) ---
    section("Backtest Window")
    st.info(
        f"Competition window: {MIN_BACKTEST_DATE.date()} → {MAX_BACKTEST_DATE.date()}. "
        "The end date is fixed; only the start date is selectable."
    )
    start = st.date_input(
        "Backtest start date",
        value=DEFAULT_BACKTEST_START,
        min_value=MIN_BACKTEST_DATE,
        max_value=MAX_BACKTEST_DATE,
    )
    end = MAX_BACKTEST_DATE
    try:
        start, end = validate_backtest_range(start, end)
    except DateRangeError as exc:
        st.error(str(exc))
        st.stop()
    st.caption(f"Effective range: {start.date()} → {end.date()}")

    # --- Download / Refresh ---
    section("Download / Update Data")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Download / Refresh (incremental)", type="primary"):
            with st.spinner("Downloading NIFTY 500 ticker-by-ticker (this can take a while)..."):
                report = mdm.download_universe(
                    universe_name=universe.name, start=start, end=end,
                    full_refresh=False, provider_key=provider,
                )
            _show_report(report)
    with c2:
        if st.button("Complete Refresh (re-download all)"):
            with st.spinner("Performing complete refresh (this can take a while)..."):
                report = mdm.download_universe(
                    universe_name=universe.name, start=start, end=end,
                    full_refresh=True, provider_key=provider,
                )
            _show_report(report)

    # --- Data validation ---
    section("Data Validation")
    if st.button("Run validation"):
        with st.spinner("Validating coverage..."):
            coverage = mdm.coverage_summary(universe)
            missing = mdm.missing_data_report(universe, start, end)
        st.dataframe(coverage, use_container_width=True)
        total_missing = sum(len(v) for v in missing.values())
        if total_missing == 0:
            st.success("No business-day gaps detected in stored range.")
        else:
            st.warning(f"{total_missing} missing business-day rows across {len(missing)} tickers.")
            with st.expander("Show gaps"):
                for t, gaps in missing.items():
                    st.write(f"{t}: {len(gaps)} missing")

    # --- Storage statistics ---
    section("Storage Statistics")
    stats = storage.storage_statistics()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price rows", f"{stats['price_rows']:,}")
    col2.metric("Stored tickers", stats["stored_tickers"])
    col3.metric("Feature rows", f"{stats['feature_rows']:,}")
    col4.metric("DB size (MB)", f"{stats['db_size_bytes'] / 1e6:.2f}")


def _show_report(report) -> None:
    """Render the structured DownloadReport."""
    s = report.summary()
    section("Download Summary")
    cols = st.columns(4)
    cols[0].metric("Total Constituents", s["total_constituents"])
    cols[1].metric("Successfully Downloaded", s["successfully_downloaded"])
    cols[2].metric("Failed Downloads", s["failed_downloads"])
    cols[3].metric("Total Rows Stored", f"{s['total_rows_stored']:,}")
    st.caption(
        f"Provider: {s['provider']} · Full refresh: {s['full_refresh']} · "
        f"Range: {s['start']} → {s['end']} · Duration: {s['duration_seconds']}s"
    )

    if report.failed:
        section("Failed Symbols with Reasons")
        st.dataframe(report.failed_df()[["company_name", "original_symbol", "yahoo_symbol", "error", "retries"]],
                     use_container_width=True)

    tab_s, tab_f = st.tabs(["Successfully Downloaded", "All Records"])
    with tab_s:
        if not report.success_df().empty:
            st.dataframe(report.success_df(), use_container_width=True, height=400)
        else:
            st.info("No securities downloaded in this run.")
    with tab_f:
        st.dataframe(report.all_df(), use_container_width=True, height=400)

    # CSV export
    if st.button("Export download report to CSV"):
        paths = report.export_csv(settings.ingestion.export_abs_dir)
        for label, p in paths.items():
            st.success(f"{label}: {p}")


if __name__ == "__main__":
    render()
