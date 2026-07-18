"""Historical downloader (Market Data Ingestion layer).

Downloads NIFTY 500 constituents **ticker-by-ticker** via the configured data
provider. It is intentionally modularised from the old monolithic download:

    Load constituents -> Resolve symbols -> Download each individually
    -> Validate -> Store -> Log failures -> Generate report

Reliability features: configurable retries, small randomised delay between
ticks (rate-limit friendly), continue-on-failure, structured logging, and
incremental updates (only fetch missing history unless a full refresh is
requested).
"""

from __future__ import annotations

import random
import time
from typing import Iterable

import pandas as pd

from core.config.settings import settings
from core.data.ingestion.constituents import Constituent
from core.data.ingestion.reports import DownloadReport, TickerReport
from core.data.ingestion.ticker_resolver import TickerResolver
from core.data.ingestion.validation import validate_ohlcv
from core.data.providers.base_provider import PriceColumns
from core.data.providers.registry import get_provider
from core.data.storage.storage_manager import StorageManager
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE, clamp_to_bounds
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

BUSINESS_DAY = pd.offsets.BusinessDay(1)
# Fields persisted to the download_metadata table per ticker.
META_FIELDS = (
    "ticker",
    "company_name",
    "provider",
    "status",
    "rows",
    "earliest_date",
    "latest_date",
    "error",
    "retries",
)


class HistoricalDownloader:
    def __init__(
        self,
        storage: StorageManager,
        resolver: TickerResolver | None = None,
        provider_key: str | None = None,
    ) -> None:
        self.storage = storage
        self.resolver = resolver or TickerResolver()
        self.provider_key = provider_key or settings.providers.default
        self.retries = settings.ingestion.retries
        self.backoff = settings.ingestion.retry_backoff_seconds
        self.min_delay = settings.ingestion.min_delay_seconds
        self.max_delay = settings.ingestion.max_delay_seconds

    # -- public API ---------------------------------------------------------
    def download(
        self,
        constituents: list[Constituent],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        full_refresh: bool = False,
        provider_key: str | None = None,
    ) -> DownloadReport:
        start = clamp_to_bounds(start) if start else MIN_BACKTEST_DATE
        end = clamp_to_bounds(end) if end else MAX_BACKTEST_DATE
        start = max(start, MIN_BACKTEST_DATE)
        end = min(end, MAX_BACKTEST_DATE)
        if start > end:  # defensive
            start = end

        key = provider_key or self.provider_key
        provider = get_provider(key)

        resolved = [(c, self.resolver.resolve(c.base_symbol)) for c in constituents]
        symbols = [s for _, s in resolved]

        if full_refresh and symbols:
            logger.info("Full refresh requested: clearing stored data for %d tickers", len(symbols))
            self.storage.delete_ticker_data(symbols, include_features=True)

        # Incremental planning (skip if full refresh).
        latest = {} if full_refresh else self.storage.latest_date_per_ticker(symbols)

        t0 = time.time()
        success: list[TickerReport] = []
        failed: list[TickerReport] = []
        metas: list[dict] = []
        all_anomalies: list[dict] = []

        for constituent, symbol in resolved:
            report, meta, anomalies = self._download_one(
                provider, constituent, symbol, start, end, latest.get(symbol), key
            )
            if report.status == "success":
                success.append(report)
            else:
                failed.append(report)
            metas.append(meta)
            all_anomalies.extend(anomalies)
            # Small randomised delay to avoid hammering the provider.
            time.sleep(random.uniform(self.min_delay, self.max_delay))

        self.storage.upsert_download_metadata(metas)
        self.storage.record_anomalies(all_anomalies)

        duration = time.time() - t0
        report = DownloadReport(
            start=str(start.date()),
            end=str(end.date()),
            full_refresh=full_refresh,
            provider=key,
            duration_seconds=duration,
            total_constituents=len(constituents),
            success=success,
            failed=failed,
        )
        logger.info(
            "Download complete: %d/%d succeeded, %d failed, %d rows, %.1fs",
            report.successfully_downloaded,
            report.total_constituents,
            report.failed_downloads,
            report.total_rows_stored,
            duration,
        )
        return report

    # -- internals ----------------------------------------------------------
    def _download_one(self, provider, constituent, symbol, start, end, stored_latest, provider_key):
        company = constituent.company_name
        base = constituent.base_symbol

        # Already up to date (incremental): report as success from storage.
        if stored_latest is not None and stored_latest >= end:
            rows = int(self.storage.get_prices([symbol]).shape[0])
            earliest = self.storage.earliest_date_per_ticker([symbol]).get(symbol)
            meta = self._meta(symbol, company, provider_key, "success", rows, earliest, stored_latest, None, 0)
            report = TickerReport(
                company_name=company,
                original_symbol=base,
                yahoo_symbol=symbol,
                status="success",
                rows=rows,
                earliest=str(earliest.date()) if earliest else None,
                latest=str(stored_latest.date()),
                retries=0,
            )
            return report, meta, []

        fetch_start = start if stored_latest is None else (stored_latest + BUSINESS_DAY)
        fetch_end = end
        if fetch_start > fetch_end:
            rows = int(self.storage.get_prices([symbol]).shape[0])
            meta = self._meta(symbol, company, provider_key, "success", rows, stored_latest, stored_latest, None, 0)
            report = TickerReport(
                company_name=company,
                original_symbol=base,
                yahoo_symbol=symbol,
                status="success",
                rows=rows,
                earliest=str(stored_latest.date()) if stored_latest else None,
                latest=str(stored_latest.date()) if stored_latest else None,
                retries=0,
            )
            return report, meta, []

        # Retry loop with backoff.
        attempts = 0
        last_error: str | None = None
        raw = None
        for attempt in range(1, self.retries + 1):
            attempts = attempt
            try:
                raw = provider.fetch_prices([symbol], fetch_start, fetch_end)
                break
            except Exception as exc:  # noqa: BLE001 - isolation per ticker
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Retry %d/%d for %s: %s", attempt, self.retries, symbol, last_error)
                time.sleep(self.backoff * attempt)

        if raw is None or raw.empty:
            status = "no_data" if last_error is None else "failed"
            meta = self._meta(symbol, company, provider_key, status, 0, None, None, last_error, attempts)
            report = TickerReport(
                company_name=company,
                original_symbol=base,
                yahoo_symbol=symbol,
                status=status,
                rows=0,
                error=last_error or "Provider returned no rows",
                retries=attempts,
            )
            return report, meta, []

        # Validate + store.
        result = validate_ohlcv(raw)
        anomalies = [
            {"ticker": a.get("ticker", symbol), "issue": a["issue"], "detail": a["detail"]}
            for a in result.anomalies
        ]
        if result.clean.empty:
            meta = self._meta(symbol, company, provider_key, "failed", 0, None, None, "Validation produced no usable rows", attempts)
            report = TickerReport(
                company_name=company,
                original_symbol=base,
                yahoo_symbol=symbol,
                status="failed",
                rows=0,
                error="Validation produced no usable rows",
                retries=attempts,
                anomalies="; ".join(a["issue"] for a in anomalies),
            )
            return report, meta, anomalies

        written = self.storage.upsert_prices(result.clean)
        earliest = result.clean[PriceColumns.DATE].min()
        latest_date = result.clean[PriceColumns.DATE].max()
        meta = self._meta(
            symbol, company, provider_key, "success", written,
            earliest, latest_date, None, attempts,
        )
        report = TickerReport(
            company_name=company,
            original_symbol=base,
            yahoo_symbol=symbol,
            status="success",
            rows=written,
            earliest=str(pd.Timestamp(earliest).date()),
            latest=str(pd.Timestamp(latest_date).date()),
            retries=attempts,
            anomalies="; ".join(a["issue"] for a in anomalies),
        )
        return report, meta, anomalies

    @staticmethod
    def _meta(ticker, company, provider, status, rows, earliest, latest, error, retries) -> dict:
        return {
            "ticker": ticker,
            "company_name": company,
            "provider": provider,
            "status": status,
            "rows": rows,
            "earliest_date": pd.Timestamp(earliest).date() if earliest is not None else None,
            "latest_date": pd.Timestamp(latest).date() if latest is not None else None,
            "error": error,
            "retries": retries,
        }
