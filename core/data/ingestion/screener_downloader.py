"""Screener batch downloader (single source of truth for fundamentals).

Iterates over screener.in company URLs (from the uploaded CSV / stored
universe) and pulls the complete financial history for each via
:class:`~core.data.providers.apify_screener_provider.ApifyScreenerProvider`.
The full payload is normalised into the seven ``fundamentals_*`` tables.

Design mirrors :mod:`core.data.ingestion.fundamental_downloader`:
    Plan (resume-aware) -> Batch of N -> Fetch (concurrent) -> Store -> Log

There is no extraction window: the screener actor returns the entire history.

Key properties:
  * Batched by ``batch_size``; store-then-advance.
  * Resume: URLs already in ``fundamentals_company`` are skipped unless forced.
  * Retry: failed URLs are recorded and re-run via ``retry_failed``.
  * Continue-on-failure: a bad URL is logged and skipped.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from core.config.providers_config import get_provider_config
from core.data.providers.apify_screener_provider import ApifyScreenerProvider
from core.data.providers.base_provider import BaseFundamentalProvider
from core.data.storage.storage_manager import StorageManager
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ScreenerJob:
    """A single screener URL together with its display metadata."""

    url: str
    ticker: str
    company_name: str = ""


class ScreenerDownloader:
    def __init__(
        self,
        storage: StorageManager,
        jobs: list[ScreenerJob] | None = None,
        provider: BaseFundamentalProvider | None = None,
        config: dict | None = None,
    ) -> None:
        self.storage = storage
        self.jobs = list(jobs or [])
        self.cfg = config or get_provider_config("apify")
        self.batch_size = int(self.cfg.get("batch_size", 25))
        self.max_concurrency = int(self.cfg.get("max_concurrency", 3))
        self.provider = provider or ApifyScreenerProvider()
        self.api_calls = 0

    # -- planning -----------------------------------------------------------
    def plan(self, force_refresh: bool = False) -> list[ScreenerJob]:
        if force_refresh:
            return list(self.jobs)
        done = self.storage.tickers_with_screener_data()
        return [j for j in self.jobs if j.ticker not in done]

    def failed_jobs(self) -> list[ScreenerJob]:
        meta = self.storage.get_fundamental_download_metadata()
        if meta.empty:
            return []
        failed = set(meta[meta["status"].isin(["failed", "no_data", "partial"])]["ticker"])
        return [j for j in self.jobs if j.ticker in failed]

    # -- per-URL fetch ------------------------------------------------------
    def _fetch_one(self, job: ScreenerJob):
        try:
            result = self.provider.fetch(job.url)
            self.api_calls += 1
            return result, None
        except Exception as exc:  # noqa: BLE001 - isolate per URL
            return None, f"{type(exc).__name__}: {exc}"

    # -- batch run ----------------------------------------------------------
    def run_batch(self, jobs: list[ScreenerJob]) -> list[dict]:
        results: dict[str, tuple] = {}
        with ThreadPoolExecutor(max_workers=max(1, self.max_concurrency)) as ex:
            futures = {ex.submit(self._fetch_one, j): j for j in jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    results[j.ticker] = fut.result()
                except Exception as exc:  # noqa: BLE001
                    results[j.ticker] = (None, f"fetch: {exc}")

        records = []
        for job in jobs:
            res, error = results.get(job.ticker, (None, "no result"))
            records.append(self._store_one(job, res, error))
        return records

    # -- storage of one URL -------------------------------------------------
    def _store_one(self, job: ScreenerJob, result, error) -> dict:
        status = "success"
        try:
            if result is None:
                status = "failed"
            else:
                if result.company:
                    company = dict(result.company)
                    company["url"] = job.url
                    self.storage.upsert_fundamentals_company(pd.DataFrame([company]))
                if result.income_annual:
                    self.storage.upsert_fundamentals_income_annual(
                        pd.DataFrame(result.income_annual)
                    )
                if result.income_quarterly:
                    self.storage.upsert_fundamentals_income_quarterly(
                        pd.DataFrame(result.income_quarterly)
                    )
                if result.balance_sheet:
                    self.storage.upsert_fundamentals_balance_sheet(
                        pd.DataFrame(result.balance_sheet)
                    )
                if result.cashflow:
                    self.storage.upsert_fundamentals_cashflow(pd.DataFrame(result.cashflow))
                if result.dividends:
                    self.storage.upsert_fundamentals_dividends(pd.DataFrame(result.dividends))
                if result.ratios:
                    yearly = result.ratios.pop("_yearly", [])
                    if yearly:
                        yearly_df = pd.DataFrame(yearly)
                        key = ["ticker", "financial_year"]
                        val_cols = [c for c in yearly_df.columns if c not in key]
                        yearly_df = yearly_df[yearly_df[val_cols].notna().any(axis=1)] if val_cols else yearly_df.iloc[0:0]
                        if not yearly_df.empty:
                            self.storage.upsert_fundamentals_ratios(yearly_df)
                    else:
                        snap = dict(result.ratios)
                        snap["financial_year"] = 0
                        sn = {k: v for k, v in snap.items() if k != "ticker" and pd.notna(v)}
                        if sn:
                            self.storage.upsert_fundamentals_ratios(pd.DataFrame([snap]))
                if not (result.company or result.income_annual or result.ratios):
                    status = "no_data"
        except Exception as exc:  # noqa: BLE001
            error = f"store: {exc}"
            status = "failed"
            logger.error("Failed to store screener data for %s: %s", job.ticker, exc)

        record = {
            "ticker": job.ticker,
            "company_name": job.company_name or job.ticker,
            "status": status,
            "financials_status": "success" if result else "failed",
            "ratios_status": "success" if (result and result.ratios) else "failed",
            "error": error,
            "retries": 0,
        }
        self.storage.upsert_fundamental_download_metadata([record])
        logger.info("Screener completed %s -> %s", job.ticker, status)
        return record

    # -- top-level ----------------------------------------------------------
    def download(
        self,
        force_refresh: bool = False,
        jobs: Iterable[ScreenerJob] | None = None,
    ) -> list[dict]:
        targets = list(jobs) if jobs is not None else self.plan(force_refresh)
        all_records: list[dict] = []
        t0 = time.time()
        for i in range(0, len(targets), self.batch_size):
            batch = targets[i : i + self.batch_size]
            logger.info(
                "Screener batch %d/%d (%d URLs)",
                i // self.batch_size + 1,
                (len(targets) + self.batch_size - 1) // self.batch_size,
                len(batch),
            )
            all_records.extend(self.run_batch(batch))
        logger.info(
            "Screener download finished: %d URLs in %.1fs", len(targets), time.time() - t0
        )
        return all_records

    def retry_failed(self) -> list[dict]:
        failed = self.failed_jobs()
        if not failed:
            return []
        return self.download(jobs=failed)
