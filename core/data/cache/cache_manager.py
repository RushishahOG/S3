"""Cache / incremental-update manager.

Implements the "download only what is missing" policy so the platform never
re-pulls an entire history unnecessarily:

  * For tickers with no stored data -> fetch the full requested range.
  * For tickers with stored data -> fetch only from (last stored date + 1) to
    the requested end date.
  * :meth:`detect_missing_dates` reports business-day gaps inside the stored
    range so the UI can surface data-quality issues.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.data.storage.storage_manager import StorageManager
from core.utils.dates import date_range_business
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DownloadPlan:
    """Resolved per-ticker fetch ranges for an incremental update."""

    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    full_fetch: list[str]
    incremental: list[str]

    @property
    def needs_download(self) -> bool:
        return bool(self.ranges)


class CacheManager:
    def __init__(self, storage: StorageManager) -> None:
        self.storage = storage

    def compute_incremental_ranges(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> DownloadPlan:
        """Determine which (ticker, date-range) pairs must be fetched."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        latest = self.storage.latest_date_per_ticker(tickers)

        ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        full_fetch: list[str] = []
        incremental: list[str] = []

        for ticker in tickers:
            last = latest.get(ticker)
            if last is None:
                ranges[ticker] = (start, end)
                full_fetch.append(ticker)
            elif last < end:
                # Resume one business day after the last stored observation.
                nxt = last + pd.offsets.BusinessDay(1)
                if nxt <= end:
                    ranges[ticker] = (nxt, end)
                    incremental.append(ticker)
            # else: fully up to date, nothing to fetch.

        logger.info(
            "Incremental plan: %d full, %d incremental, %d up-to-date",
            len(full_fetch),
            len(incremental),
            len(tickers) - len(full_fetch) - len(incremental),
        )
        return DownloadPlan(ranges=ranges, full_fetch=full_fetch, incremental=incremental)

    def detect_missing_dates(
        self,
        tickers: list[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> dict[str, list[pd.Timestamp]]:
        """Return business-day dates that should exist but are not stored."""
        start = pd.Timestamp(start) if start else None
        end = pd.Timestamp(end) if end else None
        latest = self.storage.latest_date_per_ticker(tickers)
        earliest = self.storage.earliest_date_per_ticker(tickers)

        missing: dict[str, list[pd.Timestamp]] = {}
        for ticker in tickers:
            first = earliest.get(ticker)
            last = latest.get(ticker)
            if first is None or last is None:
                continue
            span_start = first if start is None else max(first, start)
            span_end = last if end is None else min(last, end)
            expected = date_range_business(span_start, span_end)
            stored = set(
                pd.to_datetime(
                    self.storage.get_prices([ticker], span_start, span_end)["date"]
                ).dt.date
            )
            gaps = [d for d in expected if d.date() not in stored]
            if gaps:
                missing[ticker] = gaps
        return missing
