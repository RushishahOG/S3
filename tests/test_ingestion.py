"""Network-free smoke test of the Market Data Ingestion layer.

Uses a fake in-memory provider so we can exercise the full pipeline
(constituents -> resolver -> downloader -> validation -> storage -> report)
without hitting Yahoo Finance.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd

from core.data.ingestion.constituents import load_constituents
from core.data.ingestion.downloader import HistoricalDownloader
from core.data.ingestion.ticker_resolver import TickerResolver
from core.data.providers.base_provider import BaseDataProvider, PriceColumns
from core.data.storage.storage_manager import StorageManager
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE

CONSTITUENTS_PATH = "nifty_500_constituents/ind_nifty500list_2026.csv"


class FakeProvider(BaseDataProvider):
    name = "fake"

    def fetch_prices(self, tickers, start, end):
        dates = pd.bdate_range(start, end)
        rows = []
        rng = np.random.default_rng(0)
        for t in tickers:
            p = 100.0
            for d in dates:
                p *= 1 + rng.normal(0.0004, 0.02)
                rows.append((t, pd.Timestamp(d).date(), p, p, p, p, p, 1_000_000))
        return pd.DataFrame(rows, columns=PriceColumns.LONG_COLUMNS)


def test_ingestion_pipeline():
    consts = load_constituents(CONSTITUENTS_PATH)
    assert len(consts) == 500

    resolver = TickerResolver()
    assert resolver.resolve("RELIANCE") == "RELIANCE.NS"

    db = tempfile.mktemp(suffix=".duckdb")
    sm = StorageManager(db)
    downloader = HistoricalDownloader(sm, resolver=resolver, provider_key="fake")

    # Patch get_provider so the downloader uses our fake.
    import core.data.ingestion.downloader as dl

    orig = dl.get_provider
    dl.get_provider = lambda key: FakeProvider()

    try:
        report = downloader.download(consts[:5], start=MIN_BACKTEST_DATE, end=MAX_BACKTEST_DATE)
    finally:
        dl.get_provider = orig

    assert report.total_constituents == 5
    assert report.successfully_downloaded == 5
    assert report.failed_downloads == 0
    assert report.total_rows_stored > 0

    # Metadata persisted.
    meta = sm.get_download_metadata()
    assert len(meta) == 5
    assert set(meta["status"]) == {"success"}

    # Prices stored for resolved symbols.
    first_symbol = resolver.resolve(consts[0].base_symbol)
    prices = sm.get_prices([first_symbol])
    assert not prices.empty

    # Full refresh path runs and marks metadata.
    dl.get_provider = lambda key: FakeProvider()
    try:
        report2 = downloader.download(consts[:5], start=MIN_BACKTEST_DATE, end=MAX_BACKTEST_DATE, full_refresh=True)
    finally:
        dl.get_provider = orig
    assert report2.full_refresh is True
    assert report2.successfully_downloaded == 5

    sm.close()
    os.remove(db)
    print("INGESTION SMOKE TEST PASSED")


if __name__ == "__main__":
    test_ingestion_pipeline()
