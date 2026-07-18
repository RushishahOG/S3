"""Market Data Manager (Application Service).

Top-level orchestrator for the Version 1 ingestion workflow. It wires the
:class:`HistoricalDownloader` (ticker-by-ticker, reliable, incremental) to the
configured data provider and local storage, and exposes clean retrieval /
validation helpers consumed by the factor engine and the UI.

The downloader never performs a single bulk pull; it downloads each security
individually, validates, stores, logs failures and produces a structured
:class:`DownloadReport`. New factors / portfolios / backtests simply call
``download_universe`` and read clean panels from storage.
"""

from __future__ import annotations

import pandas as pd

from core.config import settings
from core.data.cache.cache_manager import CacheManager
from core.data.ingestion.downloader import HistoricalDownloader
from core.data.ingestion.reports import DownloadReport
from core.data.ingestion.ticker_resolver import TickerResolver
from core.data.providers.base_provider import PriceColumns
from core.data.storage.storage_manager import StorageManager
from core.data.universe.universe_manager import Universe, UniverseManager
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE, clamp_to_bounds
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


class MarketDataManager:
    def __init__(
        self,
        storage: StorageManager | None = None,
        universe_manager: UniverseManager | None = None,
        provider_key: str | None = None,
    ) -> None:
        self.storage = storage or StorageManager()
        self.universe_manager = universe_manager or UniverseManager()
        self.provider_key = provider_key or settings.providers.default
        self.cache = CacheManager(self.storage)
        self.resolver = TickerResolver()

    # -- ingestion ----------------------------------------------------------
    def download_universe(
        self,
        universe_name: str | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        full_refresh: bool = False,
        provider_key: str | None = None,
    ) -> DownloadReport:
        """Download (incrementally or fully) a universe's historical data.

        Returns a structured :class:`DownloadReport` with per-ticker results.
        Failed tickers never abort the run.
        """
        universe_name = universe_name or settings.universe.default
        universe = self.universe_manager.get_universe(universe_name)
        constituents = self.universe_manager.get_constituents(universe_name)
        downloader = HistoricalDownloader(
            self.storage, resolver=self.resolver, provider_key=provider_key or self.provider_key
        )
        return downloader.download(
            constituents,
            start=start,
            end=end,
            full_refresh=full_refresh,
            provider_key=provider_key or self.provider_key,
        )

    def update_universe(
        self,
        universe_name: str | None = None,
        end: pd.Timestamp | None = None,
        provider_key: str | None = None,
    ) -> DownloadReport:
        """Incremental refresh: fetch only missing history up to ``end``."""
        return self.download_universe(
            universe_name=universe_name,
            start=MIN_BACKTEST_DATE,
            end=end or MAX_BACKTEST_DATE,
            full_refresh=False,
            provider_key=provider_key,
        )

    # -- retrieval ----------------------------------------------------------
    def get_price_panel(
        self,
        tickers: Iterable[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self.storage.get_adjusted_price_panel(tickers=tickers, start=start, end=end)

    def get_ohlcv(
        self,
        tickers: Iterable[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        return self.storage.get_prices(tickers=tickers, start=start, end=end)

    def missing_data_report(
        self, universe: Universe, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
    ) -> dict[str, list[pd.Timestamp]]:
        return self.cache.detect_missing_dates(list(universe), start=start, end=end)

    def coverage_summary(self, universe: Universe) -> pd.DataFrame:
        """Per-ticker coverage: first/last date and row count."""
        latest = self.storage.latest_date_per_ticker(list(universe))
        earliest = self.storage.earliest_date_per_ticker(list(universe))
        rows = []
        for t in universe:
            rows.append(
                {
                    "ticker": t,
                    "first_date": earliest.get(t),
                    "last_date": latest.get(t),
                    "rows": int(self.storage.get_prices([t]).shape[0]) if t in latest else 0,
                }
            )
        return pd.DataFrame(rows)
