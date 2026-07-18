"""Abstract data provider contract.

The factor engine and every downstream module depend ONLY on this interface,
never on ``yfinance`` or any concrete vendor SDK. Adding a new provider (Alpha
Vantage, Polygon, NSE, ...) means implementing this class and registering it in
``registry.py``; no other code needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from core.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PriceRecord:
    """Canonical shape of a single price observation row."""

    ticker: str
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


class BaseDataProvider(ABC):
    """Vendor-agnostic market data source."""

    #: Unique registry key, e.g. ``"yahoo_finance"``.
    name: str = "base"

    @abstractmethod
    def fetch_prices(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """Return price data for ``tickers`` in ``[start, end]``.

        The returned frame MUST have a ``DatetimeIndex`` (dates) and columns
        matching :data:`PriceColumns.FIELDS` (open, high, low, close,
        adj_close, volume) with one column group per ticker, OR a long frame
        with a ``ticker`` column. The storage layer normalises either shape.
        """

    def is_available(self) -> bool:
        """Hook for providers that require credentials or network checks."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"


class PriceColumns:
    """Column names used everywhere downstream (storage, features, factors)."""

    TICKER = "ticker"
    DATE = "date"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    ADJ_CLOSE = "adj_close"
    VOLUME = "volume"

    FIELDS = [OPEN, HIGH, LOW, CLOSE, ADJ_CLOSE, VOLUME]
    LONG_COLUMNS = [TICKER, DATE, *FIELDS]


class BaseFundamentalProvider(ABC):
    """Vendor-agnostic fundamental / corporate-action data source.

    Mirrors :class:`BaseDataProvider` but is dedicated to *fundamental* data
    (financial statements, ratios, dividends, ...). The ingestion layer and the
    feature store depend only on this interface, never on a concrete vendor SDK.
    Adding a new fundamental vendor means implementing this class and registering
    it in ``registry.py`` - nothing else changes.
    """

    #: Unique registry key, e.g. ``"apify_financial"``.
    name: str = "base_fundamental"

    @abstractmethod
    def fetch(self, symbol: str):
        """Fetch the fundamental payload for a single ``symbol``.

        The return type is provider-specific (a dataclass / dict). The caller
        (batch downloader) is responsible for normalising it into the storage
        schema.
        """

    def fetch_many(self, symbols: list[str]) -> dict[str, object]:
        """Fetch fundamentals for many symbols; never raises per symbol.

        Returns a mapping ``symbol -> result`` (result may be ``None`` on
        failure). Failures are logged, not propagated, so the pipeline can
        continue. Concrete providers may override for batch efficiency.
        """
        out: dict[str, object] = {}
        for sym in symbols:
            try:
                out[sym] = self.fetch(sym)
            except Exception as exc:  # noqa: BLE001 - isolate per symbol
                logger.warning("fetch_many failed for %s: %s", sym, exc)
                out[sym] = None
        return out

    def is_available(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"
