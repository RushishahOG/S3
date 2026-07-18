"""Yahoo Finance data provider.

This is the only concrete provider shipped in V1. It translates Yahoo's
multi-index frame into the platform's canonical long format. The rest of the
platform never imports ``yfinance`` directly.

Adjusted Close is treated as the **primary** price series for all
return-based calculations (Momentum, cumulative returns, backtests). Yahoo only
emits the ``Adj Close`` column when ``auto_adjust=False``; when ``auto_adjust``
is enabled the column is folded into ``Close`` and lost. This provider therefore
always requests the un-adjusted frame and preserves ``Adj Close`` explicitly.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from core.config.settings import settings
from core.data.providers.base_provider import BaseDataProvider, PriceColumns
from core.utils.decorators import retry
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

# Standard Yahoo Finance OHLCV schema we expect to receive. ``Adj Close`` is
# mandatory for this platform; its absence is a data-quality warning, not silent.
YAHOO_EXPECTED_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")

try:  # Import is isolated so the provider can be discovered even offline.
    import yfinance as yf
except Exception as exc:  # pragma: no cover - depends on environment
    yf = None
    logger.warning("yfinance unavailable: %s", exc)


class YahooFinanceProvider(BaseDataProvider):
    name = "yahoo_finance"

    def __init__(
        self,
        auto_adjust: bool = False,
        adj_close_fallback_to_close: bool | None = None,
        progress: bool = False,
    ) -> None:
        # ``auto_adjust`` MUST stay False: enabling it makes Yahoo fold the
        # ``Adj Close`` column into ``Close`` and drop it entirely, which would
        # remove the series momentum/returns depend on.
        self.auto_adjust = auto_adjust
        if adj_close_fallback_to_close is None:
            adj_close_fallback_to_close = settings.providers.adj_close_fallback
        self.adj_close_fallback_to_close = adj_close_fallback_to_close
        self.progress = progress

    def is_available(self) -> bool:
        return yf is not None

    @retry(max_attempts=3, backoff_seconds=2.0)
    def fetch_prices(
        self,
        tickers: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is not installed; cannot fetch data.")

        tickers = sorted(set(tickers))
        logger.info(
            "Fetching %d tickers from %s to %s via Yahoo Finance",
            len(tickers),
            start.date(),
            end.date(),
        )

        raw = yf.download(
            tickers,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=self.auto_adjust,
            progress=self.progress,
            group_by="column",
            threads=True,
        )

        long_df = self._normalise(raw, tickers)
        logger.info("Yahoo Finance returned %d rows", len(long_df))
        return long_df

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    def _normalise(self, raw: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
        """Convert Yahoo's frame into the canonical long format.

        Guarantees that the ``Adj Close`` -> ``adj_close`` mapping is applied,
        validates the returned schema against :data:`YAHOO_EXPECTED_FIELDS`, and
        applies the configurable fallback when ``Adj Close`` is absent.
        """
        if raw is None or raw.empty:
            logger.warning("Yahoo Finance returned an empty frame for %s", tickers)
            return pd.DataFrame(columns=PriceColumns.LONG_COLUMNS)

        # yfinance returns a MultiIndex (field, ticker) for multiple tickers.
        if isinstance(raw.columns, pd.MultiIndex):
            frames = []
            for ticker in tickers:
                if ticker not in raw.columns.get_level_values(1):
                    continue
                sub = raw.xs(ticker, axis=1, level=1).copy()
                sub[PriceColumns.TICKER] = ticker
                frames.append(sub)
            if not frames:
                logger.warning("No columns matched for tickers %s", tickers)
                return pd.DataFrame(columns=PriceColumns.LONG_COLUMNS)
            df = pd.concat(frames)
        else:
            # Single ticker: flat columns.
            df = raw.copy()
            df[PriceColumns.TICKER] = tickers[0] if tickers else "UNKNOWN"

        df = df.reset_index()
        # yfinance index may be named 'Date' or 'index'.
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: PriceColumns.DATE})

        # Validate the raw Yahoo schema (capitalised) before renaming.
        self._validate_schema(df, tickers)

        # yfinance emits capitalised OHLCV field names; map them to the
        # platform's lowercase canonical names before filtering.
        field_map = {
            "Open": PriceColumns.OPEN,
            "High": PriceColumns.HIGH,
            "Low": PriceColumns.LOW,
            "Close": PriceColumns.CLOSE,
            "Adj Close": PriceColumns.ADJ_CLOSE,
            "Volume": PriceColumns.VOLUME,
        }
        df = df.rename(columns={k: v for k, v in field_map.items() if k in df.columns})

        self._ensure_adj_close(df, tickers)

        keep = [c for c in PriceColumns.LONG_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df[PriceColumns.DATE] = pd.to_datetime(df[PriceColumns.DATE])
        for col in PriceColumns.FIELDS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values([PriceColumns.TICKER, PriceColumns.DATE]).reset_index(
            drop=True
        )

    def _validate_schema(self, df: pd.DataFrame, tickers: Sequence[str]) -> None:
        """Warn if the raw Yahoo frame is missing any expected OHLCV field."""
        present = [f for f in YAHOO_EXPECTED_FIELDS if f in df.columns]
        missing = [f for f in YAHOO_EXPECTED_FIELDS if f not in df.columns]
        if missing:
            logger.warning(
                "Yahoo Finance schema mismatch for %s - missing fields: %s "
                "(present: %s). This may indicate an unsupported instrument.",
                list(tickers), missing, present,
            )

    def _ensure_adj_close(self, df: pd.DataFrame, tickers: Sequence[str]) -> None:
        """Guarantee an ``adj_close`` series, applying the configured fallback.

        If Yahoo omits ``Adj Close`` (some indices/ETFs/cash instruments), we
        either fall back to the raw ``close`` (configurable) or leave it NULL,
        logging the reason either way so the gap is never silent.
        """
        if PriceColumns.ADJ_CLOSE in df.columns:
            return

        # No Adj Close column returned at all.
        if self.adj_close_fallback_to_close and PriceColumns.CLOSE in df.columns:
            logger.warning(
                "Adj Close missing for %s - falling back to Close per config.",
                list(tickers),
            )
            df[PriceColumns.ADJ_CLOSE] = df[PriceColumns.CLOSE]
        else:
            logger.warning(
                "Adj Close missing for %s - storing NULL (fallback disabled).",
                list(tickers),
            )
            df[PriceColumns.ADJ_CLOSE] = pd.NA
