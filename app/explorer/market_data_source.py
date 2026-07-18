"""Market OHLCV dataset source for the Dataset Explorer.

Reads exclusively from the local analytical store (DuckDB) and the incremental
cache manager - it never triggers a network/API call. This is the Version 1
implementation of the :class:`~app.explorer.base.DatasetSource` contract; future
dataset families (financial statements, ETFs, benchmarks, macro) follow the
same shape and register alongside it.
"""

from __future__ import annotations

import os

import pandas as pd

from app.explorer.base import DATASET_SOURCES, DatasetSource, HealthIssue, Severity
from core.config.settings import settings
from core.data.cache.cache_manager import CacheManager
from core.data.providers.base_provider import PriceColumns
from core.data.storage.storage_manager import StorageManager
from core.utils.dates import MAX_BACKTEST_DATE, date_range_business


class MarketDataDatasetSource(DatasetSource):
    key = "market_data"
    label = "Market Data (OHLCV)"
    description = "Downloaded historical OHLCV prices for the investment universe."

    #: Status values considered a "complete" download reaching the end date.
    COMPLETE_STATUS = "success"

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()
        self.cache = CacheManager(self.storage)

    # -- catalogue --------------------------------------------------------
    def security_summary(self) -> pd.DataFrame:
        meta = self.storage.get_download_metadata()
        meta = meta.copy() if not meta.empty else pd.DataFrame(
            columns=["ticker", "company_name", "status", "rows",
                     "earliest_date", "latest_date", "downloaded_at"]
        )
        meta_tickers = set(meta["ticker"])

        # Include any ticker stored in prices but missing from download metadata.
        for t in self.storage.stored_tickers():
            if t not in meta_tickers:
                meta = pd.concat(
                    [meta, pd.DataFrame([{
                        "ticker": t, "company_name": None, "status": "unknown",
                        "rows": int(self.storage.get_prices([t]).shape[0]),
                        "earliest_date": None, "latest_date": None,
                        "downloaded_at": None,
                    }])],
                    ignore_index=True,
                )

        if meta.empty:
            return pd.DataFrame(columns=[
                "ticker", "company_name", "records", "earliest",
                "latest", "availability_pct", "last_updated", "status",
            ])

        rows = []
        for _, r in meta.iterrows():
            earliest = r.get("earliest_date")
            latest = r.get("latest_date")
            n = int(r.get("rows") or 0)
            availability = self._availability(earliest, latest, n)
            rows.append({
                "ticker": r["ticker"],
                "company_name": r.get("company_name") or r["ticker"],
                "records": n,
                "earliest": pd.Timestamp(earliest).date() if earliest else None,
                "latest": pd.Timestamp(latest).date() if latest else None,
                "availability_pct": round(availability, 1),
                "last_updated": pd.Timestamp(r["downloaded_at"]).to_pydatetime()
                if r.get("downloaded_at") else None,
                "status": r.get("status") or "unknown",
            })
        out = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
        return out

    # -- inspection -------------------------------------------------------
    def fetch_dataset(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_prices([ticker])
        if df.empty:
            return df
        df = df.sort_values(PriceColumns.DATE).reset_index(drop=True)
        df[PriceColumns.DATE] = pd.to_datetime(df[PriceColumns.DATE])
        return df

    def display_columns(self) -> list[str]:
        return [
            PriceColumns.DATE,
            PriceColumns.OPEN,
            PriceColumns.HIGH,
            PriceColumns.LOW,
            PriceColumns.CLOSE,
            PriceColumns.ADJ_CLOSE,
            PriceColumns.VOLUME,
        ]

    def dataset_statistics(self, df: pd.DataFrame, ticker: str) -> dict:
        if df.empty:
            return {
                "ticker": ticker,
                "total_trading_days": 0,
                "first_trading_date": None,
                "last_trading_date": None,
                "min_price": None,
                "max_price": None,
                "avg_volume": None,
                "missing_values": 0,
                "duplicate_rows": 0,
            }
        num_cols = [PriceColumns.OPEN, PriceColumns.HIGH, PriceColumns.LOW,
                    PriceColumns.CLOSE, PriceColumns.ADJ_CLOSE]
        missing = int(df[num_cols + [PriceColumns.VOLUME]].isna().sum().sum())
        dups = int(df.duplicated(subset=[PriceColumns.DATE]).sum())
        return {
            "ticker": ticker,
            "total_trading_days": int(df.shape[0]),
            "first_trading_date": pd.Timestamp(df[PriceColumns.DATE].min()).date(),
            "last_trading_date": pd.Timestamp(df[PriceColumns.DATE].max()).date(),
            "min_price": round(float(df[num_cols].min().min()), 2),
            "max_price": round(float(df[num_cols].max().max()), 2),
            "avg_volume": int(df[PriceColumns.VOLUME].fillna(0).mean()),
            "missing_values": missing,
            "duplicate_rows": dups,
        }

    # -- storage / health -------------------------------------------------
    def storage_statistics(self) -> dict:
        stats = self.storage.storage_statistics()
        last_dl = self.storage.last_download_time()
        cache_bytes = _dir_size(settings.storage.parquet_abs_dir)
        return {
            "total_securities": stats["stored_tickers"],
            "total_rows": stats["price_rows"],
            "db_size_bytes": stats["db_size_bytes"],
            "cache_size_bytes": cache_bytes,
            "last_download_time": last_dl,
        }

    def health_issues(self, tickers: list[str]) -> list[HealthIssue]:
        issues: list[HealthIssue] = []
        if not tickers:
            return issues

        meta = self.storage.get_download_metadata(tickers)
        quality = self.storage.price_quality_report()
        if not quality.empty:
            quality = quality.set_index("ticker")

        # Failed / no-data downloads requiring attention.
        for _, r in meta.iterrows():
            status = (r.get("status") or "unknown")
            if status in ("failed", "no_data"):
                issues.append(HealthIssue(
                    ticker=r["ticker"],
                    issue="Failed download",
                    severity=Severity.ERROR,
                    detail=str(r.get("error") or "No data returned by provider"),
                ))
            elif status == "unknown":
                issues.append(HealthIssue(
                    ticker=r["ticker"],
                    issue="Missing metadata",
                    severity=Severity.WARNING,
                    detail="Stored prices but no download metadata record.",
                ))

        # Per-ticker quality + completeness + missing-date scan.
        for ticker in tickers:
            q = quality.loc[ticker] if ticker in quality.index else None
            if q is not None:
                dups = int(q.get("duplicate_rows", 0) or 0)
                invalid = int(q.get("invalid_prices", 0) or 0)
                if dups:
                    issues.append(HealthIssue(
                        ticker=ticker, issue="Duplicate records",
                        severity=Severity.WARNING,
                        detail=f"{dups} duplicate (ticker, date) rows.",
                    ))
                if invalid:
                    issues.append(HealthIssue(
                        ticker=ticker, issue="Invalid prices",
                        severity=Severity.ERROR,
                        detail=f"{invalid} rows with non-positive or inconsistent OHLC.",
                    ))

            latest = self.storage.latest_date_per_ticker([ticker]).get(ticker)
            if latest is not None and latest < MAX_BACKTEST_DATE:
                issues.append(HealthIssue(
                    ticker=ticker, issue="Incomplete dataset",
                    severity=Severity.WARNING,
                    detail=f"Latest stored date {latest.date()} precedes "
                           f"competition end {MAX_BACKTEST_DATE.date()}.",
                ))

            missing = self.cache.detect_missing_dates([ticker])
            gaps = missing.get(ticker, [])
            if gaps:
                issues.append(HealthIssue(
                    ticker=ticker, issue="Missing trading dates",
                    severity=Severity.WARNING,
                    detail=f"{len(gaps)} missing business-day(s) within stored span.",
                ))
        return issues

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _availability(earliest, latest, rows: int) -> float:
        if not earliest or not latest or rows <= 0:
            return 0.0
        expected = len(date_range_business(pd.Timestamp(earliest), pd.Timestamp(latest)))
        if expected <= 0:
            return 100.0
        return min(100.0, rows / expected * 100.0)


def _dir_size(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                continue
    return total


# Register the Version 1 source on import (lazy: no connection opened yet).
MarketDataDatasetSource.register(MarketDataDatasetSource)
