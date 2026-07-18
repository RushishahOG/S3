"""Structured download reports.

Every download produces a :class:`DownloadReport` aggregating per-ticker
:class:`TickerReport` records. Reports are surfaced in the Streamlit UI and can
be exported to CSV. This is the single, reusable contract the ingestion layer
exposes to the presentation layer and to future modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from core.utils.paths import ensure_dir


@dataclass
class TickerReport:
    company_name: str
    original_symbol: str
    yahoo_symbol: str
    status: str  # "success" | "failed" | "no_data"
    rows: int = 0
    earliest: str | None = None
    latest: str | None = None
    error: str | None = None
    retries: int = 0
    anomalies: str = ""


@dataclass
class DownloadReport:
    start: str
    end: str
    full_refresh: bool
    provider: str
    duration_seconds: float
    total_constituents: int
    success: list[TickerReport] = field(default_factory=list)
    failed: list[TickerReport] = field(default_factory=list)

    # -- aggregates --------------------------------------------------------
    @property
    def total_rows_stored(self) -> int:
        return sum(r.rows for r in self.success)

    @property
    def successfully_downloaded(self) -> int:
        return len(self.success)

    @property
    def failed_downloads(self) -> int:
        return len(self.failed)

    def summary(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "full_refresh": self.full_refresh,
            "provider": self.provider,
            "duration_seconds": round(self.duration_seconds, 2),
            "total_constituents": self.total_constituents,
            "successfully_downloaded": self.successfully_downloaded,
            "failed_downloads": self.failed_downloads,
            "total_rows_stored": self.total_rows_stored,
        }

    def failed_symbols_with_reasons(self) -> list[tuple[str, str]]:
        return [(r.original_symbol, r.error or "unknown") for r in self.failed]

    # -- dataframes --------------------------------------------------------
    def _to_df(self, rows: list[TickerReport]) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in rows])

    def success_df(self) -> pd.DataFrame:
        return self._to_df(self.success)

    def failed_df(self) -> pd.DataFrame:
        return self._to_df(self.failed)

    def all_df(self) -> pd.DataFrame:
        return self._to_df(self.success + self.failed)

    # -- export ------------------------------------------------------------
    def export_csv(self, directory: str, prefix: str = "download_report") -> dict[str, str]:
        """Write summary, success and failed CSVs. Returns paths written."""
        ensure_dir(directory)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        paths: dict[str, str] = {}
        summary_path = f"{directory}/{prefix}_summary_{ts}.csv"
        pd.DataFrame([self.summary()]).to_csv(summary_path, index=False)
        paths["summary"] = summary_path
        if self.success:
            p = f"{directory}/{prefix}_success_{ts}.csv"
            self.success_df().to_csv(p, index=False)
            paths["success"] = p
        if self.failed:
            p = f"{directory}/{prefix}_failed_{ts}.csv"
            self.failed_df().to_csv(p, index=False)
            paths["failed"] = p
        return paths
