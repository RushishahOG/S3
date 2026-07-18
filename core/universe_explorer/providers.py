"""Membership providers.

A provider turns whatever source data exists into a list of
:class:`ConstituentMembership` records. This is the single extension point for
new universes and for ingesting historical membership files.

Currently:
  * :class:`CurrentSnapshotProvider` — reads the official *current* NIFTY 500
    constituent file. Historical entry/exit dates are unavailable, so every
    member is marked ``Active`` and the entry date is proxied by the first
    available price (or the competition start) and flagged via
    ``entry_is_proxy``.
  * :class:`CsvMembershipProvider` — ready for future historical membership
    files (CSV/Excel) carrying real entry/exit dates and statuses; once such a
    file is supplied, all analytics and visualisations populate automatically
    with no other code changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from core.data.ingestion.ticker_resolver import TickerResolver
from core.data.universe.universe_manager import UniverseManager
from core.universe_explorer.membership import (
    ConstituentMembership,
    ConstituentStatus,
)
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE


class MembershipProvider(ABC):
    """Turns a source into membership records."""

    @abstractmethod
    def get_memberships(self) -> list[ConstituentMembership]:
        ...


class CurrentSnapshotProvider(MembershipProvider):
    """Build memberships from the current NIFTY 500 constituent snapshot.

    Reads the constituent CSV directly (to capture the ``Industry`` field the
    generic loader drops) and resolves symbols to Yahoo tickers. Entry dates are
    proxied from stored price history when available.
    """

    def __init__(
        self,
        universe=None,
        constituents_path: Optional[str] = None,
        storage=None,
        min_date=MIN_BACKTEST_DATE,
        max_date=MAX_BACKTEST_DATE,
    ) -> None:
        if universe is None:
            universe = UniverseManager().default_universe()
        self.universe = universe
        self.constituents_path = constituents_path
        self.storage = storage
        self.min_date = pd.Timestamp(min_date)
        self.max_date = pd.Timestamp(max_date)

    def get_memberships(self) -> list[ConstituentMembership]:
        if self.constituents_path is None:
            raise ValueError(
                "CurrentSnapshotProvider requires `constituents_path`. Obtain it "
                "from the universe provider, e.g. "
                "`get_universe_manager().get_provider(name).constituents_path`."
            )
        path = self.constituents_path
        df = pd.read_csv(path)
        col = {str(c).strip().lower(): c for c in df.columns}
        name_col = col.get("company name") or df.columns[0]
        sym_col = col.get("symbol") or df.columns[1]
        ind_col = col.get("industry")

        resolver = TickerResolver()
        tickers = self.universe.tickers
        earliest_map = {}
        if self.storage is not None:
            try:
                earliest_map = self.storage.earliest_date_per_ticker(tickers)
            except Exception:
                earliest_map = {}

        memberships: list[ConstituentMembership] = []
        for _, row in df.iterrows():
            symbol = str(row[sym_col]).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            ticker = resolver.resolve(symbol)
            company = str(row[name_col]).strip()
            industry = (
                str(row[ind_col]).strip()
                if ind_col and ind_col in row and pd.notna(row[ind_col])
                else None
            )
            ed = earliest_map.get(ticker)
            if ed is not None:
                entry = pd.Timestamp(ed)
                proxy = True
            else:
                entry = self.min_date
                proxy = False
            memberships.append(
                ConstituentMembership(
                    ticker=ticker,
                    symbol=symbol,
                    company_name=company,
                    sector=industry,
                    industry=industry,
                    entry_date=entry,
                    exit_date=None,
                    status=ConstituentStatus.ACTIVE.value,
                    entry_is_proxy=proxy,
                )
            )
        return memberships


class CsvMembershipProvider(MembershipProvider):
    """Ingest a historical membership file (CSV/Excel).

    Expected columns (tolerant of alternate names / capitalisation):
      ``ticker`` / ``symbol``, ``company`` / ``company name``,
      ``sector`` (optional), ``industry`` (optional),
      ``entry_date`` (required), ``exit_date`` (optional; blank = still active),
      ``status`` (optional; Active/Removed/Delisted/Merged/Renamed).

    Supplying such a file automatically populates every analysis and
    visualisation in the Universe Explorer.
    """

    STATUS_MAP = {
        "active": ConstituentStatus.ACTIVE.value,
        "removed": ConstituentStatus.REMOVED.value,
        "delisted": ConstituentStatus.DELISTED.value,
        "merged": ConstituentStatus.MERGED_RENAMED.value,
        "renamed": ConstituentStatus.MERGED_RENAMED.value,
        "merged/renamed": ConstituentStatus.MERGED_RENAMED.value,
    }

    def __init__(self, path: str) -> None:
        self.path = path

    def get_memberships(self) -> list[ConstituentMembership]:
        ext = str(self.path).lower().rsplit(".", 1)[-1]
        df = (
            pd.read_excel(self.path)
            if ext in ("xlsx", "xls")
            else pd.read_csv(self.path)
        )
        col = {str(c).strip().lower(): c for c in df.columns}
        sym_col = col.get("ticker") or col.get("symbol") or df.columns[0]
        name_col = (
            col.get("company name")
            or col.get("company")
            or df.columns[1]
        )
        sector_col = col.get("sector")
        ind_col = col.get("industry")
        entry_col = col.get("entry_date") or col.get("entry")
        exit_col = col.get("exit_date") or col.get("exit")

        memberships: list[ConstituentMembership] = []
        for _, row in df.iterrows():
            symbol = str(row[sym_col]).strip()
            if not symbol or symbol.lower() == "nan":
                continue
            raw_status = (
                str(row[col["status"]]).strip().lower()
                if "status" in col and pd.notna(row[col["status"]])
                else "active"
            )
            status = self.STATUS_MAP.get(raw_status, ConstituentStatus.ACTIVE.value)
            entry = pd.to_datetime(row[entry_col], errors="coerce")
            exit_v = (
                pd.to_datetime(row[exit_col], errors="coerce")
                if exit_col and exit_col in col and pd.notna(row[exit_col])
                else None
            )
            memberships.append(
                ConstituentMembership(
                    ticker=symbol,
                    symbol=symbol,
                    company_name=str(row[name_col]).strip()
                    if name_col in row
                    else symbol,
                    sector=str(row[sector_col]).strip()
                    if sector_col and sector_col in col and pd.notna(row[sector_col])
                    else None,
                    industry=str(row[ind_col]).strip()
                    if ind_col and ind_col in col and pd.notna(row[ind_col])
                    else None,
                    entry_date=entry if pd.notna(entry) else None,
                    exit_date=exit_v,
                    status=status,
                    entry_is_proxy=False,
                )
            )
        return memberships
