"""Universe Explorer analytics.

Consumes a list of :class:`ConstituentMembership` records (from any provider)
and produces every analysis required by the UI: period explorers, longest
continuous members, the membership Gantt timeline data, and annual summaries.
It is completely decoupled from Streamlit and from any specific data source, so
future universes plug in by supplying their own provider.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from core.universe_explorer.membership import (
    ConstituentMembership,
    contextual_status,
    is_present,
    years_in_index,
)
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE


class UniverseExplorer:
    def __init__(
        self,
        memberships: list[ConstituentMembership],
        min_date=MIN_BACKTEST_DATE,
        max_date=MAX_BACKTEST_DATE,
    ) -> None:
        self.memberships = memberships
        self.min_date = pd.Timestamp(min_date)
        self.max_date = pd.Timestamp(max_date)

    # ------------------------------------------------------------------ #
    # Periods
    # ------------------------------------------------------------------ #
    def periods(self) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
        """Two-year windows covering the competition period, e.g. 2006-2007."""
        out = []
        y = self.min_date.year
        last = self.max_date.year
        while y <= last:
            start = pd.Timestamp(f"{y}-01-01")
            end = min(pd.Timestamp(f"{y + 1}-12-31"), self.max_date)
            out.append((f"{y}-{y + 1}", start, end))
            y += 1
        return out

    # ------------------------------------------------------------------ #
    # Membership queries
    # ------------------------------------------------------------------ #
    def constituents_at(self, date) -> list[ConstituentMembership]:
        date = pd.Timestamp(date)
        return [m for m in self.memberships if is_present(m, date)]

    def constituents_in_period(self, start, end) -> list[tuple[ConstituentMembership, str]]:
        """Members present at any point in ``[start, end]`` with period status."""
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        out = []
        for m in self.memberships:
            entered = m.entry_date is not None and m.entry_date <= end
            not_exited = m.exit_date is None or m.exit_date > start
            if entered and not_exited:
                out.append((m, contextual_status(m, start, end)))
        return out

    # ------------------------------------------------------------------ #
    # Longest continuous
    # ------------------------------------------------------------------ #
    def longest_continuous_df(self) -> pd.DataFrame:
        rows = []
        for m in self.memberships:
            yrs = years_in_index(m, self.max_date)
            present_throughout = (
                m.entry_date is not None
                and m.entry_date <= self.min_date
                and (m.exit_date is None or m.exit_date >= self.max_date)
            )
            rows.append(
                {
                    "Company Name": m.company_name,
                    "Ticker": m.ticker,
                    "Entry Date": m.entry_date.date() if m.entry_date else None,
                    "Exit Date": m.exit_date.date() if m.exit_date else "Present",
                    "Years in Index": round(yrs, 2),
                    "Sector": m.sector or "N/A",
                    "Present Throughout": "Yes" if present_throughout else "No",
                }
            )
        df = pd.DataFrame(rows).sort_values("Years in Index", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Timeline (Gantt) data
    # ------------------------------------------------------------------ #
    def timeline_df(self) -> pd.DataFrame:
        rows = []
        for m in self.memberships:
            exit_eff = m.exit_date if m.exit_date is not None else self.max_date
            yrs = years_in_index(m, self.max_date)
            present_throughout = (
                m.entry_date is not None
                and m.entry_date <= self.min_date
                and (m.exit_date is None or m.exit_date >= self.max_date)
            )
            rows.append(
                {
                    "Ticker": m.ticker,
                    "Company": m.company_name,
                    "Sector": m.sector or "N/A",
                    "Entry": m.entry_date,
                    "Exit": exit_eff,
                    "Exit Label": m.exit_date.date() if m.exit_date else "Present",
                    "Years in Index": round(yrs, 2),
                    "Status": m.status,
                    "Entry Is Proxy": m.entry_is_proxy,
                    "Present Throughout": "Yes" if present_throughout else "No",
                }
            )
        df = pd.DataFrame(rows).sort_values("Entry").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Annual summary
    # ------------------------------------------------------------------ #
    def year_summary_df(self) -> pd.DataFrame:
        rows = []
        prev_total = 0
        for y in range(self.min_date.year, self.max_date.year + 1):
            y_start = pd.Timestamp(f"{y}-01-01")
            y_end = min(pd.Timestamp(f"{y}-12-31"), self.max_date)
            total = 0
            new_additions = 0
            removals = 0
            delistings = 0
            corp_actions = 0
            for m in self.memberships:
                present = (
                    m.entry_date is not None
                    and m.entry_date <= y_end
                    and (m.exit_date is None or m.exit_date > y_start)
                )
                if present:
                    total += 1
                if m.entry_date is not None and y_start < m.entry_date <= y_end:
                    new_additions += 1
                if m.exit_date is not None and y_start < m.exit_date <= y_end:
                    removals += 1
                    if m.status == "Delisted":
                        delistings += 1
                    elif m.status == "Merged/Renamed":
                        corp_actions += 1
            net_change = total - prev_total
            rows.append(
                {
                    "Year": y,
                    "Total Constituents": total,
                    "New Additions": new_additions,
                    "Removals": removals,
                    "Delistings": delistings,
                    "Corporate Actions": corp_actions,
                    "Net Change": net_change,
                }
            )
            prev_total = total
        return pd.DataFrame(rows)

    def sector_distribution(self, year: int) -> pd.DataFrame:
        y_start = pd.Timestamp(f"{year}-01-01")
        y_end = min(pd.Timestamp(f"{year}-12-31"), self.max_date)
        counts: dict[str, int] = {}
        for m in self.memberships:
            present = (
                m.entry_date is not None
                and m.entry_date <= y_end
                and (m.exit_date is None or m.exit_date > y_start)
            )
            if present:
                key = m.sector or "N/A"
                counts[key] = counts.get(key, 0) + 1
        df = pd.DataFrame(
            [{"Sector": k, "Constituents": v} for k, v in counts.items()]
        ).sort_values("Constituents", ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------ #
    # Integration helper (section 6)
    # ------------------------------------------------------------------ #
    def universe_at_date(self, date) -> dict:
        """Breakdown of the universe at a specific backtest start date."""
        date = pd.Timestamp(date)
        present = self.constituents_at(date)
        # Newly added = entered within the trailing 12 months before `date`.
        cutoff = date - pd.DateOffset(months=12)
        newly_added = [
            m for m in present
            if m.entry_date is not None and cutoff < m.entry_date <= date
        ]
        # Removed by this date = exited on/before `date`.
        removed = [
            m for m in self.memberships
            if m.exit_date is not None and m.exit_date <= date
        ]
        return {
            "date": date,
            "universe_size": len(present),
            "historical_constituents": present,
            "newly_added": newly_added,
            "removed": removed,
        }
