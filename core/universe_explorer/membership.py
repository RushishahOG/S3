"""Data model for NIFTY 500 (and future) universe membership.

A :class:`ConstituentMembership` captures everything the explorer needs about a
single stock's tenure in an index: identity, classification, entry/exit dates
and status. Historical membership data is frequently unavailable (as it is for
the current NIFTY 500 snapshot), so every field degrades gracefully and the
provenance of the ``entry_date`` is tracked via ``entry_is_proxy``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd


class ConstituentStatus(str, Enum):
    """Base membership status sourced from (future) historical data."""

    ACTIVE = "Active"
    REMOVED = "Removed"
    DELISTED = "Delisted"
    MERGED_RENAMED = "Merged/Renamed"


# Statuses computed *relative to a selected period* (section 1 of the spec).
CONTEXTUAL_ACTIVE = "Active"
CONTEXTUAL_ADDED = "Added during period"
CONTEXTUAL_PRESENT_THROUGHOUT = "Present Throughout"
CONTEXTUAL_REMOVED = "Removed"
CONTEXTUAL_DELISTED = "Delisted"
CONTEXTUAL_MERGED = "Merged/Renamed"


@dataclass
class ConstituentMembership:
    """One stock's membership record in an index."""

    ticker: str
    company_name: str
    symbol: str = ""
    sector: Optional[str] = None
    industry: Optional[str] = None
    entry_date: Optional[pd.Timestamp] = None
    exit_date: Optional[pd.Timestamp] = None
    status: str = ConstituentStatus.ACTIVE.value
    entry_is_proxy: bool = False


def is_present(m: ConstituentMembership, date: pd.Timestamp) -> bool:
    """True if ``m`` is a constituent at ``date`` (inclusive of entry)."""
    if m.entry_date is None:
        return False
    date = pd.Timestamp(date)
    if m.entry_date > date:
        return False
    if m.exit_date is not None and m.exit_date <= date:
        return False
    return True


def years_in_index(m: ConstituentMembership, as_of: pd.Timestamp) -> float:
    """Years between entry and exit (or ``as_of`` if still active)."""
    if m.entry_date is None:
        return 0.0
    end = m.exit_date if m.exit_date is not None else pd.Timestamp(as_of)
    days = (end - m.entry_date).days
    return max(days, 0) / 365.25


def contextual_status(
    m: ConstituentMembership, start: pd.Timestamp, end: pd.Timestamp
) -> str:
    """Status of ``m`` relative to a selected period ``[start, end]``."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    entered_before_start = (
        m.entry_date is not None and m.entry_date <= start
    )
    exited_after_end = (
        m.exit_date is None or m.exit_date >= end
    )
    if entered_before_start and exited_after_end:
        return CONTEXTUAL_PRESENT_THROUGHOUT

    entered_in_period = (
        m.entry_date is not None and start < m.entry_date <= end
    )
    exited_in_period = (
        m.exit_date is not None and start < m.exit_date <= end
    )

    if exited_in_period and m.status == ConstituentStatus.DELISTED.value:
        return CONTEXTUAL_DELISTED
    if exited_in_period and m.status == ConstituentStatus.MERGED_RENAMED.value:
        return CONTEXTUAL_MERGED
    if entered_in_period:
        return CONTEXTUAL_ADDED
    if exited_in_period:
        return CONTEXTUAL_REMOVED
    return CONTEXTUAL_ACTIVE
