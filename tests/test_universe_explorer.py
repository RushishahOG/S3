"""Tests for the NIFTY 500 Universe Explorer analytics.

Uses synthetic :class:`ConstituentMembership` records so the analytics can be
exercised without a live DuckDB connection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.universe_explorer import UniverseExplorer
from core.universe_explorer.membership import ConstituentMembership, ConstituentStatus
from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE


def _m(ticker, company, entry, exit=None, sector="Banks", status=None):
    return ConstituentMembership(
        ticker=ticker,
        company_name=company,
        sector=sector,
        entry_date=pd.Timestamp(entry),
        exit_date=pd.Timestamp(exit) if exit else None,
        status=status or ("Removed" if exit else "Active"),
        entry_is_proxy=exit is None,
    )


@pytest.fixture
def memberships():
    return [
        _m("AAA.NS", "Alpha", "2005-06-01"),  # present throughout
        _m("BBB.NS", "Beta", "2010-01-01"),   # enters mid-window
        _m("CCC.NS", "Gamma", "2004-01-01", "2015-06-01"),  # removed mid-window
        _m("DDD.NS", "Delta", "2020-03-01", sector="IT"),
    ]


@pytest.fixture
def explorer(memberships):
    return UniverseExplorer(memberships)


def test_periods_cover_window(explorer):
    periods = explorer.periods()
    assert periods[0][1] == pd.Timestamp("2006-01-01")
    assert periods[-1][0].startswith(str(MAX_BACKTEST_DATE.year))


def test_constituents_at_date(explorer):
    # 2007-01-01: Alpha (2005), Gamma (2004); Beta/Ccc not yet; DDD not yet.
    present = {m.ticker for m in explorer.constituents_at("2007-01-01")}
    assert present == {"AAA.NS", "CCC.NS"}
    # 2012-01-01: Alpha, Gamma (exit 2015), Beta (2010). DDD not yet.
    present = {m.ticker for m in explorer.constituents_at("2012-01-01")}
    assert present == {"AAA.NS", "CCC.NS", "BBB.NS"}


def test_constituents_in_period_status(explorer):
    # Period 2006-2007: Alpha present throughout, Gamma present throughout.
    members = dict(
        (m.ticker, s) for m, s in explorer.constituents_in_period("2006-01-01", "2007-12-31")
    )
    assert "AAA.NS" in members and "CCC.NS" in members
    assert members["AAA.NS"] == "Present Throughout"
    assert "DDD.NS" not in members


def test_longest_continuous(explorer):
    df = explorer.longest_continuous_df()
    assert df.iloc[0]["Ticker"] == "AAA.NS"  # longest tenure first
    throughout = df[df["Present Throughout"] == "Yes"]["Ticker"].tolist()
    assert throughout == ["AAA.NS"]


def test_timeline_present_throughout(explorer):
    tl = explorer.timeline_df()
    row = tl[tl["Ticker"] == "AAA.NS"].iloc[0]
    assert row["Present Throughout"] == "Yes"
    assert row["Exit Label"] == "Present"
    row2 = tl[tl["Ticker"] == "CCC.NS"].iloc[0]
    assert row2["Exit Label"] == pd.Timestamp("2015-06-01").date()


def test_year_summary(explorer):
    ys = explorer.year_summary_df()
    # 2006: Alpha + Gamma present -> total 2.
    row2006 = ys[ys["Year"] == 2006].iloc[0]
    assert row2006["Total Constituents"] == 2
    # 2015: Gamma removed this year -> Removals 1.
    row2015 = ys[ys["Year"] == 2015].iloc[0]
    assert row2015["Removals"] == 1  # Gamma
    # 2016: Gamma already gone -> Alpha + Beta present = 2, no removals.
    row2016 = ys[ys["Year"] == 2016].iloc[0]
    assert row2016["Total Constituents"] == 2
    assert row2016["Removals"] == 0


def test_sector_distribution(explorer):
    sec = explorer.sector_distribution(2012)
    total = sec["Constituents"].sum()
    assert total == 3  # Alpha, Beta, Gamma


def test_universe_at_date(explorer):
    snap = explorer.universe_at_date("2021-01-01")
    assert snap["universe_size"] == 3  # Alpha, Beta, DDD (Gamma exited 2015)
    assert snap["universe_size"] == len(snap["historical_constituents"])
    # Newly added within trailing 12 months (2020-01-01..2021-01-01): DDD (2020-03).
    assert "DDD.NS" in [m.ticker for m in snap["newly_added"]]
    # Removed by 2021: Gamma.
    assert "CCC.NS" in [m.ticker for m in snap["removed"]]


def test_min_max_dates_used(explorer):
    assert explorer.min_date == pd.Timestamp(MIN_BACKTEST_DATE)
    assert explorer.max_date == pd.Timestamp(MAX_BACKTEST_DATE)
