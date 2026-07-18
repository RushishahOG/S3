"""Eligibility Analyzer.

Determines, for every stock in the (NIFTY 500) universe and for *every*
rebalance date across the competition window (01-01-2006 → 31-05-2026), whether
the stock is eligible given actual data availability and the lookback required
by the selected factors.

Eligibility is evaluated **independently for each rebalance date** (never from a
single pre-computed "first eligible date"). For every date the analyzer reports:

  * Universe Size (fixed)
  * Stocks With Data (covering the date)
  * Eligible Stocks
  * Coverage % (eligible / universe)
  * Excluded (Missing Data)
  * Excluded (Insufficient Lookback)

The warm-up requirement is driven entirely by the factor registry
(:mod:`core.eligibility.registry`): the analyzer uses the *maximum* lookback
among the selected factors, so adding a new factor (Quality, Value, Growth, ...)
never requires touching this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core.utils.dates import MAX_BACKTEST_DATE, MIN_BACKTEST_DATE

# Rebalance cadences. Add new entries here to extend supported frequencies.
# Keys are user-facing labels; values are pandas offset aliases.
REBALANCE_FREQUENCIES = {
    "Daily": "D",
    "Weekly": "W",
    "Fortnightly": "2W",
    "Monthly": "ME",
    "Quarterly": "QE",
    "Semi-Annual": "2QE",
    "Annual": "YE",
}


@dataclass
class EligibilityResult:
    """Container for one eligibility analysis run."""

    timeline: pd.DataFrame
    per_stock: pd.DataFrame
    recommended_start: pd.Timestamp | None
    required_lookback_months: int
    threshold: float
    universe_size: int
    data_universe_size: int
    rebalance_freq: str
    global_last_trading: pd.Timestamp

    def eligible_at(self, date) -> list[str]:
        """Return the tickers eligible at a specific rebalance ``date``."""
        date = pd.Timestamp(date)
        eff = min(date, self.global_last_trading)
        sub = self.per_stock
        mask = (
            sub["has_data"]
            & (sub["latest_date"] >= eff)
            & (
                sub["first_trading_date"]
                <= (eff - pd.DateOffset(months=self.required_lookback_months))
            )
        )
        return sub.loc[mask, "ticker"].tolist()

    def summary(self) -> dict:
        """Headline numbers for the UI metric cards."""
        out: dict = {
            "recommended_start": self.recommended_start,
            "required_lookback_months": self.required_lookback_months,
            "universe_size": self.universe_size,
            "data_universe_size": self.data_universe_size,
            "threshold": self.threshold,
        }
        tl = self.timeline
        if self.recommended_start is not None and not tl.empty:
            idx = (tl["date"] - self.recommended_start).abs().argmin()
            row = tl.iloc[idx]
            out["eligible_at_start"] = int(row["eligible_count"])
            out["coverage_at_start"] = float(row["coverage_pct"])
            out["with_data_at_start"] = int(row["stocks_with_data"])
        else:
            out["eligible_at_start"] = 0
            out["coverage_at_start"] = 0.0
            out["with_data_at_start"] = 0
        if not tl.empty:
            last = tl.iloc[-1]
            out["eligible_at_end"] = int(last["eligible_count"])
            out["coverage_at_end"] = float(last["coverage_pct"])
            out["with_data_at_end"] = int(last["stocks_with_data"])
        else:
            out["eligible_at_end"] = 0
            out["coverage_at_end"] = 0.0
            out["with_data_at_end"] = 0
        return out


class EligibilityAnalyzer:
    """Compute per-stock and timeline eligibility from data availability.

    Parameters
    ----------
    universe_tickers:
        Ordered list of tickers defining the universe (e.g. NIFTY 500).
    earliest_dates:
        Mapping ``ticker -> first trading date`` (from storage). Tickers absent
        from the mapping are treated as having no data.
    latest_dates:
        Mapping ``ticker -> last available date`` (from storage).
    min_date, max_date:
        Bounds of the analysis window (defaults to the competition window).
    """

    def __init__(
        self,
        universe_tickers: Iterable[str],
        earliest_dates: Mapping[str, object] | None = None,
        latest_dates: Mapping[str, object] | None = None,
        min_date=MIN_BACKTEST_DATE,
        max_date=MAX_BACKTEST_DATE,
    ) -> None:
        self.universe_tickers = list(universe_tickers)
        self.earliest = {
            t: pd.Timestamp(v) for t, v in (earliest_dates or {}).items() if v is not None
        }
        self.latest = {
            t: pd.Timestamp(v) for t, v in (latest_dates or {}).items() if v is not None
        }
        self.min_date = pd.Timestamp(min_date)
        self.max_date = pd.Timestamp(max_date)
        # Latest trading day present anywhere in the data; used to clamp
        # rebalance dates that fall after the last available trading day so we
        # never treat every stock as unavailable at the tail of the window.
        self._global_last = (
            max(self.latest.values()) if self.latest else self.max_date
        )

    def analyze(
        self,
        required_lookback_months: int = 12,
        threshold: float = 0.8,
        rebalance_freq: str = "ME",
    ) -> EligibilityResult:
        """Run the analysis over the full competition window.

        Parameters
        ----------
        required_lookback_months:
            Warm-up in months. A stock needs at least this much history *before*
            the rebalance date to be eligible.
        threshold:
            Minimum universe coverage (fraction, e.g. ``0.8`` for 80%) that the
            recommended start date must achieve.
        rebalance_freq:
            Pandas offset alias for candidate rebalance dates (e.g. ``"ME"``).
        """
        lookback = int(required_lookback_months)
        tickers = self.universe_tickers
        n_stocks = len(tickers)

        # --- Per-stock base arrays -----------------------------------------
        has_data = np.array([t in self.earliest for t in tickers])
        first_t = pd.to_datetime(
            [self.earliest.get(t) for t in tickers], errors="coerce"
        ).values
        latest_t = pd.to_datetime(
            [self.latest.get(t) for t in tickers], errors="coerce"
        ).values

        # --- Rebalance dates across the FULL competition window ------------
        dates = pd.date_range(self.min_date, self.max_date, freq=rebalance_freq)
        n_dates = len(dates)

        # Effective evaluation date per rebalance: clamp to the last trading day
        # so dates beyond the data's end still evaluate against available data.
        eff = pd.Series(dates).clip(upper=self._global_last)
        req_start = (eff - pd.DateOffset(months=lookback)).values  # datetime64[N]

        # Broadcast to (n_dates, n_stocks) for independent per-date evaluation.
        has_v = has_data[None, :]
        latest_v = latest_t[None, :]
        first_v = first_t[None, :]
        eff_v = eff.values[:, None]
        req_v = req_start[:, None]

        # Covering = has data AND that data extends to/through the eval date.
        covering = has_v & (latest_v >= eff_v)
        # Sufficient lookback = first trading is at or before the window start.
        sufficient = first_v <= req_v
        eligible = covering & sufficient

        eligible_count = eligible.sum(axis=1).astype(int)
        covering_count = covering.sum(axis=1).astype(int)
        insufficient_count = (covering_count - eligible_count).astype(int)
        missing_count = (n_stocks - covering_count).astype(int)

        universe_size = n_stocks
        denom = universe_size if universe_size > 0 else 1
        coverage = eligible_count / denom * 100.0

        timeline = pd.DataFrame(
            {
                "date": dates,
                "universe_size": universe_size,
                "stocks_with_data": covering_count,
                "eligible_count": eligible_count,
                "coverage_pct": coverage,
                "excluded_missing": missing_count,
                "excluded_insufficient": insufficient_count,
            }
        )

        # --- Per-stock reference (earliest possible eligibility) -----------
        recs = []
        for t in tickers:
            ed = self.earliest.get(t)
            ld = self.latest.get(t)
            has = ed is not None
            first_trading = ed if has else pd.NaT
            latest = ld if has else pd.NaT
            first_eligible = (
                first_trading + pd.DateOffset(months=lookback) if has else pd.NaT
            )
            recs.append(
                {
                    "ticker": t,
                    "has_data": has,
                    "first_trading_date": first_trading,
                    "first_eligible_date": first_eligible,
                    "latest_date": latest,
                }
            )
        per_stock = pd.DataFrame(recs)

        # --- Recommendation: FIRST date meeting the threshold --------------
        thr_pct = float(threshold) * 100.0
        above = coverage >= thr_pct
        recommended_start = dates[above.argmax()] if above.any() else None

        return EligibilityResult(
            timeline=timeline,
            per_stock=per_stock,
            recommended_start=recommended_start,
            required_lookback_months=lookback,
            threshold=float(threshold),
            universe_size=universe_size,
            data_universe_size=int(has_data.sum()),
            rebalance_freq=rebalance_freq,
            global_last_trading=pd.Timestamp(self._global_last),
        )
