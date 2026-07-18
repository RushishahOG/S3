"""Unit tests for the Eligibility Analyzer (no DB / Streamlit needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from core.eligibility import EligibilityAnalyzer, REBALANCE_FREQUENCIES, list_factors, max_lookback_for
from core.eligibility.registry import register_default_factors

register_default_factors()


def _build(size=200, seed=7, max_date="2026-05-31"):
    np.random.seed(seed)
    tickers = [f"T{i}" for i in range(size)]
    base = pd.Timestamp("2005-01-01")
    earliest = {
        t: base + pd.Timedelta(days=int(np.random.randint(0, 4000))) for t in tickers
    }
    latest = {t: pd.Timestamp(max_date) for t in tickers}
    return tickers, earliest, latest


class TestEligibilityAnalyzer(unittest.TestCase):
    def test_registry_has_v1_factors(self):
        names = set(list_factors().keys())
        self.assertTrue({"Momentum", "Beta", "Low Volatility"} <= names)

    def test_max_lookback_combines_selected_factors(self):
        self.assertEqual(max_lookback_for(["Beta", "Momentum"]), 12)
        self.assertEqual(max_lookback_for(["Beta", "Low Volatility"]), 12)
        self.assertEqual(max_lookback_for([]), 0)

    def test_rebalance_frequencies_extensible(self):
        # All required cadences are present and map to pandas aliases.
        for label in (
            "Daily",
            "Weekly",
            "Fortnightly",
            "Monthly",
            "Quarterly",
            "Semi-Annual",
            "Annual",
        ):
            self.assertIn(label, REBALANCE_FREQUENCIES)

    def test_timeline_spans_full_competition_window(self):
        tickers, earliest, latest = _build()
        a = EligibilityAnalyzer(tickers, earliest, latest)
        res = a.analyze(12, 0.8, "ME")
        self.assertEqual(res.timeline["date"].iloc[0].date(), pd.Timestamp("2006-01-31").date())
        self.assertEqual(res.timeline["date"].iloc[-1].date(), pd.Timestamp("2026-05-31").date())
        # Every month present (no truncation).
        self.assertGreater(len(res.timeline), 200)

    def test_breakdown_sums_to_universe_every_row(self):
        tickers, earliest, latest = _build()
        a = EligibilityAnalyzer(tickers, earliest, latest)
        res = a.analyze(12, 0.8, "ME")
        tl = res.timeline
        accounted = (
            tl["eligible_count"]
            + tl["excluded_insufficient"]
            + tl["excluded_missing"]
        )
        self.assertTrue((accounted == tl["universe_size"]).all())

    def test_recommended_start_reaches_threshold(self):
        tickers, earliest, latest = _build()
        a = EligibilityAnalyzer(tickers, earliest, latest)
        res = a.analyze(required_lookback_months=12, threshold=0.8, rebalance_freq="ME")
        self.assertIsNotNone(res.recommended_start)
        s = res.summary()
        self.assertGreaterEqual(s["coverage_at_start"], 80.0)
        self.assertEqual(
            len(res.eligible_at(res.recommended_start)), s["eligible_at_start"]
        )

    def test_higher_threshold_starts_later(self):
        tickers, earliest, latest = _build()
        a = EligibilityAnalyzer(tickers, earliest, latest)
        r80 = a.analyze(12, 0.8, "ME").recommended_start
        r95 = a.analyze(12, 0.95, "ME").recommended_start
        self.assertIsNotNone(r80)
        self.assertIsNotNone(r95)
        self.assertGreaterEqual(r95, r80)

    def test_no_data_yields_no_recommendation(self):
        tickers = [f"T{i}" for i in range(10)]
        a = EligibilityAnalyzer(tickers, {}, {})
        res = a.analyze(required_lookback_months=12, threshold=0.8, rebalance_freq="ME")
        self.assertIsNone(res.recommended_start)
        self.assertEqual(res.data_universe_size, 0)
        self.assertEqual(res.timeline["coverage_pct"].iloc[-1], 0.0)

    def test_daily_timeline_does_not_drop_to_zero_at_end(self):
        # Last available trading day (29-May-2026) is before the window end, so
        # the final rebalance dates must clamp and keep eligibility (not 0).
        tickers, earliest, latest = _build(max_date="2026-05-29")
        a = EligibilityAnalyzer(tickers, earliest, latest)
        res = a.analyze(12, 0.8, "D")
        self.assertGreater(len(res.timeline), 7000)
        self.assertGreater(res.timeline["coverage_pct"].iloc[-1], 0.0)

    def test_2006_stock_becomes_eligible_after_lookback(self):
        # A stock trading continuously from 2006 must be eligible once it has
        # the required lookback of history.
        ticker = "REL2006"
        a = EligibilityAnalyzer(
            [ticker],
            {ticker: pd.Timestamp("2006-01-02")},
            {ticker: pd.Timestamp("2026-05-29")},
        )
        res = a.analyze(12, 0.0, "ME")
        # Before 12 months of history (mid-2006) it is NOT eligible.
        self.assertNotIn(ticker, res.eligible_at(pd.Timestamp("2006-06-01")))
        # After 12 months it IS eligible.
        self.assertIn(ticker, res.eligible_at(pd.Timestamp("2007-02-01")))

    def test_timeline_has_full_coverage_at_end(self):
        tickers, earliest, latest = _build()
        a = EligibilityAnalyzer(tickers, earliest, latest)
        res = a.analyze(12, 0.8, "ME")
        self.assertEqual(res.summary()["coverage_at_end"], 100.0)


if __name__ == "__main__":
    unittest.main()
