"""Unit tests for the global competition backtesting date constraints."""

from __future__ import annotations

import unittest

import pandas as pd

from core.utils.dates import (
    DateRangeError,
    MAX_BACKTEST_DATE,
    MIN_BACKTEST_DATE,
    clamp_to_bounds,
    default_backtest_range,
    validate_backtest_range,
)


class TestBacktestConstraints(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(MIN_BACKTEST_DATE, pd.Timestamp("2006-01-01"))
        self.assertEqual(MAX_BACKTEST_DATE, pd.Timestamp("2026-05-31"))

    def test_default_range(self):
        start, end = default_backtest_range()
        self.assertEqual(start, MIN_BACKTEST_DATE)
        self.assertEqual(end, MAX_BACKTEST_DATE)

    def test_valid_range_accepted(self):
        s, e = validate_backtest_range("2010-06-01", "2020-12-31")
        self.assertEqual(s, pd.Timestamp("2010-06-01"))
        self.assertEqual(e, pd.Timestamp("2020-12-31"))

    def test_start_before_min_rejected(self):
        with self.assertRaises(DateRangeError):
            validate_backtest_range("2005-12-31", MAX_BACKTEST_DATE)

    def test_end_after_max_rejected(self):
        with self.assertRaises(DateRangeError):
            validate_backtest_range("2010-01-01", "2026-06-01")

    def test_start_after_end_rejected(self):
        with self.assertRaises(DateRangeError):
            validate_backtest_range("2020-01-01", "2019-01-01")

    def test_missing_end_defaults_to_max(self):
        s, e = validate_backtest_range("2015-01-01")
        self.assertEqual(e, MAX_BACKTEST_DATE)

    def test_clamp_to_bounds(self):
        self.assertEqual(clamp_to_bounds("2000-01-01"), MIN_BACKTEST_DATE)
        self.assertEqual(clamp_to_bounds("2030-01-01"), MAX_BACKTEST_DATE)
        self.assertEqual(clamp_to_bounds("2012-03-03"), pd.Timestamp("2012-03-03"))


if __name__ == "__main__":
    unittest.main()
