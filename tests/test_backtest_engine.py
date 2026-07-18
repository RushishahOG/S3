"""Tests for the ARQM backtest engine and its pure helpers.

These run without a live database by constructing a small synthetic
``StorageManager`` shim, so they are fast and deterministic. They lock in the
contract that the engine (a) produces a NAV series, (b) emits trade log + rebalance
snapshots, and (c) never raises on edge cases (empty universe, missing factors).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtesting import normalization as norm
from core.backtesting.engine import run_backtest
from core.backtesting.metrics import sharpe, max_drawdown, sortino
from core.backtesting.regime import detect_regime
from core.config.backtest_schema import BacktestParameters, GeneralConfig


@pytest.fixture(autouse=True)
def _patch_universe(monkeypatch):
    """Point the engine's universe loader at the synthetic tickers so the
    storage shim (rather than the real NIFTY 500 constituents file) is used."""
    from core.backtesting import data as bt_data

    def _fake_load_universe(storage, params):
        return list(storage.tickers)

    monkeypatch.setattr(bt_data, "_load_universe", _fake_load_universe)


class _FakeStorage:
    """Minimal storage shim returning synthetic frames."""

    def __init__(self, prices, quality, lowvol, company, tickers):
        self.prices = prices
        self.quality = quality
        self.lowvol = lowvol
        self.company = company
        self.tickers = tickers

    def get_adjusted_price_panel(self, tickers=None, start=None, end=None):
        cols = [c for c in self.prices.columns if c in (tickers or self.prices.columns)]
        out = self.prices[cols]
        if start is not None:
            out = out[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out[out.index <= pd.Timestamp(end)]
        return out

    def get_fundamentals_company(self, t):
        return self.company.reindex(t)

    def get_fundamental_quality_features(self, t):
        q = self.quality.reset_index().rename(columns={"index": "ticker"})
        q["financial_year"] = 2024
        return q

    def feature_columns(self):
        return list(self.lowvol.columns)

    def get_features(self, t, columns=None):
        lf = self.lowvol.reindex(t).reset_index().rename(columns={"index": "ticker"})
        lf.insert(1, "date", pd.Timestamp("2024-01-01"))
        return lf

    def tickers_with_screener_data(self):
        return self.tickers


def _make_data(n=40, start="2001-01-01", end="2009-01-01", seed=1):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="B")
    tickers = [f"T{i}.NS" for i in range(n)]
    prices = pd.DataFrame(
        {t: 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(dates)))) for t in tickers},
        index=dates,
    )
    prices["NIFTY_500"] = prices.mean(axis=1)
    quality = pd.DataFrame(
        {
            "roe": rng.uniform(0.10, 0.30, n),
            "roce": rng.uniform(0.10, 0.30, n),
            "roa": rng.uniform(0.05, 0.20, n),
            "ocf_to_ebitda": rng.uniform(0.10, 0.40, n),
            "eps_growth": rng.uniform(-0.1, 0.3, n),
            "revenue_growth": rng.uniform(-0.05, 0.25, n),
            "interest_coverage_ratio": rng.uniform(1.5, 12.0, n),
            "equity_to_total_capital": rng.uniform(0.35, 0.80, n),
        },
        index=tickers,
    )
    lowvol = pd.DataFrame(
        {"semi_deviation": rng.uniform(0.01, 0.04, n),
         "beta": rng.uniform(0.6, 1.4, n)},
        index=tickers,
    )
    company = pd.DataFrame({"market_cap": rng.uniform(1e3, 5e4, n), "sector": ["X"] * n}, index=tickers)
    return _FakeStorage(prices, quality, lowvol, company, tickers)


def test_normalization_methods():
    s = pd.Series([1.0, 2, 3, 4, 100], index=list("ABCDE"))
    assert norm.normalize(s, "zscore").notna().all()
    assert norm.normalize(s, "percentile").between(0, 1).all()
    assert norm.normalize(s, "minmax").between(0, 1).all()
    lower = norm.score_lower_is_better(s, "percentile")
    assert lower.iloc[0] > lower.iloc[-1]  # lower raw -> higher score


def test_regime_detection_basic():
    dates = pd.date_range("2005-01-01", "2008-01-01", freq="B")
    up = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    df = detect_regime(up, BacktestParameters().regime)
    assert not df.empty
    assert set(df["state"]).issubset({"invested", "flat"})
    assert (df["state"] == "invested").any()


def test_engine_runs_on_synthetic_data():
    fs = _make_data()
    params = BacktestParameters(general=GeneralConfig(start_date="2006-01-01", end_date="2008-12-31"))
    res = run_backtest(params, fs)
    assert len(res.nav) > 0
    assert res.nav.iloc[0] == params.general.initial_capital
    assert len(res.snapshots) > 0
    assert "overall" in next(iter(res.snapshots.values())).columns
    # Metrics computable
    assert np.isfinite(res.metrics["max_drawdown"])
    assert np.isfinite(res.metrics["annual_volatility"])


def test_engine_empty_universe_safe():
    dates = pd.date_range("2006-01-01", "2007-01-01", freq="B")
    prices = pd.DataFrame({"NIFTY_500": np.linspace(100, 110, len(dates))}, index=dates)
    fs = _FakeStorage(prices, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), [])
    params = BacktestParameters(general=GeneralConfig(start_date="2006-01-01", end_date="2007-01-01"))
    # Should not raise; universe empty -> no trades.
    res = run_backtest(params, fs)
    assert res.trades.empty


def test_metrics_edge_cases():
    nav = pd.Series([100, 101, 99, 102], dtype=float)
    assert np.isfinite(sharpe(nav.pct_change().fillna(0)))
    assert max_drawdown(nav) <= 0
    flat = pd.Series([100.0] * 10)
    assert np.isnan(sharpe(flat.pct_change().fillna(0)))  # zero vol -> nan
