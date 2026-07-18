"""Tests for core/feature_engineering modules."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.feature_engineering.return_engine import (
    ReturnEngine,
    prepare_price_panel,
    compute_all_returns,
    merge_returns_into_panel,
)
from core.feature_engineering.risk_engine import (
    RiskEngine,
    compute_all_risk,
    MARKET_RISK_FEATURES,
)
from core.feature_engineering.feature_store import FeatureStore
from core.feature_engineering.feature_validator import (
    validate_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_prices():
    """3 tickers, ~5 years of daily data."""
    np.random.seed(42)
    tickers = ["AAPL", "GOOGL", "MSFT"]
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="B")
    rows = []
    for t in tickers:
        price = 100.0
        for d in dates:
            price *= 1 + np.random.normal(0.0005, 0.015)
            rows.append({"Date": d, "Ticker": t, "Adj Close": round(price, 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_benchmark(synthetic_prices):
    """Single benchmark series aligned to the price date range."""
    dates = synthetic_prices["Date"].unique()
    np.random.seed(99)
    price = 5000.0
    rows = []
    for d in sorted(dates):
        price *= 1 + np.random.normal(0.0003, 0.01)
        rows.append({"Date": d, "Adj Close": round(price, 2)})
    return pd.DataFrame(rows)


@pytest.fixture
def returns_panel(synthetic_prices):
    """Pre-computed returns panel."""
    panel = prepare_price_panel(synthetic_prices, "Adj Close")
    returns = compute_all_returns(panel, "Adj Close")
    return merge_returns_into_panel(panel, returns)


@pytest.fixture
def benchmark_returns(synthetic_benchmark):
    """Benchmark daily returns."""
    b = synthetic_benchmark.copy()
    b["Date"] = pd.to_datetime(b["Date"])
    b = b.sort_values("Date").set_index("Date")
    ret = b["Adj Close"].pct_change().dropna().reset_index()
    ret.columns = ["Date", "benchmark_return"]
    return ret


# ---------------------------------------------------------------------------
# RiskEngine column name tests
# ---------------------------------------------------------------------------

class TestRiskEngineColumnNames:
    """Verify the daily risk/momentum feature columns."""

    def test_compute_all_risk_returns_daily_features(self, returns_panel, benchmark_returns):
        """compute_all_risk should produce the four daily feature columns."""
        result = compute_all_risk(returns_panel, benchmark_returns)

        for col in MARKET_RISK_FEATURES:
            assert col in result.columns, f"Missing {col} in {list(result.columns)}"

        assert "Ticker" in result.columns
        assert "Date" in result.columns

        # No legacy monthly/weekly matrix columns should leak through.
        for col in result.columns:
            assert not col.startswith("std_"), f"Legacy column leaked: {col}"
            assert "_weekly" not in col, f"Weekly column leaked: {col}"

    def test_risk_engine_class_returns_valid_columns(self, synthetic_prices, synthetic_benchmark):
        """RiskEngine.compute_rolling_volatility should output the daily features."""
        from core.feature_engineering.return_engine import (
            prepare_price_panel,
            compute_all_returns,
            merge_returns_into_panel,
        )

        panel = prepare_price_panel(synthetic_prices, "Adj Close")
        returns = compute_all_returns(panel, "Adj Close")
        returns_panel = merge_returns_into_panel(panel, returns)

        engine = RiskEngine()
        result = engine.compute_rolling_volatility(
            returns_df=returns_panel,
            price_df=synthetic_prices,
            benchmark_df=synthetic_benchmark,
        )

        for col in MARKET_RISK_FEATURES:
            assert col in result.columns, f"Missing {col} in {list(result.columns)}"


# ---------------------------------------------------------------------------
# FeatureStore tests
# ---------------------------------------------------------------------------

class TestFeatureStore:
    """DuckDB-backed persistence layer."""

    def test_upsert_and_query(self, returns_panel, benchmark_returns):
        from core.feature_engineering.risk_engine import compute_all_risk

        risk = compute_all_risk(returns_panel, benchmark_returns)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_features.duckdb"
            store = FeatureStore(str(db_path))

            # Upsert
            store.upsert_features(risk)

            # Query back for first ticker
            ticker = risk["Ticker"].iloc[0]
            queried = store.get_features(tickers=[ticker])

            assert not queried.empty
            # Store normalises column name to lowercase
            assert set(queried["ticker"].unique()) == {ticker}

            # Check a risk column exists
            assert "beta" in queried.columns, (
                f"Missing 'beta' in queried: {list(queried.columns)}"
            )

    def test_metadata_registration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_meta.duckdb"
            store = FeatureStore(str(db_path))

            # Mock metadata dataclass
            from dataclasses import dataclass

            @dataclass
            class MockSpec:
                key: str
                description: str
                factor_category: str
                frequency: str
                lookback_months: int
                formula: str

            specs = [
                MockSpec("semi_deviation", "downside dev", "Risk", "daily", 12, "sqrt(mean(min(r,0)^2))*sqrt(252)"),
                MockSpec("beta", "12m beta", "Risk", "daily", 12, "cov/var"),
            ]

            store.register_feature_metadata(specs)

            # Verify by querying metadata table
            import duckdb
            con = duckdb.connect(str(db_path))
            result = con.execute("SELECT feature_key FROM feature_metadata").fetchdf()
            assert "semi_deviation" in result["feature_key"].values
            assert "beta" in result["feature_key"].values
            con.close()

    def test_export_parquet(self, returns_panel, benchmark_returns):
        from core.feature_engineering.risk_engine import compute_all_risk

        risk = compute_all_risk(returns_panel, benchmark_returns)

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_export.duckdb"
            parquet_path = Path(tmpdir) / "export.parquet"
            store = FeatureStore(str(db_path))

            store.upsert_features(risk)

            store.export_parquet(str(parquet_path))
            assert parquet_path.exists()
            assert parquet_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# FeatureValidator tests
# ---------------------------------------------------------------------------

class TestFeatureValidator:
    """Validation checks on feature DataFrames."""

    def test_validate_daily_features_pass(self, returns_panel, benchmark_returns):
        from core.feature_engineering.risk_engine import compute_all_risk

        risk = compute_all_risk(returns_panel, benchmark_returns)
        ok, issues, stats = validate_features(risk)
        assert ok, f"Daily validation failed: {issues}"

    def test_validate_with_nan_fails(self):
        df = pd.DataFrame({
            "Ticker": ["A", "A", "B", "B"],
            "Date": pd.date_range("2020-01-01", periods=4, freq="D"),
            "beta": [0.1, np.nan, np.nan, np.nan],
        })
        ok, issues, _ = validate_features(df)
        assert not ok

    def test_validate_with_inf_fails(self):
        df = pd.DataFrame({
            "ticker": ["A", "A"],
            "date": pd.date_range("2020-01-01", periods=2, freq="D"),
            "beta": [0.1, np.inf],
        })
        ok, issues, _ = validate_features(df)
        assert not ok

    def test_validate_duplicates_fails(self):
        df = pd.DataFrame({
            "ticker": ["A", "A"],
            "date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
            "std_3m_daily": [0.1, 0.2],
        })
        ok, issues, _ = validate_features(df)
        assert not ok
