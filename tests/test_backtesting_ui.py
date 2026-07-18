"""Smoke test that the backtest UI page and exports import / run without error.

The Streamlit ``render()`` function requires a running runtime, so we test the
pieces that can run head-less: module import, export byte generation, and the
parameter round-trip used by the config sidebar.
"""

from __future__ import annotations

import pandas as pd

from app.pages import backtesting
from core.backtesting.export import export_dataframe
from core.config.backtest_schema import BacktestParameters


def test_backtesting_page_imports():
    # Page module must import (Streamlit functions are lazily called).
    assert hasattr(backtesting, "render")
    assert hasattr(backtesting, "_config_panel")


def test_export_formats():
    df = pd.DataFrame({"a": [1, 2, 3]})
    csv = export_dataframe(df, "csv")
    assert isinstance(csv, bytes) and len(csv) > 0
    parquet = export_dataframe(df, "parquet")
    assert isinstance(parquet, bytes) and len(parquet) > 0


def test_config_sidebar_roundtrip():
    # The sidebar builds a BacktestParameters; ensure it round-trips via schema.
    p = BacktestParameters()
    d = p.to_dict()
    p2 = BacktestParameters.from_dict(d)
    assert p2.general.rebalance_frequency == p.general.rebalance_frequency
    assert p2.scoring.momentum_weight == p.scoring.momentum_weight
