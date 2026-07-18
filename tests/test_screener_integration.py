"""End-to-end tests for the Screener integration: provider parsing, storage,
downloader and the 16-factor Quality engine (synthetic data, no network)."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from core.data.ingestion.screener_downloader import ScreenerDownloader, ScreenerJob
from core.data.providers.apify_screener_provider import ApifyScreenerProvider
from core.data.storage.storage_manager import StorageManager
from core.factors.fundamental import FundamentalQualityEngine

SCREENER_URL = "https://www.screener.in/company/TCS/consolidated/"


def _synthetic_payload():
    """A realistic screener.in response (tabular sections, wrapped in items[])."""
    return {
        "payload": {"mode": "getstockdetails", "url": SCREENER_URL},
        "items": [
            {
                "company_name": "Tata Consultancy Services",
                "profit_and_loss": {
                    "annual_data": [
                        {"Metric": "Sales", "Mar 2023": 200000.0, "Mar 2022": 180000.0},
                        {"Metric": "Operating Profit", "Mar 2023": 50000.0, "Mar 2022": 45000.0},
                        {"Metric": "Expenses", "Mar 2023": 150000.0, "Mar 2022": 135000.0},
                        {"Metric": "Other Income", "Mar 2023": 2000.0, "Mar 2022": 1800.0},
                        {"Metric": "Interest", "Mar 2023": 1000.0, "Mar 2022": 1100.0},
                        {"Metric": "Depreciation", "Mar 2023": 8000.0, "Mar 2022": 7500.0},
                        {"Metric": "Profit before tax", "Mar 2023": 41000.0, "Mar 2022": 37000.0},
                        {"Metric": "Tax %", "Mar 2023": 25.0, "Mar 2022": 25.0},
                        {"Metric": "Net Profit", "Mar 2023": 30000.0, "Mar 2022": 27000.0},
                        {"Metric": "EPS in Rs", "Mar 2023": 80.0, "Mar 2022": 72.0},
                        {"Metric": "Dividend Payout %", "Mar 2023": 40.0, "Mar 2022": 38.0},
                    ],
                    "Return on Equity": [
                        {"10 Years:": 13.0},
                        {"5 Years:": 14.0},
                        {"3 Years:": 15.0},
                        {"Last Year:": 45.0},
                    ],
                },
                "quarters": [
                    {"Metric": "Sales", "Mar 2023": 50000.0, "Jun 2023": 52000.0,
                     "Sep 2023": 51000.0, "Dec 2023": 54000.0, "Mar 2024": 53000.0},
                    {"Metric": "Operating Profit", "Mar 2023": 12000.0, "Jun 2023": 12500.0,
                     "Sep 2023": 12300.0, "Dec 2023": 13000.0, "Mar 2024": 12800.0},
                    {"Metric": "Expenses", "Mar 2023": 38000.0, "Jun 2023": 39500.0,
                     "Sep 2023": 38700.0, "Dec 2023": 41000.0, "Mar 2024": 40200.0},
                    {"Metric": "Interest", "Mar 2023": 250.0, "Jun 2023": 240.0,
                     "Sep 2023": 245.0, "Dec 2023": 255.0, "Mar 2024": 250.0},
                    {"Metric": "Net Profit", "Mar 2023": 7500.0, "Jun 2023": 7700.0,
                     "Sep 2023": 7600.0, "Dec 2023": 8000.0, "Mar 2024": 7800.0},
                    {"Metric": "EPS in Rs", "Mar 2023": 20.0, "Jun 2023": 20.5,
                     "Sep 2023": 20.3, "Dec 2023": 21.2, "Mar 2024": 20.8},
                ],
                "balance_sheet": [
                    {"Metric": "Equity Capital", "Mar 2023": 1000.0, "Mar 2022": 1000.0},
                    {"Metric": "Reserves", "Mar 2023": 60000.0, "Mar 2022": 55000.0},
                    {"Metric": "Borrowings", "Mar 2023": 5000.0, "Mar 2022": 4500.0},
                    {"Metric": "Other Liabilities", "Mar 2023": 20000.0, "Mar 2022": 18000.0},
                    {"Metric": "Total Liabilities", "Mar 2023": 86000.0, "Mar 2022": 78500.0},
                    {"Metric": "Fixed Assets", "Mar 2023": 30000.0, "Mar 2022": 28000.0},
                    {"Metric": "CWIP", "Mar 2023": 2000.0, "Mar 2022": 1500.0},
                    {"Metric": "Investments", "Mar 2023": 15000.0, "Mar 2022": 14000.0},
                    {"Metric": "Other Assets", "Mar 2023": 8000.0, "Mar 2022": 7500.0},
                    {"Metric": "Total Assets", "Mar 2023": 100000.0, "Mar 2022": 92000.0},
                    {"Metric": "Current Liabilities", "Mar 2023": 10000.0, "Mar 2022": 9000.0},
                ],
                "cash_flow": [
                    {"Metric": "Cash from Operating Activity", "Mar 2023": 40000.0, "Mar 2022": 36000.0},
                    {"Metric": "Cash from Investing Activity", "Mar 2023": -10000.0, "Mar 2022": -9000.0},
                    {"Metric": "Cash from Financing Activity", "Mar 2023": -5000.0, "Mar 2022": -4500.0},
                    {"Metric": "Free Cash Flow", "Mar 2023": 30000.0, "Mar 2022": 27000.0},
                    {"Metric": "Net Cash Flow", "Mar 2023": 25000.0, "Mar 2022": 22500.0},
                    {"Metric": "CFO/OP", "Mar 2023": 0.8, "Mar 2022": 0.8},
                ],
                "ratios": [
                    {"Metric": "ROCE %", "Mar 2023": 50.0, "Mar 2022": 49.0},
                    {"Metric": "Working Capital Days", "Mar 2023": 10.0, "Mar 2022": 11.0},
                    {"Metric": "Debtor Days", "Mar 2023": 60.0, "Mar 2022": 62.0},
                    {"Metric": "Cash Conversion Cycle", "Mar 2023": 70.0, "Mar 2022": 72.0},
                ],
                "dividends": [
                    {"exDate": "2023-05-15", "amount": 32.0},
                    {"exDate": "2022-05-10", "amount": 28.0},
                ],
            }
        ],
    }


class _FakeScreenerProvider:
    name = "fake_screener"

    def fetch(self, url: str):
        real = ApifyScreenerProvider()
        return real._parse(_synthetic_payload(), url)

    def is_available(self):
        return True


@pytest.fixture
def tmp_storage():
    fd, path = tempfile.mkstemp(suffix=".duckdb")
    os.close(fd)
    os.remove(path)
    store = StorageManager(path)
    yield store
    try:
        os.remove(path)
    except OSError:
        pass


def test_provider_parses_all_tables():
    res = ApifyScreenerProvider()._parse(_synthetic_payload(), SCREENER_URL)
    assert res.ticker == "TCS.NS"
    assert res.company["company_name"] == "Tata Consultancy Services"
    assert len(res.income_annual) == 2
    assert len(res.income_quarterly) == 5
    assert res.income_quarterly[0]["quarter_index"] == 1
    assert len(res.balance_sheet) == 2
    assert len(res.cashflow) == 2
    assert len(res.dividends) == 2
    yearly = res.ratios.get("_yearly", [])
    assert len(yearly) == 2
    by_year = {r["financial_year"]: r for r in yearly}
    # ROE is read from profit_and_loss -> 'Return on Equity' (Last Year) and
    # attached to the most recent ratio year (2023 here).
    assert by_year[2023]["roe"] == pytest.approx(45.0)
    assert by_year[2023]["roce"] == pytest.approx(50.0)


def test_downloader_stores_and_engineering_runs(tmp_storage):
    dl = ScreenerDownloader(
        tmp_storage,
        jobs=[ScreenerJob(SCREENER_URL, "TCS.NS", "TCS")],
        provider=_FakeScreenerProvider(),
    )
    recs = dl.download()
    assert recs[0]["status"] == "success"
    assert tmp_storage.tickers_with_screener_data() == {"TCS.NS"}

    assert len(tmp_storage.get_fundamentals_income_annual(["TCS.NS"])) == 2
    assert len(tmp_storage.get_fundamentals_balance_sheet(["TCS.NS"])) == 2
    assert len(tmp_storage.get_fundamentals_dividends(["TCS.NS"])) == 2

    ratios = tmp_storage.get_fundamentals_ratios(["TCS.NS"])
    assert len(ratios) == 2
    assert ratios[ratios["financial_year"] == 2023].iloc[0]["roe"] == pytest.approx(45.0)

    engine = FundamentalQualityEngine(tmp_storage)
    out = engine.compute(store=True)
    assert not out.empty
    tcs = out[out["ticker"] == "TCS.NS"]
    assert (tcs["financial_year"] == 2023).any()
    row23 = tcs[tcs["financial_year"] == 2023].iloc[0]
    # ROE now computed as Net Profit / (Equity Capital + Reserves) = 30000/61000.
    assert row23["roe"] == pytest.approx(30000 / 61000, abs=1e-4)
    assert row23["roce"] == pytest.approx(50.0)
    assert row23["interest_coverage_ratio"] == pytest.approx(50.0)
    assert row23["roa"] == pytest.approx(30000 / 100000, abs=1e-3)
    assert row23["equity_to_total_capital"] == pytest.approx(61000 / 66000, abs=1e-3)
    # SGR = ROE x (1 - payout) = (30000/61000) x (1 - 0.40).
    assert row23["sustainable_growth_rate"] == pytest.approx((30000 / 61000) * 0.60, abs=1e-4)
    assert tcs[tcs["financial_year"] == 2023]["eps_growth"].iloc[0] == pytest.approx(0.1111, abs=1e-3)
    assert pd.notna(row23["eps_growth_median"])
    stored = tmp_storage.get_fundamental_quality_features(["TCS.NS"])
    assert len(stored) == 2
