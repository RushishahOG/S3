"""Fundamental data dataset source for the Dataset Explorer.

Registers alongside the market-data source and exposes the normalised
Screener-derived fundamental tables (income statements, balance sheet, cash
flow, dividends, ratios and the engineered Quality features) through the uniform
:class:`~app.explorer.base.DatasetSource` contract. Selecting a security renders
inspectable tables, each with sorting / filtering / search / CSV export -
exactly like the OHLCV explorer.
"""

from __future__ import annotations

import pandas as pd

from app.explorer.base import DATASET_SOURCES, DatasetSource, HealthIssue, Severity
from core.data.storage.storage_manager import StorageManager
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


class FundamentalDatasetSource(DatasetSource):
    key = "fundamentals"
    label = "Fundamental Data (Quality)"
    description = (
        "Normalised Screener fundamental data: annual / quarterly financial "
        "statements, balance sheet, cash flow, dividend history, ratio snapshots "
        "and engineered Quality features."
    )

    COMPANY_COLUMNS = [
        "ticker", "company_name", "last_updated",
    ]
    QUALITY_COLUMNS = [
        "ticker", "financial_year", "roe", "roce", "roa",
        "interest_coverage_ratio", "equity_to_total_capital",
        "dividend_payout_ratio", "ocf_to_ebitda", "cash_roce",
        "sustainable_growth_rate",
    ]

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()

    # -- catalogue --------------------------------------------------------
    def security_summary(self) -> pd.DataFrame:
        company = self.storage.get_fundamentals_company()
        if company.empty:
            return pd.DataFrame(columns=[
                "ticker", "company_name", "records", "earliest",
                "latest", "availability_pct", "last_updated", "status",
            ])
        meta = self.storage.get_fundamental_download_metadata()
        meta_idx = meta.set_index("ticker") if not meta.empty else pd.DataFrame()

        income = self.storage.get_fundamentals_income_annual()
        counts = income.groupby("ticker")["financial_year"].agg(
            records="count", earliest="min", latest="max"
        ) if not income.empty else pd.DataFrame()

        rows = []
        for ticker, c in company.groupby("ticker"):
            company_name = ticker
            status = "success"
            if ticker in meta_idx.index:
                company_name = meta_idx.loc[ticker].get("company_name") or ticker
                status = meta_idx.loc[ticker].get("status") or "success"
            rec = counts.loc[ticker] if ticker in counts.index else None
            rows.append({
                "ticker": ticker,
                "company_name": company_name,
                "records": int(rec["records"]) if rec is not None else 0,
                "earliest": int(rec["earliest"]) if rec is not None else None,
                "latest": int(rec["latest"]) if rec is not None else None,
                "availability_pct": 100.0,
                "last_updated": pd.Timestamp.now() if ticker in meta_idx.index else None,
                "status": status,
            })
        return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    # -- inspection -------------------------------------------------------
    def fetch_dataset(self, ticker: str) -> pd.DataFrame:
        return self.fetch_screener_income_annual(ticker)

    def fetch_income(self, ticker: str) -> pd.DataFrame:
        return self.fetch_screener_income_annual(ticker)

    def fetch_dividends(self, ticker: str) -> pd.DataFrame:
        return self.fetch_screener_dividends(ticker)

    def fetch_ratios(self, ticker: str) -> pd.DataFrame:
        return self.fetch_screener_ratios(ticker)

    def fetch_quality(self, ticker: str) -> pd.DataFrame:
        return self.fetch_screener_quality(ticker)

    # -- screener (single source of truth) tables ---------------------------
    def fetch_screener_company(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_company([ticker])
        cols = [c for c in self.COMPANY_COLUMNS if c in df.columns]
        return df[cols] if not df.empty else df

    def fetch_screener_income_annual(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_income_annual([ticker])
        return df.sort_values("financial_year") if not df.empty else df

    def fetch_screener_income_quarterly(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_income_quarterly([ticker])
        return df.sort_values("quarter_index") if not df.empty else df

    def fetch_screener_balance_sheet(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_balance_sheet([ticker])
        return df.sort_values("financial_year") if not df.empty else df

    def fetch_screener_cashflow(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_cashflow([ticker])
        return df.sort_values("financial_year") if not df.empty else df

    def fetch_screener_dividends(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_dividends([ticker])
        return df.sort_values("ex_date") if not df.empty else df

    def fetch_screener_ratios(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamentals_ratios([ticker])
        return df.sort_values("financial_year") if not df.empty else df

    def fetch_screener_roe(self, ticker: str) -> pd.DataFrame:
        """Return ROE as trailing averages (Last / 3 / 5 / 10 Years).

        The Screener actor exposes ROE under ``profit_and_loss -> Return on
        Equity`` as trailing averages; these are stored on the anchor financial
        years in ``fundamentals_ratios.roe``. Reconstruct the human-readable
        period label by the year's recency position *among all ratio years*
        (so the 3rd/5th/10th most recent year is labelled correctly) and drop
        NULL years.
        """
        df = self.storage.get_fundamentals_ratios([ticker])
        if df.empty or "roe" not in df.columns:
            return pd.DataFrame(columns=["period", "financial_year", "roe"])
        all_years = df.sort_values("financial_year")["financial_year"].tolist()
        n_total = len(all_years)
        roe = df[df["roe"].notna()].sort_values("financial_year")
        if roe.empty:
            return pd.DataFrame(columns=["period", "financial_year", "roe"])
        offsets = {0: "Last Year", 2: "3 Years", 4: "5 Years", 9: "10 Years"}
        rows = []
        for r in roe.itertuples(index=False):
            yr = int(r.financial_year)
            pos_from_latest = (n_total - 1) - all_years.index(yr)
            label = offsets.get(pos_from_latest, f"{pos_from_latest + 1}Y ago")
            rows.append({"period": label, "financial_year": yr, "roe": float(r.roe)})
        out = pd.DataFrame(rows)
        order = {"Last Year": 0, "3 Years": 1, "5 Years": 2, "10 Years": 3}
        out["_o"] = out["period"].map(lambda p: order.get(p, 9))
        out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        return out

    def fetch_screener_quality(self, ticker: str) -> pd.DataFrame:
        df = self.storage.get_fundamental_quality_features([ticker])
        return df.sort_values("financial_year") if not df.empty else df

    def display_columns(self) -> list[str]:
        return self.COMPANY_COLUMNS

    def dataset_statistics(self, df: pd.DataFrame, ticker: str) -> dict:
        if df.empty:
            return {"ticker": ticker, "financial_years": 0,
                    "first_year": None, "last_year": None}
        years = pd.to_numeric(df["financial_year"], errors="coerce").dropna()
        return {
            "ticker": ticker,
            "financial_years": int(len(df)),
            "first_year": int(years.min()) if not years.empty else None,
            "last_year": int(years.max()) if not years.empty else None,
        }

    # -- storage / health -------------------------------------------------
    def storage_statistics(self) -> dict:
        stats = self.storage.storage_statistics()
        income = self.storage.get_fundamentals_income_annual()
        meta = self.storage.get_fundamental_download_metadata()
        last_dl = None
        if not meta.empty and "updated_at" in meta.columns:
            last_dl = pd.to_datetime(meta["updated_at"]).max()
        return {
            "total_securities": income["ticker"].nunique() if not income.empty else 0,
            "total_rows": int(len(income)),
            "db_size_bytes": stats["db_size_bytes"],
            "cache_size_bytes": 0,
            "last_download_time": last_dl,
        }

    def health_issues(self, tickers: list[str]) -> list[HealthIssue]:
        issues: list[HealthIssue] = []
        for ticker in tickers:
            income = self.storage.get_fundamentals_income_annual([ticker])
            ratios = self.storage.get_fundamentals_ratios([ticker])
            quality = self.storage.get_fundamental_quality_features([ticker])
            if income.empty:
                issues.append(HealthIssue(
                    ticker=ticker, issue="Missing financial statements",
                    severity=Severity.WARNING, detail="No annual income data stored."))
            if ratios.empty:
                issues.append(HealthIssue(
                    ticker=ticker, issue="Missing ratio snapshot",
                    severity=Severity.WARNING, detail="No ROE/ROCE/P/E/P/B snapshot stored."))
            if quality.empty and not income.empty:
                issues.append(HealthIssue(
                    ticker=ticker, issue="Missing Quality features",
                    severity=Severity.WARNING,
                    detail="Fundamentals present but Quality features not engineered."))
        return issues


# Register on import (lazy: no connection opened until first access).
FundamentalDatasetSource.register(FundamentalDatasetSource)
