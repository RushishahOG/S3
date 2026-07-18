"""Apify Screener provider — single source of truth for fundamental data.

Wraps the Apify *Screener* actor (config key ``actors.screener``) and normalises
its complete output into the platform's canonical screener tables:

    * fundamentals_company
    * fundamentals_income_statement_annual
    * fundamentals_income_statement_quarterly
    * fundamentals_balance_sheet
    * fundamentals_cashflow
    * fundamentals_dividends
    * fundamentals_ratios

The actor is driven by a screener.in company URL (e.g.
``https://www.screener.in/company/TCS/consolidated/``) and returns the full
financial history — there is no extraction window.

The live Screener actor returns a **tabular** payload: the response is wrapped in
``{"payload": ..., "items": [{...}]}`` and each financial section is a list of
``{"Metric": <name>, "Mar 2015": <value>, "Mar 2016": <value>, ...}`` rows
(``profit_and_loss.annual_data``, ``quarters``, ``balance_sheet``, ``cash_flow``,
``ratios``). This module pivots those wide rows into the platform's long
``fundamentals_*`` tables. Parsing is schema-tolerant: sections that are absent
or use a different shape degrade to NULL rather than failing the run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from core.config.providers_config import get_provider_config
from core.data.providers.apify_client import run_actor
from core.data.providers.base_provider import BaseFundamentalProvider
from core.data.providers.fundamental_parsing import (
    build_lookup,
    extract_field,
    extract_year,
    iter_dicts,
    norm_key,
    to_float,
)
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Canonical field maps (screener.in naming -> storage column)                 #
# --------------------------------------------------------------------------- #
INCOME_ANNUAL_FIELDS: dict[str, list[str]] = {
    "sales": ["Sales", "sales", "totalRevenue", "Total Revenue", "revenue", "netSales", "netRevenue"],
    "operating_profit": ["Operating Profit", "operatingProfit", "operatingProfit", "ebit", "EBIT", "operatingIncome"],
    "expenses": ["Expenses", "expenses", "Expenses", "totalExpenses", "totalExpense"],
    "other_income": ["Other Income", "otherIncome", "Other Income"],
    "interest": ["Interest", "interest", "Interest", "interestExpense", "financeCosts"],
    "depreciation": ["Depreciation", "depreciation", "Depreciation", "depreciationAmortisation"],
    "profit_before_tax": ["Profit before tax", "profitBeforeTax", "Profit Before Tax", "pbt", "pretaxIncome"],
    "tax_percent": ["Tax %", "taxPercentage", "taxPercent", "Tax %", "taxPct"],
    "net_profit": ["Net Profit", "netProfit", "Net Profit", "pat", "profitAfterTax", "netIncome", "Net Income"],
    "eps": ["EPS in Rs", "eps", "EPS", "epsBasic", "basicEps", "earningsPerShare"],
    "dividend_payout_percent": ["Dividend Payout %", "dividendPayoutPercentage", "dividendPayoutPercent", "Dividend Payout %"],
}

INCOME_QUARTERLY_FIELDS: dict[str, list[str]] = {
    "quarterly_sales": ["Sales", "sales", "Sales", "totalRevenue", "revenue", "netSales"],
    "quarterly_operating_profit": ["Operating Profit", "operatingProfit", "Operating Profit", "ebit", "EBIT", "operatingIncome"],
    "quarterly_expenses": ["Expenses", "expenses", "Expenses", "totalExpenses"],
    "quarterly_interest": ["Interest", "interest", "Interest", "interestExpense", "financeCosts"],
    "quarterly_net_profit": ["Net Profit", "netProfit", "Net Profit", "pat", "netIncome", "Net Income"],
    "quarterly_eps": ["EPS in Rs", "eps", "EPS", "epsBasic", "basicEps"],
}

BALANCE_FIELDS: dict[str, list[str]] = {
    "equity_capital": ["Equity Capital", "equityCapital", "Equity Capital", "shareCapital", "Share Capital"],
    "reserves": ["Reserves", "reserves", "Reserves", "reserveSurplus", "Reserves & Surplus", "reservesAndSurplus"],
    "borrowings": ["Borrowings", "borrowings", "Borrowings", "totalBorrowings", "totalDebt"],
    "other_liabilities": ["Other Liabilities", "otherLiabilities", "Other Liabilities"],
    "total_liabilities": ["Total Liabilities", "totalLiabilities", "Total Liabilities"],
    "fixed_assets": ["Fixed Assets", "fixedAssets", "Fixed Assets", "grossBlock", "Gross Block"],
    "cwip": ["CWIP", "cwip", "CWIP", "capitalWorkInProgress"],
    "investments": ["Investments", "investments", "Investments", "totalInvestments"],
    "other_assets": ["Other Assets", "otherAssets", "Other Assets"],
    "total_assets": ["Total Assets", "totalAssets", "Total Assets"],
    "current_liabilities": ["Current Liabilities", "currentLiabilities", "Current Liabilities", "totalCurrentLiabilities"],
}

CASHFLOW_FIELDS: dict[str, list[str]] = {
    "operating_cash_flow": ["Cash from Operating Activity", "operatingCashFlow", "Operating Cash Flow", "cashFromOperations", "cfo"],
    "investing_cash_flow": ["Cash from Investing Activity", "investingCashFlow", "Investing Cash Flow", "cashFromInvesting"],
    "financing_cash_flow": ["Cash from Financing Activity", "financingCashFlow", "Financing Cash Flow", "cashFromFinancing"],
    "free_cash_flow": ["Free Cash Flow", "freeCashFlow", "Free Cash Flow", "fcf"],
    "net_cash_flow": ["Net Cash Flow", "netCashFlow", "Net Cash Flow", "netCash"],
    "cfo_per_op": ["CFO/OP", "cfoPerOp", "CFO/OP", "cfoOpRatio", "cfoToOperatingProfit"],
}

RATIO_FIELDS: dict[str, list[str]] = {
    "roe": ["ROE %", "roe", "roePercent", "returnOnEquity", "ROE %"],
    "roce": ["ROCE %", "roce", "rocePercent", "returnOnCapitalEmployed", "ROCE %"],
    "working_capital_days": ["Working Capital Days", "workingCapitalDays", "Working Capital Days", "wcDays"],
    "debtor_days": ["Debtor Days", "debtorDays", "Debtor Days", "receivableDays"],
    "cash_conversion_cycle": ["Cash Conversion Cycle", "cashConversionCycle", "Cash Conversion Cycle", "ccc"],
}

COMPANY_FIELDS: dict[str, list[str]] = {
    "company_name": ["companyName", "name", "company"],
    "sector": ["sector", "industryGroup"],
    "industry": ["industry"],
    "market_cap": ["marketCap", "marketCapCrore", "marketCapitalization", "mktCap"],
    "pe": ["peRatio", "pe", "priceToEarnings", "PERatio"],
    "pb": ["pbRatio", "pb", "priceToBook", "PBRatio"],
    "dividend_yield": ["dividendYield", "dividendYieldPercent"],
    "current_price": ["currentPrice", "price", "lastPrice", "cmp"],
}

#: Keys whose value is expressed in *crores* (1 crore = 1e7) and must be scaled.
CRORE_FIELDS = {"market_cap": {"marketcapcrore"}}

YEAR_CANDIDATES = ["year", "fiscalYear", "financialYear", "date", "period", "quarter", "periodEnding"]
QUARTER_CANDIDATES = ["quarter", "q", "period", "date", "label"]
ANNUAL_LIST_CANDIDATES = ["annualResults", "annualResult", "yearlyResults", "results"]
QUARTERLY_LIST_CANDIDATES = ["quarterlyResults", "quarterlyResult", "qResults"]
BALANCE_CANDIDATES = ["balanceSheet", "balance"]
CASHFLOW_CANDIDATES = ["cashFlow", "cashflow", "cashFlowStatement"]
DIVIDEND_LIST_CANDIDATES = ["dividends", "dividend", "dividendHistory", "dividendPayout"]


@dataclass
class ScreenerResult:
    """Normalised screener payload for a single company URL."""

    url: str
    ticker: str
    company: dict = field(default_factory=dict)
    income_annual: list[dict] = field(default_factory=list)
    income_quarterly: list[dict] = field(default_factory=list)
    balance_sheet: list[dict] = field(default_factory=list)
    cashflow: list[dict] = field(default_factory=list)
    dividends: list[dict] = field(default_factory=list)
    ratios: dict = field(default_factory=dict)


class ApifyScreenerProvider(BaseFundamentalProvider):
    name = "apify_screener"

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or get_provider_config("apify")
        self.token = self.cfg.get("api_token") or ""
        self.timeout = int(self.cfg.get("timeout_seconds", 120))
        self.actor_id = (self.cfg.get("actors") or {}).get("screener", {}).get("id")

    def is_available(self) -> bool:
        return bool(self.token) and bool(self.actor_id)

    # -- public API ---------------------------------------------------------
    def fetch(self, url: str) -> ScreenerResult:
        result, _raw = self.fetch_with_raw(url)
        return result

    def fetch_with_raw(self, url: str) -> tuple[ScreenerResult, list[dict]]:
        """Like :meth:`fetch` but returns the raw actor items for inspection."""
        if not self.actor_id:
            raise RuntimeError("Apify screener actor id is not configured.")
        payload = {
            "mode": "getstockdetails",
            "url": url,
            "queryString": "",
            "username": "",
            "password": "",
        }
        logger.info("Downloading screener data for %s", url)
        items = run_actor(self.actor_id, payload, token=self.token, timeout_seconds=self.timeout)
        result = self._parse(items, url)
        n_rows = (
            len(result.income_annual) + len(result.income_quarterly)
            + len(result.balance_sheet) + len(result.cashflow) + len(result.dividends)
        )
        logger.info(
            "Screener parse: %s | company=%s income_years=%d qtr=%d bal=%d cf=%d div=%d",
            url, bool(result.company), len(result.income_annual),
            len(result.income_quarterly), len(result.balance_sheet),
            len(result.cashflow), len(result.dividends),
        )
        return result, (items or [])

    # -- parsing helpers ----------------------------------------------------
    @staticmethod
    def _resolve_root(items: Any) -> dict | None:
        """Locate the company payload dict inside an actor response.

        The live Screener actor wraps its output as
        ``{"payload": ..., "items": [ {company dict} ]}``; earlier/variant actors
        may return the company dict directly, or a bare list. Normalise to the
        single company dict.
        """
        if isinstance(items, dict):
            if items.get("items") and isinstance(items["items"], list) and items["items"]:
                cand = items["items"][0]
                return cand if isinstance(cand, dict) else items
            return items
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return items[0]
        for _p, d in iter_dicts(items):
            if isinstance(d, dict):
                return d
        return None

    @staticmethod
    def _year_from_col(col: str) -> int | None:
        """Extract a fiscal year (e.g. 'Mar 2023' -> 2023) from a column name."""
        return extract_year(col)

    @classmethod
    def _pivot_table(
        cls,
        rows: list[dict],
        ticker: str,
        metric_map: dict[str, str],
        year_cols: list[str],
        period_label: str | None = None,
    ) -> list[dict]:
        """Pivot wide ``Metric``/year-row data into one row per period.

        ``rows`` is a list of ``{"Metric": <name>, "Mar 2015": <v>, ...}``.
        ``metric_map`` maps the *exact* Metric label -> storage column. For each
        year column a long row ``{ticker, financial_year, <col>: value, ...}`` is
        produced (only periods that have at least one non-null mapped metric).
        """
        collected: dict[int, dict] = {}
        order: list[int] = []
        # Invert metric_map (col -> [candidate labels]) into a label -> col map
        # keyed by the normalised candidate, so we tolerate spacing/case.
        label_to_col: dict[str, str] = {}
        for col, cands in metric_map.items():
            for c in cands:
                label_to_col[norm_key(c)] = col
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric = row.get("Metric")
            if metric is None:
                continue
            col = label_to_col.get(norm_key(metric))
            if col is None:
                continue
            for yc in year_cols:
                val = to_float(row.get(yc))
                if val is None:
                    continue
                yr = cls._year_from_col(yc)
                if yr is None:
                    continue
                if yr not in collected:
                    collected[yr] = {"ticker": ticker, "financial_year": yr}
                    order.append(yr)
                collected[yr][col] = val
        out = [collected[yr] for yr in sorted(order)]
        if period_label is not None:
            for i, r in enumerate(out, start=1):
                r[period_label] = i
        return out

    def _parse(self, items: Any, url: str) -> ScreenerResult:
        root = self._resolve_root(items)
        ticker = self._ticker_from_url(url)
        res = ScreenerResult(url=url, ticker=ticker)
        if root is None:
            logger.warning("No parseable screener payload for %s", url)
            return res

        res.company = self._parse_company(root, ticker)
        res.income_annual = self._parse_annual(root, ticker)
        res.income_quarterly = self._parse_quarterly(root, ticker)
        res.balance_sheet = self._parse_balance(root, ticker)
        res.cashflow = self._parse_cashflow(root, ticker)
        res.dividends = self._parse_dividends(root, ticker)
        res.ratios = self._parse_ratios(root, ticker)
        return res

    def _parse_company(self, root: dict, ticker: str) -> dict:
        row: dict[str, Any] = {"ticker": ticker}
        # Company-level fields live under the canonical screener keys when
        # present (older/variant actor shapes); the live tabular actor only
        # exposes the company name at the top level.
        lookup = build_lookup(root)
        for col, cands in COMPANY_FIELDS.items():
            val = extract_field(root, cands, lookup)
            if col in CRORE_FIELDS and val is not None:
                used = {norm_key(c) for c in cands} & CRORE_FIELDS[col]
                if used:
                    val = val * 1e7
            row[col] = val
        row["last_updated"] = pd.Timestamp.now().date()
        return row

    @staticmethod
    def _tabular_section(root: dict, *keys: str) -> list[dict]:
        """Return a tabular section (list of Metric rows) found under ``keys``."""
        node = root
        for k in keys:
            if not isinstance(node, dict):
                return []
            nk = build_lookup(node).get(norm_key(k))
            if nk is None:
                return []
            node = node[nk]
        if isinstance(node, list):
            return [r for r in node if isinstance(r, dict)]
        if isinstance(node, dict):
            # Some actors nest annual_data inside the section dict.
            inner = node.get("annual_data")
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
        return []

    @staticmethod
    def _year_columns(rows: list[dict]) -> list[str]:
        """Collect the year/period columns (everything but 'Metric') in order."""
        cols: list[str] = []
        for r in rows:
            for c in r.keys():
                if c == "Metric":
                    continue
                if c not in cols:
                    cols.append(c)
        return cols

    def _parse_annual(self, root: dict, ticker: str) -> list[dict]:
        rows = self._tabular_section(root, "profit_and_loss", "annual_data") or \
            self._tabular_section(root, "annualResults")
        if not rows:
            return []
        ycols = self._year_columns(rows)
        return self._pivot_table(rows, ticker, INCOME_ANNUAL_FIELDS, ycols)

    def _parse_quarterly(self, root: dict, ticker: str) -> list[dict]:
        rows = self._tabular_section(root, "quarters") or \
            self._tabular_section(root, "quarterlyResults")
        if not rows:
            return []
        ycols = self._year_columns(rows)
        label_to_col = {}
        for col, cands in INCOME_QUARTERLY_FIELDS.items():
            for c in cands:
                label_to_col[norm_key(c)] = col
        collected: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            metric = row.get("Metric")
            if metric is None:
                continue
            col = label_to_col.get(norm_key(metric))
            if col is None:
                continue
            for yc in ycols:
                val = to_float(row.get(yc))
                if val is None:
                    continue
                if yc not in collected:
                    collected[yc] = {"ticker": ticker, "quarter_label": str(yc)}
                collected[yc][col] = val
        out = [collected[yc] for yc in ycols if yc in collected]
        for i, r in enumerate(out, start=1):
            r["quarter_index"] = i
        return out

    def _parse_balance(self, root: dict, ticker: str) -> list[dict]:
        rows = self._tabular_section(root, "balance_sheet") or \
            self._tabular_section(root, "balanceSheet")
        if not rows:
            return []
        ycols = self._year_columns(rows)
        return self._pivot_table(rows, ticker, BALANCE_FIELDS, ycols)

    def _parse_cashflow(self, root: dict, ticker: str) -> list[dict]:
        rows = self._tabular_section(root, "cash_flow") or \
            self._tabular_section(root, "cashFlow")
        if not rows:
            return []
        ycols = self._year_columns(rows)
        return self._pivot_table(rows, ticker, CASHFLOW_FIELDS, ycols)

    def _parse_dividends(self, root: dict, ticker: str) -> list[dict]:
        rows = self._tabular_section(root, "dividends") or \
            self._tabular_section(root, "dividend")
        out: list[dict] = []
        for d in rows:
            if not isinstance(d, dict):
                continue
            lookup = build_lookup(d)
            dt = extract_field(d, ["exDate", "exDividendDate", "date", "recordDate", "paymentDate"], lookup)
            amt = to_float(extract_field(d, ["amount", "dividend", "dividendAmount", "dividendPerShare", "dps"], lookup))
            if dt is not None and amt is not None:
                out.append({"ticker": ticker, "ex_date": str(dt), "dividend_amount": amt})
        return out

    def _parse_roe(self, root: dict) -> dict[str, float | None]:
        """Read Return on Equity from ``profit_and_loss['Return on Equity']``.

        The Screener actor exposes ROE as a list of ``{period: value}`` dicts
        (``Last Year:``, ``3 Years:``, ``5 Years:``, ``10 Years:``) — *not* as a
        top-level or ratios-section field. Returns a mapping of the normalised
        period -> value (all coerced to float, missing -> None).
        """
        pl = root.get("profit_and_loss") if isinstance(root, dict) else None
        if isinstance(pl, dict):
            node = pl.get("Return on Equity")
        else:
            node = None
        out: dict[str, float | None] = {}
        if isinstance(node, list):
            for item in node:
                if not isinstance(item, dict):
                    continue
                for k, v in item.items():
                    period = str(k).replace(":", "").strip().lower().replace(" ", "_")
                    out[period] = to_float(v)
        return out

    def _parse_ratios(self, root: dict, ticker: str) -> dict:
        rows = self._tabular_section(root, "ratios") or \
            self._tabular_section(root, "ratioHistory")
        row: dict[str, Any] = {"ticker": ticker}
        yearly_rows: list[dict] = []
        if rows:
            ycols = self._year_columns(rows)
            # RATIO_FIELDS maps storage column -> candidate Metric labels; for the
            # tabular shape the *keys* are the metric labels, so invert per row.
            label_to_col = {str(v): k for k, vs in RATIO_FIELDS.items() for v in vs}
            for yr in sorted(
                {self._year_from_col(c) for c in ycols if self._year_from_col(c)}
            ):
                r = {"ticker": ticker, "financial_year": yr}
                for mrow in rows:
                    metric = mrow.get("Metric")
                    col = label_to_col.get(str(metric).strip()) if metric else None
                    if col is None:
                        continue
                    val = to_float(mrow.get(f"Mar {yr}")) if f"Mar {yr}" in mrow \
                        else to_float(mrow.get(str(yr)))
                    if val is not None:
                        r[col] = val
                if len(r) > 2:
                    yearly_rows.append(r)
        # ROE lives under profit_and_loss -> 'Return on Equity' (Last/3/5/10 Years).
        # The Screener actor reports ROE as trailing averages, so map each period
        # to its corresponding ratio year by recency position: the most recent
        # year gets 'Last Year', the 3rd most recent gets '3 Years', etc.
        roe_map = self._parse_roe(root)
        if roe_map and yearly_rows:
            period_offsets = {
                "last_year": 0,
                "3_years": 2,
                "5_years": 4,
                "10_years": 9,
            }
            for period, offset in period_offsets.items():
                val = roe_map.get(period)
                if val is None:
                    continue
                idx = len(yearly_rows) - 1 - offset
                if 0 <= idx < len(yearly_rows):
                    yearly_rows[idx]["roe"] = val
        elif roe_map.get("last_year") is not None:
            # No tabular ratio years; carry ROE on a single snapshot row.
            row["roe"] = roe_map["last_year"]
        row["_yearly"] = yearly_rows
        return row

    @staticmethod
    def _ticker_from_url(url: str) -> str:
        m = re.search(r"/company/([^/]+)", str(url))
        sym = m.group(1) if m else str(url)
        sym = sym.split("/")[0].strip().upper()
        if not sym.endswith(".NS"):
            sym = f"{sym}.NS"
        return sym
