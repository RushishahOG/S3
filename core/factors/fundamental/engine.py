"""Quality Factor feature engineering (Screener data pipeline).

Computes the 16 supported Quality factors from the normalised ``fundamentals_*``
tables and persists them into ``fundamental_quality_features``. This is a
dedicated, precomputed pipeline (like Momentum / Low Volatility): features are
NEVER calculated on dashboard load.

Raw Data -> DB Storage -> Feature Engineering -> Engineered Tables -> Strategy

Design
------
* Each factor is computed by a standalone helper operating on one stock's
  yearly frames, returning a ``financial_year``-indexed Series/DataFrame.
* Growth factors additionally expose **Median** and **Weighted Average**
  roll-ups (recency-weighted) computed across the available history, matching
  the Momentum-engine rolling-lookback contract (latest / median / weighted).
* Missing previous-year values -> NULL (never substituted with zero); the stock
  is simply excluded from that factor's ranking input.
* Interest expense == 0 -> NULL for ICR. Quarterly EPS missing -> skip Q EPS growth.

Extensibility: add a ``compute_*`` helper and register it in
:data:`QUALITY_FACTOR_FUNCTIONS`.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from core.data.storage.storage_manager import StorageManager
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #
def _yr(df: pd.DataFrame) -> pd.DataFrame:
    """Sort a frame by financial_year ascending and return indexed by year."""
    d = df.copy()
    d["financial_year"] = pd.to_numeric(d["financial_year"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["financial_year"]).sort_values("financial_year")
    return d.set_index("financial_year")


def _safe_div(num, den):
    """Element-wise divide guarding zero/NA -> NULL."""
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    return (num / den.where(den != 0)).where(den.notna() & (den != 0))


def _yoy_growth(series: pd.Series) -> pd.Series:
    """Year-on-year growth of a per-year series. Previous-year missing -> NULL."""
    s = pd.to_numeric(series, errors="coerce").sort_index()
    prev = s.shift(1)
    g = (s - prev) / prev.abs()
    return g.where(prev.notna() & (prev != 0) & g.abs() != float("inf"))


def _weighted_mean(series: pd.Series) -> float | None:
    """Recency-weighted mean: most recent year has the highest weight."""
    s = series.dropna()
    if s.empty:
        return None
    weights = pd.Series(range(1, len(s) + 1), index=s.index)
    return float((s * weights).sum() / weights.sum())


def _median(vals: Iterable[float]) -> float | None:
    s = pd.Series(list(vals)).dropna()
    return float(s.median()) if not s.empty else None


# --------------------------------------------------------------------------- #
# Per-factor computations (one stock, yearly frames)                         #
# --------------------------------------------------------------------------- #
def compute_roe(income: pd.DataFrame, balance: pd.DataFrame) -> pd.Series:
    """ROE = Net Profit / Shareholders' Equity (year-end).

    Shareholders' Equity = Equity Capital + Reserves (balance sheet). Net
    Profit is taken from the Annual Income Statement; both are aligned by
    financial year.
    """
    if income is None or balance is None or income.empty or balance.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    bal = _yr(balance)
    net = inc["net_profit"] if "net_profit" in inc else pd.Series(None, index=inc.index)
    equity = bal["equity_capital"].fillna(0) + bal["reserves"].fillna(0)
    return _safe_div(net, equity).rename("roe")


def compute_roce(ratios: pd.DataFrame, years=None) -> pd.Series:
    if ratios is None or ratios.empty or "roce" not in ratios:
        return pd.Series(dtype="float64")
    s = _yr(ratios)["roce"]
    if years is not None and len(s) == 1:
        return pd.Series([s.iloc[0]] * len(years), index=list(years), name="roce")
    return s.rename("roce")


def compute_roa(income: pd.DataFrame, balance: pd.DataFrame) -> pd.Series:
    """ROA = Net Profit / Total Assets (same financial year)."""
    if income is None or balance is None or income.empty or balance.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    bal = _yr(balance)
    net = inc["net_profit"] if "net_profit" in inc else pd.Series(None, index=inc.index)
    assets = bal["total_assets"] if "total_assets" in bal else pd.Series(None, index=bal.index)
    return _safe_div(net, assets).rename("roa")


def compute_interest_coverage_ratio(income: pd.DataFrame) -> pd.Series:
    """ICR = EBIT (operating_profit) / Interest Expense."""
    if income is None or income.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    ebit = inc["operating_profit"] if "operating_profit" in inc else pd.Series(None, index=inc.index)
    ie = inc["interest"] if "interest" in inc else pd.Series(None, index=inc.index)
    return _safe_div(ebit, ie).rename("interest_coverage_ratio")


def compute_equity_to_total_capital(balance: pd.DataFrame) -> pd.Series:
    """(Equity Capital + Reserves) / (Equity Capital + Reserves + Borrowings)."""
    if balance is None or balance.empty:
        return pd.Series(dtype="float64")
    bal = _yr(balance)
    eq = bal["equity_capital"].fillna(0) + bal["reserves"].fillna(0)
    cap = eq + bal["borrowings"].fillna(0)
    return _safe_div(eq, cap).rename("equity_to_total_capital")


def compute_dividend_payout_ratio(income: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    """Dividend amount = dividend_payout_percent x Net Profit (year-wise);
    payout ratio = Dividend Amount / Net Profit."""
    if income is None or income.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    net = inc["net_profit"] if "net_profit" in inc else pd.Series(None, index=inc.index)
    if "dividend_payout_percent" in inc and inc["dividend_payout_percent"].notna().any():
        pct = inc["dividend_payout_percent"]
        # Tolerate both 0-1 and 0-100 encodings.
        pct = pct.where(pct <= 1.0, pct / 100.0)
        div_amount = pct * net
        return _safe_div(div_amount, net).rename("dividend_payout_ratio")
    # Fallback: derive payout % from DPS / EPS, then dividend amount = payout x net.
    eps = inc["eps"] if "eps" in inc else pd.Series(None, index=inc.index)
    dps = _yearly_dps(dividends, inc.index)
    payout = _safe_div(dps, eps)
    div_amount = payout * net
    return _safe_div(div_amount, net).rename("dividend_payout_ratio")


def compute_dividend_payout_ratio_cumulative(income: pd.DataFrame, dividends: pd.DataFrame) -> pd.Series:
    """Rolling cumulative payout = cumsum(Dividend Amount) / cumsum(Net Profit).

    Dividend Amount per year = dividend_payout_percent x Net Profit. The value
    evolves each year as the cumulative dividend total is divided by the
    cumulative net profit up to that year.
    """
    if income is None or income.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    net = inc["net_profit"] if "net_profit" in inc else pd.Series(None, index=inc.index)
    if "dividend_payout_percent" in inc and inc["dividend_payout_percent"].notna().any():
        pct = inc["dividend_payout_percent"]
        pct = pct.where(pct <= 1.0, pct / 100.0)
        div_amount = pct * net
    else:
        eps = inc["eps"] if "eps" in inc else pd.Series(None, index=inc.index)
        dps = _yearly_dps(dividends, inc.index)
        payout = _safe_div(dps, eps)
        div_amount = payout * net
    cum_div = div_amount.fillna(0).cumsum()
    cum_net = net.fillna(0).cumsum()
    return _safe_div(cum_div, cum_net).rename("dividend_payout_ratio_cumulative")


def compute_ocf_to_ebitda(income: pd.DataFrame, cashflow: pd.DataFrame) -> pd.Series:
    """Operating Cash Flow / EBITDA (EBITDA = operating_profit + depreciation)."""
    if income is None or cashflow is None or income.empty or cashflow.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    cf = _yr(cashflow)
    ebitda = inc["operating_profit"].fillna(0) + inc["depreciation"].fillna(0)
    ocf = cf["operating_cash_flow"] if "operating_cash_flow" in cf else pd.Series(None, index=cf.index)
    return _safe_div(ocf, ebitda).rename("ocf_to_ebitda")


def compute_cash_roce(balance: pd.DataFrame, cashflow: pd.DataFrame) -> pd.Series:
    """OCF / Average Capital Employed (Equity + Reserves + Borrowings)."""
    if balance is None or cashflow is None or balance.empty or cashflow.empty:
        return pd.Series(dtype="float64")
    bal = _yr(balance)
    cf = _yr(cashflow)
    ce = bal["equity_capital"].fillna(0) + bal["reserves"].fillna(0) + bal["borrowings"].fillna(0)
    avg_ce = (ce + ce.shift(1)) / 2
    ocf = cf["operating_cash_flow"] if "operating_cash_flow" in cf else pd.Series(None, index=cf.index)
    return _safe_div(ocf, avg_ce).rename("cash_roce")


def compute_eps_growth(income: pd.DataFrame) -> pd.Series:
    if income is None or income.empty or "eps" not in income:
        return pd.Series(dtype="float64")
    return _yoy_growth(_yr(income)["eps"]).rename("eps_growth")


def compute_roe_growth(income: pd.DataFrame, balance: pd.DataFrame) -> pd.Series:
    """Year-on-year growth of the computed ROE (Net Profit / Shareholders' Equity)."""
    roe = compute_roe(income, balance)
    if roe is None or roe.dropna().empty:
        return pd.Series(dtype="float64")
    return _yoy_growth(roe).rename("roe_growth")


def compute_roce_growth(ratios: pd.DataFrame, years=None) -> pd.Series:
    if ratios is None or ratios.empty or "roce" not in ratios:
        return pd.Series(dtype="float64")
    s = _yr(ratios)["roce"]
    if years is not None and len(s) == 1:
        return pd.Series([None] * len(years), index=list(years), name="roce_growth")
    return _yoy_growth(s).rename("roce_growth")


def compute_revenue_growth(income: pd.DataFrame) -> pd.Series:
    if income is None or income.empty or "sales" not in income:
        return pd.Series(dtype="float64")
    return _yoy_growth(_yr(income)["sales"]).rename("revenue_growth")


def compute_dps_growth(dividends: pd.DataFrame, income: pd.DataFrame) -> pd.Series:
    """Year-on-year growth of DPS, where DPS = EPS x Dividend Payout Ratio."""
    if income is None or income.empty:
        return pd.Series(dtype="float64")
    inc = _yr(income)
    dps = _yearly_dps_from_eps(income, dividends)
    return _yoy_growth(dps).rename("dps_growth")


def _yearly_dps_from_eps(income: pd.DataFrame, dividends: pd.DataFrame = None) -> pd.Series:
    """Per-year DPS derived as EPS x Dividend Payout Ratio.

    Payout ratio is taken from ``dividend_payout_percent`` (normalized to 0-1)
    when present; otherwise it falls back to actual DPS / EPS so the formula
    degrades gracefully.
    """
    inc = _yr(income)
    eps = inc["eps"] if "eps" in inc else pd.Series(None, index=inc.index)
    if "dividend_payout_percent" in inc and inc["dividend_payout_percent"].notna().any():
        pct = inc["dividend_payout_percent"]
        pct = pct.where(pct <= 1.0, pct / 100.0)
    else:
        pct = _safe_div(_yearly_dps(dividends, inc.index), eps)
    return (eps * pct).rename("dps")


def compute_sustainable_growth_rate(income: pd.DataFrame, balance: pd.DataFrame, dividends: pd.DataFrame, ratios: pd.DataFrame, years=None) -> pd.Series:
    """SGR = ROE x (1 - Dividend Payout Ratio)."""
    roe = compute_roe(income, balance)
    payout = compute_dividend_payout_ratio(income, dividends)
    df = pd.concat([roe, payout], axis=1)
    return (df["roe"] * (1.0 - df["dividend_payout_ratio"])).where(
        df["roe"].notna() & df["dividend_payout_ratio"].notna()
    ).rename("sustainable_growth_rate")


# --------------------------------------------------------------------------- #
# Internal utilities                                                          #
# --------------------------------------------------------------------------- #
def _yearly_dps(dividends: pd.DataFrame, years) -> pd.Series:
    """Aggregate dividend amounts into a per-year Series aligned to ``years``."""
    if dividends is None or dividends.empty:
        return pd.Series(None, index=pd.Index(years, name="financial_year"), dtype="float64")
    d = dividends.copy()
    d["year"] = pd.to_datetime(d["ex_date"], errors="coerce").dt.year
    d = d.dropna(subset=["year"])
    per_year = d.groupby("year")["dividend_amount"].sum(min_count=1)
    per_year.index = per_year.index.astype("int64")
    return per_year.reindex(pd.Index(years)).rename("dps")


QUALITY_FACTOR_FUNCTIONS = {
    "roe": compute_roe,
    "roce": compute_roce,
    "roa": compute_roa,
    "interest_coverage_ratio": compute_interest_coverage_ratio,
    "equity_to_total_capital": compute_equity_to_total_capital,
    "dividend_payout_ratio": compute_dividend_payout_ratio,
    "dividend_payout_ratio_cumulative": compute_dividend_payout_ratio_cumulative,
    "ocf_to_ebitda": compute_ocf_to_ebitda,
    "cash_roce": compute_cash_roce,
    "eps_growth": compute_eps_growth,
    "roe_growth": compute_roe_growth,
    "roce_growth": compute_roce_growth,
    "revenue_growth": compute_revenue_growth,
    "dps_growth": compute_dps_growth,
    "sustainable_growth_rate": compute_sustainable_growth_rate,
}


# --------------------------------------------------------------------------- #
# Engine                                                                      #
# --------------------------------------------------------------------------- #
class FundamentalQualityEngine:
    """Computes + persists the 16 Quality factors from the screener tables."""

    GROWTH_FACTORS = {
        "eps_growth": ("eps_growth_median", "eps_growth_weighted"),
        "roe_growth": ("roe_growth_median", "roe_growth_weighted"),
        "roce_growth": ("roce_growth_median", "roce_growth_weighted"),
        "revenue_growth": ("revenue_growth_median", "revenue_growth_weighted"),
        "dps_growth": ("dps_growth_median", None),
    }

    def __init__(self, storage: StorageManager | None = None) -> None:
        self.storage = storage or StorageManager()

    # -- public -------------------------------------------------------------
    def compute(
        self,
        tickers: list[str] | None = None,
        store: bool = True,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        if tickers is None:
            tickers = sorted(self.storage.tickers_with_screener_data())
        if not tickers:
            logger.warning("No screener data available to engineer quality factors.")
            return pd.DataFrame()

        selected = set(features) if features else set(QUALITY_FACTOR_FUNCTIONS)
        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            rows = self._compute_ticker(ticker, selected)
            if rows is not None and not rows.empty:
                frames.append(rows)

        if not frames:
            logger.warning("No quality feature rows produced.")
            return pd.DataFrame()
        result = pd.concat(frames, ignore_index=True)
        if store:
            self.storage.upsert_fundamental_quality_features(result)
            logger.info("Engineered %d quality factor rows across %d tickers", len(result), len(frames))
        return result

    # -- per-ticker ---------------------------------------------------------
    def _compute_ticker(self, ticker: str, selected: set[str]) -> pd.DataFrame | None:
        income = self.storage.get_fundamentals_income_annual([ticker])
        balance = self.storage.get_fundamentals_balance_sheet([ticker])
        cashflow = self.storage.get_fundamentals_cashflow([ticker])
        dividends = self.storage.get_fundamentals_dividends([ticker])
        ratios = self.storage.get_fundamentals_ratios([ticker])
        quarterly = self.storage.get_fundamentals_income_quarterly([ticker])

        years = (
            pd.to_numeric(income["financial_year"], errors="coerce").dropna().astype("int").tolist()
            if income is not None and not income.empty
            else []
        )
        if not years:
            return None

        base = pd.DataFrame({"financial_year": years}).set_index("financial_year")

        series: dict[str, pd.Series] = {}
        # Factors needing only one frame.
        if "roe" in selected:
            series["roe"] = compute_roe(income, balance)
        if "roce" in selected:
            series["roce"] = compute_roce(ratios, years)
        if "roa" in selected:
            series["roa"] = compute_roa(income, balance)
        if "interest_coverage_ratio" in selected:
            series["interest_coverage_ratio"] = compute_interest_coverage_ratio(income)
        if "equity_to_total_capital" in selected:
            series["equity_to_total_capital"] = compute_equity_to_total_capital(balance)
        if "dividend_payout_ratio" in selected:
            series["dividend_payout_ratio"] = compute_dividend_payout_ratio(income, dividends)
        if "dividend_payout_ratio_cumulative" in selected:
            series["dividend_payout_ratio_cumulative"] = compute_dividend_payout_ratio_cumulative(income, dividends)
        if "ocf_to_ebitda" in selected:
            series["ocf_to_ebitda"] = compute_ocf_to_ebitda(income, cashflow)
        if "cash_roce" in selected:
            series["cash_roce"] = compute_cash_roce(balance, cashflow)
        if "eps_growth" in selected:
            series["eps_growth"] = compute_eps_growth(income)
        if "roe_growth" in selected:
            series["roe_growth"] = compute_roe_growth(income, balance)
        if "roce_growth" in selected:
            series["roce_growth"] = compute_roce_growth(ratios, years)
        if "revenue_growth" in selected:
            series["revenue_growth"] = compute_revenue_growth(income)
        if "dps_growth" in selected:
            series["dps_growth"] = compute_dps_growth(dividends, income)
        if "sustainable_growth_rate" in selected:
            series["sustainable_growth_rate"] = compute_sustainable_growth_rate(income, balance, dividends, ratios, years)

        out = base
        for name, s in series.items():
            if s is None or s.empty:
                continue
            s = s.rename(name) if s.name != name else s
            out = out.join(s, how="left")

        # Growth roll-ups: median + weighted, broadcast across all year rows.
        for gname, (med_col, wt_col) in self.GROWTH_FACTORS.items():
            if gname in selected and gname in out.columns:
                vals = out[gname].dropna()
                med = _median(vals)
                wt = _weighted_mean(vals)
                if med_col:
                    out[med_col] = med
                if wt_col:
                    out[wt_col] = wt

        out = out.reset_index()
        out.insert(0, "ticker", ticker)
        return out

    def compute_ticker(self, ticker: str, store: bool = False, features: list[str] | None = None) -> pd.DataFrame:
        return self.compute(tickers=[ticker], store=store, features=features)
