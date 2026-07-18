"""One-off re-ingest: re-parse local Screener raw JSONs into fundamentals_* tables.

Re-parses every stored raw response (no API calls) using the current parser so
that updates like the multi-year ROE mapping land in the live DB.
"""
from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

from core.data.providers.apify_screener_provider import ApifyScreenerProvider
from core.data.storage.storage_manager import StorageManager

RAW_DIR = "storage/raw/apify"
RAW_GLOB = os.path.join(RAW_DIR, "XargjiJ7dQUj2q2Bx_*.json")


def _url_from_payload(it: dict) -> str:
    try:
        return it.get("payload", {}).get("url", "") or ""
    except Exception:
        return ""


def main() -> None:
    storage = StorageManager()
    provider = ApifyScreenerProvider()
    files = sorted(glob.glob(RAW_GLOB))
    print(f"Found {len(files)} raw files")

    stats = {"company": 0, "income": 0, "qtr": 0, "bal": 0, "cf": 0, "div": 0, "rat": 0}
    seen_tickers = set()
    for f in files:
        try:
            it = json.load(open(f, encoding="utf-8"))
        except Exception as exc:
            print(f"SKIP {os.path.basename(f)}: {exc}")
            continue
        url = _url_from_payload(it)
        if not url:
            continue
        res = provider._parse(it, url)
        t = res.ticker
        if t in seen_tickers:
            continue  # latest raw file per ticker wins; skip older duplicates
        seen_tickers.add(t)

        if res.company:
            c = dict(res.company)
            c["url"] = url
            storage.upsert_fundamentals_company(pd.DataFrame([c]))
            stats["company"] += 1
        if res.income_annual:
            storage.upsert_fundamentals_income_annual(pd.DataFrame(res.income_annual))
            stats["income"] += 1
        if res.income_quarterly:
            storage.upsert_fundamentals_income_quarterly(pd.DataFrame(res.income_quarterly))
            stats["qtr"] += 1
        if res.balance_sheet:
            storage.upsert_fundamentals_balance_sheet(pd.DataFrame(res.balance_sheet))
            stats["bal"] += 1
        if res.cashflow:
            storage.upsert_fundamentals_cashflow(pd.DataFrame(res.cashflow))
            stats["cf"] += 1
        if res.dividends:
            storage.upsert_fundamentals_dividends(pd.DataFrame(res.dividends))
            stats["div"] += 1
        if res.ratios:
            yearly = res.ratios.pop("_yearly", [])
            if yearly:
                yearly_df = pd.DataFrame(yearly)
                key = ["ticker", "financial_year"]
                val_cols = [c for c in yearly_df.columns if c not in key]
                if val_cols:
                    yearly_df = yearly_df[yearly_df[val_cols].notna().any(axis=1)]
                if not yearly_df.empty:
                    storage.upsert_fundamentals_ratios(yearly_df)
                    stats["rat"] += 1
            else:
                snap = dict(res.ratios)
                snap["ticker"] = t
                snap["financial_year"] = 0
                snap = {k: v for k, v in snap.items() if pd.notna(v)}
                if snap:
                    storage.upsert_fundamentals_ratios(pd.DataFrame([snap]))
                    stats["rat"] += 1

    print("Re-ingested tickers:", len(seen_tickers))
    print("Per-table ticker counts:", stats)


if __name__ == "__main__":
    main()
