---
- task: Fundamental Data Integration Update (Screener API) — major refactor
- context: >
    Replacing the two Apify actors (financials + ratios) with a single Screener
    actor (XargjiJ7dQUj2q2Bx) that returns complete financial history per company
    URL. New normalized schema (7 tables), 16 quality factors with rolling
    lookback, dry-run via uploaded CSV, bulk controls, validation dashboard.
- decisions:
    - Keep legacy `fundamental_*` tables + `ApifyFinancialProvider`/`ApifyRatioProvider`
      + existing `fundamental_features` + tests intact (no breakage).
    - NEW screener flow is additive: new tables `fundamentals_*` (company, income
      annual, income quarterly, balance_sheet, cashflow, dividends, ratios),
      new provider `ApifyScreenerProvider`, new downloader `ScreenerDownloader`,
      new feature module `core/factors/fundamental/`, new dashboard page.
    - Screener URL is the single identifier; symbol derived from CSV Symbol column
      (bare, e.g. 360ONE) then resolved to 360ONE.NS for storage key consistency.
    - Extraction window REMOVED from screener flow (actor returns full history).
    - Feature module lives at `core/factors/fundamental/` (repo uses `core/`, not
      `backend/` as spec literally says).
    - Provider parsing is schema-tolerant (candidate keys) since exact JSON shape
      of new actor is unknown; tests use synthetic screener-style payload.
- links:
    - config/providers.yaml
    - core/data/providers/apify_screener_provider.py
    - core/data/ingestion/screener_downloader.py
    - core/factors/fundamental/
    - app/pages/fundamental_downloader.py (rewrite)
    - app/pages/fundamental_dashboard.py (new)
    - nifty_500_constituents/nifty500_screener_urls.csv (Company Name,Symbol,Screener URL)
