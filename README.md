# Smart Beta Research Platform

> **Version 1** — a modular, layered quantitative research platform (not a
> trading app). Built to be the primary codebase for all future factor
> research, portfolio construction, and machine-learning milestones.

## Features (V1)

- **Provider-abstracted data ingestion** — Yahoo Finance shipped; swap in Alpha
  Vantage / Polygon / NSE by implementing one interface.
- **Competition backtesting window** — fixed, configurable global constraint:
  `2006-01-01` → `2026-05-31` (end date non-configurable). All data ingestion,
  feature engineering and factor calculations are clamped to this window; user
  start-date input is validated and cannot precede the minimum.
- **Universe management** — Version 1 operates exclusively on the **NIFTY 500**
  constituents (via a generic `UniverseManager` + `NIFTY500Universe` provider),
  with NIFTY 50 as the Beta benchmark. Multi-universe / custom uploads are
  deferred but architecturally supported.
- **Historical data ingestion** — modular `HistoricalDownloader` reads the
  official NIFTY 500 constituent file, resolves Yahoo symbols via an external
  mapping, downloads **ticker-by-ticker** with retries + rate-limit delays,
  validates, stores to DuckDB, and produces structured download reports
  (success / failed / statistics) exportable as CSV.
- **Feature engineering pipeline** — returns, log returns, rolling returns,
  rolling volatility, semi-deviation, beta, rolling momentum.
- **Reusable Feature Store** — wide `(ticker, date)` schema; new factors simply
  add columns.
- **Momentum factor** — 3/6/9/12 month horizons with lag & scaled variants.
- **Low Volatility factor** — rolling std-dev & semi-deviation over
  6/12/24/36/48 month windows; **beta computed at 3/6/9/12 month horizons**
  (configurable via `low_volatility.beta_windows_months`).
- **Interactive dashboard** — Dashboard, Data Manager, Dataset Explorer,
  **Eligibility Analyzer**, Developer Logs.
- **Eligibility Analyzer** — derives the earliest valid backtest start date from
  *actual data availability* rather than a fixed constant. For every NIFTY 500
  constituent it computes the first trading date and the first eligible date
  (first trading + the maximum factor lookback), builds a monthly eligibility
  timeline with universe coverage %, and recommends the earliest rebalance date
  that clears a configurable coverage threshold (default 80%). Factor
  lookbacks are sourced from a central registry, so new factors (Quality,
  Value, Growth, ...) join the framework automatically.

## Quickstart

```bash
pip install streamlit pandas numpy scipy pyarrow duckdb yfinance pyyaml
streamlit run main.py
```

Then, in the UI:

1. **Data Manager** → *Download / Refresh* (downloads NIFTY 500; *Complete
   Refresh* re-downloads all).
2. **Feature Engineering** → select factor categories → *Generate Features*.
3. **Factor Explorer** → inspect rankings, distributions and historical
   evolution.

> Version 1 is scoped to the **NIFTY 500** universe only. Constituents are read
> from `nifty_500_constituents/ind_nifty500list_2026.csv` at runtime; symbol
> resolution (e.g. renamed companies) is configured in
> `config/ticker_mapping.yaml`.

## Project Layout

```
app/        Streamlit presentation (pages, components, layouts)
core/       Business logic: data, features, factors, engine, analytics, viz
config/     settings.yaml + universe CSVs
storage/    Generated DuckDB / parquet / logs (git-ignored)
tests/      Unit + smoke tests
docs/       Architecture documentation
```

See [`docs/architecture.md`](docs/architecture.md) for the full design and the
guide to adding providers, factors and universes.

## Testing

```bash
python -m unittest tests.test_features tests.test_universe tests.test_constraints tests.test_ingestion
python -m tests.smoke_pipeline                                # end-to-end (synthetic data)
python -m tests.test_ingestion                                # ingestion pipeline (fake provider)
```

## Next Milestones

Composite scoring, smart-beta portfolio construction, backtesting, simulations,
persona-based optimisation, efficient-frontier analysis, Gold/Debt ETF
integration, multi-asset allocation, and ML models — all fit the existing
layers without architectural change.
