# Smart Beta Research Platform — Architecture

Version 1 of a modular, layered **quantitative research platform**. The system
is intentionally *not* a trading application: it is an extensible foundation for
factor research, with portfolio construction / optimisation / ML deliberately
deferred to later milestones.

---

## 1. Layered Architecture

```
Presentation Layer (Streamlit)        app/pages, app/components, app/layouts
                │
                ▼
Application Services                  core/data/market_data_manager.py
                │
                ▼
Research / Factor Engine              core/factor_engine.py
                │
                ▼
Factor Framework                      core/factors/*
                │
                ▼
Feature Engineering                   core/features/*
                │
                ▼
Market Data Layer                     core/data/providers, universe, cache
                │
                ▼
Local Storage / Cache                 core/data/storage (DuckDB)
```

Every layer has a single responsibility and depends only on the layer beneath
it. **No business logic is coupled to Streamlit** — the UI only calls service
objects that live in `core.*`.

---

## 2. Module Map

| Path | Responsibility |
|------|----------------|
| `app/` | Streamlit presentation only (pages, components, layouts). |
| `core/config/` | Typed settings loaded from `config/settings.yaml`. No hardcoded constants elsewhere. |
| `core/utils/` | Paths, logging configuration, retry decorator, date helpers. |
| `core/data/providers/` | Vendor abstraction (`BaseDataProvider`) + Yahoo Finance impl + registry. |
| `core/data/storage/` | `StorageManager` — DuckDB persistence of prices & feature store. |
| `core/data/cache/` | `CacheManager` — incremental download plans & gap detection. |
| `core/data/universe/` | `UniverseManager` + `BaseUniverseProvider`/`NIFTY500Universe` (V1: NIFTY 500 only). |
| `core/data/market_data_manager.py` | Orchestrates the ingestion pipeline + clean retrieval/validation. |
| `core/data/ingestion/` | Constituents loader, `TickerResolver`, `HistoricalDownloader`, validation, `DownloadReport`. |
| `core/features/` | Reusable primitives (returns, volatility, beta, rolling) + `FeatureStore`. |
| `core/factors/` | `BaseFactor` interface, registry, momentum & low_volatility factors. |
| `core/factor_engine.py` | Builds `PricePanel`, runs factors, persists to Feature Store. |
| `core/analytics/` | Cross-sectional ranks / composite scores. |
| `core/visualization/` | Decoupled Altair chart builders. |
| `config/` | `settings.yaml` + universe CSVs. |
| `storage/` | Generated DuckDB DB, parquet exports, custom universes, logs (git-ignored). |
| `tests/` | Unit + smoke tests. |

---

## 3. Key Design Decisions

### Data Provider Abstraction
Factors and downstream code **never** import `yfinance`. They receive a
`BaseDataProvider` whose only contract is `fetch_prices(tickers, start, end)`
returning a long frame `[ticker, date, open, high, low, close, adj_close,
volume]`. Adding Alpha Vantage / Polygon / NSE means implementing this class
and calling `register()` — nothing else changes.

### Local Storage & Incremental Cache
`StorageManager` uses **DuckDB**. `CacheManager` computes, per ticker, the
missing date range so re-downloads only fetch deltas (from *last stored date +
1* to *today*). `detect_missing_dates` surfaces business-day gaps for
validation.

### Market Data Ingestion Layer
The `HistoricalDownloader` (`core/data/ingestion/downloader.py`) is the
permanent V1 ingestion pipeline. It is intentionally modularised:

```
Load constituents -> Resolve symbols -> Download each individually
-> Validate -> Store -> Log failures -> Generate report
```

- **Constituents** are read from the official file
  (`nifty_500_constituents/ind_nifty500list_2026.csv`) — never hardcoded.
- **TickerResolver** converts a base symbol to a provider ticker (appends
  `.NS`) using an external mapping (`config/ticker_mapping.yaml`); renamed
  companies / other exchanges are overrides, not code changes.
- Downloads happen **ticker-by-ticker** with configurable retries (default 3),
  a small randomised delay (rate-limit friendly), and continue-on-failure so a
  bad ticker never blocks the rest.
- **Validation** (`validation.py`) checks required columns, removes duplicate
  `(ticker, date)` rows, sorts chronologically and flags missing trading days.
- **Storage** persists normalised OHLCV to DuckDB plus per-ticker
  `download_metadata` (provider, status, rows, earliest/latest, retries,
  error) and `validation_anomalies`.
- A structured **DownloadReport** (summary / success / failed / statistics,
  with company name, symbols, status, rows, dates, error, retries) is returned
  and rendered + CSV-exported in the Data Manager UI.

### Feature Store (wide schema)
The feature store is a wide table keyed by `(ticker, date)`. Factors declare
their output columns; when a column does not yet exist, the store issues
`ALTER TABLE ... ADD COLUMN`. **Adding a factor therefore only adds new
columns** — existing data is preserved.

### Factor Framework
`BaseFactor` requires only `feature_specs()` and `compute(panel)`. A factor
receives a `PricePanel` (prices + benchmark returns) and returns a long frame
`[ticker, date, <columns>]`. The engine merges all factor outputs and persists
them. The registry auto-discovers factors so new modules plug in with zero core
changes.

---

## 4. Extending the Platform

### Add a new data provider
```python
from core.data.providers.base_provider import BaseDataProvider
from core.data.providers.registry import register

class AlphaVantageProvider(BaseDataProvider):
    name = "alpha_vantage"
    def fetch_prices(self, tickers, start, end): ...

register(AlphaVantageProvider.name, AlphaVantageProvider)
```

### Add a new factor (e.g. Quality)
1. Create `core/factors/quality/quality_factor.py` subclassing `BaseFactor`.
2. Implement `feature_specs()` and `compute(panel)`.
3. Register the instance in `core/factors/registry.py`.
4. (Optional) add config knobs to `config/settings.yaml`.

No change to the engine, UI, or storage is required — the new columns appear
automatically in the Feature Store and Factor Explorer.

### Add a new universe
Implement a `BaseUniverseProvider` subclass (e.g. `NIFTY50Universe`,
`Midcap150Universe`, a custom-CSV provider, or an ETF provider), then register
it in `UniverseManager._register_defaults()`. The UI stays locked to the
default universe, so no page changes are needed.

---

## 4. Competition Backtesting Window

The platform enforces a single, fixed, configurable backtest window declared
once in `config/settings.yaml` under `backtest:`:

```
backtest:
  min_date: "2006-01-01"
  max_date: "2026-05-31"   # FIXED, non-configurable in the UI
  default_start: "2006-01-01"
```

These are exposed as global constants in `core/utils/dates.py`
(`MIN_BACKTEST_DATE`, `MAX_BACKTEST_DATE`) and a `validate_backtest_range()` /
`clamp_to_bounds()` helper. Every layer honours them:

- **Data Manager** — the end date is always clamped to `MAX_BACKTEST_DATE`;
  users pick only a start date (>= `MIN_BACKTEST_DATE`) via a bounded
  `date_input`, and invalid input raises `DateRangeError`.
- **Factor Engine** — `build_panel` clamps the requested range into the window.
- **Feature store / factors / future portfolio & simulations** — all derive
  their window from these constants, so nothing can exceed 2026-05-31.

Changing competition rules later means editing `settings.yaml` only.

---

## 5. Future Milestones (intentionally excluded from V1)
Composite factor scoring, smart-beta portfolio construction, rebalancing,
historical backtesting, research simulations, persona-based optimisation,
efficient-frontier analysis, Gold/Debt ETF integration, multi-asset allocation,
and ML models all slot into the existing layers (`core/portfolio`,
`core/backtesting`, `core/optimization`, `core/simulation`) without
reshaping the architecture.
