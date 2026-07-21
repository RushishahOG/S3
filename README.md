# Smart Beta Research Platform

> A modular, end-to-end quantitative investment research and portfolio construction platform for the Indian equity market (NIFTY 500). Built for factor research, systematic strategy development, backtesting, and portfolio analytics.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![Storage](https://img.shields.io/badge/storage-DuckDB-FFF000.svg)](https://duckdb.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

The Smart Beta Research Platform is a full-stack quantitative research system that takes you from raw market data ingestion through feature engineering, factor construction, systematic strategy backtesting, parameter optimization, Monte Carlo simulation, and portfolio analytics — all within an interactive Streamlit interface.

The platform implements a **gate-based pipeline architecture** called ARQM (Adaptive Regime-based Quality Momentum), where each stage of the stock selection process (eligibility, momentum scoring, stability scoring, quality assessment, persistence filtering) is a configurable, independently operable gate. This design makes the system modular by construction — gates can be reordered, disabled, or new ones registered without touching any other component.

---

## Key Features

- **Data Ingestion Pipeline** — automated download of historical prices from Yahoo Finance and fundamental data via Apify Screener, with retry logic, rate limiting, incremental updates, and comprehensive validation.
- **Universe Management** — NIFTY 500 constituents with dynamic ticker resolution and mapping.
- **Feature Engineering Engine** — returns (simple/log at daily/weekly/monthly), risk metrics (beta, momentum, semi-deviation), and a persistent wide-format feature store in DuckDB.
- **Fundamental Quality Engine** — 15+ quality factor computations across 5 pillars (profitability, growth, financial strength, cash flow, shareholder return).
- **Eligibility Analyzer** — data-driven determination of earliest valid backtest start dates based on actual per-stock data availability.
- **ARQM Backtesting Engine** — a configurable, gate-based systematic strategy simulator with 4 core gates and 50+ configurable parameters.
- **Parameter Optimization** — automated search across the parameter space with multiple algorithms, objectives, and constraint validation.
- **Monte Carlo Simulation** — 4 resampling methodologies (i.i.d. bootstrap, block bootstrap, regime-conditional bootstrap, trade sequence randomization) for robust strategy validation.
- **Sensitivity Analysis** — one-way, two-way, and multi-parameter sensitivity grids with importance, correlation, interaction, and robustness analytics.
- **Efficient Frontier / Portfolio Analytics** — mean-variance optimization with 10 objectives, 6 solvers, and comprehensive constraint types.
- **Strategy Comparison** — side-by-side performance, risk, holdings, and statistical comparison across any combination of completed strategies.
- **Research Lab** — integrated toolkit combining parameter optimization, Monte Carlo, sensitivity analysis, efficient frontier, and strategy comparison in a unified workspace.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐ │
│  │Dashboard │ │Data Mgr  │ │Feature   │ │Backtesting     │ │
│  │          │ │Explorer  │ │Engineer  │ │Manual │Queue   │ │
│  └──────────┘ └──────────┘ └──────────┘ └───────┬────────┘ │
│                                                  │          │
│  ┌───────────────────────────────────────────────┴────────┐ │
│  │                  Research Lab                           │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │ │
│  │  │Parameter │ │ Monte    │ │Sensitivity│ │Strategy  │  │ │
│  │  │Optimiz.  │ │ Carlo    │ │Analysis   │ │Compare   │  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │ │
│  │  ┌──────────┐                                          │ │
│  │  │ Efficient│                                          │ │
│  │  │ Frontier │                                          │ │
│  │  └──────────┘                                          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Core Business Logic                      │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Data     │  │ Feature  │  │ Backtest │  │ Portfolio   │ │
│  │Injection │→│ Engineer │→│  Engine   │→│  Optimizer  │ │
│  └──────────┘  └──────────┘  └────┬─────┘  └─────────────┘ │
│                                   │                         │
│                        ┌──────────┴──────────┐              │
│                        │                     │              │
│                   ┌────▼────┐          ┌─────▼─────┐       │
│                   │ Monte   │          │ Sensitivity│       │
│                   │ Carlo   │          │ Analysis   │       │
│                   └─────────┘          └───────────┘       │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  Optimization │  │  Strategy    │  │  Eligibility     │ │
│  │  Engine       │  │  Comparison  │  │  Analyzer        │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     Storage Layer                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐              │
│  │ DuckDB   │  │ YAML     │  │  CSV / Parquet│              │
│  │ (Market  │  │ Config   │  │  (Exports)   │              │
│  │  Data)   │  │          │  │              │              │
│  └──────────┘  └──────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

---

## Strategy Methodology: ARQM Pipeline

The backtesting engine implements a 4-stage gate pipeline executed sequentially at each rebalance date:

### Gate Pipeline

| Gate | Stage | What It Does | Configurable Parameters |
|------|-------|-------------|------------------------|
| 0 | **Eligibility** | Filters stocks by universe membership, minimum trading history, and availability of required factor data as of the rebalance date. | `min_trading_history_days`, `require_quality_features`, `require_lowvol_features`, `require_momentum_data` |
| 1 | **Momentum** | Scores remaining stocks on momentum factors (raw and risk-adjusted). Normalizes, weights, and selects top candidates. | `selection_mode` (top_pct/top_n), `top_pct`, `top_n`, `combine_method`, `normalization` (zscore/robust_zscore/percentile/minmax) |
| 2 | **Stability** | Scores on low-volatility factors (beta, semi-deviation). Inverts risk metrics so lower risk = higher score. | Same as Momentum (independent configuration) |
| 3 | **Quality** | Evaluates 5 quality pillars (profitability, growth, financial strength, cash flow, shareholder return) with configurable pillar weights and minimum quality thresholds. | `pillar_weights`, `min_quality_score`, `use_rollup`, per-factor `min_threshold` |
| — | **Persistence** | (Optional) Filters for stocks that consistently score well across consecutive rebalance periods. | `required_periods`, `momentum_quantile`, `stability_quantile` |
| — | **Final Scoring** | Combines momentum, stability, and quality scores with user-defined weights for final ranking. | `momentum_weight`, `quality_weight`, `stability_weight` |

After scoring, stocks are segmented by market capitalization tier (large/mid/small) with configurable allocation weights, and the target portfolio is constructed with position-sizing constraints.

### Regime Management

The engine supports market-regime-aware execution:
- **Reference benchmark** for regime state detection
- **Buy trigger**: invest when benchmark exceeds threshold from trough
- **Sell trigger**: exit when benchmark drops below threshold from peak
- **Swing low detection**: invest after sustained recovery from drawdown
- **Peak detection**: exit on newly confirmed peaks

---

## Factor Engineering

### Risk & Momentum Features (Daily)

| Feature | Calculation | Window |
|---------|------------|--------|
| Beta | Rolling covariance(returns, benchmark) / variance(benchmark) | 252 trading days |
| Momentum (Unscaled) | `P_t / P_{t-252} - 1` with 21-day lag | 12-month horizon |
| Momentum (Scaled) | Unscaled momentum / annualized volatility | 12-month horizon |
| Semi-Deviation | Rolling downside deviation (annualized) | 252 trading days |

### Fundamental Quality Factors (Annual)

| Pillar | Factors |
|--------|---------|
| **Profitability** | ROE, ROCE, ROA, Cash ROCE |
| **Growth** | EPS Growth, ROE Growth, ROCE Growth, Revenue Growth, DPS Growth |
| **Financial Strength** | Interest Coverage Ratio, Equity to Total Capital |
| **Cash Flow** | OCF to EBITDA, Cash ROCE |
| **Shareholder Return** | Dividend Payout Ratio, Sustainable Growth Rate |

---

## Backtesting Framework

The backtest engine (`core/backtesting/engine.py`) simulates portfolio evolution day-by-day across the specified rebalance schedule.

**Key characteristics:**
- **Deterministic** — identical parameters and database state produce identical results
- **Rebalance frequencies**: monthly, quarterly, semi-annual
- **Transaction costs**: configurable as percentage of trade value
- **Slippage**: configurable percentage applied to each trade
- **Benchmark tracking**: NAV comparison against NIFTY 500 benchmark
- **Complete audit trail**: per-gate pipeline snapshots with input/output universes, scores, and rejection reasons
- **Performance metrics** (computed by `core/backtesting/metrics.py`):

| Metric | Definition |
|--------|-----------|
| Total Return | `NAV_final / NAV_initial - 1` |
| CAGR | Annualized compound return |
| Annual Volatility | Standard deviation of daily returns × √252 |
| Sharpe Ratio | (Return − RF) / Volatility |
| Sortino Ratio | (Return − RF) / Downside Deviation |
| Calmar Ratio | CAGR / Max Drawdown |
| Treynor Ratio | (Return − RF) / Beta |
| Information Ratio | Active Return / Tracking Error |
| Alpha (Annual) | Actual Return − (RF + β × (Benchmark − RF)) |
| Max Drawdown | Maximum peak-to-trough decline |
| Ulcer Index | Root-mean-square of drawdown series |
| Beta | Covariance(portfolio, benchmark) / variance(benchmark) |
| Win Rate | Fraction of profitable trades |
| Profit Factor | Gross Profit / Gross Loss |

---

## Research Lab Capabilities

### Parameter Optimization

Automated search across the ARQM strategy parameter space using `core/optimization/`.

- **Parameter types**: continuous, discrete, categorical, boolean
- **Algorithms**: grid search, random search, and extensible algorithm interface
- **Objectives**: 10+ configurable objective functions (Sharpe, Sortino, Calmar, CAGR, etc.)
- **Constraints**: parameter-level and cross-parameter validation
- **Sum-groups**: automatically normalized parameter groups (cap weights, scoring weights, quality pillar weights)
- **Results**: ranked candidates with full performance metrics, persistable to disk

Optimizable parameters include:
- Market timing triggers (buy/sell thresholds)
- Cap segment allocation weights
- Momentum gate parameters (top %, selection mode, scoring weights)
- Stability gate parameters
- Quality pillar weights and minimum thresholds
- Portfolio construction (size, max position %)
- Rebalance frequency

### Monte Carlo Simulation

Four methodologies for strategy robustness validation (`core/monte_carlo/`):

| Method | What It Does | Best For |
|--------|-------------|----------|
| **Return Bootstrap** | i.i.d. resampling of daily returns with replacement | Quick distribution of outcomes under stationary returns |
| **Block Bootstrap** | Sampling contiguous return blocks | Preserving autocorrelation structure |
| **Regime Bootstrap** | Bootstrap within each regime state, preserving regime order | Testing strategy resilience across market regimes |
| **Trade Randomization** | Permutation of holding-period trade legs | Testing trade timing luck |

**Outputs per simulation:** equity curves, CAGR, total return, Sharpe, Sortino, Calmar, max drawdown, ulcer index, win rate, profit factor, expectancy, and several risk metrics. Aggregate statistics include confidence intervals, probability distributions, and risk summaries.

### Efficient Frontier & Portfolio Analytics

Mean-variance optimization framework (`core/portfolio/`):

- **10 optimization objectives**: max Sharpe, min volatility, max return, max Calmar, max Sortino, min drawdown, risk parity, equal weight, max diversification, min correlation
- **6 solvers**: quadratic programming, SLSQP, differential evolution, particle swarm, genetic algorithm, simulated annealing
- **Constraint types**: weight bounds, portfolio size, sector limits, market cap tiers, cash allocation, turnover limits, liquidity thresholds
- **Visualization**: efficient frontier curve, allocation pie/treemap, risk contribution breakdown

### Sensitivity Analysis

Multi-parameter sensitivity engine (`core/sensitivity/`):

- One-way, two-way, and multi-parameter perturbation grids
- In-process result caching (repeated analyses reuse prior runs)
- Full metric suite collected per combination
- **Analytics**: sensitivity scores, stability statistics, parameter importance, correlation analysis, interaction effects, robustness metrics
- **Output**: robust-parameter recommendations based on empirical stability

### Strategy Comparison

Side-by-side comparison engine (`core/strategy_comparison/`):

- Config comparison across strategies (side-by-side parameter tables)
- 25+ performance and risk metrics automatically derived from cached equity curves
- Return correlation matrix and holdings overlap (Jaccard similarity)
- Composite ranking with user-configurable metric weights
- Automatic recommendations (best CAGR, lowest risk, highest Sharpe, most stable, etc.)
- Statistical tests: paired t-test, bootstrap confidence intervals, outperformance frequency
- Equity, drawdown, rolling return, annual return, and monthly return curves

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Framework** | Streamlit 1.59+ | Interactive web UI |
| **Storage** | DuckDB 1.5+ | Analytical database for market data, features, and metadata |
| **Market Data** | Yahoo Finance (`yfinance`) | Historical price data for NIFTY 500 constituents |
| **Fundamentals** | Apify Screener | Financial statements, ratios, and company data |
| **Numerical** | NumPy, Pandas, SciPy, PyArrow | Data processing, statistics, optimization |
| **Portfolio Opt.** | SciPy (`minimize`, `quadratic`) | Constrained portfolio optimization |
| **Visualization** | Altair, Streamlit native charts | Interactive charts and dashboards |
| **Serialization** | PyYAML, JSON | Configuration and results persistence |
| **Execution** | ThreadPoolExecutor, joblib | Parallel simulation and computation |

---

## Project Structure

```
asset_class_selection_system/
├── main.py                           # Application entry point
├── requirements.txt                  # Python dependencies
├── .env                              # Environment variables (API keys, DB URIs)
│
├── app/                              # Streamlit presentation layer
│   ├── services.py                   # Service locator (StorageManager, UniverseManager)
│   ├── components/                   # Reusable UI components (sidebar, logs, metrics)
│   ├── explorer/                     # Data explorer components
│   ├── layouts/                      # Page layout templates
│   └── pages/
│       ├── backtesting.py            # ARQM backtesting page (4-tab interface)
│       ├── backtest/                 # Backtest sub-pages
│       │   ├── state.py             # Backtest session state & worker management
│       │   ├── manual_testing/      # Manual strategy configuration & submission
│       │   ├── portfolio_queue/     # Running/queued backtest dashboard
│       │   └── results/             # Completed backtest analysis
│       ├── research_lab/            # Research Lab modules
│       │   ├── research_lab.py      # Tab navigation & orchestration
│       │   ├── parameter_optimization.py
│       │   ├── monte_carlo.py
│       │   ├── efficient_frontier.py
│       │   ├── sensitivity_analysis.py
│       │   └── strategy_comparison.py
│       ├── dashboard.py             # System overview dashboard
│       ├── data_extractor.py        # Data ingestion & management
│       ├── dataset_explorer.py      # Dataset browsing & analysis
│       ├── eligibility_analyzer.py  # Data availability & start-date analysis
│       ├── feature_engineering.py   # Feature computation pipeline
│       ├── mongo_cloud.py           # MongoDB/GridFS backup management
│       └── universe_explorer.py     # Universe membership explorer
│
├── core/                             # Business logic layer
│   ├── backtesting/                  # ARQM backtest engine
│   │   ├── engine.py                # Main backtest orchestrator
│   │   ├── data.py                  # Data loading for backtesting
│   │   ├── gates.py                 # Gate 0-3 implementations
│   │   ├── gate_registry.py         # Pluggable gate system
│   │   ├── metrics.py               # 20+ performance & risk metrics
│   │   ├── momentum.py              # Momentum factor computation
│   │   ├── normalization.py         # Cross-sectional normalization methods
│   │   ├── regime.py                # Market regime detection
│   │   └── export.py                # Backtest result export
│   │
│   ├── config/                       # Configuration schema & settings
│   │   ├── backtest_schema.py       # BacktestParameters dataclasses
│   │   ├── settings.py              # Centralized YAML-driven settings
│   │   └── providers_config.py      # Data provider configuration
│   │
│   ├── data/                         # Data layer
│   │   ├── market_data_manager.py   # Orchestrates data downloads
│   │   ├── cache/                   # Caching layer
│   │   ├── ingestion/               # Data downloaders & validators
│   │   │   ├── downloader.py        # Historical price downloader
│   │   │   ├── screener_downloader.py # Fundamental data downloader
│   │   │   ├── ticker_resolver.py   # Symbol resolution & mapping
│   │   │   ├── constituents.py      # Universe constituent loader
│   │   │   ├── reports.py           # Download reporting
│   │   │   └── validation.py        # OHLCV data validation
│   │   ├── providers/               # Data provider abstraction
│   │   │   ├── base_provider.py     # Abstract provider interface
│   │   │   ├── yahoo_finance.py     # Yahoo Finance implementation
│   │   │   ├── apify_client.py      # Apify API client
│   │   │   ├── apify_screener_provider.py
│   │   │   ├── fundamental_parsing.py
│   │   │   └── registry.py          # Provider registry
│   │   ├── storage/                 # Persistent storage
│   │   │   ├── storage_manager.py   # DuckDB storage manager
│   │   │   └── provisioning.py      # MongoDB/GridFS provisioning
│   │   └── universe/                # Universe management
│   │       ├── universe_manager.py  # Registry & orchestration
│   │       ├── base_universe.py     # Abstract universe provider
│   │       └── nifty500.py          # NIFTY 500 implementation
│   │
│   ├── eligibility/                  # Eligibility analysis
│   │   ├── analyzer.py              # Data-driven start-date analysis
│   │   └── registry.py              # Factor lookback registry
│   │
│   ├── factors/                      # Factor computation
│   │   └── fundamental/             # Quality factors (5 pillars, 15+ metrics)
│   │
│   ├── feature_engineering/          # Feature pipeline
│   │   ├── return_engine.py         # Return computation (daily/weekly/monthly)
│   │   ├── risk_engine.py           # Beta, momentum, semi-deviation
│   │   ├── feature_store.py         # Wide-format DuckDB feature store
│   │   └── feature_validator.py     # Data quality validation
│   │
│   ├── monte_carlo/                  # Monte Carlo simulation
│   │   ├── engine.py                # 4 simulation methods
│   │   ├── types.py                 # Config & result dataclasses
│   │   ├── statistics.py            # Aggregate statistics & confidence intervals
│   │   ├── plotting.py              # Simulation visualization
│   │   ├── runner.py                # Runner management
│   │   └── export.py                # Simulation export
│   │
│   ├── optimization/                 # Parameter optimization
│   │   ├── engine.py                # Optimization orchestrator
│   │   ├── spec.py                  # Parameter specification catalogue
│   │   ├── algorithms.py            # Search algorithms
│   │   ├── constraints.py           # Constraint validation
│   │   ├── objectives.py            # Objective functions
│   │   ├── candidate.py             # Candidate construction
│   │   ├── search_space.py          # Search space definition
│   │   ├── param_registry.py        # Parameter metadata
│   │   └── results.py               # Results persistence
│   │
│   ├── portfolio/                    # Portfolio optimization
│   │   ├── optimizer.py             # Efficient frontier & 6-solver engine
│   │   ├── risk_models.py           # Risk model implementations
│   │   └── visualization.py         # Portfolio visualization
│   │
│   ├── sensitivity/                  # Sensitivity analysis
│   │   ├── engine.py                # Multi-parameter sensitivity engine
│   │   └── export.py                # Results export
│   │
│   ├── strategy_comparison/          # Strategy comparison
│   │   ├── comparison.py            # Side-by-side comparison engine
│   │   ├── repository.py            # Strategy record storage & retrieval
│   │   ├── export.py                # Comparison export
│   │   └── visualization.py         # Comparison charts
│   │
│   ├── universe_explorer/           # Universe exploration
│   │   ├── explorer.py              # Universe data exploration
│   │   ├── membership.py            # Membership timeline analysis
│   │   └── providers.py             # Provider implementations
│   │
│   ├── utils/                        # Utilities
│   │   ├── dates.py                 # Date range handling & validation
│   │   ├── decorators.py            # Utility decorators
│   │   ├── logging_config.py        # Structured logging setup
│   │   └── paths.py                 # Project path resolution
│   │
│   └── visualization/               # Core visualization
│       └── charts.py                # Reusable chart functions
│
├── config/                           # Configuration files
│   ├── settings.yaml                # Application settings
│   ├── providers.yaml               # Data provider configuration
│   ├── ticker_mapping.yaml          # Symbol alias mapping
│   └── universe/                    # Universe definition files
│
├── nifty_500_constituents/          # Reference constituent data
├── storage/                          # Generated data (git-ignored)
│   ├── market_data.duckdb           # Analytical database
│   └── logs/                        # Application logs
│
├── tests/                            # Unit & integration tests
└── docs/                             # Documentation
```

---

## Installation & Setup

### Prerequisites

- Python 3.11 or later
- (Optional) [Apify API token](https://console.apify.com/settings/integrations) for fundamental data ingestion

### Clone & Install

```bash
git clone <repository-url>
cd asset_class_selection_system

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Environment variables** — copy the provided `.env` file and configure:
   - `APIFY_API_TOKEN` — your Apify API token (for screener fundamental data)
   - `MONGO_URI` — MongoDB connection string (optional, for cloud backup)

2. **Application settings** — edit `config/settings.yaml` to customize:
   - Data provider preferences
   - Backtest date window (default: 2006-01-01 to 2026-05-31)
   - Feature computation parameters
   - Logging verbosity

3. **Provider configuration** — `config/providers.yaml` controls data vendor settings (API endpoints, rate limits, batch sizes)

### First Run

```bash
streamlit run main.py
```

The application opens in your default browser at `http://localhost:8501`.

---

## Usage Workflow

### 1. Data Ingestion

Navigate to **Data Manager** → configure your data source → click **Download / Refresh** to download historical prices for all NIFTY 500 constituents. Use **Screener Download** for fundamental data.

### 2. Feature Engineering

Go to **Feature Engineering** → select factor categories (returns, risk metrics, fundamental quality) → **Generate Features**. Engineered features are stored in the DuckDB feature store with full metadata tracking.

### 3. Backtesting

Switch to **Backtesting** → **Manual Testing** tab:
- Configure strategy parameters across all gates (universe, momentum, stability, quality, persistence, scoring, portfolio construction)
- Set market timing triggers and cap allocation weights
- Submit the strategy for execution

Monitor progress in the **Portfolio Queue** tab, then analyze results in **Results** tab.

### 4. Research & Analysis

Navigate to the **Research Lab** for advanced analysis:
- **Parameter Optimization**: automatically search the parameter space for optimal configurations
- **Monte Carlo Simulation**: validate strategy robustness across 4 resampling methodologies
- **Sensitivity Analysis**: understand which parameters drive performance
- **Efficient Frontier**: optimize portfolio weights across the risk-return spectrum
- **Strategy Comparison**: compare any combination of completed strategies side-by-side

### 5. Export & Backup

Results can be exported as CSV, and the entire database can be backed up to MongoDB/GridFS for cloud persistence.

---

## Configuration Reference

### Backtest Date Window

The competition backtesting window is globally constrained (default: 2006-01-01 to 2026-05-31). The end date is fixed; users select a start date after the minimum. Configure in `config/settings.yaml`:

```yaml
backtest:
  min_date: "2006-01-01"
  max_date: "2026-05-31"
  default_start: "2006-01-01"
```

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `storage.database_path` | `storage/market_data.duckdb` | DuckDB database file |
| `providers.default` | `yahoo_finance` | Default market data provider |
| `universe.default` | `nifty500` | Investment universe |
| `universe.benchmark` | `NIFTY_500` | Benchmark for beta/alpha |
| `features.trading_days_per_year` | `252` | Trading days convention |
| `ingestion.retries` | `3` | Download retry count |
| `ingestion.ticker_mapping_file` | `config/ticker_mapping.yaml` | Symbol alias mapping |
| `logging.level` | `INFO` | Log verbosity |

---

## Testing

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test modules
python -m unittest tests.test_backtest_engine
python -m unittest tests.test_feature_engineering
python -m unittest tests.test_ingestion
python -m unittest tests.test_eligibility
```

---

## Future Enhancements

- Additional data providers (Alpha Vantage, Polygon, NSE direct feed)
- Multi-universe support (Midcap 150, custom CSVs, ETFs)
- Multi-asset allocation (Gold, Debt ETFs, international equities)
- Machine learning models for factor combination and regime prediction
- Alternative data integration (sentiment, ESG, macro indicators)
- Live trading signal generation and broker API integration
- Performance attribution and decomposition analysis
- WebSocket-based real-time data streaming

---

## License

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project is developed for the NJ Factor Investing Olympiad. Licensed under the MIT License.

---

## Contributing

Contributions are welcome. Please ensure:
- All tests pass before submitting changes
- New features include appropriate test coverage
- Configuration changes go in YAML files, not in source code
- The gate-based pipeline architecture is respected for strategy modifications
