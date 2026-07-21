"""Backtest data access layer (ARQM simulation engine).

This module is the *only* bridge between the simulation engine and the stored,
already-engineered datasets. It never performs raw financial calculations -- it
pulls prices, low-volatility features, quality features, company metadata and
the universe membership, and adds two derived conveniences that the spec needs
but the DB does not store:

* point-in-time **cap-tier** labels (large / mid / small) from the current
  ``market_cap`` snapshot, ranked within the universe; and
* a wide adjusted-price panel used by the momentum signal generator.

The engine consumes the objects returned here and nothing else.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from core.config.backtest_schema import BacktestParameters
from core.data.storage.storage_manager import StorageManager
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS = 252


@dataclass
class BacktestData:
    """Container of every engineered dataset the engine needs."""

    params: BacktestParameters
    universe_tickers: list[str]
    prices: pd.DataFrame  # wide: index=date, columns=tickers, values=adj_close
    benchmark_prices: pd.Series  # adj_close of the benchmark pseudo-ticker
    quality: pd.DataFrame  # ticker -> quality factor columns (latest rollup)
    lowvol: pd.DataFrame  # ticker -> low-vol feature columns (latest available)
    company: pd.DataFrame  # ticker -> sector, market_cap, company_name
    cap_tier: pd.Series  # ticker -> "large" | "mid" | "small"
    market_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_ts: pd.DataFrame = field(default_factory=pd.DataFrame)
    start: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    end: pd.Timestamp = field(default_factory=pd.Timestamp.now)

    # -- derived helpers -----------------------------------------------------
    @property
    def tickers(self) -> list[str]:
        return self.universe_tickers

    def quality_for(self, tickers: Iterable[str]) -> pd.DataFrame:
        return self.quality.reindex(list(tickers))

    def lowvol_for(self, tickers: Iterable[str]) -> pd.DataFrame:
        return self.lowvol.reindex(list(tickers))


def _parse_date(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def load_backtest_data(
    storage: StorageManager,
    params: BacktestParameters,
    warmup_years: int = 2,
    progress: Callable[[str, str, float | None, int | None], None] | None = None,
) -> BacktestData:
    """Pull all engineered datasets required by the engine from storage.

    Prices are fetched with a ``warmup_years`` look-back buffer *before* the
    backtest start so that momentum windows and the minimum-history eligibility
    check have data to work with. The simulation itself still only runs over
    ``[start, end]`` (returned in ``start``/``end``); the extra history is only
    used for factor computation at the first rebalance date.

    ``progress`` (optional) is called as ``progress(stage, status, duration_s, n_rows)``
    where ``status`` is ``"start"`` or ``"done"`` -- this lets the UI advance past a
    static "loading" spinner and surface exactly which substep is running.
    """
    start = _parse_date(params.general.start_date)
    end = _parse_date(params.general.end_date)
    warm_start = start - pd.DateOffset(years=warmup_years)

    def _step(stage: str, fn, *args):
        if progress is not None:
            progress(stage, "start", None, None)
        t0 = time.perf_counter()
        result = fn(*args)
        dt = time.perf_counter() - t0
        n = _row_count(result)
        logger.info("load_backtest_data[%s] completed in %.2fs (%s rows)", stage, dt, n)
        if progress is not None:
            progress(stage, "done", dt, n)
        return result

    # --- Universe (current snapshot proxy) ---------------------------------
    universe = _step("universe", _load_universe, storage, params)
    universe = [t for t in universe if t != params.general.benchmark]

    # --- Prices (wide adj_close panel, with warm-up buffer) ----------------
    prices = _step(
        "prices",
        lambda: storage.get_adjusted_price_panel(tickers=universe, start=warm_start, end=end),
    )
    prices = prices.sort_index()
    # Align universe to tickers that actually have price history.
    universe = [t for t in universe if t in prices.columns]
    prices = prices[universe]

    # --- Benchmark pseudo-ticker -------------------------------------------
    bench = params.general.benchmark
    bench_panel = _step(
        "benchmark_prices",
        lambda: storage.get_adjusted_price_panel(tickers=[bench], start=start, end=end),
    )
    benchmark_prices = (
        bench_panel[bench].sort_index() if bench in bench_panel.columns else pd.Series(dtype="float64")
    )

    # --- Quality features (latest rollup per ticker) -----------------------
    quality = _step("quality_features", _load_quality, storage, universe, params)

    # --- Low-volatility features (latest available per ticker) -------------
    lowvol = _step("lowvol_features", _load_lowvol, storage, universe)

    # --- Company metadata + cap tiers --------------------------------------
    company = _step("company_metadata", storage.get_fundamentals_company, universe)
    cap_tier = _assign_cap_tiers(company, params)

    # --- Point-in-time time-series (daily market + yearly quality) ---------
    # These carry *every* stored observation (not just the latest rollup) so
    # the engine can take an as-of snapshot at each rebalance date and honour
    # the rolling 252-day windows / yearly fundamental vintage.
    market_features = _step("market_features", _load_market_features, storage, universe, warm_start, end)
    quality_ts = _step("quality_time_series", _load_quality_ts, storage, universe)

    logger.info(
        "load_backtest_data FINISHED: universe=%d prices=%s market_features=%s quality_ts=%s",
        len(universe), prices.shape, market_features.shape, quality_ts.shape,
    )

    return BacktestData(
        params=params,
        universe_tickers=universe,
        prices=prices,
        benchmark_prices=benchmark_prices,
        quality=quality,
        lowvol=lowvol,
        company=company,
        cap_tier=cap_tier,
        market_features=market_features,
        quality_ts=quality_ts,
        start=start,
        end=end,
    )


def _row_count(obj) -> int:
    """Best-effort row count for logging (DataFrame / Series / list)."""
    try:
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return int(obj.shape[0])
        if isinstance(obj, (list, tuple, set)):
            return len(obj)
    except Exception:
        return 0
    return 0


def _load_universe(storage: StorageManager, params: BacktestParameters) -> list[str]:
    """Current NIFTY 500 snapshot proxy (no point-in-time history available)."""
    try:
        from core.data.universe.universe_manager import UniverseManager

        um = UniverseManager()
        tickers = list(um.default_universe().tickers)
        if tickers:
            return list(tickers)
    except Exception as exc:  # pragma: no cover - fall back to storage
        logger.warning("Universe manager unavailable (%s); falling back to stored tickers", exc)
    return list(storage.tickers_with_screener_data())


def _load_quality(storage: StorageManager, tickers: list[str], params: BacktestParameters) -> pd.DataFrame:
    """Return one row per ticker of the chosen quality rollup."""
    q = storage.get_fundamental_quality_features(tickers)
    if q.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    rollup = params.quality.use_rollup
    base_cols = [f.name for f in params.quality.factors]
    keep = {"ticker", "financial_year"}
    for base in base_cols:
        if rollup == "median" and f"{base}_median" in q.columns:
            keep.add(f"{base}_median")
        elif rollup == "weighted" and f"{base}_weighted" in q.columns:
            keep.add(f"{base}_weighted")
        elif base in q.columns:
            keep.add(base)
        else:
            for suf in ("_median", "_weighted"):
                if f"{base}{suf}" in q.columns:
                    keep.add(f"{base}{suf}")
                    break
    q = q[[c for c in q.columns if c in keep]]
    q = q.sort_values(["ticker", "financial_year"])
    q = q.groupby("ticker").tail(1).set_index("ticker")
    rename = {c: c.replace("_median", "").replace("_weighted", "") for c in q.columns if c != "ticker"}
    q = q.rename(columns=rename)
    return q


def _load_lowvol(storage: StorageManager, tickers: list[str]) -> pd.DataFrame:
    """Return one row per ticker of the latest low-vol feature snapshot."""
    cols = storage.feature_columns()
    if not cols:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    lf = storage.get_features(tickers, columns=cols)
    if lf.empty:
        return pd.DataFrame(index=pd.Index(tickers, name="ticker"))
    lf = lf.sort_values(["ticker", "date"])
    lf = lf.groupby("ticker").tail(1).set_index("ticker")
    return lf


def _load_market_features(
    storage: StorageManager,
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return the full daily feature_store time-series (long format).

    Columns: ``ticker``, ``date`` plus every engineered daily market feature
    (``beta``, ``momentum_unscaled``, ``momentum_scaled``, ``semi_deviation``).
    Each value is, by construction, the trailing 252-day rolling window estimate
    as of that day, so the engine only needs to take the latest row on/before a
    rebalance date to obtain the point-in-time factor.
    """
    cols = ["beta", "momentum_unscaled", "momentum_scaled", "semi_deviation"]
    existing = [c for c in cols if c in storage.feature_columns()]
    if not existing:
        return pd.DataFrame(columns=["ticker", "date", *cols])
    mf = storage.get_features(tickers, start=start, end=end, columns=existing)
    if mf is None or mf.empty:
        return pd.DataFrame(columns=["ticker", "date", *existing])
    mf = mf.copy()
    mf["date"] = pd.to_datetime(mf["date"])
    return mf.sort_values(["ticker", "date"]).reset_index(drop=True)


def _load_quality_ts(storage: StorageManager, tickers: list[str]) -> pd.DataFrame:
    """Return the full yearly fundamental quality time-series (long format).

    Columns: ``ticker``, ``financial_year`` plus every engineered quality factor
    (incl. the ``*_weighted`` / ``*_median`` growth variants). The engine takes
    the latest financial-year vintage available as-of a rebalance date for
    point-in-time, vintage-correct quality scoring.
    """
    q = storage.get_fundamental_quality_features(tickers)
    if q.empty:
        return pd.DataFrame()
    q = q.copy()
    return q.sort_values(["ticker", "financial_year"]).reset_index(drop=True)


def _assign_cap_tiers(company: pd.DataFrame, params: BacktestParameters) -> pd.Series:
    """Rank universe by current market_cap into large/mid/small tiers.

    The DB only stores a current ``market_cap`` scalar (no historical tiers), so
    we rank the *eligible* universe and split at the conventional 70/90
    percentiles of the cap distribution (top 30% = large, next 30% = mid, rest =
    small). This is the documented snapshot-proxy behaviour chosen for the build.

    Fallback: if ``market_cap`` is unavailable for (almost) the whole universe,
    every stock is labelled ``large`` so the engine degrades to a single bucket
    instead of producing an empty book. A warning is logged by the caller.
    """
    empty_idx = company.index if company is not None and not company.empty else pd.Index([])
    if company is None or company.empty or "market_cap" not in company.columns:
        return pd.Series(index=empty_idx, dtype=object)
    caps = pd.to_numeric(company["market_cap"], errors="coerce")
    if caps.notna().sum() < max(10, 0.1 * len(caps)):
        # Insufficient cap coverage -> single bucket fallback.
        return pd.Series("large", index=caps.index)
    rank = caps.rank(pct=True, method="first")
    tiers = pd.Series(index=caps.index, dtype=object)
    tiers[rank >= 0.70] = "large"
    tiers[(rank >= 0.40) & (rank < 0.70)] = "mid"
    tiers[rank < 0.40] = "small"
    tiers[caps.isna()] = "large"
    return tiers
