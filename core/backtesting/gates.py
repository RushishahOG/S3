"""ARQM backtest gates & scoring (Gates 0-3 + final ranking).

All functions here operate on a *single rebalance date cross-section*: they take
the engineered datasets (already loaded) restricted to the eligible universe and
return scores / rankings / selection masks. The backtest engine loops over
rebalance dates and calls these; no date logic lives here.

Conventions
-----------
* A stock is only scored on factors it actually has data for. If a *mandatory*
  factor is NaN the stock is rejected at the relevant gate (never scored 0).
* Higher score = better. Risk factors (volatility, beta, semi-dev) are inverted
  so lower raw risk -> higher stability score.
* Normalization is applied cross-sectionally via ``core.backtesting.normalization``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from core.backtesting import normalization as norm
from core.config.backtest_schema import (
    BacktestParameters,
    QualityConfig,
    SelectionMode,
    StabilityConfig,
)
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Gate 0: Eligibility                                                          #
# --------------------------------------------------------------------------- #
def eligibility_filter(
    prices: pd.DataFrame,
    quality: pd.DataFrame,
    lowvol: pd.DataFrame,
    cap_tier: pd.Series,
    universe: list[str],
    date: pd.Timestamp,
    params: BacktestParameters,
) -> tuple[list[str], dict[str, str]]:
    """Return (eligible tickers, {ticker: rejection_reason}).

    Filters applied (all configurable in ``params.universe``):
      * present in the universe snapshot;
      * has >= min_trading_history_days of price history up to ``date``;
      * (optional, point-in-time) has quality features / low-vol features /
        momentum data *available as of ``date``* — i.e. the as-of snapshots
        passed in already carry only data on/before ``date``, so a stock that
        only starts reporting quality in 2015 is simply ineligible before then.
        This keeps the momentum, low-vol and quality windows aligned per
        rebalance date instead of mixing heterogeneous start years.
    """
    cfg = params.universe
    # Columns that the gates actually consume from each as-of snapshot.
    quality_cols = [f.name for f in params.quality.factors]
    lowvol_cols = [params.stability.column_map.get(f.name, f.name) for f in params.stability.factors]
    mom_cols = [f.name for f in params.momentum.factors]
    reasons: dict[str, str] = {}
    eligible: list[str] = []

    hist = prices.loc[:date]
    for t in universe:
        if t not in prices.columns:
            reasons[t] = "not_in_price_panel"
            continue
        series = hist[t].dropna()
        if len(series) < cfg.min_trading_history_days:
            reasons[t] = f"insufficient_history({len(series)}<{cfg.min_trading_history_days})"
            continue
        if cfg.require_quality_features:
            if t not in quality.index or quality.loc[t].reindex(quality_cols).isna().all():
                reasons[t] = "missing_quality_asof"
                continue
        if cfg.require_lowvol_features:
            if t not in lowvol.index or lowvol.loc[t].reindex(lowvol_cols).isna().all():
                reasons[t] = "missing_lowvol_asof"
                continue
        if cfg.require_momentum_data:
            if t not in lowvol.index or lowvol.loc[t].reindex(mom_cols).isna().all():
                reasons[t] = "missing_momentum_asof"
                continue
        eligible.append(t)

    return eligible, reasons


# --------------------------------------------------------------------------- #
# Gate 1: Momentum discovery                                                   #
# --------------------------------------------------------------------------- #
def momentum_gate(
    momentum_raw: pd.DataFrame,
    eligible: list[str],
    params: BacktestParameters,
) -> tuple[pd.Series, list[str]]:
    """Normalize + combine momentum factors into a single score, then select.

    Returns (momentum_score [0..1-ish per ticker], selected_tickers).
    """
    cfg = params.momentum
    sub = momentum_raw.reindex(eligible)
    score = pd.Series(np.nan, index=eligible)

    enabled = [f for f in cfg.factors if f.enabled]
    if not enabled:
        return score, []

    per_factor: list[pd.Series] = []
    weights: list[float] = []
    for f in enabled:
        raw = sub[f.name]
        if f.normalize:
            ns = norm.normalize(raw, cfg.normalization)
        else:
            ns = raw
        per_factor.append(ns)
        weights.append(f.weight)

    mat = pd.concat(per_factor, axis=1)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    if cfg.combine_method == "weighted_score":
        combined = mat.mul(w, axis=1).sum(axis=1)
    else:  # rank_average
        ranks = mat.rank(axis=0, pct=True)
        combined = ranks.mean(axis=1)

    # Normalize the combined score to a stable 0..1 scale for downstream mixing.
    score = norm.minmax(combined) if combined.notna().any() else combined

    selected = _select(score.dropna(), cfg.selection_mode, cfg.top_pct, cfg.top_n)
    return score, selected


# --------------------------------------------------------------------------- #
# Gate 2: Momentum stability                                                    #
# --------------------------------------------------------------------------- #
def stability_gate(
    lowvol: pd.DataFrame,
    eligible: list[str],
    params: BacktestParameters,
) -> tuple[pd.Series, list[str]]:
    """Combine low-volatility features into a stability score (lower risk=better)."""
    cfg = params.stability
    sub = lowvol.reindex(eligible)
    enabled = [f for f in cfg.factors if f.enabled]
    if not enabled:
        return pd.Series(np.nan, index=eligible), []

    per_factor: list[pd.Series] = []
    weights: list[float] = []
    for f in enabled:
        col = cfg.column_map.get(f.name, f.name)
        if col not in sub.columns:
            logger.warning(
                "Stability factor '%s' mapped to column '%s' which is not present "
                "in the low-vol feature store; skipping it.", f.name, col
            )
            continue
        raw = sub[col]
        ns = norm.score_lower_is_better(raw, cfg.normalization) if f.normalize else raw
        per_factor.append(ns)
        weights.append(f.weight)

    if not per_factor:
        logger.error(
            "No low-volatility features found for any enabled stability factor "
            "(expected columns: %s). Stability pillar will be neutral-filled.",
            {f.name: cfg.column_map.get(f.name, f.name) for f in enabled},
        )
        return pd.Series(np.nan, index=eligible), []

    mat = pd.concat(per_factor, axis=1)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    if cfg.combine_method == "weighted_score":
        combined = mat.mul(w, axis=1).sum(axis=1)
    else:
        combined = mat.rank(axis=0, pct=True).mean(axis=1)

    score = norm.minmax(combined) if combined.notna().any() else combined
    selected = _select(score.dropna(), cfg.selection_mode, cfg.top_pct, cfg.top_n)
    return score, selected


# --------------------------------------------------------------------------- #
# Gate 3: Quality validation                                                    #
# --------------------------------------------------------------------------- #
def quality_gate(
    quality: pd.DataFrame,
    eligible: list[str],
    params: BacktestParameters,
) -> tuple[dict[str, pd.Series], pd.Series, list[str]]:
    """Compute pillar scores, the final quality score, and selection.

    Returns (pillar_scores dict, quality_score, selected_tickers).
    """
    cfg: QualityConfig = params.quality
    sub = quality.reindex(eligible)
    enabled = [f for f in cfg.factors if f.enabled]

    # Per-factor normalized matrix, split by pillar.
    pillar_norm: dict[str, list[pd.Series]] = {p: [] for p in cfg.pillar_weights}
    pillar_w: dict[str, list[float]] = {p: [] for p in cfg.pillar_weights}
    for f in enabled:
        if f.name not in sub.columns:
            continue
        raw = sub[f.name]
        if f.normalize:
            ns = norm.normalize(raw, cfg.normalization)
        else:
            ns = raw
        pillar_norm[f.pillar].append(ns)
        pillar_w[f.pillar].append(f.weight)

    pillar_scores: dict[str, pd.Series] = {}
    for p, series_list in pillar_norm.items():
        if not series_list:
            pillar_scores[p] = pd.Series(np.nan, index=eligible)
            continue
        mat = pd.concat(series_list, axis=1)
        w = np.array(pillar_w[p], dtype=float)
        w = w / w.sum()
        combined = mat.mul(w, axis=1).sum(axis=1)
        pillar_scores[p] = norm.minmax(combined) if combined.notna().any() else combined

    # Weighted combination of pillar scores (weights sum to 1.0 by schema guard).
    pw = pd.Series(cfg.pillar_weights)
    active = [p for p in pw.index if pillar_scores[p].notna().any()]
    if not active:
        return pillar_scores, pd.Series(np.nan, index=eligible), []
    wvec = pw[active] / pw[active].sum()
    qual = sum(pillar_scores[p] * wvec[p] for p in active)
    qual = norm.minmax(qual) if qual.notna().any() else qual

    # Apply per-factor minimum thresholds: a stock breaching any mandatory
    # threshold is rejected (score set to NaN) regardless of composite.
    for f in enabled:
        if f.min_threshold is None or f.name not in sub.columns:
            continue
        breach = sub[f.name] < f.min_threshold
        qual = qual.mask(breach, np.nan)

    qual = qual[qual >= params.quality.min_quality_score] if params.quality.min_quality_score > 0 else qual
    selected = list(qual.dropna().index)
    return pillar_scores, qual, selected


# --------------------------------------------------------------------------- #
# Persistence filter (optional)                                                 #
# --------------------------------------------------------------------------- #
def apply_persistence(
    momentum_score: pd.Series,
    stability_score: pd.Series,
    history: list[tuple[pd.Series, pd.Series]],
    params: BacktestParameters,
) -> list[str]:
    """Require momentum & stability to stay above quantile for N consecutive periods.

    ``history`` is the list of (momentum_score, stability_score) from the previous
    rebalance dates, oldest first. Returns the tickers that pass.
    """
    cfg = params.persistence
    if not cfg.enabled:
        return list(momentum_score.dropna().index)
    if len(history) < cfg.required_periods:
        return list(momentum_score.dropna().index)

    recent = history[-cfg.required_periods:]
    passed: list[str] = []
    for t in momentum_score.dropna().index:
        ok = True
        for m_prev, s_prev in recent:
            mq = m_prev.quantile(cfg.momentum_quantile) if m_prev.notna().any() else np.nan
            sq = s_prev.quantile(cfg.stability_quantile) if s_prev.notna().any() else np.nan
            if pd.isna(mq) or pd.isna(sq):
                continue
            if momentum_score.get(t, np.nan) < mq or stability_score.get(t, np.nan) < sq:
                ok = False
                break
        if ok:
            passed.append(t)
    return passed


# --------------------------------------------------------------------------- #
# Final ranking                                                                 #
# --------------------------------------------------------------------------- #
def final_scores(
    momentum_score: pd.Series,
    stability_score: pd.Series,
    quality_score: pd.Series,
    eligible: Iterable[str],
    params: BacktestParameters,
) -> pd.Series:
    """Combine the three pillar scores into the overall ARQM score (0..1).

    A pillar that is entirely unavailable for the current cross-section (e.g.
    low-vol features not yet generated) is filled with a neutral 0.5 so a single
    missing dataset does not empty the entire book; a warning is logged. Stocks
    still must pass quality validation (NaN -> excluded) since quality is the
    hard gate.
    """
    idx = pd.Index(eligible)
    m = momentum_score.reindex(idx)
    s = stability_score.reindex(idx)
    q = quality_score.reindex(idx)
    w = params.scoring.weights

    def _neutral(series: pd.Series, name: str) -> pd.Series:
        if series.isna().all():
            logger.warning("Pillar '%s' entirely unavailable; using neutral fill.", name)
            return pd.Series(0.5, index=idx)
        return series

    m = _neutral(m, "momentum")
    s = _neutral(s, "stability")
    # Quality is the hard gate: keep NaN (excludes the stock) rather than fill.
    combined = m * w["momentum"] + s * w["stability"] + q * w["quality"]
    return norm.minmax(combined)


def _select(score: pd.Series, mode: SelectionMode, top_pct: float, top_n: int) -> list[str]:
    """Select top stocks by score using pct or N."""
    if score.empty:
        return []
    if mode == "top_pct":
        k = max(1, int(round(len(score) * top_pct)))
    else:
        k = min(top_n, len(score))
    return list(score.sort_values(ascending=False).head(k).index)
