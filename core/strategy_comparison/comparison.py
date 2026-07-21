"""Comparison engine for strategy analytics.

Computes performance, risk, configuration, holdings and statistical comparisons
across multiple strategies. All inputs are read-only (cached backtest outputs);
no backtests are re-run. Missing metrics (win rate, profit factor, VaR, holding
period, capture ratios, ...) are derived from the cached equity curve and trade
log so the comparison is complete even when the original metrics dict is sparse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from core.backtesting.metrics import TRADING_DAYS, compute_all_metrics
from core.strategy_comparison.repository import StrategyRecord

_DEFAULT_RANK_WEIGHTS = {
    "CAGR": 0.25,
    "Sharpe": 0.20,
    "Max DD": 0.20,
    "Calmar": 0.15,
    "Sortino": 0.15,
    "Volatility": 0.05,
}


@dataclass
class ComparisonResult:
    """Result of comparing multiple strategies."""

    strategies: list[StrategyRecord]
    config_comparison: pd.DataFrame
    performance_table: pd.DataFrame
    equity_curves: pd.DataFrame
    drawdown_curves: pd.DataFrame
    rolling_returns: pd.DataFrame
    annual_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    risk_table: pd.DataFrame
    correlation_matrix: pd.DataFrame
    rankings: pd.DataFrame
    recommendations: dict
    allocation_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    holdings_overlap_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    benchmark_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    radar_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats_tests: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Metric enrichment
# --------------------------------------------------------------------------


def _daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _enrich_metrics(record: StrategyRecord) -> dict[str, float]:
    """Build a complete metric dict, preferring stored values then deriving."""
    m = dict(record.metrics or {})
    eq = record.equity
    has_eq = eq is not None and not eq.empty and len(eq) > 1
    bench = record.benchmark
    has_bench = bench is not None and not bench.empty and len(bench) > 1

    if has_eq:
        base = compute_all_metrics(eq, bench if has_bench else eq)
        for k, v in base.items():
            m.setdefault(k, v)

    if has_eq:
        ret = _daily_returns(eq)
        if "annual_volatility" not in m or pd.isna(m.get("annual_volatility", np.nan)):
            m["annual_volatility"] = float(ret.std() * np.sqrt(TRADING_DAYS))
        if "sharpe" not in m or pd.isna(m.get("sharpe", np.nan)):
            excess = ret - 0.0
            sd = excess.std()
            m["sharpe"] = float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd else np.nan
        if "max_drawdown" not in m or pd.isna(m.get("max_drawdown", np.nan)):
            m["max_drawdown"] = float(_drawdown_series(eq).min())
        if "ulcer_index" not in m or pd.isna(m.get("ulcer_index", np.nan)):
            dd = _drawdown_series(eq) * 100.0
            m["ulcer_index"] = float(np.sqrt((dd[dd < 0] ** 2).mean())) if (dd < 0).any() else 0.0
        if "total_return" not in m or pd.isna(m.get("total_return", np.nan)):
            m["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        if "annual_return" not in m or pd.isna(m.get("annual_return", np.nan)):
            years = len(eq) / TRADING_DAYS
            m["annual_return"] = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0) if years > 0 else np.nan

        dd = _drawdown_series(eq)
        downside = ret[ret < 0]
        m["downside_volatility"] = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) else np.nan
        var = float(np.percentile(ret, 5))
        m["var_5"] = var
        tail = ret[ret <= var]
        m["cvar_5"] = float(tail.mean()) if len(tail) else var
        m["longest_dd_duration"] = float(_longest_dd_duration(eq))
        m["avg_drawdown"] = float(dd.mean())
        m["current_drawdown"] = float(dd.iloc[-1])
        m["recovery_factor"] = float(m["total_return"] / abs(m["max_drawdown"])) if m.get("max_drawdown") else np.nan
        m["mar_ratio"] = m["calmar"]

        ann = ret.resample("YE").apply(lambda x: np.prod(1 + x) - 1)
        ann = ann.dropna()
        if len(ann):
            m["avg_annual_return"] = float(ann.mean())
            m["median_annual_return"] = float(ann.median())
            m["worst_year"] = float(ann.min())
            m["best_year"] = float(ann.max())

        if has_bench:
            try:
                bret = _daily_returns(bench).reindex(ret.index).fillna(0.0)
                active = ret - bret
                m["tracking_error"] = float(active.std() * np.sqrt(TRADING_DAYS))
                m["information_ratio"] = float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)) if active.std() else np.nan
                up = bret > 0
                down = bret < 0
                su = ret[up].mean() if up.any() else np.nan
                bu = bret[up].mean() if up.any() else np.nan
                sd = ret[down].mean() if down.any() else np.nan
                bd = bret[down].mean() if down.any() else np.nan
                m["up_capture"] = float(su / bu) if bu else np.nan
                m["down_capture"] = float(sd / bd) if bd else np.nan
            except Exception:
                pass

    # Trade-derived metrics.
    tm = _trade_metrics(record.trades, eq, eq.index if has_eq else None)
    m.update(tm)
    if "n_trades" not in m or pd.isna(m.get("n_trades", np.nan)):
        m["n_trades"] = float(len(record.trades)) if record.trades is not None else 0.0
    if "final_value" not in m or pd.isna(m.get("final_value", np.nan)):
        m["final_value"] = float(eq.iloc[-1]) if has_eq else np.nan
    return m


def _longest_dd_duration(equity: pd.Series) -> int:
    peak = equity.iloc[0]
    cur = 0
    longest = 0
    for v in equity.values:
        if v >= peak:
            peak = v
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
    return longest


def _trade_metrics(trades, equity, dates) -> dict[str, float]:
    if trades is None or trades.empty or equity is None or len(equity) == 0:
        return {}
    try:
        tdates = pd.to_datetime(trades["date"].dropna().unique())
        pos = equity.index.searchsorted(tdates)
        pos = np.clip(pos, 0, len(equity))
        edges = np.unique(np.concatenate([[0], pos, [len(equity)]]))
        legs = []
        for k in range(len(edges) - 1):
            a, b = edges[k], edges[k + 1]
            if b - a > 1:
                legs.append(equity.iloc[b - 1] / equity.iloc[a] - 1.0)
        if not legs:
            return {}
        legs = np.array(legs, dtype=float)
        wins = legs[legs > 0]
        losses = legs[legs < 0]
        pf = float(wins.sum() / (-losses.sum())) if len(losses) and losses.sum() < 0 else (np.inf if len(wins) else np.nan)
        out = {
            "win_rate": float(len(wins) / len(legs)),
            "profit_factor": pf,
            "expectancy": float(legs.mean()),
            "avg_gain": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "largest_winner": float(legs.max()),
            "largest_loser": float(legs.min()),
        }
        # Average holding period (days) per ticker.
        try:
            sub = trades.dropna(subset=["date", "ticker"]).copy()
            sub["date"] = pd.to_datetime(sub["date"])
            hp = []
            for _, g in sub.groupby("ticker"):
                ds = g["date"].sort_values()
                if len(ds) > 1:
                    hp.append(float(ds.diff().dropna().dt.days.mean()))
            if hp:
                out["avg_holding_days"] = float(np.mean(hp))
        except Exception:
            pass
        return out
    except Exception:
        return {}


# --------------------------------------------------------------------------
# Config / allocation / quality resolution
# --------------------------------------------------------------------------


def _cfg(config: dict, *keys, default=0.0):
    node = config
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, {})
    if isinstance(node, dict):
        return default
    return node


def _build_config_comparison(strategies: list[StrategyRecord]) -> pd.DataFrame:
    rows = []
    for s in strategies:
        cfg = s.config or {}
        pillar = _cfg(cfg, "quality", "pillar_weights", default={}) or {}
        rows.append({
            "Strategy": s.name,
            "Large Cap Weight": _cfg(cfg, "cap_segment", "large_cap_weight"),
            "Mid Cap Weight": _cfg(cfg, "cap_segment", "mid_cap_weight"),
            "Small Cap Weight": _cfg(cfg, "cap_segment", "small_cap_weight"),
            "Single Stock Max": _cfg(cfg, "portfolio", "max_position_pct"),
            "Portfolio Size": _cfg(cfg, "portfolio", "total_size"),
            "Rebalance Freq": _cfg(cfg, "general", "rebalance_frequency", default=""),
            "Momentum Top %": _cfg(cfg, "momentum", "top_pct"),
            "Momentum Top N": _cfg(cfg, "momentum", "top_n"),
            "Stability Top %": _cfg(cfg, "stability", "top_pct"),
            "Profitability W": pillar.get("profitability", 0) if isinstance(pillar, dict) else 0,
            "Growth W": pillar.get("growth", 0) if isinstance(pillar, dict) else 0,
            "Efficiency W": pillar.get("efficiency", 0) if isinstance(pillar, dict) else 0,
            "Fin. Strength W": pillar.get("financial_strength", 0) if isinstance(pillar, dict) else 0,
            "Shareholder W": pillar.get("shareholder_return", 0) if isinstance(pillar, dict) else 0,
            "Gate Top N": _cfg(cfg, "pipeline", "gates", default=0) if isinstance(_cfg(cfg, "pipeline", "gates", default=0), (int, float)) else 0,
            "Monte Carlo Seed": _cfg(cfg, "general", "monte_carlo_seed", default=""),
            "Optimization Method": _cfg(cfg, "general", "optimization_method", default=""),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def _build_allocation_df(strategies: list[StrategyRecord]) -> pd.DataFrame:
    rows = []
    for s in strategies:
        cfg = s.config or {}
        lc = _cfg(cfg, "cap_segment", "large_cap_weight")
        mc = _cfg(cfg, "cap_segment", "mid_cap_weight")
        sc = _cfg(cfg, "cap_segment", "small_cap_weight")
        cash = max(0.0, 1.0 - (lc + mc + sc))
        rows.append({
            "Strategy": s.name,
            "Large Cap %": lc,
            "Mid Cap %": mc,
            "Small Cap %": sc,
            "Cash %": cash,
            "Avg Stock Weight": _cfg(cfg, "portfolio", "max_position_pct"),
            "Portfolio Size": _cfg(cfg, "portfolio", "total_size"),
            "Exposure %": s.metrics.get("exposure", np.nan),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def _build_quality_df(strategies: list[StrategyRecord]) -> pd.DataFrame:
    rows = []
    for s in strategies:
        cfg = s.config or {}
        pillar = _cfg(cfg, "quality", "pillar_weights", default={}) or {}
        if isinstance(pillar, dict):
            rows.append({
                "Strategy": s.name,
                "Profitability": pillar.get("profitability", 0),
                "Growth": pillar.get("growth", 0),
                "Efficiency": pillar.get("efficiency", 0),
                "Financial Strength": pillar.get("financial_strength", 0),
                "Shareholder Return": pillar.get("shareholder_return", 0),
            })
        else:
            rows.append({"Strategy": s.name})
    return pd.DataFrame(rows).set_index("Strategy")


def _build_holdings_overlap(strategies: list[StrategyRecord]) -> pd.DataFrame:
    tickers = {}
    for s in strategies:
        tick = set()
        if s.snapshots:
            last = next(reversed(s.snapshots.values())) if isinstance(s.snapshots, dict) else None
            if last is not None and not last.empty and "ticker" in last.columns:
                tick = set(last["ticker"].astype(str))
        tickers[s.name] = tick
    names = [s.name for s in strategies]
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            sa, sb = tickers[a], tickers[b]
            union = sa | sb
            mat.loc[a, b] = float(len(sa & sb) / len(union)) if union else np.nan
    return mat


def _build_trade_df(metrics_list: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metrics_list:
        rows.append({
            "Strategy": m["_name"],
            "Trades": m.get("n_trades", np.nan),
            "Win Rate": m.get("win_rate", np.nan),
            "Profit Factor": m.get("profit_factor", np.nan),
            "Expectancy": m.get("expectancy", np.nan),
            "Avg Holding Days": m.get("avg_holding_days", np.nan),
            "Avg Gain": m.get("avg_gain", np.nan),
            "Avg Loss": m.get("avg_loss", np.nan),
            "Largest Winner": m.get("largest_winner", np.nan),
            "Largest Loser": m.get("largest_loser", np.nan),
            "Turnover": m.get("turnover", np.nan),
        })
    return pd.DataFrame(rows).set_index("Strategy")


def _build_benchmark_df(strategies: list[StrategyRecord], metrics_list: list[dict]) -> pd.DataFrame:
    rows = []
    for s, m in zip(strategies, metrics_list):
        bench_cagr = np.nan
        if s.benchmark is not None and not s.benchmark.empty and len(s.benchmark) > 1:
            bench_cagr = float(s.benchmark.iloc[-1] / s.benchmark.iloc[0] - 1.0)
            years = len(s.benchmark) / TRADING_DAYS
            if years > 0:
                bench_cagr = float((s.benchmark.iloc[-1] / s.benchmark.iloc[0]) ** (1 / years) - 1.0)
        rows.append({
            "Strategy": s.name,
            "Benchmark CAGR": bench_cagr,
            "Excess CAGR": (m.get("annual_return", np.nan) - bench_cagr) if not pd.isna(bench_cagr) else np.nan,
            "Tracking Error": m.get("tracking_error", np.nan),
            "Info Ratio": m.get("information_ratio", np.nan),
            "Up Capture": m.get("up_capture", np.nan),
            "Down Capture": m.get("down_capture", np.nan),
        })
    return pd.DataFrame(rows).set_index("Strategy")


# --------------------------------------------------------------------------
# Rankings / recommendations / stats
# --------------------------------------------------------------------------


def _normalize_series(s: pd.Series, method: Literal["minmax", "zscore", "rank"] = "minmax") -> pd.Series:
    if method == "minmax":
        mn, mx = s.min(), s.max()
        if mx == mn or pd.isna(mx) or pd.isna(mn):
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - mn) / (mx - mn)
    elif method == "zscore":
        std = s.std()
        if std == 0 or pd.isna(std):
            return pd.Series([0.0] * len(s), index=s.index)
        return (s - s.mean()) / std
    return s.rank(pct=True)


def _build_rankings(metrics_list: list[dict], weights: dict) -> pd.DataFrame:
    names = [m["_name"] for m in metrics_list]
    def col(key):
        return pd.Series([m.get(key, np.nan) for m in metrics_list], index=names)
    specs = {
        "CAGR": ("annual_return", "higher"),
        "Sharpe": ("sharpe", "higher"),
        "Sortino": ("sortino", "higher"),
        "Calmar": ("calmar", "higher"),
        "Max DD": ("max_drawdown", "lower"),
        "Volatility": ("annual_volatility", "lower"),
    }
    norm = {}
    for label, (key, direction) in specs.items():
        series = col(key)
        if direction == "lower":
            series = -series
        norm[label] = _normalize_series(series.fillna(series.min() if direction == "higher" else series.max()))
    composite = pd.Series(0.0, index=names)
    for label, w in weights.items():
        composite = composite + norm.get(label, pd.Series(0.0, index=names)) * w
    total_w = sum(weights.values()) or 1.0
    composite = composite / total_w
    df = pd.DataFrame({label: norm[label] for label in norm})
    df["composite"] = composite
    df.index = names
    return df.sort_values("composite", ascending=False)


def _build_recommendations(strategies: list[StrategyRecord], metrics_list: list[dict], rankings: pd.DataFrame) -> dict:
    names = [m["_name"] for m in metrics_list]
    def c(key):
        return pd.Series([m.get(key, np.nan) for m in metrics_list], index=names)
    out = {}
    if not names:
        return out
    cagr = c("annual_return")
    vol = c("annual_volatility")
    sharpe = c("sharpe")
    calmar = c("calmar")
    mdd = c("max_drawdown")
    best_comp = rankings.index[0]

    def pick(series, ascending):
        s = series.dropna()
        if s.empty:
            return None
        idx = s.idxmin() if ascending else s.idxmax()
        return idx, float(s.loc[idx])

    r = pick(cagr, ascending=False)
    if r: out["highest_cagr"] = {"strategy": r[0], "value": r[1], "reason": f"Highest compound annual growth rate ({r[1]*100:.1f}%)."}
    r = pick(vol, ascending=True)
    if r: out["lowest_risk"] = {"strategy": r[0], "value": r[1], "reason": f"Lowest annualized volatility ({r[1]*100:.1f}%)."}
    r = pick(sharpe, ascending=False)
    if r: out["highest_sharpe"] = {"strategy": r[0], "value": r[1], "reason": f"Best risk-adjusted return (Sharpe {r[1]:.2f})."}
    r = pick(calmar, ascending=False)
    if r: out["highest_calmar"] = {"strategy": r[0], "value": r[1], "reason": f"Highest return-to-drawdown (Calmar {r[1]:.2f})."}
    r = pick(mdd, ascending=True)
    if r: out["most_stable"] = {"strategy": r[0], "value": r[1], "reason": f"Smallest peak-to-trough drawdown ({r[1]*100:.1f}%)."}
    out["best_risk_adjusted"] = {"strategy": best_comp, "value": float(rankings.loc[best_comp, "composite"]),
                                 "reason": "Highest composite score across user-weighted metrics."}
    r = pick(cagr, ascending=False)
    if r: out["best_long_term_compounder"] = {"strategy": r[0], "value": r[1], "reason": f"Highest CAGR ({r[1]*100:.1f}%) for long-horizon compounding."}
    cons = c("max_drawdown").where(sharpe >= 1.0)
    r = pick(cons, ascending=True)
    if r: out["best_conservative"] = {"strategy": r[0], "value": r[1], "reason": f"Lowest drawdown ({r[1]*100:.1f}%) among strategies with Sharpe ≥ 1."}
    return out


def _build_stats_tests(strategies: list[StrategyRecord]) -> dict:
    out = {}
    for s in strategies:
        if s.equity is None or s.equity.empty or s.benchmark is None or s.benchmark.empty:
            continue
        ret = _daily_returns(s.equity)
        bret = _daily_returns(s.benchmark).reindex(ret.index).fillna(0.0)
        common = ret.index.intersection(bret.index)
        if len(common) < 30:
            continue
        r = ret.loc[common]
        b = bret.loc[common]
        active = r - b
        ttest = scipy_stats.ttest_rel(r, b) if len(r) > 1 else None
        # Bootstrap CI of mean active daily return.
        rng = np.random.default_rng(0)
        boot = np.array([rng.choice(active.values, len(active), replace=True).mean() for _ in range(300)])
        ci_lo, ci_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        roll = (1 + r.rolling(TRADING_DAYS).apply(lambda x: np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1 if len(x) else 0))
        broll = (1 + b.rolling(TRADING_DAYS).apply(lambda x: np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1 if len(x) else 0))
        roll = roll.dropna()
        broll = broll.reindex(roll.index).dropna()
        outperf_freq = float((r > b).mean()) if len(r) else np.nan
        roll_outperf = float((roll > broll).mean()) if len(roll) and len(broll) else np.nan
        out[s.name] = {
            "excess_cagr": float(r.mean() * TRADING_DAYS - b.mean() * TRADING_DAYS),
            "paired_t": float(ttest.statistic) if ttest else np.nan,
            "paired_p": float(ttest.pvalue) if ttest else np.nan,
            "bootstrap_ci": (ci_lo * TRADING_DAYS, ci_hi * TRADING_DAYS),
            "outperformance_frequency": outperf_freq,
            "rolling_outperformance_pct": roll_outperf,
        }
    return out


def _build_radar(metrics_list: list[dict]) -> pd.DataFrame:
    names = [m["_name"] for m in metrics_list]
    specs = {
        "CAGR": ("annual_return", "higher"),
        "Sharpe": ("sharpe", "higher"),
        "Sortino": ("sortino", "higher"),
        "Calmar": ("calmar", "higher"),
        "Drawdown": ("max_drawdown", "lower"),
        "Win Rate": ("win_rate", "higher"),
        "Volatility": ("annual_volatility", "lower"),
        "Recovery": ("recovery_factor", "higher"),
    }
    df = pd.DataFrame(index=names)
    for label, (key, direction) in specs.items():
        series = pd.Series([m.get(key, np.nan) for m in metrics_list], index=names)
        if direction == "lower":
            series = -series
        df[label] = _normalize_series(series.fillna(series.min() if direction == "higher" else series.max()))
    return df


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------


def compare_strategies(
    strategies: list[StrategyRecord],
    ranking_weights: dict | None = None,
) -> ComparisonResult:
    """Compare multiple strategies and return all analytics."""
    if not strategies:
        raise ValueError("No strategies to compare")

    weights = dict(_DEFAULT_RANK_WEIGHTS)
    if ranking_weights:
        weights.update(ranking_weights)

    metrics_list = []
    for s in strategies:
        m = _enrich_metrics(s)
        m["_name"] = s.name
        metrics_list.append(m)

    names = [s.name for s in strategies]

    perf_rows = []
    for m in metrics_list:
        perf_rows.append({
            "CAGR": m.get("annual_return", np.nan),
            "Annualized Return": m.get("annual_return", np.nan),
            "Volatility": m.get("annual_volatility", np.nan),
            "Sharpe": m.get("sharpe", np.nan),
            "Sortino": m.get("sortino", np.nan),
            "Calmar": m.get("calmar", np.nan),
            "Max DD": m.get("max_drawdown", np.nan),
            "Ulcer Index": m.get("ulcer_index", np.nan),
            "MAR": m.get("mar_ratio", np.nan),
            "Info Ratio": m.get("information_ratio", np.nan),
            "Treynor": m.get("treynor", np.nan),
            "Beta": m.get("beta", np.nan),
            "Alpha": m.get("alpha_annual", np.nan),
            "Win Rate": m.get("win_rate", np.nan),
            "Profit Factor": m.get("profit_factor", np.nan),
            "Recovery Factor": m.get("recovery_factor", np.nan),
            "Avg Annual Return": m.get("avg_annual_return", np.nan),
            "Median Annual Return": m.get("median_annual_return", np.nan),
            "Worst Year": m.get("worst_year", np.nan),
            "Best Year": m.get("best_year", np.nan),
            "Std Dev": m.get("annual_volatility", np.nan),
            "Avg Holding Days": m.get("avg_holding_days", np.nan),
            "Turnover": m.get("turnover", np.nan),
            "Trades": m.get("n_trades", np.nan),
            "Final Value": m.get("final_value", np.nan),
        })
    perf_df = pd.DataFrame(perf_rows, index=names)

    # Equity / drawdown / returns.
    equity_frames, dd_frames, ret_frames = {}, {}, {}
    for s in strategies:
        if s.equity is not None and not s.equity.empty:
            equity_frames[s.name] = s.equity
            dd_frames[s.name] = _drawdown_series(s.equity)
            ret_frames[s.name] = _daily_returns(s.equity)
    equity_df = pd.DataFrame(equity_frames)
    dd_df = pd.DataFrame(dd_frames)

    rolling_frames, annual_frames, monthly_frames = {}, {}, {}
    for s in strategies:
        if s.equity is not None and len(s.equity) >= TRADING_DAYS:
            ret = _daily_returns(s.equity)
            rolling = (1 + ret.rolling(TRADING_DAYS).apply(
                lambda x: np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1 if len(x) else 0))
            rolling_frames[s.name] = rolling
            annual = ret.resample("YE").apply(lambda x: np.prod(1 + x) - 1)
            annual_frames[s.name] = annual
            monthly = ret.resample("ME").apply(lambda x: np.prod(1 + x) - 1)
            monthly_frames[s.name] = monthly
    rolling_df = pd.DataFrame(rolling_frames)
    annual_df = pd.DataFrame(annual_frames)
    monthly_df = pd.DataFrame(monthly_frames)

    # Correlation on aligned returns.
    if ret_frames:
        common = None
        for ser in ret_frames.values():
            common = ser.index if common is None else common.intersection(ser.index)
        aligned = pd.DataFrame({k: v.reindex(common).fillna(0.0) for k, v in ret_frames.items()})
        corr_df = aligned.corr()
    else:
        corr_df = pd.DataFrame()

    risk_rows = []
    for m in metrics_list:
        risk_rows.append({
            "Volatility": m.get("annual_volatility", np.nan),
            "Downside Vol": m.get("downside_volatility", np.nan),
            "Beta": m.get("beta", np.nan),
            "Tracking Error": m.get("tracking_error", np.nan),
            "VaR 5%": m.get("var_5", np.nan),
            "CVaR 5%": m.get("cvar_5", np.nan),
            "Ulcer Index": m.get("ulcer_index", np.nan),
            "Semi Deviation": m.get("downside_volatility", np.nan),
            "Max DD": m.get("max_drawdown", np.nan),
            "Longest DD": m.get("longest_dd_duration", np.nan),
            "Avg Drawdown": m.get("avg_drawdown", np.nan),
            "Current DD": m.get("current_drawdown", np.nan),
        })
    risk_df = pd.DataFrame(risk_rows, index=names)

    rankings = _build_rankings(metrics_list, weights)
    recommendations = _build_recommendations(strategies, metrics_list, rankings)
    allocation_df = _build_allocation_df(strategies)
    quality_df = _build_quality_df(strategies)
    holdings_overlap_df = _build_holdings_overlap(strategies)
    trade_df = _build_trade_df(metrics_list)
    benchmark_df = _build_benchmark_df(strategies, metrics_list)
    radar_df = _build_radar(metrics_list)
    stats_tests = _build_stats_tests(strategies)

    return ComparisonResult(
        strategies=strategies,
        config_comparison=_build_config_comparison(strategies),
        performance_table=perf_df,
        equity_curves=equity_df,
        drawdown_curves=dd_df,
        rolling_returns=rolling_df,
        annual_returns=annual_df,
        monthly_returns=monthly_df,
        risk_table=risk_df,
        correlation_matrix=corr_df,
        rankings=rankings,
        recommendations=recommendations,
        allocation_df=allocation_df,
        quality_df=quality_df,
        holdings_overlap_df=holdings_overlap_df,
        trade_df=trade_df,
        benchmark_df=benchmark_df,
        radar_df=radar_df,
        stats_tests=stats_tests,
    )


__all__ = ["ComparisonResult", "compare_strategies"]
