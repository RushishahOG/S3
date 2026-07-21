"""Sensitivity Analysis computation engine.

This module is the Streamlit-independent core of the Sensitivity Analysis
Research Lab module. It:

* builds a curated catalogue of strategy parameters that map to real fields in
  :class:`~core.config.backtest_schema.BacktestParameters`;
* generates one-way / two-way / multi-parameter grids from user ranges;
* runs the ARQM backtest for every combination (with an in-process result cache
  so repeated analyses reuse prior runs and only re-run modified combinations);
* collects the full performance & risk metric suite;
* computes sensitivity scores, stability statistics, parameter importance,
  correlation, interaction and robustness analytics;
* derives robust-parameter recommendations.

Heavy backtest I/O is serialised on the engine's shared read lock so it does
not contend on the DuckDB store (see :mod:`core.optimization.engine`).
"""

from __future__ import annotations

import hashlib
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from dataclasses import replace

from core.backtesting.engine import _rebalance_dates, run_backtest
from core.config.backtest_schema import BacktestParameters
from core.optimization.engine import _BACKTEST_IO_LOCK
from core.optimization.spec import OptimizerParamSpec, ParamKind

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

#: Display labels for every metric emitted per combination.
METRIC_LABELS: dict[str, str] = {
    "cagr": "CAGR",
    "annual_return": "Annualized Return",
    "sharpe": "Sharpe Ratio",
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "max_drawdown": "Maximum Drawdown",
    "annual_volatility": "Volatility",
    "win_rate": "Win Rate",
    "profit_factor": "Profit Factor",
    "portfolio_turnover": "Portfolio Turnover",
    "recovery_factor": "Recovery Factor",
    "avg_holding_period_days": "Avg Holding Period (days)",
    "final_portfolio_value": "Final Portfolio Value",
    "number_of_trades": "Number of Trades",
}

#: Whether a higher value of the metric is better (used for tornado direction
#: and recommendation logic).
HIGHER_IS_BETTER: dict[str, bool] = {
    "cagr": True,
    "annual_return": True,
    "sharpe": True,
    "sortino": True,
    "calmar": True,
    "max_drawdown": False,
    "annual_volatility": False,
    "win_rate": True,
    "profit_factor": True,
    "portfolio_turnover": False,
    "recovery_factor": True,
    "avg_holding_period_days": True,
    "final_portfolio_value": True,
    "number_of_trades": True,
}

#: Metrics used for the composite parameter-importance score.
IMPORTANCE_METRICS = ["cagr", "sharpe", "max_drawdown", "calmar"]


# ---------------------------------------------------------------------------
# Parameter catalogue
# ---------------------------------------------------------------------------
def build_catalog() -> list[OptimizerParamSpec]:
    """Return the curated sensitivity-parameter catalogue.

    Grouped parameters reuse the **same keys** as the optimization registry so
    that :func:`core.optimization.candidate.build_candidate` correctly
    normalises sum-groups (cap / scoring / quality pillars). ``current`` values
    are placeholders overwritten at runtime from the selected base strategy.
    """
    return [
        # --- Portfolio Construction ------------------------------------
        OptimizerParamSpec("large_cap_weight", "Large Cap Allocation", "Portfolio Construction",
                           "cap_segment", "large_cap_weight", ParamKind.CONTINUOUS, 0.60, 0.0, 1.0, 0.05,
                           help="Large cap allocation fraction (0-1)"),
        OptimizerParamSpec("mid_cap_weight", "Mid Cap Allocation", "Portfolio Construction",
                           "cap_segment", "mid_cap_weight", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           help="Mid cap allocation fraction (0-1)"),
        OptimizerParamSpec("small_cap_weight", "Small Cap Allocation", "Portfolio Construction",
                           "cap_segment", "small_cap_weight", ParamKind.CONTINUOUS, 0.10, 0.0, 1.0, 0.05,
                           help="Small cap allocation fraction (0-1)"),
        OptimizerParamSpec("max_position_pct", "Single Stock Weight Limit", "Portfolio Construction",
                           "portfolio", "max_position_pct", ParamKind.CONTINUOUS, 0.07, 0.01, 0.5, 0.01,
                           help="Maximum single-stock position weight (0-1)"),
        OptimizerParamSpec("total_size", "Portfolio Size", "Portfolio Construction",
                           "portfolio", "total_size", ParamKind.DISCRETE, 50, 5, 200, 5,
                           allowed=(10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200),
                           help="Total number of holdings"),
        OptimizerParamSpec("rebalance_frequency", "Rebalancing Frequency", "Portfolio Construction",
                           "general", "rebalance_frequency", ParamKind.CATEGORICAL, "quarterly",
                           allowed=("monthly", "quarterly", "semi_annual"),
                           help="Portfolio rebalancing frequency"),
        # --- Momentum ----------------------------------------------------
        OptimizerParamSpec("mom_lookback", "Momentum Lookback (Months)", "Momentum",
                           "momentum", "horizon_months", ParamKind.DISCRETE, 12, 1, 36, 1,
                           allowed=(3, 6, 9, 12, 18, 24, 36),
                           help="Momentum lookback horizon (months)"),
        OptimizerParamSpec("w_momentum", "Momentum Weight", "Momentum",
                           "scoring", "momentum_weight", ParamKind.CONTINUOUS, 0.40, 0.0, 1.0, 0.05,
                           help="Final scoring momentum weight (0-1)"),
        OptimizerParamSpec("mom_top_pct", "Momentum Gate Top %", "Momentum",
                           "momentum", "top_pct", ParamKind.CONTINUOUS, 0.30, 0.05, 1.0, 0.05,
                           help="Momentum gate top-pct selection threshold (0-1)"),
        OptimizerParamSpec("mom_top_n", "Momentum Gate Top N", "Momentum",
                           "momentum", "top_n", ParamKind.DISCRETE, 50, 5, 200, 5,
                           allowed=(10, 20, 30, 40, 50, 75, 100, 150, 200),
                           help="Momentum gate top-N selection count"),
        # --- Quality Gates ----------------------------------------------
        OptimizerParamSpec("w_profitability", "Profitability Weight", "Quality Gates",
                           "quality", "profitability_pillar", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           help="Quality profitability pillar weight (0-1)"),
        OptimizerParamSpec("w_growth", "Growth Weight", "Quality Gates",
                           "quality", "growth_pillar", ParamKind.CONTINUOUS, 0.30, 0.0, 1.0, 0.05,
                           help="Quality growth pillar weight (0-1)"),
        OptimizerParamSpec("w_fin_strength", "Financial Strength Weight", "Quality Gates",
                           "quality", "fin_strength_pillar", ParamKind.CONTINUOUS, 0.15, 0.0, 1.0, 0.05,
                           help="Quality financial-strength pillar weight (0-1)"),
        OptimizerParamSpec("w_cashflow", "Efficiency / Cash Flow Weight", "Quality Gates",
                           "quality", "cashflow_pillar", ParamKind.CONTINUOUS, 0.15, 0.0, 1.0, 0.05,
                           help="Quality cash-flow / efficiency pillar weight (0-1)"),
        OptimizerParamSpec("w_shareholder", "Shareholder Return Weight", "Quality Gates",
                           "quality", "shareholder_pillar", ParamKind.CONTINUOUS, 0.10, 0.0, 1.0, 0.05,
                           help="Quality shareholder-return pillar weight (0-1)"),
        # --- Market Timing ----------------------------------------------
        OptimizerParamSpec("buy_trigger_pct", "Buy Trigger %", "Market Timing",
                           "regime", "buy_trigger_pct", ParamKind.CONTINUOUS, 5.0, -50.0, 50.0, 0.5,
                           help="Regime buy-trigger percentage"),
        OptimizerParamSpec("sell_trigger_pct", "Sell Trigger %", "Market Timing",
                           "regime", "sell_trigger_pct", ParamKind.CONTINUOUS, -15.0, -50.0, 0.0, 0.5,
                           help="Regime sell-trigger percentage"),
    ]


def catalog_by_key() -> dict[str, OptimizerParamSpec]:
    return {s.key: s for s in build_catalog()}


# ---------------------------------------------------------------------------
# Value / grid generation
# ---------------------------------------------------------------------------
def generate_values(spec: OptimizerParamSpec, rng: dict[str, Any]) -> list[Any]:
    """Generate the list of values for one parameter from a user range."""
    mn = float(rng.get("min", spec.min if spec.min is not None else 0.0))
    mx = float(rng.get("max", spec.max if spec.max is not None else 1.0))
    step = float(rng.get("step", spec.step if spec.step is not None else 1.0))

    if spec.kind == ParamKind.CATEGORICAL:
        allowed = rng.get("choices") or list(spec.allowed or [])
        return [a for a in allowed if mn <= _ord(a) <= mx] if allowed else list(spec.allowed or [])

    if spec.kind == ParamKind.DISCRETE:
        if spec.allowed:
            vals = [a for a in spec.allowed if mn - 1e-9 <= float(a) <= mx + 1e-9]
            if vals:
                return [int(v) for v in vals]
        raw = np.arange(int(round(mn)), int(round(mx)) + 1, max(1, int(round(step))))
        return [int(v) for v in raw]

    # Continuous
    if step <= 0:
        step = 0.01
    raw = np.arange(mn, mx + step * 0.5, step)
    return [round(float(v), 6) for v in raw]


def build_combos(
    mode: str,
    selected_keys: list[str],
    value_map: dict[str, list[Any]],
    max_combinations: int = 1000,
) -> list[dict[str, Any]]:
    """Build the list of parameter-combination dicts for the chosen mode."""
    if not selected_keys:
        return []
    if mode == "one_way":
        k0 = selected_keys[0]
        return [{k0: v} for v in value_map[k0]]
    if mode == "two_way":
        k0, k1 = selected_keys[0], selected_keys[1]
        return [{k0: a, k1: b} for a in value_map[k0] for b in value_map[k1]]
    # multi
    keys = list(selected_keys)
    grids = [value_map[k] for k in keys]
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grids)]
    if len(combos) > max_combinations:
        idx = np.linspace(0, len(combos) - 1, max_combinations).astype(int)
        combos = [combos[i] for i in idx]
    return combos


# ---------------------------------------------------------------------------
# Backtest evaluation + metrics
# ---------------------------------------------------------------------------
_RESULT_CACHE: dict[str, dict[str, float]] = {}
_CACHE_LOCK = threading.Lock()


def _cache_key(base: BacktestParameters, vals: dict[str, Any]) -> str:
    base_repr = repr(base.to_dict())
    val_repr = tuple(sorted((k, _hashable_val(v)) for k, v in vals.items()))
    blob = base_repr + "::" + repr(val_repr)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _hashable_val(v: Any) -> Any:
    if isinstance(v, float):
        return round(v, 8)
    return v


def _collect_metrics(result, cfg: BacktestParameters) -> dict[str, float]:
    """Augment engine metrics with final value, turnover and trade analytics."""
    m = dict(result.metrics)
    nav = result.nav
    if nav is not None and not nav.empty:
        m["final_portfolio_value"] = float(nav.iloc[-1])
        total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if nav.iloc[0] else float("nan")
    else:
        m["final_portfolio_value"] = float("nan")
        total_ret = float("nan")
    m["cagr"] = m.get("annual_return", float("nan"))
    trades = result.trades
    if trades is not None and not trades.empty and "weight" in trades.columns:
        m["portfolio_turnover"] = float(trades["weight"].abs().sum() / 2.0)
        m["number_of_trades"] = float(len(trades))
        m["win_rate"], m["profit_factor"], m["avg_holding_period_days"] = _trade_stats(nav, trades, cfg)
    else:
        m["portfolio_turnover"] = 0.0
        m["number_of_trades"] = 0.0
        m["win_rate"], m["profit_factor"], m["avg_holding_period_days"] = float("nan"), float("nan"), float("nan")

    mdd = m.get("max_drawdown", float("nan"))
    if pd.notna(mdd) and mdd != 0:
        m["recovery_factor"] = total_ret / abs(mdd)
    else:
        m["recovery_factor"] = float("nan")
    return m


def _trade_stats(nav: pd.Series, trades: pd.DataFrame, cfg: BacktestParameters):
    """Rebalance-period win rate / profit factor and average holding period."""
    # Win rate / profit factor on rebalance-period returns.
    win_rate = profit_factor = float("nan")
    if nav is not None and not nav.empty:
        try:
            rb = _rebalance_dates(
                pd.Timestamp(cfg.general.start_date),
                pd.Timestamp(cfg.general.end_date),
                cfg.general.rebalance_frequency,
            )
            rb = [d for d in rb if d in nav.index]
            if len(rb) >= 2:
                sub = nav.reindex(rb)
                rets = sub.pct_change().dropna()
                if len(rets) > 0:
                    pos = rets[rets > 0]
                    neg = rets[rets < 0]
                    win_rate = float((rets > 0).mean())
                    if len(neg) > 0 and neg.sum() < 0:
                        profit_factor = float(pos.sum() / abs(neg.sum()))
        except Exception:
            pass

    # Average holding period from paired BUY -> SELL events per ticker.
    avg_holding = float("nan")
    try:
        tr = trades.sort_values("date")
        holdings = []
        open_buys: dict[str, pd.Timestamp] = {}
        for _, row in tr.iterrows():
            t = row["ticker"]
            d = row["date"]
            if row["action"] == "BUY" and float(row["weight"]) > 0:
                open_buys[t] = d
            elif row["action"] == "SELL" and t in open_buys:
                delta = (pd.Timestamp(d) - pd.Timestamp(open_buys[t])).days
                if delta >= 0:
                    holdings.append(delta)
                del open_buys[t]
        if holdings:
            avg_holding = float(np.mean(holdings))
    except Exception:
        pass
    return win_rate, profit_factor, avg_holding


def build_sensitivity_candidate(base: BacktestParameters, vals: dict[str, Any]) -> BacktestParameters:
    """Build an isolated candidate config from ``base`` + ``vals``.

    Unlike the generic optimizer :func:`build_candidate`, this builder correctly
    handles parameters that live on *nested* schema fields (momentum factor
    horizon, quality pillar-weight dictionary) and re-normalises the cap and
    scoring sum-groups so the resulting frozen config always validates.
    """
    general = base.general
    regime = base.regime
    cap = base.cap_segment
    portfolio = base.portfolio
    scoring = base.scoring
    quality = base.quality
    momentum = base.momentum

    rg: dict[str, Any] = {}
    if "buy_trigger_pct" in vals:
        rg["buy_trigger_pct"] = float(vals["buy_trigger_pct"])
    if "sell_trigger_pct" in vals:
        rg["sell_trigger_pct"] = float(vals["sell_trigger_pct"])
    if rg:
        regime = replace(regime, **rg)

    if "rebalance_frequency" in vals:
        general = replace(general, rebalance_frequency=vals["rebalance_frequency"])

    pf: dict[str, Any] = {}
    if "max_position_pct" in vals:
        pf["max_position_pct"] = float(vals["max_position_pct"])
    if "total_size" in vals:
        new_total = int(vals["total_size"])
        old = portfolio.large_size + portfolio.mid_size + portfolio.small_size
        if old > 0:
            ls = round(portfolio.large_size * new_total / old)
            ms = round(portfolio.mid_size * new_total / old)
            ss = max(0, new_total - ls - ms)
        else:
            ls, ms, ss = new_total, 0, 0
        pf.update(total_size=new_total, large_size=ls, mid_size=ms, small_size=ss)
    if pf:
        portfolio = replace(portfolio, **pf)

    cap_vals: dict[str, float] = {}
    if "large_cap_weight" in vals:
        cap_vals["large_cap_weight"] = float(vals["large_cap_weight"])
    if "mid_cap_weight" in vals:
        cap_vals["mid_cap_weight"] = float(vals["mid_cap_weight"])
    if "small_cap_weight" in vals:
        cap_vals["small_cap_weight"] = float(vals["small_cap_weight"])
    if cap_vals:
        l = cap_vals.get("large_cap_weight", cap.large_cap_weight)
        m = cap_vals.get("mid_cap_weight", cap.mid_cap_weight)
        s = cap_vals.get("small_cap_weight", cap.small_cap_weight)
        tot = l + m + s
        if tot > 0:
            l, m, s = l / tot, m / tot, s / tot
        cap = replace(cap, large_cap_weight=l, mid_cap_weight=m, small_cap_weight=s)

    sc_vals: dict[str, float] = {}
    if "w_momentum" in vals:
        sc_vals["momentum_weight"] = float(vals["w_momentum"])
    if "w_quality" in vals:
        sc_vals["quality_weight"] = float(vals["w_quality"])
    if "w_stability" in vals:
        sc_vals["stability_weight"] = float(vals["w_stability"])
    if sc_vals:
        mw = sc_vals.get("momentum_weight", scoring.momentum_weight)
        qw = sc_vals.get("quality_weight", scoring.quality_weight)
        sw = sc_vals.get("stability_weight", scoring.stability_weight)
        tot = mw + qw + sw
        if tot > 0:
            mw, qw, sw = mw / tot, qw / tot, sw / tot
        scoring = replace(scoring, momentum_weight=mw, quality_weight=qw, stability_weight=sw)

    qp = dict(quality.pillar_weights)
    qmap = {"w_profitability": "profitability", "w_growth": "growth",
            "w_fin_strength": "financial_strength", "w_cashflow": "cash_flow",
            "w_shareholder": "shareholder_return"}
    changed = False
    for k, pillar in qmap.items():
        if k in vals:
            qp[pillar] = float(vals[k])
            changed = True
    if changed:
        tot = sum(qp.values())
        if tot > 0:
            qp = {p: v / tot for p, v in qp.items()}
        quality = replace(quality, pillar_weights=qp)

    mo: dict[str, Any] = {}
    if "mom_top_pct" in vals:
        mo["top_pct"] = float(vals["mom_top_pct"])
    if "mom_top_n" in vals:
        mo["top_n"] = int(vals["mom_top_n"])
    if "mom_lookback" in vals:
        h = int(vals["mom_lookback"])
        momentum = replace(momentum, factors=tuple(
            replace(f, horizon_months=h) for f in momentum.factors))
    if mo:
        momentum = replace(momentum, **mo)

    return BacktestParameters(
        general=general, regime=regime, universe=base.universe, cap_segment=cap,
        momentum=momentum, stability=base.stability, persistence=base.persistence,
        quality=quality, scoring=scoring, portfolio=portfolio,
        management=base.management, pipeline=base.pipeline,
    )


def _evaluate_one(
    base: BacktestParameters,
    vals: dict[str, Any],
    specs: list[OptimizerParamSpec],
    storage,
    shared_data=None,
) -> dict[str, float]:
    key = _cache_key(base, vals)
    with _CACHE_LOCK:
        hit = _RESULT_CACHE.get(key)
    if hit is not None:
        return hit
    with _BACKTEST_IO_LOCK:
        cfg = build_sensitivity_candidate(base, vals)
        result = run_backtest(cfg, storage, data=shared_data)
    metrics = _collect_metrics(result, cfg)
    with _CACHE_LOCK:
        _RESULT_CACHE[key] = metrics
    return metrics


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
@dataclass
class SensitivityResult:
    base: BacktestParameters
    baseline_metrics: dict[str, float]
    specs: list[OptimizerParamSpec]
    mode: str
    primary_metric: str
    combos: list[dict[str, Any]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)

    @property
    def selected_keys(self) -> list[str]:
        return [s.key for s in self.specs]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)


def run_sensitivity(
    base: BacktestParameters,
    selected_keys: list[str],
    ranges: dict[str, dict[str, Any]],
    mode: str,
    max_combinations: int = 1000,
    primary_metric: str = "sharpe",
    workers: int = 1,
    progress_callback: Callable[[dict], None] | None = None,
    storage_factory: Callable[[], Any] | None = None,
) -> SensitivityResult:
    """Run a full sensitivity analysis and return a :class:`SensitivityResult`."""
    all_specs = catalog_by_key()
    specs = [all_specs[k] for k in selected_keys if k in all_specs]
    value_map = {k: generate_values(all_specs[k], ranges.get(k, {})) for k in selected_keys}
    combos = build_combos(mode, selected_keys, value_map, max_combinations)

    if storage_factory is None:
        from app.services import get_storage
        storage_factory = get_storage

    storage = storage_factory()

    # Load the engineered dataset ONCE and reuse it for every combination
    # (only strategy params vary, not universe / date range). This satisfies
    # the "reuse cached computations" requirement and is the key performance
    # lever for sensitivity sweeps.
    from core.backtesting.data import load_backtest_data
    shared_data = load_backtest_data(storage, base)

    # Baseline (base configuration) — cached.
    baseline_metrics = _evaluate_one(base, {}, specs, storage, shared_data)

    records: list[dict[str, Any]] = []
    result = SensitivityResult(
        base=base, baseline_metrics=baseline_metrics, specs=specs,
        mode=mode, primary_metric=primary_metric, combos=combos,
    )

    if not combos:
        return result

    total = len(combos)
    start = time.time()
    done = 0
    lock = threading.Lock()

    def _work(i: int) -> None:
        nonlocal done
        vals = combos[i]
        m = _evaluate_one(base, vals, specs, storage, shared_data)
        rec = dict(vals)
        rec.update(m)
        with lock:
            records.append(rec)
            done += 1
            d = done
        if progress_callback is not None:
            elapsed = time.time() - start
            rate = d / elapsed if elapsed > 0 else 0.0
            eta = (total - d) / rate if rate > 0 else None
            progress_callback({
                "event": "combo_done", "done": d, "total": total,
                "current": vals, "metrics": m, "eta": eta,
            })

    if progress_callback is not None:
        progress_callback({"event": "start", "total": total})

    n_workers = max(1, min(workers, total))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(_work, range(total)))

    result.records = records
    if progress_callback is not None:
        progress_callback({"event": "done", "result": result})
    return result


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def _frame(result: SensitivityResult) -> pd.DataFrame:
    if not result.records:
        return pd.DataFrame()
    return pd.DataFrame(result.records)


def sensitivity_scores(result: SensitivityResult, metric: str | None = None) -> pd.DataFrame:
    """Compute the normalized sensitivity score (0-100) for each parameter."""
    metric = metric or result.primary_metric
    df = _frame(result)
    if df.empty:
        return pd.DataFrame()
    base_val = result.baseline_metrics.get(metric, float("nan"))
    base_denom = abs(base_val) if pd.notna(base_val) and base_val != 0 else 1.0

    rows = []
    raw_scores = {}
    for spec in result.specs:
        k = spec.key
        if k not in df.columns:
            continue
        sub = df.groupby(k)[metric].mean()
        if sub.empty:
            continue
        perf_range = float(sub.max() - sub.min())
        perf_change_pct = (perf_range / base_denom) * 100.0 if base_denom else 0.0
        vals = list(sub.index)
        if spec.kind == ParamKind.CATEGORICAL:
            param_change_pct = 100.0
        else:
            vmin, vmax = float(min(vals)), float(max(vals))
            bv = float(getattr(getattr(result.base, spec.block), spec.field, 0.0))
            param_denom = abs(bv) if bv != 0 else (abs(vmax) if vmax != 0 else 1.0)
            param_change_pct = ((vmax - vmin) / param_denom) * 100.0 if param_denom else 100.0
        raw = perf_change_pct / param_change_pct if param_change_pct > 0 else 0.0
        raw_scores[k] = raw
        rows.append({
            "parameter": spec.name, "key": k,
            "perf_change_pct": perf_change_pct,
            "param_change_pct": param_change_pct,
            "raw_sensitivity": raw,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    mx = out["raw_sensitivity"].max()
    out["sensitivity_score"] = (out["raw_sensitivity"] / mx * 100.0) if mx > 0 else 0.0
    out["sensitivity_score"] = out["sensitivity_score"].round(2)
    return out.sort_values("sensitivity_score", ascending=False).reset_index(drop=True)


def stability_analysis(result: SensitivityResult, metric: str | None = None) -> pd.DataFrame:
    """Per-parameter stability statistics and classification."""
    metric = metric or result.primary_metric
    df = _frame(result)
    if df.empty:
        return pd.DataFrame()
    base_val = result.baseline_metrics.get(metric, float("nan"))
    rows = []
    for spec in result.specs:
        k = spec.key
        if k not in df.columns:
            continue
        sub = df.groupby(k)[metric].mean()
        vals = list(sub.index)
        std = float(sub.std()) if len(sub) > 1 else 0.0
        mean = float(sub.mean())
        median = float(sub.median())
        cv = (std / abs(mean)) * 100.0 if mean != 0 else float("nan")
        best = float(sub.max()) if HIGHER_IS_BETTER.get(metric, True) else float(sub.min())
        worst = float(sub.min()) if HIGHER_IS_BETTER.get(metric, True) else float(sub.max())
        max_imp = best - (base_val if pd.notna(base_val) else best)
        max_deg = worst - (base_val if pd.notna(base_val) else worst)
        # Stable / optimal range: values within 5% of best metric.
        thresh = best - 0.05 * abs(best) if best != 0 else best - 1e-9
        if HIGHER_IS_BETTER.get(metric, True):
            ok = sub[sub >= thresh]
        else:
            ok = sub[sub <= thresh]
        stable_range = f"{_fmt_v(min(ok.index))} – {_fmt_v(max(ok.index))}" if len(ok) else "—"
        # Classification by CV.
        if pd.isna(cv):
            cls = "Highly Stable"
        elif cv < 5:
            cls = "Highly Stable"
        elif cv < 15:
            cls = "Moderately Stable"
        else:
            cls = "Highly Sensitive"
        rows.append({
            "parameter": spec.name, "key": k,
            "mean": mean, "median": median, "std": std, "cv_pct": cv,
            "max_improvement": max_imp, "max_degradation": max_deg,
            "stable_range": stable_range, "classification": cls,
        })
    return pd.DataFrame(rows)


def parameter_importance(result: SensitivityResult) -> pd.DataFrame:
    """Composite parameter-importance ranking across the key metrics."""
    df = _frame(result)
    if df.empty:
        return pd.DataFrame()
    impact: dict[str, dict[str, float]] = {}
    norm: dict[str, dict[str, float]] = {}
    for metric in IMPORTANCE_METRICS:
        if metric not in df.columns:
            continue
        spans = {}
        for spec in result.specs:
            k = spec.key
            if k not in df.columns:
                continue
            sub = df.groupby(k)[metric].mean()
            if sub.empty:
                continue
            spans[k] = abs(float(sub.max() - sub.min()))
        if not spans:
            continue
        mx = max(spans.values())
        for k, v in spans.items():
            impact.setdefault(k, {})[metric] = v
            norm.setdefault(k, {})[metric] = (v / mx) if mx > 0 else 0.0

    rows = []
    for spec in result.specs:
        k = spec.key
        if k not in norm:
            continue
        comp = float(np.mean(list(norm[k].values())))
        rows.append({"parameter": spec.name, "key": k, "composite_impact": comp,
                     **{f"imp_{m}": impact[k].get(m, 0.0) for m in IMPORTANCE_METRICS}})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["composite_impact"] = (out["composite_impact"] / out["composite_impact"].max() * 100.0) \
        if out["composite_impact"].max() > 0 else 0.0
    out["composite_impact"] = out["composite_impact"].round(2)
    return out.sort_values("composite_impact", ascending=False).reset_index(drop=True)


def correlation_analysis(result: SensitivityResult) -> dict[str, pd.DataFrame]:
    """Pearson / Spearman / Kendall correlation of params vs metrics."""
    df = _frame(result)
    if df.empty:
        return {}
    metrics = [m for m in METRIC_LABELS if m in df.columns]
    enc: dict[str, list[float]] = {}
    for spec in result.specs:
        k = spec.key
        if k not in df.columns:
            continue
        if spec.kind == ParamKind.CATEGORICAL:
            allowed = list(spec.allowed or sorted(set(df[k])))
            enc[k] = [allowed.index(v) if v in allowed else 0 for v in df[k]]
        else:
            enc[k] = df[k].astype(float).tolist()
    enc_df = pd.DataFrame(enc)
    out = {}
    for method in ("pearson", "spearman", "kendall"):
        try:
            corr = enc_df.corr(method=method)
        except Exception:
            corr = pd.DataFrame()
        # Restrict to param-vs-metric view.
        mat = pd.DataFrame(index=list(enc_df.columns), columns=metrics, dtype=float)
        for pk in enc_df.columns:
            for m in metrics:
                try:
                    mat.loc[pk, m] = corr.loc[pk, m] if (pk in corr.index and m in corr.columns) else np.nan
                except Exception:
                    mat.loc[pk, m] = np.nan
        out[method] = mat.astype(float)
    return out


def interaction_analysis(result: SensitivityResult) -> dict[str, Any]:
    """Quantify two-parameter interaction effects.

    For *two-way* mode the true 2-D interaction surface is computed. For other
    modes a proxy (correlation of marginal impacts) is returned and flagged.
    """
    df = _frame(result)
    metric = result.primary_metric
    base_val = result.baseline_metrics.get(metric, float("nan"))
    pairs = list(itertools.combinations([s.key for s in result.specs], 2))
    rows = []
    precise = result.mode == "two_way"
    for a, b in pairs:
        if a not in df.columns or b not in df.columns:
            continue
        sub = df[[a, b, metric]].dropna()
        if len(sub) < 4:
            continue
        if precise:
            grid = sub.pivot_table(index=a, columns=b, values=metric)
            main_a = sub.groupby(a)[metric].mean()
            main_b = sub.groupby(b)[metric].mean()
            inter = pd.DataFrame(index=grid.index, columns=grid.columns, dtype=float)
            for ai in grid.index:
                for bi in grid.columns:
                    inter.loc[ai, bi] = grid.loc[ai, bi] - (
                        main_a[ai] + main_b[bi] - (base_val if pd.notna(base_val) else main_a[ai])
                    )
            mag = float(inter.abs().mean().mean())
            mean_sign = float(inter.mean().mean())
            kind = "Synergistic" if mean_sign > 0 else ("Conflicting" if mean_sign < 0 else "Neutral")
        else:
            # Proxy: correlation between the two params across combos, and the
            # spread each adds on its own.
            mag = float(abs(sub[a].corr(sub[b])))
            mean_sign = mag
            kind = "Proxy (see note)"
        rows.append({"param_a": a, "param_b": b, "interaction_magnitude": mag,
                     "direction": mean_sign, "classification": kind})
    out = pd.DataFrame(rows)
    return {"precise": precise, "table": out}


def robustness_analysis(result: SensitivityResult, metric: str | None = None) -> pd.DataFrame:
    """Detect flat-optimum (robust) vs sharp-peak (overfit-prone) regions."""
    metric = metric or result.primary_metric
    df = _frame(result)
    if df.empty:
        return pd.DataFrame()
    rows = []
    for spec in result.specs:
        k = spec.key
        if k not in df.columns:
            continue
        sub = df.groupby(k)[metric].mean()
        if len(sub) < 3:
            rows.append({"parameter": spec.name, "key": k, "flatness_pct": float("nan"),
                         "profile": "Insufficient variation", "recommendation": "—"})
            continue
        best = float(sub.max()) if HIGHER_IS_BETTER.get(metric, True) else float(sub.min())
        thresh = best - 0.05 * abs(best) if best != 0 else best - 1e-9
        if HIGHER_IS_BETTER.get(metric, True):
            near = sub[sub >= thresh]
        else:
            near = sub[sub <= thresh]
        flatness = len(near) / len(sub) * 100.0
        if flatness >= 50:
            profile = "Flat optimum (robust)"
            rec = "Safe to operate across a wide range"
        elif flatness >= 20:
            profile = "Moderate plateau"
            rec = "Prefer the centre of the good region"
        else:
            profile = "Sharp peak (overfit-prone)"
            rec = "Lock to the optimum; avoid drift"
        rows.append({"parameter": spec.name, "key": k, "flatness_pct": round(flatness, 1),
                     "profile": profile, "recommendation": rec})
    return pd.DataFrame(rows)


def recommendations(result: SensitivityResult, metric: str | None = None) -> dict[str, Any]:
    """Derive robust-parameter recommendations with explanations."""
    metric = metric or result.primary_metric
    df = _frame(result)
    out: dict[str, Any] = {"best_single": None, "best_stable_region": [], "safest_ranges": [],
                           "fix_params": [], "optimize_params": [], "explanations": []}
    if df.empty:
        return out

    best_row = df.loc[df[metric].idxmax() if HIGHER_IS_BETTER.get(metric, True) else df[metric].idxmin()]
    out["best_single"] = {k: best_row[k] for k in result.selected_keys}

    sens = sensitivity_scores(result, metric)
    stab = stability_analysis(result, metric)
    rob = robustness_analysis(result, metric)
    if not sens.empty:
        for _, r in sens.iterrows():
            key = r["key"]
            srow = stab[stab["key"] == key].iloc[0] if len(stab[stab["key"] == key]) else None
            rrow = rob[rob["key"] == key].iloc[0] if len(rob[rob["key"] == key]) else None
            if r["sensitivity_score"] < 20:
                out["fix_params"].append(r["parameter"])
                out["explanations"].append(
                    f"**{r['parameter']}** is insensitive (sensitivity {r['sensitivity_score']:.0f}/100) — "
                    f"keep it fixed at its current value; tuning adds risk without reward."
                )
            else:
                out["optimize_params"].append(r["parameter"])
                out["explanations"].append(
                    f"**{r['parameter']}** is sensitive (sensitivity {r['sensitivity_score']:.0f}/100) — "
                    f"requires careful optimization; its stable range is "
                    f"{srow['stable_range'] if srow is not None else 'n/a'}."
                )
            if rrow is not None and pd.notna(rrow["flatness_pct"]):
                out["safest_ranges"].append(
                    f"{r['parameter']}: {rrow['profile']} (flatness {rrow['flatness_pct']:.0f}%) — "
                    f"{rrow['recommendation']}"
                )
    out["best_stable_region"] = [
        f"{k} = {_fmt_v(v)}" for k, v in out["best_single"].items()
    ]
    return out


def _fmt_v(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def _ord(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(abs(hash(str(v))) % 1000)
