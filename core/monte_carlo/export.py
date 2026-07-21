"""Export engine for Monte Carlo results.

Produces CSV / JSON / Excel / PDF byte blobs for download. All functions are
pure (return ``bytes``) so the UI only has to wire up download buttons.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from core.monte_carlo.plotting import _pdf_figures
from core.monte_carlo.types import SimulationResult


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and (obj != obj):
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _summary_frame(result: SimulationResult) -> pd.DataFrame:
    risk = result.risk_summary
    prob = result.probabilities
    rows = []
    for k, v in risk.items():
        rows.append({"section": "risk", "metric": k, "value": v})
    for k, v in prob.items():
        rows.append({"section": "probability", "metric": k, "value": v})
    return pd.DataFrame(rows)


def export_metrics_csv(result: SimulationResult) -> bytes:
    return result.metrics_df.to_csv(index=False).encode("utf-8")


def export_equity_csv(result: SimulationResult) -> bytes:
    cols = [f"sim_{i}" for i in range(result.equity_curves.shape[0])]
    df = pd.DataFrame(result.equity_curves.T, index=result.sim_dates, columns=cols)
    return df.to_csv(index=True).encode("utf-8")


def export_summary_csv(result: SimulationResult) -> bytes:
    return _summary_frame(result).to_csv(index=False).encode("utf-8")


def export_trade_stats_csv(result: SimulationResult) -> bytes:
    cols = [c for c in ["win_rate", "profit_factor", "expectancy", "n_trades"]
            if c in result.metrics_df.columns]
    return result.metrics_df[cols].to_csv(index=False).encode("utf-8")


def export_json(result: SimulationResult) -> bytes:
    payload = {
        "config": asdict(result.config),
        "method": result.method,
        "n_simulations": result.n_simulations,
        "horizon_used": result.horizon_used,
        "seed": result.seed,
        "aggregate": result.aggregate.reset_index().to_dict("records"),
        "probabilities": result.probabilities,
        "risk_summary": result.risk_summary,
        "confidence_intervals": result.confidence_intervals,
        "original_metrics": result.original_metrics,
        "metrics": result.metrics_df.replace([np.inf, -np.inf], np.nan).to_dict("records"),
    }
    return json.dumps(_jsonify(payload), indent=2, default=str).encode("utf-8")


def export_excel(result: SimulationResult) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        _summary_frame(result).to_excel(xw, sheet_name="Summary", index=False)
        result.aggregate.to_excel(xw, sheet_name="AggregateStats")
        result.metrics_df.to_excel(xw, sheet_name="Metrics", index=False)
        cols = [f"sim_{i}" for i in range(result.equity_curves.shape[0])]
        eq_df = pd.DataFrame(result.equity_curves.T, index=result.sim_dates, columns=cols)
        eq_df.to_excel(xw, sheet_name="EquityCurves")
    return buf.getvalue()


def export_pdf(result: SimulationResult) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.4 * cm, bottomMargin=1.2 * cm)
    styles = getSampleStyleSheet()
    story: list[Any] = []

    story.append(Paragraph("Monte Carlo Simulation Report", styles["Title"]))
    meta = (
        f"Method: {result.method} &nbsp;|&nbsp; Simulations: {result.n_simulations:,} "
        f"&nbsp;|&nbsp; Horizon: {result.horizon_used} days &nbsp;|&nbsp; "
        f"Seed: {result.seed}"
    )
    story.append(Paragraph(meta, styles["Normal"]))
    story.append(Spacer(1, 0.3 * cm))

    def _table(data: list[list[Any]], col_widths: list[float] | None = None) -> Table:
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.4, (0.8, 0.8, 0.8)),
            ("BACKGROUND", (0, 0), (-1, 0), (0.93, 0.95, 0.98)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        return t

    def _pc(v: float) -> str:
        return f"{v*100:.1f}%" if isinstance(v, (int, float)) and v == v else "—"

    def _num(v: float) -> str:
        return f"{v:,.2f}" if isinstance(v, (int, float)) and v == v else "—"

    risk = result.risk_summary
    risk_rows = [["Risk Metric", "Value"]]
    for k in [
        "probability_of_profit", "probability_of_loss", "expected_cagr",
        "median_cagr", "best_cagr", "worst_case_95_cagr", "best_case_95_cagr",
        "expected_final_portfolio", "worst_case_95_final", "best_case_95_final",
        "expected_sharpe", "expected_sortino", "worst_drawdown",
        "var_95", "cvar_95",
    ]:
        v = risk.get(k)
        if k.startswith("probability") or "cagr" in k or k in ("var_95", "cvar_95", "worst_drawdown"):
            risk_rows.append([k, _pc(v) if "probability" in k else (_pc(v) if "cagr" not in k else _pc(v))])
        else:
            risk_rows.append([k, _num(v)])
    story.append(Paragraph("Risk Summary", styles["Heading2"]))
    story.append(_table(risk_rows, [7 * cm, 5 * cm]))

    prob = result.probabilities
    prob_rows = [["Probability", "Value"]]
    for k, v in prob.items():
        prob_rows.append([k, _pc(v)])
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Probability Thresholds", styles["Heading2"]))
    story.append(_table(prob_rows, [7 * cm, 5 * cm]))

    ci = result.confidence_intervals
    ci_rows = [["Metric", "95% CI", "99% CI"]]
    for k, v in ci.items():
        ci_rows.append([k, f"[{_num(v['ci95'][0])}, {_num(v['ci95'][1])}]",
                        f"[{_num(v['ci99'][0])}, {_num(v['ci99'][1])}]"])
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("Confidence Intervals", styles["Heading2"]))
    story.append(_table(ci_rows, [5 * cm, 5.5 * cm, 5.5 * cm]))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Charts", styles["Heading2"]))
    for title, png in _pdf_figures(result):
        story.append(Image(io.BytesIO(png), width=16 * cm, height=7.2 * cm))
        story.append(Spacer(1, 0.2 * cm))

    doc.build(story)
    return buf.getvalue()
