"""Export helpers for Sensitivity Analysis results.

Produces CSV / Excel / JSON / PDF byte blobs for download. CSV/Excel/JSON use
pandas + openpyxl; the PDF is rendered with matplotlib (reportlab/kaleido are
not available in this environment) so it remains dependency-light.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import pandas as pd

from core.sensitivity.engine import METRIC_LABELS, SensitivityResult, _fmt_v


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


def records_frame(result: SensitivityResult) -> pd.DataFrame:
    return result.to_frame()


def export_csv(result: SensitivityResult) -> bytes:
    return records_frame(result).to_csv(index=False).encode("utf-8")


def export_json(result: SensitivityResult, analytics: dict[str, Any] | None = None) -> bytes:
    payload: dict[str, Any] = {
        "mode": result.mode,
        "primary_metric": result.primary_metric,
        "base_config": result.base.to_dict(),
        "baseline_metrics": result.baseline_metrics,
        "records": result.records,
    }
    if analytics:
        payload["analytics"] = _jsonify(analytics)
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def export_excel(
    result: SensitivityResult,
    analytics: dict[str, Any] | None = None,
) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        records_frame(result).to_excel(xw, sheet_name="Results", index=False)
        pd.DataFrame([{
            "parameter": s.name, "key": s.key, "block": s.block, "field": s.field,
        } for s in result.specs]).to_excel(xw, sheet_name="Parameters", index=False)
        if analytics:
            if analytics.get("sensitivity") is not None:
                analytics["sensitivity"].to_excel(xw, sheet_name="Sensitivity", index=False)
            if analytics.get("stability") is not None:
                analytics["stability"].to_excel(xw, sheet_name="Stability", index=False)
            if analytics.get("importance") is not None:
                analytics["importance"].to_excel(xw, sheet_name="Importance", index=False)
            if analytics.get("robustness") is not None:
                analytics["robustness"].to_excel(xw, sheet_name="Robustness", index=False)
            inter = analytics.get("interaction")
            if isinstance(inter, dict) and inter.get("table") is not None:
                inter["table"].to_excel(xw, sheet_name="Interaction", index=False)
    return buf.getvalue()


def export_pdf(
    result: SensitivityResult,
    analytics: dict[str, Any] | None = None,
) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    analytics = analytics or {}
    metric = result.primary_metric
    metric_label = METRIC_LABELS.get(metric, metric)

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # --- Title page -----------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.92, "Sensitivity Analysis Report", ha="center", fontsize=18, weight="bold")
        fig.text(0.5, 0.88, f"Mode: {result.mode}   |   Primary metric: {metric_label}",
                 ha="center", fontsize=11)
        fig.text(0.5, 0.85, f"Combinations evaluated: {len(result.records)}", ha="center", fontsize=10)
        if analytics.get("recommendations"):
            rec = analytics["recommendations"]
            lines = ["Robust-Parameter Recommendations", ""]
            if rec.get("best_single"):
                lines.append("Best combination: " + ", ".join(
                    f"{k}={_fmt_v(v)}" for k, v in rec["best_single"].items()))
            if rec.get("fix_params"):
                lines.append("Fix (insensitive): " + ", ".join(rec["fix_params"]))
            if rec.get("optimize_params"):
                lines.append("Optimize (sensitive): " + ", ".join(rec["optimize_params"]))
            for e in rec.get("explanations", [])[:8]:
                lines.append("• " + e.replace("**", ""))
            fig.text(0.08, 0.80, "\n".join(lines), va="top", fontsize=9, wrap=True)
        pdf.savefig(fig)
        plt.close(fig)

        # --- Sensitivity & importance charts --------------------------------
        sens = analytics.get("sensitivity")
        imp = analytics.get("importance")
        if sens is not None and not sens.empty:
            fig, ax = plt.subplots(figsize=(8.27, 5.5))
            ax.barh(sens["parameter"][::-1], sens["sensitivity_score"][::-1], color="#3b6ea5")
            ax.set_xlabel("Sensitivity Score (0-100)")
            ax.set_title("Parameter Sensitivity Ranking")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
        if imp is not None and not imp.empty:
            fig, ax = plt.subplots(figsize=(8.27, 5.5))
            ax.barh(imp["parameter"][::-1], imp["composite_impact"][::-1], color="#7a3b9d")
            ax.set_xlabel("Composite Importance (0-100)")
            ax.set_title("Parameter Importance Ranking")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Two-way heatmap ------------------------------------------------
        if result.mode == "two_way" and len(result.specs) >= 2 and not records_frame(result).empty:
            df = records_frame(result)
            k0, k1 = result.specs[0].key, result.specs[1].key
            try:
                grid = df.pivot_table(index=k0, columns=k1, values=metric)
                fig, ax = plt.subplots(figsize=(8.27, 6.0))
                im = ax.imshow(grid.values, aspect="auto", cmap="viridis")
                ax.set_xticks(range(len(grid.columns)))
                ax.set_xticklabels([_fmt_v(c) for c in grid.columns], rotation=45, ha="right")
                ax.set_yticks(range(len(grid.index)))
                ax.set_yticklabels([_fmt_v(c) for c in grid.index])
                ax.set_xlabel(result.specs[1].name)
                ax.set_ylabel(result.specs[0].name)
                ax.set_title(f"{metric_label} Surface")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            except Exception:
                pass

        # --- Data tables ----------------------------------------------------
        for name, tbl in (("Sensitivity Scores", sens),
                          ("Stability", analytics.get("stability")),
                          ("Robustness", analytics.get("robustness")),
                          ("Parameter Importance", imp)):
            if tbl is None or getattr(tbl, "empty", True):
                continue
            _pdf_table(pdf, name, tbl)

        # --- Results table (first N rows sorted by metric) ------------------
        df = records_frame(result)
        if not df.empty and metric in df.columns:
            top = df.sort_values(metric, ascending=False).head(40)
            cols = [s.key for s in result.specs] + [metric]
            cols = [c for c in cols if c in top.columns]
            _pdf_table(pdf, f"Top Results by {metric_label}", top[cols].reset_index(drop=True))
    return buf.getvalue()


def _pdf_table(pdf, title: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    data = df.head(max_rows).copy()
    for c in data.columns:
        if data[c].dtype.kind in "fc":
            data[c] = data[c].map(lambda v: f"{v:.4g}" if isinstance(v, (int, float)) and v == v else "—")
        else:
            data[c] = data[c].map(lambda v: str(v))
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.95, title, fontsize=12, weight="bold")
    n_rows = len(data) + 1
    n_cols = len(data.columns)
    tbl = fig.add_axes((0.05, 0.05, 0.9, 0.87))
    tbl.axis("off")
    cell_text = [list(data.columns)] + data.values.tolist()
    t = tbl.table(cellText=cell_text, loc="upper center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(6)
    t.scale(1, 1.2)
    for j in range(n_cols):
        t[0, j].set_facecolor("#dfe6ef")
    pdf.savefig(fig)
    plt.close(fig)
