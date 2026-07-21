"""Export strategies to various formats."""

from __future__ import annotations

import io
import json
from datetime import datetime

import pandas as pd

from core.backtesting.export import export_dataframe
from core.strategy_comparison.comparison import ComparisonResult


def export_csv(result: ComparisonResult) -> bytes:
    """Export performance table to CSV."""
    return export_dataframe(result.performance_table, "csv")


def export_excel(result: ComparisonResult) -> bytes:
    """Export all comparison data to Excel workbook."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        result.config_comparison.to_excel(xw, sheet_name="Configuration")
        result.performance_table.to_excel(xw, sheet_name="Performance")
        if not result.equity_curves.empty:
            result.equity_curves.to_excel(xw, sheet_name="Equity Curves")
        if not result.drawdown_curves.empty:
            result.drawdown_curves.to_excel(xw, sheet_name="Drawdowns")
        result.risk_table.to_excel(xw, sheet_name="Risk")
        result.rankings.to_excel(xw, sheet_name="Rankings")
        result.correlation_matrix.to_excel(xw, sheet_name="Correlation")
        result.allocation_df.to_excel(xw, sheet_name="Allocation")
        result.quality_df.to_excel(xw, sheet_name="Quality Gates")
        result.trade_df.to_excel(xw, sheet_name="Trade Analysis")
        result.benchmark_df.to_excel(xw, sheet_name="Benchmark")
        if not result.annual_returns.empty:
            result.annual_returns.to_excel(xw, sheet_name="Annual Returns")
        if not result.monthly_returns.empty:
            result.monthly_returns.to_excel(xw, sheet_name="Monthly Returns")
        if result.holdings_overlap_df is not None and not result.holdings_overlap_df.empty:
            result.holdings_overlap_df.to_excel(xw, sheet_name="Holdings Overlap")
    return buf.getvalue()


def export_json(result: ComparisonResult) -> bytes:
    """Export comparison data to JSON."""
    data = {
        "generated_at": datetime.now().isoformat(),
        "strategies": [s.to_dict() for s in result.strategies],
        "performance": result.performance_table.to_dict(),
        "config": result.config_comparison.to_dict(),
        "rankings": result.rankings.to_dict(),
        "recommendations": result.recommendations,
    }
    return json.dumps(data, default=str, indent=2).encode("utf-8")


def export_pdf(result: ComparisonResult) -> bytes:
    """Generate PDF report."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    except ImportError:
        raise RuntimeError("reportlab required for PDF export")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("t", parent=styles["Title"], fontSize=16)
    story = [
        Paragraph("Strategy Comparison Report", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
        Spacer(1, 12),
    ]

    # Strategy list
    story.append(Paragraph("Strategies Compared", styles["Heading2"]))
    for s in result.strategies:
        story.append(Paragraph(f"• {s.name} ({s.source.value})", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Performance table
    story.append(Paragraph("Performance Summary", styles["Heading2"]))
    perf = result.performance_table
    rows = [["Strategy"] + list(perf.columns)]
    for name in perf.index:
        row = [name] + [f"{perf.loc[name, c]:.2f}" for c in perf.columns]
        rows.append(row)
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#4C9AFF")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Recommendations
    if result.recommendations:
        story.append(Paragraph("Recommendations", styles["Heading2"]))
        for k, v in result.recommendations.items():
            if isinstance(v, dict) and "strategy" in v:
                reason = f" — {v['reason']}" if v.get("reason") else ""
                story.append(Paragraph(
                    f"• <b>{k.replace('_', ' ').title()}</b>: {v['strategy']}{reason}",
                    styles["Normal"]))

    # Charts
    try:
        for title, png in _pdf_charts(result):
            story.append(Spacer(1, 8))
            story.append(Paragraph(title, styles["Heading3"]))
            story.append(Image(io.BytesIO(png), width=16 * cm, height=8 * cm))
    except Exception:
        pass

    doc.build(story)
    return buf.getvalue()


def _pdf_charts(result: ComparisonResult) -> list[tuple[str, bytes]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out: list[tuple[str, bytes]] = []

    def _save(fig):
        b = io.BytesIO()
        fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return b.getvalue()

    eq = result.equity_curves
    if eq is not None and not eq.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for col in eq.columns:
            series = eq[col]
            series = series / series.iloc[0] * 100
            ax.plot(series.index, series.values, lw=1.2, label=col)
        ax.set_title("Equity Curves (Normalized)")
        ax.legend(fontsize=6)
        fig.autofmt_xdate()
        out.append(("Equity Curves", _save(fig)))

    corr = result.correlation_matrix
    if corr is not None and not corr.empty:
        fig, ax = plt.subplots(figsize=(5.5, 4.6))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title("Return Correlation")
        out.append(("Correlation Matrix", _save(fig)))

    perf = result.performance_table
    if perf is not None and not perf.empty:
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        ax.scatter(perf["Volatility"] * 100, perf["CAGR"] * 100, s=30, c=perf["Sharpe"], cmap="viridis")
        for name in perf.index:
            ax.annotate(name, (perf.loc[name, "Volatility"] * 100, perf.loc[name, "CAGR"] * 100),
                        fontsize=6)
        ax.set_xlabel("Volatility %")
        ax.set_ylabel("CAGR %")
        ax.set_title("Risk-Return Scatter")
        out.append(("Risk-Return Scatter", _save(fig)))

    return out



__all__ = ["export_csv", "export_excel", "export_json", "export_pdf"]