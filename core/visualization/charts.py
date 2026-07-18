"""Visualisation helpers.

Returns Altair chart objects (a charting library, not Streamlit) so the
presentation layer stays decoupled. Pages render them with ``st.altair_chart``.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from core.data.providers.base_provider import PriceColumns


def distribution_chart(series: pd.Series, title: str, bins: int = 30) -> alt.Chart:
    df = series.dropna().rename("value").reset_index(drop=True)
    df["bucket"] = pd.cut(df["value"], bins=bins)
    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("bucket:O", title=title),
            y=alt.Y("count()", title="Count"),
            tooltip=[alt.Tooltip("count()")],
        )
        .properties(title=title, width="container", height=300)
    )
    return hist


def ranking_bar(top: pd.DataFrame, column: str, title: str, ascending: bool = True) -> alt.Chart:
    data = top.copy()
    if not ascending:
        data = data.iloc[::-1]
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(f"{column}:Q", title=column),
            y=alt.Y(f"{PriceColumns.TICKER}:N", sort=None, title="Ticker"),
            tooltip=[PriceColumns.TICKER, column],
        )
        .properties(title=title, width="container", height=max(300, 20 * len(data)))
    )


def time_series_panel(panel: pd.DataFrame, title: str) -> alt.Chart:
    """``panel`` is date x ticker wide; render as small-multiples / overlay."""
    long = panel.reset_index().melt(id_vars=panel.index.name or "date", var_name="ticker", value_name="value")
    long = long.rename(columns={long.columns[0]: "date"})
    return (
        alt.Chart(long)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("value:Q", title=title),
            color=alt.Color("ticker:N", legend=alt.Legend(orient="right")),
        )
        .properties(title=title, width="container", height=350)
        .interactive()
    )


def coverage_heatmap(coverage: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(coverage)
        .mark_bar()
        .encode(
            x=alt.X("ticker:N", title="Ticker"),
            y=alt.Y("rows:Q", title="Stored rows"),
            tooltip=["ticker", "first_date", "last_date", "rows"],
        )
        .properties(title="Per-ticker data coverage", width="container", height=300)
    )
