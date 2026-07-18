"""Interactive chart helpers for the Dataset Explorer (Altair based).

Altair is already a first-class dependency of the platform, so the candlestick
+ volume view is built without introducing a new charting library.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from core.data.providers.base_provider import PriceColumns


def candlestick_chart(df: pd.DataFrame, ticker: str, height: int = 360) -> alt.Chart:
    """Return an interactive candlestick chart with a volume subplot.

    ``df`` must contain OHLCV columns from :data:`PriceColumns`. The two
    sub-charts share the date axis so brushing/panning stays aligned.
    """
    data = df.copy()
    data[PriceColumns.DATE] = pd.to_datetime(data[PriceColumns.DATE])
    data = data.dropna(subset=[PriceColumns.OPEN, PriceColumns.HIGH,
                               PriceColumns.LOW, PriceColumns.CLOSE])

    if data.empty:
        return alt.Chart(pd.DataFrame({"note": ["No data"]})).mark_text()

    domain = (data[PriceColumns.DATE].min(), data[PriceColumns.DATE].max())
    shared_x = alt.X("date:T", title="Date", scale=alt.Scale(domain=domain))

    base = alt.Chart(data)
    rule = base.mark_rule(color="#888").encode(x=shared_x, y="low:Q", y2="high:Q")
    color = alt.condition(
        alt.datum.close >= alt.datum.open,
        alt.value("#26a69a"),
        alt.value("#ef5350"),
    )
    bar = base.mark_bar().encode(
        x=shared_x,
        y=alt.Y("open:Q", scale=alt.Scale(zero=False)),
        y2="close:Q",
        color=color,
    )

    # One shared zoom selection keeps both subplots aligned without emitting
    # duplicate-selection warnings.
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    candles = (rule + bar).add_selection(zoom).properties(
        title=f"{ticker} - Candlestick", height=height, width="container"
    )

    vol = base.mark_bar().encode(
        x=shared_x,
        y=alt.Y("volume:Q", title="Volume"),
        color=color,
    ).add_selection(zoom).properties(title="Volume", height=120, width="container")

    chart = (
        alt.vconcat(candles, vol)
        .configure_view(strokeWidth=0)
        .configure_title(anchor="start")
    )
    return chart
