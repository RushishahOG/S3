"""Market-regime signal detection (ARQM regime module).

Generates a daily regime state from a reference price series (benchmark index by
default, or the live portfolio NAV). Two configurable mechanisms:

* **Rolling swing-low detection** -- a *buy* signal fires when the reference
  price recovers at least ``buy_trigger_pct`` above its recent swing low
  (the lowest close within the trailing ``swing_low_window`` trading days).
* **Rolling peak detection** -- a *sell* signal fires when the reference price
  falls at least ``sell_trigger_pct`` below its recent peak (highest close within
  the trailing ``peak_window`` trading days).

The engine maintains an ``invested`` boolean: it enters (buy) on a buy signal
while flat, and exits (sell) on a sell signal while invested. This drives the
"buy entire portfolio on buy signal / exit together on sell signal" behaviour.

All logic is point-in-time: at day t only information up to t is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config.backtest_schema import RegimeConfig
from core.utils.logging_config import get_logger

logger = get_logger(__name__)


def detect_regime(
    reference: pd.Series,
    config: RegimeConfig,
) -> pd.DataFrame:
    """Return a daily regime frame for ``reference`` (indexed by date).

    Columns
    -------
    close : reference price
    swing_low : trailing min over swing_low_window
    peak : trailing max over peak_window
    buy_signal : bool (price >= swing_low * (1 + buy_trigger/100))
    sell_signal : bool (price <= peak * (1 + sell_trigger/100))
    state : "invested" | "flat"  (cumulative, event-driven)
    """
    if reference is None or reference.dropna().empty:
        return pd.DataFrame()

    close = reference.dropna().sort_index()
    buy = config.buy_trigger_pct / 100.0
    sell = config.sell_trigger_pct / 100.0

    swing_low = (
        close.rolling(config.swing_low_window, min_periods=5).min()
        if config.enable_swing_low
        else pd.Series(np.nan, index=close.index)
    )
    peak = (
        close.rolling(config.peak_window, min_periods=5).max()
        if config.enable_peak_detection
        else pd.Series(np.nan, index=close.index)
    )

    buy_signal = (close >= swing_low * (1 + buy)) & swing_low.notna()
    sell_signal = (close <= peak * (1 + sell)) & peak.notna()

    state = []
    # Long-only quality-momentum strategies are *invested by default*; a sell
    # signal moves to cash, and a subsequent buy signal re-enters. This avoids
    # missing the entire backtest when no early buy trigger fires.
    invested = True
    for b, s in zip(buy_signal, sell_signal):
        if invested and s:
            invested = False
        elif not invested and b:
            invested = True
        state.append("invested" if invested else "flat")

    out = pd.DataFrame(
        {
            "close": close,
            "swing_low": swing_low,
            "peak": peak,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "state": state,
        },
        index=close.index,
    )
    return out
