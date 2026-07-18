"""Reusable metric / KPI display components for the presentation layer."""

from __future__ import annotations

from typing import Iterable

import streamlit as st


def kpi_cards(metrics: Iterable[tuple[str, str, str | None]]) -> None:
    """Render a row of KPI cards. ``metrics`` is (label, value, delta|None)."""
    cols = st.columns(max(1, min(len(list(metrics)), 4)))
    for col, (label, value, delta) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value, delta=delta)
