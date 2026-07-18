"""Base layout helpers for the Streamlit presentation layer."""

from __future__ import annotations

import streamlit as st

from core.config.settings import settings


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.divider()


def app_title() -> None:
    st.markdown(
        f"""
        <div style="display:flex;align-items:baseline;gap:12px">
          <h1 style="margin:0">{settings.app.name}</h1>
          <span style="color:#888">v{settings.app.version}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f"### {title}")
