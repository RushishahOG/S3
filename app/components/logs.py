"""Developer log viewer component (read-only, file-backed).

Exposes the application's existing backend log file in an interactive Streamlit
panel. It never touches the logging framework - it just reads
``settings.logging.log_abs_dir / "platform.log"`` and renders it with filtering,
colour-coding by level, auto-scroll and export.

This is a debugging aid only; all behaviour is derived from the configured log
file and its format string.
"""

from __future__ import annotations

import html
import os
import re
import time
from datetime import datetime

import streamlit as st

from core.config.settings import settings

# Absolute path of the backend log file (mirrors logging_config.configure_logging).
LOG_PATH = os.path.join(settings.logging.log_abs_dir, "platform.log")

# Example line: 2026-07-15 08:02:16,198 | INFO     | core.data.x | message
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2},\d{3})\s*\|\s*"
    r"(?P<level>\w+)\s*\|\s*"
    r"(?P<module>[\w\.]+)\s*\|\s*"
    r"(?P<msg>.*)$"
)

LEVEL_COLORS = {
    "DEBUG": "#6c8cff",
    "INFO": "#cfd8dc",
    "WARNING": "#ffb74d",
    "ERROR": "#ff5252",
    "CRITICAL": "#ff1744",
}

ALL_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def _parse_lines(raw_lines: list[str]) -> list[dict]:
    """Parse raw log lines into structured entries.

    Lines that do not match the standard format (e.g. traceback continuations)
    are appended to the preceding entry's message.
    """
    entries: list[dict] = []
    for line in raw_lines:
        line = line.rstrip("\n")
        m = _LINE_RE.match(line)
        if m:
            entries.append({
                "ts": m.group("ts"),
                "level": m.group("level"),
                "module": m.group("module"),
                "msg": m.group("msg"),
            })
        elif entries:
            entries[-1]["msg"] += "\n" + line
    return entries


def _read_tail(max_lines: int) -> list[str]:
    """Return the last ``max_lines`` lines of the log file (cheap, file-backed)."""
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return lines[-max_lines:] if max_lines else lines


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def render_log_viewer() -> None:
    """Render the full Developer Logs page (standalone entry, kept for compat)."""
    from app.layouts.base import page_header

    page_header("Developer Logs", "Live view of the backend log file (read-only)")
    render_log_panel()


def render_log_panel() -> None:
    """Render the embeddable Developer Logs panel used inside other pages.

    A toggle ("Enable logs") controls visibility; when enabled the live,
    filterable log viewer is shown inline within the host page.
    """
    if "logs_enabled" not in st.session_state:
        st.session_state["logs_enabled"] = False

    col_toggle = st.columns([1, 4])
    with col_toggle[0]:
        if st.toggle("Enable logs", value=st.session_state["logs_enabled"], key="logs_toggle"):
            st.session_state["logs_enabled"] = True
        else:
            st.session_state["logs_enabled"] = False

    if not st.session_state["logs_enabled"]:
        return

    # --- Filters & Controls (read before loading so limits apply) ----------
    from app.layouts.base import section

    section("Filters & Controls")
    col_ctrl = st.columns([1, 1])
    with col_ctrl[0]:
        levels = st.multiselect("Log Level", ALL_LEVELS, default=ALL_LEVELS, key="logs_levels")
    with col_ctrl[1]:
        max_lines = st.slider("Max lines loaded", 200, 20000, 3000, step=200, key="logs_max")

    col_ctrl2 = st.columns([1, 1, 1])
    with col_ctrl2[0]:
        keyword = st.text_input("Keyword search", "", key="logs_keyword")
    with col_ctrl2[1]:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="logs_auto")
        refresh_interval = st.number_input(
            "Refresh interval (s)", min_value=1, max_value=30, value=3, key="logs_interval"
        ) if auto_refresh else 3
    with col_ctrl2[2]:
        d_from = st.date_input("From date", value=None, key="logs_from")
        d_to = st.date_input("To date", value=None, key="logs_to")

    # --- File metadata -----------------------------------------------------
    section("Log File")
    exists = os.path.exists(LOG_PATH)
    if not exists:
        st.warning(f"Log file not found at `{LOG_PATH}`. Enable `logging.to_file` "
                   "in settings to write backend logs to disk.")
        entries_all: list[dict] = []
        module_options: list[str] = []
    else:
        size = os.path.getsize(LOG_PATH)
        mtime = os.path.getmtime(LOG_PATH)
        raw = _read_tail(max_lines)
        entries_all = _parse_lines(raw)
        module_options = sorted({e["module"] for e in entries_all})
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"**Path**\n`{LOG_PATH}`")
        c2.metric("Size", _fmt_bytes(size))
        c3.markdown(f"**Modified**\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))}")
        c4.metric("Entries", len(entries_all))

    modules = st.multiselect("Module", module_options, default=[], key="logs_modules")

    # --- Buttons -----------------------------------------------------------
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("Refresh Logs", type="primary", key="logs_refresh"):
            st.session_state["logs_cleared"] = False
            st.rerun()
    with b2:
        if st.button("Clear View", key="logs_clear"):
            st.session_state["logs_cleared"] = True
            st.rerun()
    with b3:
        if exists:
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as fh:
                st.download_button(
                    "Download Logs",
                    data=fh.read(),
                    file_name="platform_logs.txt",
                    mime="text/plain",
                    key="logs_download",
                )
    with b4:
        if st.button("Copy path", key="logs_path"):
            st.code(LOG_PATH)

    if st.session_state.get("logs_cleared"):
        st.info("Log view cleared (the file is unchanged). Click **Refresh Logs** to reload.")
        return

    if not exists:
        return

    # --- Filtering ---------------------------------------------------------
    entries = entries_all
    if levels:
        entries = [e for e in entries if e["level"] in levels]
    if modules:
        entries = [e for e in entries if e["module"] in modules]
    if keyword:
        kw = keyword.lower()
        entries = [e for e in entries if kw in (e["msg"] + e["module"] + e["level"]).lower()]
    if d_from is not None or d_to is not None:
        filtered = []
        for e in entries:
            try:
                dt = datetime.strptime(e["ts"], "%Y-%m-%d %H:%M:%S,%f")
            except ValueError:
                filtered.append(e)
                continue
            if d_from is not None and dt.date() < d_from:
                continue
            if d_to is not None and dt.date() > d_to:
                continue
            filtered.append(e)
        entries = filtered

    st.caption(f"Showing {len(entries)} of {len(entries_all)} parsed entries "
               f"(newest at bottom; auto-scrolls).")

    if not entries:
        st.info("No log entries match the current filters.")
        return

    # --- Colour-coded, scrollable viewer ----------------------------------
    parts = []
    for e in entries:
        color = LEVEL_COLORS.get(e["level"], "#cfd8dc")
        safe = html.escape(e["ts"]) + " | " + html.escape(e["level"]) + " | " + \
               html.escape(e["module"]) + " | " + html.escape(e["msg"])
        parts.append(
            f'<div style="color:{color};white-space:pre-wrap;font-family:monospace;'
            f'font-size:12px;line-height:1.4;">{safe}</div>'
        )
    viewer = (
        '<div id="logbox" style="height:520px;overflow:auto;'
        'background:#0e1117;border:1px solid #262b3a;border-radius:6px;'
        'padding:10px;">'
        + "".join(parts) +
        '</div>'
        '<script>var b=document.getElementById("logbox");'
        'if(b){b.scrollTop=b.scrollHeight;}</script>'
    )
    st.markdown(viewer, unsafe_allow_html=True)

    # --- Auto-refresh loop -------------------------------------------------
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
