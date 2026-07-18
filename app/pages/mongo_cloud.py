"""MongoDB Cloud Controls page.

Manage the DuckDB store's cloud copy (MongoDB GridFS): check connectivity,
inspect the local vs. cloud file, and upload/download the ~775MB DuckDB blob.

This is the operational counterpart to ``core.data.storage.provisioning``: the
app auto-downloads the DB at startup on hosts that don't have it, and this page
lets you publish a freshly-built store or force a re-download from the UI.
"""

from __future__ import annotations

import streamlit as st

from app.layouts.base import page_header, section
from core.data.storage.provisioning import (
    config_summary,
    ensure_database,
    gridfs_status,
    is_configured,
    local_status,
    test_connection,
    upload_to_gridfs,
)
from core.data.storage.storage_manager import StorageManager


def render() -> None:
    page_header("MongoDB Cloud Controls", "Manage the DuckDB store in MongoDB GridFS")

    cfg = config_summary()

    # --- Configuration ------------------------------------------------------
    section("Configuration")
    if not cfg["configured"]:
        st.error(
            "MONGO_URI is not configured. Add it to `.streamlit/secrets.toml` "
            "locally, or to the Streamlit Cloud **Secrets** UI."
        )
        st.code(
            'MONGO_URI = "mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/..."\n'
            'MONGO_DB_NAME = "smartbeta"\n'
            'MONGO_GRIDFS_BUCKET = "duckdb_store"\n'
            'MONGO_DUCKDB_FILE = "market_data.duckdb.gz"',
            language="toml",
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster host", cfg["host"] or "-")
        c2.metric("Database", cfg["db_name"])
        c3.metric("GridFS bucket", cfg["bucket"])
        st.caption(f"Blob filename: `{cfg['filename']}`")

    st.divider()

    # --- Status -------------------------------------------------------------
    section("Status")
    col_local, col_cloud = st.columns(2)

    with col_local:
        st.markdown("**Local DuckDB file**")
        local = local_status()
        if local["exists"]:
            st.success(f"Present · {local['size_mb']:.1f} MB")
        else:
            st.warning("Not present on this host")
        st.caption(local["path"])

    with col_cloud:
        st.markdown("**Cloud copy (GridFS)**")
        if st.button("Check cloud status", key="mc_check", disabled=not cfg["configured"]):
            with st.spinner("Querying GridFS..."):
                st.session_state["mc_gridfs"] = gridfs_status()
        gs = st.session_state.get("mc_gridfs")
        if gs is None:
            st.caption("Click **Check cloud status**.")
        elif gs.get("exists"):
            st.success(f"Present · {gs['size_mb']:.1f} MB")
            st.caption(f"Uploaded: {gs.get('upload_date', '-')}")
        else:
            st.warning(f"Not found ({gs.get('detail', 'unknown')})")

    st.divider()

    # --- Connectivity -------------------------------------------------------
    section("Connectivity")
    if st.button("Test connection", key="mc_test", disabled=not cfg["configured"]):
        with st.spinner("Pinging cluster..."):
            res = test_connection()
        if res["ok"]:
            st.success("Connection OK")
            st.caption("Databases: " + ", ".join(res["databases"]) or "-")
        else:
            st.error(f"Connection failed: {res['detail']}")

    st.divider()

    # --- Upload -------------------------------------------------------------
    section("Upload local DB to cloud")
    st.caption(
        "Publishes the local `market_data.duckdb` to GridFS (replaces any "
        "existing copy). Automatic gzip compression (~4-5x reduction) keeps the "
        "upload under 250 MB — perfect for the Atlas free tier. Large file — this "
        "may take several minutes."
    )
    local = local_status()
    up_disabled = not (cfg["configured"] and local["exists"])
    if not local["exists"]:
        st.info("No local DuckDB file to upload on this host.")
    confirm_up = st.checkbox("I understand this overwrites the cloud copy", key="mc_up_ok")
    if st.button("Upload to GridFS", key="mc_upload", type="primary",
                 disabled=up_disabled or not confirm_up):
        try:
            with st.spinner(f"Uploading {local['size_mb']:.0f} MB to GridFS..."):
                upload_to_gridfs()
            st.success("Upload complete.")
            st.session_state["mc_gridfs"] = gridfs_status()
        except Exception as exc:
            st.error(f"Upload failed: {type(exc).__name__}: {exc}")

    # Add a separate clean-up action here.
    st.divider()

    # --- Cleanup ------------------------------------------------------------
    section("Cleanup GridFS")
    st.caption(
        "Delete all files and chunks in the GridFS bucket. Use this when you need "
        "to free quota before uploading the compressed version. **This is permanent** "
        "— re-upload the DB afterwards from a local development build or via the "
        "upload section above."
    )
    confirm_cleanup = st.checkbox("I understand this deletes everything in GridFS", key="mc_clean_ok")
    if st.button("Clean up GridFS bucket", key="mc_cleanup",
                 disabled=not (cfg["configured"] and confirm_cleanup)):
        try:
            with st.spinner("Cleaning up GridFS..."):
                StorageManager.cleanup_gridfs()
            st.success("GridFS cleanup complete.")
        except Exception as exc:
            st.error(f"Cleanup failed: {type(exc).__name__}: {exc}")

    st.divider()

    # --- Download -----------------------------------------------------------
    section("Download DB from cloud")
    st.caption(
        "Fetches the compressed DuckDB blob from GridFS to local disk, decompresses "
        "in-place, and makes it available to the rest of the app. Runs automatically "
        "at startup when the file is missing; use this to force a refresh."
    )
    dl_disabled = not cfg["configured"]
    force = st.checkbox("Overwrite existing local file", key="mc_dl_force")
    if st.button("Download from GridFS", key="mc_download", disabled=dl_disabled):
        try:
            with st.spinner("Downloading from GridFS..."):
                ensure_database(force=force)
            st.success("Download complete. Reload other pages to use the data.")
        except Exception as exc:
            st.error(f"Download failed: {type(exc).__name__}: {exc}")
