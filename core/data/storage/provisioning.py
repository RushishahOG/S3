"""Database provisioning: download the DuckDB file from MongoDB GridFS.

The analytical store (``storage/market_data.duckdb``, ~775MB) is far too large to
commit to GitHub, so it is absent on hosts that only have the repo (e.g.
Streamlit Community Cloud). We keep DuckDB as the query engine everywhere and use
MongoDB GridFS purely as blob storage for the file:

* **Local dev** already has the file on disk -> provisioning is a no-op.
* **Cloud** downloads the file from GridFS to local disk on first startup, then
  opens it with DuckDB exactly as before. No query-layer changes.

Connection settings are read (in priority order) from:
    1. ``st.secrets`` (Streamlit Cloud "Secrets" UI), and
    2. environment variables.

Relevant keys::

    MONGO_URI            mongodb+srv://user:pass@cluster/...   (required on cloud)
    MONGO_DB_NAME        database name              (default: "smartbeta")
    MONGO_GRIDFS_BUCKET  GridFS bucket name         (default: "duckdb_store")
    MONGO_DUCKDB_FILE    logical filename in GridFS (default: "market_data.duckdb.gz")
"""

from __future__ import annotations

import gzip
import os
import tempfile

from core.config.settings import settings
from core.utils.logging_config import get_logger
from core.utils.paths import ensure_dir

logger = get_logger(__name__)

_DEFAULTS = {
    "MONGO_DB_NAME": "smartbeta",
    "MONGO_GRIDFS_BUCKET": "duckdb_store",
    "MONGO_DUCKDB_FILE": "market_data.duckdb.gz",
}

#: Set once provisioning has run so repeated calls (Streamlit reruns) are cheap.
_PROVISIONED = False


def _get_conf(key: str, default: str | None = None) -> str | None:
    """Read a config value from Streamlit secrets first, then the environment."""
    try:
        import streamlit as st

        # ``st.secrets`` raises if no secrets file exists; guard with try/except.
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.environ.get(key, default)


def _mongo_uri() -> str | None:
    return _get_conf("MONGO_URI")


def ensure_database(db_path: str | None = None, *, force: bool = False) -> str:
    """Ensure the DuckDB file exists locally, downloading from GridFS if needed.

    Returns the resolved local path. Safe to call repeatedly (idempotent): if the
    file already exists it returns immediately unless ``force`` is set.
    """
    global _PROVISIONED
    dest = db_path or settings.storage.database_abs_path

    if not force and os.path.exists(dest):
        _PROVISIONED = True
        return dest
    if _PROVISIONED and not force:
        return dest

    uri = _mongo_uri()
    if not uri:
        logger.warning(
            "DuckDB file missing at %s and MONGO_URI is not set; cannot download.", dest
        )
        return dest

    download_from_gridfs(dest, uri=uri)
    _PROVISIONED = True
    return dest


def is_configured() -> bool:
    """True if a MongoDB URI is available (secrets or env)."""
    return bool(_mongo_uri())


def config_summary() -> dict:
    """Non-secret view of the active Mongo configuration for display."""
    uri = _mongo_uri()
    host = ""
    if uri:
        tail = uri.split("@", 1)[-1]
        host = tail.split("/", 1)[0].split("?", 1)[0]
    return {
        "configured": bool(uri),
        "host": host,
        "db_name": _get_conf("MONGO_DB_NAME", _DEFAULTS["MONGO_DB_NAME"]),
        "bucket": _get_conf("MONGO_GRIDFS_BUCKET", _DEFAULTS["MONGO_GRIDFS_BUCKET"]),
        "filename": _get_conf("MONGO_DUCKDB_FILE", _DEFAULTS["MONGO_DUCKDB_FILE"]),
    }


def local_status(db_path: str | None = None) -> dict:
    """Local DuckDB file presence + size."""
    path = db_path or settings.storage.database_abs_path
    exists = os.path.exists(path)
    size_mb = os.path.getsize(path) / 1_048_576 if exists else 0.0
    return {"path": path, "exists": exists, "size_mb": size_mb}


def test_connection(*, uri: str | None = None) -> dict:
    """Ping the cluster. Returns {'ok': bool, 'detail': str, 'databases': [...]}"""
    try:
        from pymongo import MongoClient
    except ImportError:
        return {"ok": False, "detail": "pymongo not installed", "databases": []}

    uri = uri or _mongo_uri()
    if not uri:
        return {"ok": False, "detail": "MONGO_URI not configured", "databases": []}

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        client.admin.command("ping")
        dbs = client.list_database_names()
        return {"ok": True, "detail": "ping ok", "databases": dbs}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}", "databases": []}
    finally:
        if client is not None:
            client.close()


def gridfs_status(*, uri: str | None = None) -> dict:
    """Whether the DuckDB blob exists in GridFS, plus its size/upload date."""
    try:
        from pymongo import MongoClient
    except ImportError:
        return {"exists": False, "detail": "pymongo not installed"}

    uri = uri or _mongo_uri()
    if not uri:
        return {"exists": False, "detail": "MONGO_URI not configured"}

    db_name = _get_conf("MONGO_DB_NAME", _DEFAULTS["MONGO_DB_NAME"])
    bucket = _get_conf("MONGO_GRIDFS_BUCKET", _DEFAULTS["MONGO_GRIDFS_BUCKET"])
    filename = _get_conf("MONGO_DUCKDB_FILE", _DEFAULTS["MONGO_DUCKDB_FILE"])

    client = None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=8000)
        files_coll = client[db_name][f"{bucket}.files"]
        doc = files_coll.find_one({"filename": filename}, sort=[("uploadDate", -1)])
        if not doc:
            return {"exists": False, "detail": "no blob found", "filename": filename}
        return {
            "exists": True,
            "filename": filename,
            "size_mb": doc.get("length", 0) / 1_048_576,
            "upload_date": str(doc.get("uploadDate", "")),
        }
    except Exception as exc:
        return {"exists": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        if client is not None:
            client.close()


def upload_to_gridfs(src_path: str | None = None, *, uri: str | None = None) -> None:
    """Upload the local DuckDB file into MongoDB GridFS (run once, from dev).

    Replaces any existing blob with the same logical filename so re-uploading a
    refreshed store is safe.
    """
    try:
        from pymongo import MongoClient
        import gridfs
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pymongo is required. Add 'pymongo' to requirements.txt.") from exc

    src = src_path or settings.storage.database_abs_path
    if not os.path.exists(src):
        raise FileNotFoundError(f"Cannot upload: DuckDB file not found at {src!r}.")

    uri = uri or _mongo_uri()
    if not uri:
        raise RuntimeError("MONGO_URI is not configured; cannot upload database.")

    db_name = _get_conf("MONGO_DB_NAME", _DEFAULTS["MONGO_DB_NAME"])
    bucket = _get_conf("MONGO_GRIDFS_BUCKET", _DEFAULTS["MONGO_GRIDFS_BUCKET"])
    filename = _get_conf("MONGO_DUCKDB_FILE", _DEFAULTS["MONGO_DUCKDB_FILE"])

    client = None
    try:
        client = MongoClient(uri)
        db = client[db_name]
        fs = gridfs.GridFSBucket(db, bucket_name=bucket)

        # Delete any prior versions of this filename first.
        files_coll = db[f"{bucket}.files"]
        for doc in files_coll.find({"filename": filename}, {"_id": 1}):
            fs.delete(doc["_id"])
            logger.info("Deleted previous GridFS blob %s", doc["_id"])

        size_mb = os.path.getsize(src) / 1_048_576
        do_compress = filename.endswith(".gz")

        if do_compress:
            logger.info("Compressing and uploading %s (%.1f MB) to GridFS bucket '%s'...", src, size_mb, bucket)
            gz_path = src + ".upload_tmp.gz"
            try:
                with open(src, "rb") as f_raw, gzip.open(gz_path, "wb") as gz:
                    while chunk := f_raw.read(8192 * 1024):
                        gz.write(chunk)
                gz_size = os.path.getsize(gz_path) / 1_048_576
                logger.info("Compressed size: %.1f MB (%.1f%% reduction)", gz_size, 100.0 * gz_size / size_mb)
                with open(gz_path, "rb") as fh:
                    fs.upload_from_stream(filename, fh)
            finally:
                if os.path.exists(gz_path):
                    os.unlink(gz_path)
        else:
            logger.info("Uploading %s (%.1f MB) to GridFS bucket '%s'...", src, size_mb, bucket)
            with open(src, "rb") as fh:
                fs.upload_from_stream(filename, fh)

        logger.info("Upload complete.")
    finally:
        if client is not None:
            client.close()


def download_from_gridfs(dest_path: str, *, uri: str | None = None) -> str:
    """Download the DuckDB blob from MongoDB GridFS.

    If the stored filename ends with ``.gz`` the blob is gzip-compressed and is
    transparently decompressed after download; otherwise it is saved verbatim.
    """
    try:
        from pymongo import MongoClient
        import gridfs
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "pymongo is required to download the database from MongoDB GridFS. "
            "Add 'pymongo' to requirements.txt."
        ) from exc

    uri = uri or _mongo_uri()
    if not uri:
        raise RuntimeError("MONGO_URI is not configured; cannot download database.")

    db_name = _get_conf("MONGO_DB_NAME", _DEFAULTS["MONGO_DB_NAME"])
    bucket = _get_conf("MONGO_GRIDFS_BUCKET", _DEFAULTS["MONGO_GRIDFS_BUCKET"])
    stored_filename = _get_conf("MONGO_DUCKDB_FILE", _DEFAULTS["MONGO_DUCKDB_FILE"])
    is_compressed = stored_filename.endswith(".gz")

    # When the blob is compressed we decompress to the final .duckdb path;
    # otherwise we write directly to it.
    final_dest = dest_path
    if is_compressed:
        final_dest = dest_path.replace(".gz", "") if dest_path.endswith(".gz") else dest_path
    else:
        # dest_path is already the final .duckdb path
        pass

    ensure_dir(os.path.dirname(final_dest))
    logger.info(
        "Downloading %s from GridFS bucket '%s'%s...",
        stored_filename,
        bucket,
        " and decompressing" if is_compressed else "",
    )

    tmp = dest_path + ".part"
    client = None
    try:
        client = MongoClient(uri)
        fs = gridfs.GridFSBucket(client[db_name], bucket_name=bucket)

        with open(tmp, "wb") as fh:
            fs.download_to_stream_by_name(stored_filename, fh)

        if is_compressed:
            with gzip.open(tmp, "rb") as f_in:
                with open(final_dest, "wb") as f_out:
                    while chunk := f_in.read(8192 * 1024):
                        f_out.write(chunk)
            os.remove(tmp)
        else:
            os.replace(tmp, final_dest)

        size_mb = os.path.getsize(final_dest) / 1_048_576
        logger.info("Downloaded DuckDB to %s (%.1f MB)", final_dest, size_mb)
    finally:
        if client is not None:
            client.close()
    return final_dest
