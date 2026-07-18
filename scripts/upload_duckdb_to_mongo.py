"""Upload the local DuckDB store into MongoDB GridFS (run once from dev).

The ~775MB ``storage/market_data.duckdb`` is too large for GitHub, so the
deployed app downloads it from GridFS at startup. Run this after (re)building the
store locally to publish it.

Usage (PowerShell)::

    $env:MONGO_URI = "mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
    python scripts/upload_duckdb_to_mongo.py

Optional overrides via env: MONGO_DB_NAME, MONGO_GRIDFS_BUCKET, MONGO_DUCKDB_FILE.
"""

from __future__ import annotations

import sys

from core.data.storage.provisioning import upload_to_gridfs


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else None
    upload_to_gridfs(src)
    print("Upload finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())