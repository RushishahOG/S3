"""Local analytical storage backed by DuckDB.

Responsibilities:
  * Persist raw vendor price data (long format, one row per ticker/date).
  * Persist the wide feature store (one row per ticker/date, one column per
    feature) and auto-extend the schema when new factors are registered.
  * Answer analytical queries (price panels, feature panels, statistics).

DuckDB is the V1 engine; the rest of the platform talks only to this manager,
so swapping the engine later is localised to this module.
"""

from __future__ import annotations

import os
import time
from typing import Iterable

import duckdb
import pandas as pd

from core.config import settings
from core.data.providers.base_provider import PriceColumns
from core.utils.logging_config import get_logger
from core.utils.paths import ensure_dir

logger = get_logger(__name__)


class StorageManager:
    #: How long (seconds) to keep retrying a blocked ``duckdb.connect`` before
    #: giving up with an actionable error. Kept well under the UI's 120s load
    #: watchdog so a lock stall surfaces *as a lock stall* rather than a mute
    #: "no progress" timeout.
    _CONNECT_TIMEOUT_S = 10.0
    _CONNECT_RETRY_DELAY_S = 0.5

    def __init__(self, database_path: str | None = None, read_only: bool = False) -> None:
        self.db_path = database_path or settings.storage.database_abs_path
        self.read_only = read_only
        ensure_dir(os.path.dirname(self.db_path))
        # Read-only connections can be opened concurrently with other readers,
        # which is what lets the backtest worker read the store at the same time
        # the live Streamlit app holds its own read-only connection. On Windows,
        # however, a *read-write* connection takes an exclusive OS file handle:
        # while it is open, EVERY other connection (even read-only) fails to open
        # the file. If we simply retried forever the caller would hang until the
        # UI's 120s watchdog fired with a mute "no progress" message. Instead we
        # fail fast with a clear diagnostic naming the real cause.
        self._conn = self._connect(read_only=read_only)
        # Windows: automatic mid-transaction checkpoints can race with the OS
        # (AV / indexer) holding the .wal file, raising a FatalException that
        # kills the connection. Push the auto-checkpoint threshold very high so
        # checkpoints only happen at explicit, retry-guarded points.
        try:
            self._conn.execute("PRAGMA wal_autocheckpoint='1TB'")
        except Exception:  # pragma: no cover - older duckdb
            pass
        if not read_only:
            self._initialise_schema()

    def _connect(self, read_only: bool) -> "duckdb.DuckDBPyConnection":
        """Open the DuckDB connection, failing fast on a writer file lock.

        A file lock surfaces two different ways depending on who holds it:

        * cross-process: ``IOException`` ("... being used by another process")
          when another OS process holds a read-write handle; and
        * same-process: ``ConnectionException`` ("Can't open a connection to
          same database file with a different configuration") when this process
          already opened a read-write connection.

        We retry briefly (the lock may be a transient checkpoint) and then raise
        a message that points straight at the cause, so a blocked backtest
        reports a database lock instead of stalling until the 120s load watchdog
        trips with no explanation.

        A *read-only* connection to a non-existent file is a different failure
        (the DB was never provisioned / deployed) and must NOT be retried or
        reported as a lock -- we surface that immediately with its own message.
        """
        # Missing-file check first: a read-only connect cannot create the DB, so
        # if the file is absent this is a deployment/provisioning problem, not a
        # lock. Retrying for 10s and blaming a "writer lock" would be misleading.
        if read_only and not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"DuckDB database not found at {self.db_path!r}. The data store "
                "has not been provisioned on this host. On Streamlit Community "
                "Cloud the ~800MB .duckdb file is not in the repo; download or "
                "rebuild it at startup (see build_store()/a startup hook), or "
                "point settings.storage at a hosted copy."
            )

        deadline = time.perf_counter() + self._CONNECT_TIMEOUT_S
        last_exc: Exception | None = None
        while True:
            try:
                return duckdb.connect(self.db_path, read_only=read_only)
            except (duckdb.IOException, duckdb.ConnectionException, IOError) as exc:
                # A "does not exist" IOException is not a lock -- don't retry.
                if "does not exist" in str(exc).lower():
                    raise FileNotFoundError(
                        f"DuckDB database not found at {self.db_path!r}. The data "
                        "store has not been provisioned on this host (the ~800MB "
                        ".duckdb file is typically not committed to the repo). "
                        "Rebuild or download it at startup before opening a "
                        "read-only connection."
                    ) from exc
                last_exc = exc
                if time.perf_counter() >= deadline:
                    break
                time.sleep(self._CONNECT_RETRY_DELAY_S)
        mode = "read-only" if read_only else "read-write"
        raise RuntimeError(
            f"DuckDB is locked: could not open a {mode} connection to "
            f"{self.db_path!r} within {self._CONNECT_TIMEOUT_S:.0f}s. A "
            "read-write connection is holding an exclusive file lock (DuckDB "
            "allows either one writer OR many readers, never both). Close any "
            "ingestion/feature-engineering process or read-write StorageManager "
            "and retry."
        ) from last_exc

    def checkpoint(self, retries: int = 5, delay: float = 0.5) -> bool:
        """Force a WAL checkpoint, retrying if the OS momentarily holds the
        .wal file (common on Windows with AV / Search Indexer). Returns True on
        success, False if all retries were exhausted (data is still safe in the
        WAL and will be folded in on the next successful checkpoint)."""
        import time

        for attempt in range(1, retries + 1):
            try:
                self._conn.execute("CHECKPOINT")
                return True
            except Exception as exc:  # duckdb IOException / FatalException
                if attempt == retries:
                    logger.warning("Checkpoint failed after %d attempts: %s", retries, exc)
                    return False
                time.sleep(delay * attempt)
        return False

    @staticmethod
    def cleanup_gridfs(*, uri: str | None = None) -> None:
        """Delete orphaned files and chunks in GridFS to free quota.

        This is useful when the legacy ``market_data.duckdb`` blob still occupies
        ~775MB and prevents a new, compressed ``market_data.duckdb.gz`` upload that
        fits inside the Atlas free‑tier limit (512MB). Callers must have a valid
        MONGO_URI configuration available.
        """
        try:
            from pymongo import MongoClient
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pymongo is required. Add 'pymongo' to requirements.txt.") from exc

        # Import configuration helpers from the provisioning module.
        from core.data.storage.provisioning import _get_conf, _mongo_uri, _DEFAULTS
        from core.utils.logging_config import get_logger
        logger = get_logger(__name__)

        uri = uri or _mongo_uri()
        if not uri:
            raise RuntimeError("MONGO_URI is not configured; cannot clean up GridFS.")

        db_name = _get_conf("MONGO_DB_NAME", _DEFAULTS["MONGO_DB_NAME"])
        bucket = _get_conf("MONGO_GRIDFS_BUCKET", _DEFAULTS["MONGO_GRIDFS_BUCKET"])

        client = None
        try:
            client = MongoClient(uri)
            db = client[db_name]
            # The official GridFS collections are <bucket>.files and <bucket>.chunks.
            # Delete everything to ensure a clean slate for a fresh .gz upload.
            files_coll = db[f"{bucket}.files"]
            chunks_coll = db[f"{bucket}.chunks"]
            files_deleted = files_coll.delete_many({})
            chunks_deleted = chunks_coll.delete_many({})
            logger.info(
                "Cleaned up GridFS: %d files, %d chunks (total ~%d MB freed)",
                files_deleted.deleted_count,
                chunks_deleted.deleted_count,
                (files_deleted.deleted_count + chunks_deleted.deleted_count) * 0.26,
            )
        finally:
            if client is not None:
                client.close()

    # -- lifecycle ---------------------------------------------------------
    def _initialise_schema(self) -> None:
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.storage.prices_table} (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (ticker, date)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.storage.feature_store_table} (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                PRIMARY KEY (ticker, date)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.storage.feature_metadata_table} (
                feature_key VARCHAR NOT NULL,
                factor VARCHAR NOT NULL,
                parameter_family VARCHAR,
                smartbeta_name VARCHAR,
                formula VARCHAR,
                lookback VARCHAR,
                frequency VARCHAR,
                description VARCHAR,
                valid_obs INTEGER,
                missing_values INTEGER,
                coverage_pct DOUBLE,
                added_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (feature_key, factor)
            );
            """
        )
        # Backwards-compatible migration: older stores may lack the new columns.
        self._ensure_feature_metadata_columns()
        self._initialise_fundamentals()
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS download_metadata (
                ticker VARCHAR NOT NULL,
                company_name VARCHAR,
                provider VARCHAR,
                status VARCHAR,
                rows INTEGER,
                earliest_date DATE,
                latest_date DATE,
                downloaded_at TIMESTAMP DEFAULT now(),
                error VARCHAR,
                retries INTEGER,
                PRIMARY KEY (ticker)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS validation_anomalies (
                ticker VARCHAR,
                issue VARCHAR,
                detail VARCHAR,
                recorded_at TIMESTAMP DEFAULT now()
            );
            """
        )
        logger.debug("Storage schema initialised at %s", self.db_path)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "StorageManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- prices ------------------------------------------------------------
    def upsert_prices(self, df: pd.DataFrame) -> int:
        """Insert or replace price rows. Returns number of rows written."""
        if df is None or df.empty:
            return 0
        cols = [c for c in PriceColumns.LONG_COLUMNS if c in df.columns]
        df = df[cols].copy()
        df[PriceColumns.DATE] = pd.to_datetime(df[PriceColumns.DATE]).dt.date
        self._upsert(df, settings.storage.prices_table, [PriceColumns.TICKER, PriceColumns.DATE])
        logger.info("Upserted %d price rows", len(df))
        return len(df)

    def get_prices(
        self,
        tickers: Iterable[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return price rows (long format) filtered by ticker/date/field."""
        tbl = settings.storage.prices_table
        where, params = self._build_filters(tickers, start, end)
        def _quote(f: str) -> str:
            return f'"{f}"' if " " in f else f
        field_sql = ", ".join(_quote(f) for f in fields) if fields else "*"
        query = f"SELECT {field_sql} FROM {tbl} {where} ORDER BY ticker, date"
        return self._conn.execute(query, params).fetchdf()

    def get_adjusted_price_panel(
        self,
        tickers: Iterable[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Wide adj_close panel: index=date, columns=tickers."""
        df = self.get_prices(
            tickers=tickers, start=start, end=end, fields=[PriceColumns.TICKER, PriceColumns.DATE, PriceColumns.ADJ_CLOSE]
        )
        if df.empty:
            return pd.DataFrame()
        panel = df.pivot(index=PriceColumns.DATE, columns=PriceColumns.TICKER, values=PriceColumns.ADJ_CLOSE)
        panel.index = pd.to_datetime(panel.index)
        return panel.sort_index()

    def latest_date_per_ticker(self, tickers: Iterable[str] | None = None) -> dict[str, pd.Timestamp]:
        tbl = settings.storage.prices_table
        where, params = self._build_filters(tickers, None, None)
        query = f"SELECT ticker, max(date) AS last_date FROM {tbl} {where} GROUP BY ticker"
        rows = self._conn.execute(query, params).fetchall()
        return {t: pd.Timestamp(d) for t, d in rows}

    def earliest_date_per_ticker(self, tickers: Iterable[str] | None = None) -> dict[str, pd.Timestamp]:
        tbl = settings.storage.prices_table
        where, params = self._build_filters(tickers, None, None)
        query = f"SELECT ticker, min(date) AS first_date FROM {tbl} {where} GROUP BY ticker"
        rows = self._conn.execute(query, params).fetchall()
        return {t: pd.Timestamp(d) for t, d in rows}

    def stored_tickers(self) -> list[str]:
        tbl = settings.storage.prices_table
        return [r[0] for r in self._conn.execute(f"SELECT DISTINCT ticker FROM {tbl} ORDER BY ticker").fetchall()]

    # -- feature store -----------------------------------------------------
    def ensure_feature_columns(self, feature_keys: Iterable[str]) -> None:
        """Add any missing feature columns to the wide feature store table."""
        existing = self._column_names(settings.storage.feature_store_table)
        for key in feature_keys:
            if key in (PriceColumns.TICKER, PriceColumns.DATE):
                continue
            if key not in existing:
                self._conn.execute(
                    f"ALTER TABLE {settings.storage.feature_store_table} ADD COLUMN \"{key}\" DOUBLE"
                )
                logger.debug("Added feature column %s", key)

    def upsert_features(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        feature_keys = [c for c in df.columns if c not in (PriceColumns.TICKER, PriceColumns.DATE)]
        self.ensure_feature_columns(feature_keys)
        out = df.copy()
        out[PriceColumns.DATE] = pd.to_datetime(out[PriceColumns.DATE]).dt.date
        # Full refresh for the affected tickers: drop every pre-existing row for
        # them first so stale dates left behind by earlier engine versions (whose
        # history spans a different date range) don't linger with NULL feature
        # values. The subsequent upsert then writes only the current rows.
        tickers = list(pd.unique(out[PriceColumns.TICKER]))
        placeholders = ", ".join("?" for _ in tickers)
        self._conn.execute(
            f"DELETE FROM {settings.storage.feature_store_table} "
            f"WHERE {PriceColumns.TICKER} IN ({placeholders})",
            tickers,
        )
        self._upsert(out, settings.storage.feature_store_table, [PriceColumns.TICKER, PriceColumns.DATE])
        self.checkpoint()
        return len(out)

    def get_features(
        self,
        tickers: Iterable[str] | None = None,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        tbl = settings.storage.feature_store_table
        # Always return the key columns (ticker, date) so downstream code can
        # group / pivot by them even when only a subset of features is requested.
        if columns:
            requested = [PriceColumns.TICKER, PriceColumns.DATE] + [
                c for c in columns if c not in (PriceColumns.TICKER, PriceColumns.DATE)
            ]
            select = ", ".join(f'"{c}"' for c in requested)
        else:
            select = "*"
        where, params = self._build_filters(tickers, start, end)
        query = f"SELECT {select} FROM {tbl} {where} ORDER BY ticker, date"
        return self._conn.execute(query, params).fetchdf()

    def feature_date_range(self) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        tbl = settings.storage.feature_store_table
        row = self._conn.execute(f"SELECT min(date), max(date) FROM {tbl}").fetchone()
        lo = pd.Timestamp(row[0]) if row[0] else None
        hi = pd.Timestamp(row[1]) if row[1] else None
        return lo, hi

    def feature_columns(self) -> list[str]:
        skip = {PriceColumns.TICKER, PriceColumns.DATE}
        return [c for c in self._column_names(settings.storage.feature_store_table) if c not in skip]

    def drop_feature_columns(self, columns: Iterable[str]) -> int:
        """Remove obsolete feature columns from the wide feature store.

        Used to prune columns left behind by previous code versions (e.g. old
        ``beta_24m/36m/48m`` windows) after regenerating with the current engine.
        Returns the number of columns actually dropped.
        """
        existing = set(self._column_names(settings.storage.feature_store_table))
        dropped = 0
        for col in columns:
            if col in existing and col not in (PriceColumns.TICKER, PriceColumns.DATE):
                self._conn.execute(
                    f'ALTER TABLE {settings.storage.feature_store_table} DROP COLUMN "{col}"'
                )
                dropped += 1
                logger.debug("Dropped obsolete feature column %s", col)
        return dropped

    def drop_quality_feature_columns(self, columns: list[str]) -> int:
        """Remove obsolete columns from the quality features table."""
        table = "fundamental_quality_features"
        existing = set(self._column_names(table))
        dropped = 0
        for col in columns:
            if col in existing and col not in ("ticker", "financial_year"):
                self._conn.execute(f'ALTER TABLE {table} DROP COLUMN "{col}"')
                dropped += 1
        return dropped

    def register_feature_metadata(
        self, factor: str, specs: list, stats: dict[str, dict] | None = None
    ) -> None:
        """Persist rich per-feature metadata (incl. coverage statistics).

        ``specs`` is a list of :class:`FactorSpec`; ``stats`` maps a feature key
        to ``{"valid_obs", "missing_values", "coverage_pct"}`` computed from the
        generated frame.
        """
        if not specs:
            return
        rows = []
        for s in specs:
            st_ = (stats or {}).get(s.key, {})
            rows.append({
                "feature_key": s.key,
                "factor": factor,
                "parameter_family": s.parameter_family or None,
                "smartbeta_name": s.smartbeta_name or None,
                "formula": s.formula or None,
                "lookback": s.lookback or None,
                "frequency": s.frequency or None,
                "description": s.description or None,
                "valid_obs": int(st_.get("valid_obs", 0) or 0),
                "missing_values": int(st_.get("missing_values", 0) or 0),
                "coverage_pct": float(st_.get("coverage_pct", 0.0) or 0.0),
            })
        df = pd.DataFrame(rows)
        df["added_at"] = pd.Timestamp.now()
        df = df.where(pd.notnull(df), None)
        self._conn.execute(
            f"DELETE FROM {settings.storage.feature_metadata_table} WHERE factor = ?", [factor]
        )
        cols = ", ".join(df.columns)
        self._conn.execute(
            f"INSERT INTO {settings.storage.feature_metadata_table} ({cols}) SELECT * FROM df"
        )

    def feature_metadata(self) -> pd.DataFrame:
        return self._conn.execute(
            f"SELECT * FROM {settings.storage.feature_metadata_table} ORDER BY factor, feature_key"
        ).fetchdf()

    def _ensure_feature_metadata_columns(self) -> None:
        """Add any missing metadata columns (schema migration for old stores)."""
        desired = {
            "parameter_family": "VARCHAR",
            "smartbeta_name": "VARCHAR",
            "formula": "VARCHAR",
            "lookback": "VARCHAR",
            "frequency": "VARCHAR",
            "valid_obs": "INTEGER",
            "missing_values": "INTEGER",
            "coverage_pct": "DOUBLE",
        }
        existing = set(self._column_names(settings.storage.feature_metadata_table))
        for col, dtype in desired.items():
            if col not in existing:
                self._conn.execute(
                    f"ALTER TABLE {settings.storage.feature_metadata_table} ADD COLUMN {col} {dtype}"
                )

    # -- download metadata --------------------------------------------------
    def upsert_download_metadata(self, records: list[dict]) -> None:
        """Persist per-ticker download metadata (latest run wins)."""
        if not records:
            return
        df = pd.DataFrame(records)
        df["downloaded_at"] = pd.Timestamp.now()
        for col in ("company_name", "provider", "status", "error"):
            if col not in df.columns:
                df[col] = None
        df = df.where(pd.notnull(df), None)
        self._upsert(
            df,
            "download_metadata",
            ["ticker"],
        )

    def get_download_metadata(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        if tickers:
            placeholders = ", ".join(["?"] * len(list(tickers)))
            query = f"SELECT * FROM download_metadata WHERE ticker IN ({placeholders}) ORDER BY ticker"
            return self._conn.execute(query, list(tickers)).fetchdf()
        return self._conn.execute("SELECT * FROM download_metadata ORDER BY ticker").fetchdf()

    def record_anomalies(self, anomalies: list[dict]) -> None:
        if not anomalies:
            return
        df = pd.DataFrame(anomalies)
        df["recorded_at"] = pd.Timestamp.now()
        df = df.where(pd.notnull(df), None)
        self._conn.execute("DELETE FROM validation_anomalies")
        self._conn.execute(
            "INSERT INTO validation_anomalies (ticker, issue, detail, recorded_at) SELECT * FROM df"
        )

    def get_anomalies(self) -> pd.DataFrame:
        return self._conn.execute("SELECT * FROM validation_anomalies ORDER BY ticker").fetchdf()

    def last_download_time(self) -> pd.Timestamp | None:
        """Most recent ``downloaded_at`` across all download metadata rows."""
        row = self._conn.execute(
            "SELECT max(downloaded_at) FROM download_metadata"
        ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    def price_quality_report(self) -> pd.DataFrame:
        """Per-ticker data-quality aggregates for the validation panel.

        Returns a frame with one row per stored ticker and the columns
        ``ticker``, ``duplicate_rows`` and ``invalid_prices``.
        """
        tbl = settings.storage.prices_table
        query = f"""
            SELECT
                ticker,
                CAST(count(*) - count(DISTINCT date) AS BIGINT) AS duplicate_rows,
                CAST(sum(
                    CASE
                        WHEN close IS NULL OR close <= 0
                          OR open IS NULL OR open <= 0
                          OR high IS NULL OR low IS NULL
                          OR high < low
                        THEN 1 ELSE 0 END
                ) AS BIGINT) AS invalid_prices
            FROM {tbl}
            GROUP BY ticker
        """
        return self._conn.execute(query).fetchdf()

    # -- fundamental data (Quality factor, Screener pipeline) ---------------
    def _initialise_fundamentals(self) -> None:
        """Create the normalised fundamental tables (idempotent)."""
        # --- Screener (single source of truth) schema ------------------------
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_company (
                ticker VARCHAR NOT NULL,
                company_name VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                market_cap DOUBLE,
                pe DOUBLE,
                pb DOUBLE,
                dividend_yield DOUBLE,
                current_price DOUBLE,
                last_updated DATE,
                url VARCHAR,
                PRIMARY KEY (ticker)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_income_statement_annual (
                ticker VARCHAR NOT NULL,
                financial_year INTEGER NOT NULL,
                sales DOUBLE,
                operating_profit DOUBLE,
                expenses DOUBLE,
                other_income DOUBLE,
                interest DOUBLE,
                depreciation DOUBLE,
                profit_before_tax DOUBLE,
                tax_percent DOUBLE,
                net_profit DOUBLE,
                eps DOUBLE,
                dividend_payout_percent DOUBLE,
                PRIMARY KEY (ticker, financial_year)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_income_statement_quarterly (
                ticker VARCHAR NOT NULL,
                quarter_index INTEGER NOT NULL,
                quarter_label VARCHAR,
                quarterly_sales DOUBLE,
                quarterly_operating_profit DOUBLE,
                quarterly_expenses DOUBLE,
                quarterly_interest DOUBLE,
                quarterly_net_profit DOUBLE,
                quarterly_eps DOUBLE,
                PRIMARY KEY (ticker, quarter_index)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_balance_sheet (
                ticker VARCHAR NOT NULL,
                financial_year INTEGER NOT NULL,
                equity_capital DOUBLE,
                reserves DOUBLE,
                borrowings DOUBLE,
                other_liabilities DOUBLE,
                total_liabilities DOUBLE,
                fixed_assets DOUBLE,
                cwip DOUBLE,
                investments DOUBLE,
                other_assets DOUBLE,
                total_assets DOUBLE,
                current_liabilities DOUBLE,
                PRIMARY KEY (ticker, financial_year)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_cashflow (
                ticker VARCHAR NOT NULL,
                financial_year INTEGER NOT NULL,
                operating_cash_flow DOUBLE,
                investing_cash_flow DOUBLE,
                financing_cash_flow DOUBLE,
                free_cash_flow DOUBLE,
                net_cash_flow DOUBLE,
                cfo_per_op DOUBLE,
                PRIMARY KEY (ticker, financial_year)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_dividends (
                ticker VARCHAR NOT NULL,
                ex_date DATE NOT NULL,
                dividend_amount DOUBLE,
                PRIMARY KEY (ticker, ex_date)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamentals_ratios (
                ticker VARCHAR NOT NULL,
                financial_year INTEGER NOT NULL,
                roe DOUBLE,
                roce DOUBLE,
                working_capital_days DOUBLE,
                debtor_days DOUBLE,
                cash_conversion_cycle DOUBLE,
                PRIMARY KEY (ticker, financial_year)
            );
            """
        )
        # Engineered quality factors (16 factors + rolling-lookback aggregates).
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamental_quality_features (
                ticker VARCHAR NOT NULL,
                financial_year INTEGER NOT NULL,
                roe DOUBLE,
                roce DOUBLE,
                roa DOUBLE,
                interest_coverage_ratio DOUBLE,
                equity_to_total_capital DOUBLE,
                dividend_payout_ratio DOUBLE,
                dividend_payout_ratio_cumulative DOUBLE,
                ocf_to_ebitda DOUBLE,
                cash_roce DOUBLE,
                eps_growth DOUBLE,
                eps_growth_median DOUBLE,
                eps_growth_weighted DOUBLE,
                roe_growth DOUBLE,
                roe_growth_median DOUBLE,
                roe_growth_weighted DOUBLE,
                sustainable_growth_rate DOUBLE,
                roce_growth DOUBLE,
                roce_growth_median DOUBLE,
                roce_growth_weighted DOUBLE,
                revenue_growth DOUBLE,
                revenue_growth_median DOUBLE,
                revenue_growth_weighted DOUBLE,
                dps_growth DOUBLE,
                dps_growth_median DOUBLE,
                created_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (ticker, financial_year)
            );
            """
        )
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS fundamental_download_metadata (
                ticker VARCHAR NOT NULL,
                company_name VARCHAR,
                status VARCHAR,
                financials_status VARCHAR,
                ratios_status VARCHAR,
                error VARCHAR,
                retries INTEGER,
                updated_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (ticker)
            );
            """
        )
        logger.debug("Fundamental schema initialised")

    # -- screener (single source of truth) accessors ------------------------
    def upsert_fundamentals_company(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["last_updated"] = pd.to_datetime(df["last_updated"]).dt.date
        self._upsert(df, "fundamentals_company", ["ticker"])
        logger.info("Upserted %d company rows", len(df))
        return len(df)

    def upsert_fundamentals_income_annual(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["financial_year"] = df["financial_year"].astype("int64")
        self._upsert(df, "fundamentals_income_statement_annual", ["ticker", "financial_year"])
        return len(df)

    def upsert_fundamentals_income_quarterly(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["quarter_index"] = df["quarter_index"].astype("int64")
        self._upsert(df, "fundamentals_income_statement_quarterly", ["ticker", "quarter_index"])
        return len(df)

    def upsert_fundamentals_balance_sheet(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["financial_year"] = df["financial_year"].astype("int64")
        self._upsert(df, "fundamentals_balance_sheet", ["ticker", "financial_year"])
        return len(df)

    def upsert_fundamentals_cashflow(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["financial_year"] = df["financial_year"].astype("int64")
        self._upsert(df, "fundamentals_cashflow", ["ticker", "financial_year"])
        return len(df)

    def upsert_fundamentals_dividends(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
        self._upsert(df, "fundamentals_dividends", ["ticker", "ex_date"])
        return len(df)

    def upsert_fundamentals_ratios(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["financial_year"] = df["financial_year"].astype("int64")
        self._upsert(df, "fundamentals_ratios", ["ticker", "financial_year"])
        return len(df)

    def upsert_fundamental_quality_features(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        df = df.copy()
        df["financial_year"] = df["financial_year"].astype("int64")
        self._upsert(df, "fundamental_quality_features", ["ticker", "financial_year"])
        logger.info("Upserted %d engineered quality feature rows", len(df))
        return len(df)

    def get_fundamentals_company(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        query = f"SELECT * FROM fundamentals_company {where} ORDER BY ticker"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_income_annual(
        self, tickers: Iterable[str] | None = None, financial_year: int | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        if financial_year is not None:
            clauses = [where] if where else []
            clauses.append("financial_year = ?")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params = list(params) + [financial_year]
        query = f"SELECT * FROM fundamentals_income_statement_annual {where} ORDER BY ticker, financial_year"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_income_quarterly(
        self, tickers: Iterable[str] | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        query = f"SELECT * FROM fundamentals_income_statement_quarterly {where} ORDER BY ticker, quarter_index"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_balance_sheet(
        self, tickers: Iterable[str] | None = None, financial_year: int | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        if financial_year is not None:
            clauses = [where] if where else []
            clauses.append("financial_year = ?")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params = list(params) + [financial_year]
        query = f"SELECT * FROM fundamentals_balance_sheet {where} ORDER BY ticker, financial_year"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_cashflow(
        self, tickers: Iterable[str] | None = None, financial_year: int | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        if financial_year is not None:
            clauses = [where] if where else []
            clauses.append("financial_year = ?")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params = list(params) + [financial_year]
        query = f"SELECT * FROM fundamentals_cashflow {where} ORDER BY ticker, financial_year"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_dividends(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        query = f"SELECT * FROM fundamentals_dividends {where} ORDER BY ticker, ex_date"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamentals_ratios(
        self, tickers: Iterable[str] | None = None, financial_year: int | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        if financial_year is not None:
            clauses = [where] if where else []
            clauses.append("financial_year = ?")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params = list(params) + [financial_year]
        query = f"SELECT * FROM fundamentals_ratios {where} ORDER BY ticker, financial_year"
        return self._conn.execute(query, params).fetchdf()

    def get_fundamental_quality_features(
        self, tickers: Iterable[str] | None = None, financial_year: int | None = None
    ) -> pd.DataFrame:
        where, params = self._build_filters(tickers, None, None, ticker_col="ticker")
        if financial_year is not None:
            clauses = [where] if where else []
            clauses.append("financial_year = ?")
            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
            params = list(params) + [financial_year]
        query = f"SELECT * FROM fundamental_quality_features {where} ORDER BY ticker, financial_year"
        return self._conn.execute(query, params).fetchdf()

    def tickers_with_screener_data(self) -> set[str]:
        """Tickers that have at least one screener-derived record."""
        rows = self._conn.execute(
            "SELECT DISTINCT ticker FROM fundamentals_company"
        ).fetchall()
        return {r[0] for r in rows}

    def upsert_fundamental_download_metadata(self, records: list[dict]) -> None:
        if not records:
            return
        df = pd.DataFrame(records)
        df["updated_at"] = pd.Timestamp.now()
        for col in ("company_name", "status", "financials_status", "ratios_status", "error"):
            if col not in df.columns:
                df[col] = None
        df = df.where(pd.notnull(df), None)
        self._upsert(df, "fundamental_download_metadata", ["ticker"])

    def get_fundamental_download_metadata(
        self, tickers: Iterable[str] | None = None
    ) -> pd.DataFrame:
        if tickers:
            placeholders = ", ".join(["?"] * len(list(tickers)))
            query = (
                f"SELECT * FROM fundamental_download_metadata "
                f"WHERE ticker IN ({placeholders}) ORDER BY ticker"
            )
            return self._conn.execute(query, list(tickers)).fetchdf()
        return self._conn.execute(
            "SELECT * FROM fundamental_download_metadata ORDER BY ticker"
        ).fetchdf()

    def delete_fundamental_data(self, tickers: list[str]) -> None:
        if not tickers:
            return
        placeholders = ", ".join(["?"] * len(tickers))
        for table in (
            "fundamentals_company",
            "fundamentals_income_statement_annual",
            "fundamentals_income_statement_quarterly",
            "fundamentals_balance_sheet",
            "fundamentals_cashflow",
            "fundamentals_dividends",
            "fundamentals_ratios",
            "fundamental_quality_features",
            "fundamental_download_metadata",
        ):
            self._conn.execute(
                f"DELETE FROM {table} WHERE ticker IN ({placeholders})", tickers
            )
        logger.info("Deleted fundamental data for %d tickers", len(tickers))

    # -- refresh / export ---------------------------------------------------
    def delete_ticker_data(self, tickers: list[str], include_features: bool = True) -> int:
        """Remove stored price (and optionally feature) rows for a refresh.

        Returns the number of price rows deleted.
        """
        before = self._conn.execute(
            f"SELECT count(*) FROM {settings.storage.prices_table} "
            f"WHERE ticker IN ({','.join(['?'] * len(tickers))})",
            tickers,
        ).fetchone()[0]
        self._conn.execute(
            f"DELETE FROM {settings.storage.prices_table} "
            f"WHERE ticker IN ({','.join(['?'] * len(tickers))})",
            tickers,
        )
        if include_features:
            self._conn.execute(
                f"DELETE FROM {settings.storage.feature_store_table} "
                f"WHERE ticker IN ({','.join(['?'] * len(tickers))})",
                tickers,
            )
        logger.info("Deleted %d price rows for %d tickers (features=%s)", before, len(tickers), include_features)
        return int(before)

    def export_ticker_csv(self, ticker: str, path: str) -> str:
        """Write a single ticker's OHLCV history to CSV."""
        ensure_dir(os.path.dirname(path))
        df = self.get_prices([ticker])
        df = df.sort_values("date")
        df.to_csv(path, index=False)
        return path

    def export_ticker_csv(self, ticker: str, path: str) -> str:
        """Write a single ticker's OHLCV history to CSV."""
        ensure_dir(os.path.dirname(path))
        df = self.get_prices([ticker])
        df = df.sort_values("date")
        df.to_csv(path, index=False)
        return path

    # -- statistics --------------------------------------------------------
    def _safe_count(self, sql: str) -> int:
        """Execute a COUNT query, tolerating a missing table, no result, or a
        corrupt/unexpected connection state (returns 0 instead of crashing)."""
        try:
            row = self._conn.execute(sql).fetchone()
            if row is None:
                return 0
            return int(str(row[0]).strip())
        except (Exception, ValueError):
            return 0

    def storage_statistics(self) -> dict:
        tbl = settings.storage.prices_table
        feat = settings.storage.feature_store_table
        price_rows = self._safe_count(f"SELECT count(*) FROM {tbl}")
        tickers = self._safe_count(f"SELECT count(DISTINCT ticker) FROM {tbl}")
        feat_rows = self._safe_count(f"SELECT count(*) FROM {feat}")
        feat_cols = len(self.feature_columns())
        size_bytes = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        return {
            "price_rows": int(price_rows),
            "stored_tickers": int(tickers),
            "feature_rows": int(feat_rows),
            "feature_columns": int(feat_cols),
            "db_size_bytes": int(size_bytes),
            "db_path": self.db_path,
        }

    # -- internals ---------------------------------------------------------
    def _build_filters(self, tickers, start, end, ticker_col: str = PriceColumns.TICKER):
        clauses: list[str] = []
        params: list = []
        if tickers:
            tickers = list(tickers)
            if len(tickers) == 1:
                clauses.append(f"{ticker_col} = ?")
                params.append(tickers[0])
            else:
                clauses.append(f"{ticker_col} IN ({','.join(['?'] * len(tickers))})")
                params.extend(tickers)
        if start is not None:
            clauses.append(f"{PriceColumns.DATE} >= ?")
            params.append(pd.Timestamp(start).date())
        if end is not None:
            clauses.append(f"{PriceColumns.DATE} <= ?")
            params.append(pd.Timestamp(end).date())
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return where, params

    def _upsert(self, df: pd.DataFrame, table: str, key_cols: list[str]) -> None:
        non_key = [c for c in df.columns if c not in key_cols]
        if not non_key:
            raise ValueError(
                f"Cannot upsert into {table}: no non-key columns in the frame "
                f"(keys={key_cols}, columns={list(df.columns)})."
            )
        staging = f"_stg_{table}"
        self._conn.execute(f"DROP TABLE IF EXISTS {staging}")
        self._conn.execute(f"CREATE TEMP TABLE {staging} AS SELECT * FROM df")
        # Quote column identifiers: feature columns may start with digits
        # (e.g. "3_MONTH"), which DuckDB requires to be double-quoted.
        quoted_non_key = [f'"{c}"' for c in non_key]
        key_sql = " AND ".join(f'{table}."{c}" = {staging}."{c}"' for c in key_cols)
        col_sql = ", ".join(quoted_non_key)
        set_sql = ", ".join(f'{table}."{c}" = {staging}."{c}"' for c in non_key)
        self._conn.execute(
            f"DELETE FROM {table} USING {staging} WHERE {key_sql}"
        )
        self._conn.execute(
            f"INSERT INTO {table} ({', '.join(key_cols)}, {col_sql}) "
            f"SELECT {', '.join(key_cols)}, {col_sql} FROM {staging}"
        )
        self._conn.execute(f"DROP TABLE IF EXISTS {staging}")
        logger.debug("Upserted %d rows into %s", len(df), table)
        # Fold the WAL back into the DB now, with retries, so it never grows
        # large enough to trigger DuckDB's fatal mid-transaction checkpoint.
        self.checkpoint()

    def _column_names(self, table: str) -> list[str]:
        rows = self._conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return [r[1] for r in rows]

    def export_parquet(self, table: str, path: str) -> None:
        ensure_dir(os.path.dirname(path))
        self._conn.execute(f"COPY (SELECT * FROM {table}) TO '{path}' (FORMAT PARQUET)")
