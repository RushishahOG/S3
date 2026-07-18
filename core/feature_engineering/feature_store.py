"""Feature Store: persistence layer for engineered features.

The store uses a wide-format table in DuckDB with dynamic column addition,
mirroring the schema used by the main storage manager.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import duckdb
import pandas as pd

from core.config import settings


@dataclass
class FeatureSpec:
    """Metadata for a single feature column."""

    key: str
    description: str
    factor_category: str
    frequency: str
    lookback_months: int
    formula: str


class FeatureStore:
    """
    Persistent wide-format feature store backed by DuckDB.

    The table has a primary key of (ticker, date) and dynamically adds
    feature columns as new factor families are registered.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or settings.storage.database_abs_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_store (
                ticker VARCHAR NOT NULL,
                date DATE NOT NULL,
                PRIMARY KEY (ticker, date)
            );
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_metadata (
                feature_key VARCHAR NOT NULL,
                factor_category VARCHAR NOT NULL,
                description VARCHAR,
                frequency VARCHAR,
                lookback_months INTEGER,
                formula VARCHAR,
                created_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (feature_key)
            );
            """
        )

    def upsert_features(self, df: pd.DataFrame) -> int:
        """
        Upsert a long-format feature frame into the wide feature store.

        The input frame must have columns: [ticker, date, feature_1, feature_2, ...]
        (case-insensitive — Ticker/Date are normalised automatically).
        """
        if df is None or df.empty:
            return 0

        df = df.copy()
        # Normalise column names to lowercase
        rename = {}
        for c in df.columns:
            if c.lower() in ("ticker", "date"):
                rename[c] = c.lower()
        if rename:
            df = df.rename(columns=rename)

        cols = list(df.columns)
        key_cols = ["ticker", "date"]
        feat_cols = [c for c in cols if c not in key_cols]

        if not feat_cols:
            return 0

        df = df[key_cols + feat_cols].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # Ensure feature columns exist in the table
        for col in feat_cols:
            self._ensure_column(col)

        # Upsert via staging table
        staging = "_stg_feat"
        self._conn.execute(f"DROP TABLE IF EXISTS {staging}")
        self._conn.execute(f"CREATE TEMP TABLE {staging} AS SELECT * FROM df")

        key_sql = f"feature_store.ticker = {staging}.ticker AND feature_store.date = {staging}.date"
        col_sql = ", ".join(feat_cols)
        set_sql = ", ".join(f"feature_store.{c} = {staging}.{c}" for c in feat_cols)

        self._conn.execute(
            f"DELETE FROM feature_store USING {staging} WHERE {key_sql}"
        )
        self._conn.execute(
            f"INSERT INTO feature_store (ticker, date, {col_sql}) "
            f"SELECT ticker, date, {col_sql} FROM {staging}"
        )
        self._conn.execute(f"DROP TABLE IF EXISTS {staging}")

        return len(df)

    def _ensure_column(self, col: str) -> None:
        existing = self.get_feature_columns()
        if col not in existing:
            self._conn.execute(f'ALTER TABLE feature_store ADD COLUMN "{col}" DOUBLE')

    def get_features(
        self,
        tickers: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Query features with optional filters."""
        where, params = [], []
        if tickers:
            placeholders = ", ".join(["?"] * len(tickers))
            where.append(f"ticker IN ({placeholders})")
            params.extend(tickers)
        if start:
            where.append("date >= ?")
            params.append(start)
        if end:
            where.append("date <= ?")
            params.append(end)

        select_cols = ["ticker", "date"]
        if columns:
            select_cols.extend(columns)
        else:
            select_cols = ["*"]

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""
        select_sql = ", ".join(
            f'"{c}"' if c not in ("ticker", "date", "*") else c
            for c in select_cols
        )

        query = f"SELECT {select_sql} FROM feature_store {where_sql} ORDER BY ticker, date"
        return self._conn.execute(query, params).fetchdf()

    def get_feature_columns(self) -> list[str]:
        """Return all feature column names (excluding ticker/date)."""
        cols = self._conn.execute("PRAGMA table_info('feature_store')").fetchall()
        return [c[1] for c in cols if c[1] not in ("ticker", "date")]

    def register_feature_metadata(self, specs: list) -> None:
        """Register metadata for feature columns."""
        if not specs:
            return
        rows = []
        for s in specs:
            rows.append({
                "feature_key": s.key,
                "factor_category": s.factor_category,
                "description": s.description,
                "frequency": s.frequency,
                "lookback_months": s.lookback_months,
                "formula": s.formula,
            })
        df = pd.DataFrame(rows)
        self._conn.execute("DELETE FROM feature_metadata WHERE feature_key IN (SELECT feature_key FROM df)")
        self._conn.execute(
            "INSERT INTO feature_metadata (feature_key, factor_category, description, frequency, lookback_months, formula) "
            "SELECT feature_key, factor_category, description, frequency, lookback_months, formula FROM df"
        )

    def get_metadata(self) -> pd.DataFrame:
        return self._conn.execute(
            "SELECT * FROM feature_metadata ORDER BY factor_category, feature_key"
        ).fetchdf()

    def export_parquet(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn.execute(f"COPY (SELECT * FROM feature_store) TO '{path}' (FORMAT PARQUET)")

    def close(self) -> None:
        self._conn.close()