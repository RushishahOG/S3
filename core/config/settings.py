"""Centralised configuration loading for the Smart Beta Research Platform.

Configuration is sourced from ``config/settings.yaml`` and exposed through
dataclasses so the rest of the codebase can rely on typed attributes instead
of magic dict keys. No module should hardcode runtime constants.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml

from core.utils.paths import PROJECT_ROOT


@dataclass(frozen=True)
class StorageConfig:
    engine: str
    database_path: str
    parquet_dir: str
    prices_table: str
    feature_store_table: str
    feature_metadata_table: str

    @property
    def database_abs_path(self) -> str:
        return os.path.join(PROJECT_ROOT, self.database_path)

    @property
    def parquet_abs_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, self.parquet_dir)


@dataclass(frozen=True)
class ProvidersConfig:
    default: str
    available: list[str]
    #: If a vendor omits the adjusted-close series, fall back to the raw
    #: ``close`` price so downstream return-based factors still have a series.
    #: Set to False to store NULL instead and surface an explicit warning.
    adj_close_fallback: bool = True


@dataclass(frozen=True)
class UniverseConfig:
    config_dir: str
    default: str
    benchmark: str
    constituents_file: str

    @property
    def config_abs_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, self.config_dir)

    @property
    def constituents_abs_path(self) -> str:
        return os.path.join(PROJECT_ROOT, self.constituents_file)


@dataclass(frozen=True)
class IngestionConfig:
    ticker_mapping_file: str
    retries: int
    retry_backoff_seconds: float
    min_delay_seconds: float
    max_delay_seconds: float
    export_dir: str

    @property
    def ticker_mapping_abs_path(self) -> str:
        return os.path.join(PROJECT_ROOT, self.ticker_mapping_file)

    @property
    def export_abs_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, self.export_dir)


@dataclass(frozen=True)
class FeaturesConfig:
    trading_days_per_year: int
    default_history_years: int
    min_observations: int


@dataclass(frozen=True)
class MomentumConfig:
    horizons_months: list[int]
    lag_months: int
    scaled: bool


@dataclass(frozen=True)
class LowVolatilityConfig:
    vol_windows_months: list[int]
    semi_dev_window_months: int
    beta_windows_months: list[int]


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    log_format: str
    to_file: bool
    log_dir: str

    @property
    def log_abs_dir(self) -> str:
        return os.path.join(PROJECT_ROOT, self.log_dir)


@dataclass(frozen=True)
class BacktestConfig:
    min_date: str
    max_date: str
    default_start: str


@dataclass(frozen=True)
class AppConfig:
    name: str
    version: str
    environment: str


@dataclass(frozen=True)
class Settings:
    app: AppConfig
    storage: StorageConfig
    providers: ProvidersConfig
    universe: UniverseConfig
    features: FeaturesConfig
    momentum: MomentumConfig
    low_volatility: LowVolatilityConfig
    backtest: BacktestConfig
    ingestion: IngestionConfig
    logging: LoggingConfig


def _as_dataclass(cls, data: dict[str, Any]) -> Any:
    known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in known})


def load_settings(config_path: str | None = None) -> Settings:
    """Load YAML settings and project into typed dataclasses."""
    path = config_path or os.path.join(PROJECT_ROOT, "config", "settings.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    return Settings(
        app=_as_dataclass(AppConfig, raw["app"]),
        storage=_as_dataclass(StorageConfig, raw["storage"]),
        providers=_as_dataclass(ProvidersConfig, raw["providers"]),
        universe=_as_dataclass(UniverseConfig, raw["universe"]),
        features=_as_dataclass(FeaturesConfig, raw["features"]),
        momentum=_as_dataclass(MomentumConfig, raw["momentum"]),
        low_volatility=_as_dataclass(LowVolatilityConfig, raw["low_volatility"]),
        backtest=_as_dataclass(BacktestConfig, raw["backtest"]),
        ingestion=_as_dataclass(IngestionConfig, raw["ingestion"]),
        logging=_as_dataclass(LoggingConfig, raw["logging"]),
    )


# Module-level singleton. Importing ``settings`` anywhere yields the same
# resolved configuration object.
settings: Settings = load_settings()
