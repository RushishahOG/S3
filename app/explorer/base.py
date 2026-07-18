"""Dataset Explorer abstraction layer.

This package defines a small, provider-agnostic contract that the
``dataset_explorer`` Streamlit page renders. Every *kind* of local dataset
(market OHLCV, financial statements, ETFs, benchmark indices, macro series, ...)
is exposed through a :class:`DatasetSource` implementation and registered in
:data:`DATASET_SOURCES`.

The page never talks to DuckDB or any vendor directly; it only consumes the
uniform interface below. Adding a future dataset therefore means writing one
new ``DatasetSource`` subclass and registering it - no changes to the page or
to the storage engine are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Severity(str, Enum):
    """Severity of a data-health issue."""

    ERROR = "error"
    WARNING = "warning"


@dataclass
class HealthIssue:
    """A single data-quality / completeness finding for a security."""

    ticker: str
    issue: str
    severity: Severity
    detail: str

    @property
    def is_error(self) -> bool:
        return self.severity == Severity.ERROR


class DatasetSource(ABC):
    """Uniform contract for any inspectable local dataset."""

    #: Stable registry key (e.g. ``"market_data"``).
    key: str = "base"
    #: Human label shown in the UI selector.
    label: str = "Base Dataset"
    #: Short description of the dataset family.
    description: str = ""

    # -- registry ---------------------------------------------------------
    @classmethod
    def register(cls, source_cls: type["DatasetSource"]) -> type["DatasetSource"]:
        """Register a ``DatasetSource`` *subclass* (not an instance).

        Registration is lazy: no storage/network connection is opened until
        :func:`get_dataset_source` is called, so importing the explorer package
        is side-effect free.
        """
        DATASET_SOURCES[source_cls.key] = source_cls
        return source_cls

    # -- discovery / catalogue --------------------------------------------
    @abstractmethod
    def security_summary(self) -> pd.DataFrame:
        """Return one row per security with the catalogue columns.

        Expected columns: ``ticker``, ``company_name``, ``records``,
        ``earliest``, ``latest``, ``availability_pct``, ``last_updated``,
        ``status``.
        """

    # -- inspection -------------------------------------------------------
    @abstractmethod
    def fetch_dataset(self, ticker: str) -> pd.DataFrame:
        """Return the complete record set for ``ticker`` (one row per period)."""

    @abstractmethod
    def display_columns(self) -> list[str]:
        """Ordered column names used when rendering the dataset table."""

    @abstractmethod
    def dataset_statistics(self, df: pd.DataFrame, ticker: str) -> dict:
        """Return summary statistics for a single fetched dataset."""

    # -- storage / health -------------------------------------------------
    @abstractmethod
    def storage_statistics(self) -> dict:
        """Return storage-level aggregates for the dataset family."""

    @abstractmethod
    def health_issues(self, tickers: list[str]) -> list[HealthIssue]:
        """Return data-quality / completeness findings for ``tickers``."""

    # -- export -----------------------------------------------------------
    def export_csv(self, ticker: str) -> pd.DataFrame:
        """Return the dataset frame to be serialised by the caller."""
        return self.fetch_dataset(ticker)


#: Registered dataset sources, keyed by ``DatasetSource.key`` -> subclass.
DATASET_SOURCES: dict[str, type["DatasetSource"]] = {}


def get_dataset_source(key: str, storage=None) -> "DatasetSource":
    """Return a (cached) instance of the registered dataset source.

    If ``storage`` is supplied it is injected into the source (useful for
    reusing the application-wide storage singleton); otherwise the source
    lazily opens its own connection on first access.
    """
    if key not in DATASET_SOURCES:
        raise KeyError(
            f"Unknown dataset source {key!r}. Available: {list(DATASET_SOURCES)}"
        )
    if storage is not None:
        return DATASET_SOURCES[key](storage=storage)
    if key not in _SOURCE_INSTANCES:
        _SOURCE_INSTANCES[key] = DATASET_SOURCES[key]()
    return _SOURCE_INSTANCES[key]


#: Cache of materialised source instances (created on first access).
_SOURCE_INSTANCES: dict[str, "DatasetSource"] = {}
