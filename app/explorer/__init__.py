"""Dataset Explorer package.

Exports the base contract and registry for dataset sources.
"""

from .base import (
    DATASET_SOURCES,
    DatasetSource,
    HealthIssue,
    Severity,
    get_dataset_source,
)

# Import source modules to trigger registration
from . import market_data_source  # noqa: F401
from . import fundamental_source  # noqa: F401

__all__ = [
    "DATASET_SOURCES",
    "DatasetSource",
    "HealthIssue",
    "Severity",
    "get_dataset_source",
]