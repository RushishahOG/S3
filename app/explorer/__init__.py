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

__all__ = [
    "DATASET_SOURCES",
    "DatasetSource",
    "HealthIssue",
    "Severity",
    "get_dataset_source",
]