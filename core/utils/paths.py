"""Project path constants and small filesystem helpers."""

from __future__ import annotations

import os

# Absolute path to the repository / project root. Everything else is resolved
# relative to this so the application can be launched from any working dir.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Convenience constants for the conventional top-level directories.
APP_DIR = os.path.join(PROJECT_ROOT, "app")
CORE_DIR = os.path.join(PROJECT_ROOT, "core")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
STORAGE_DIR = os.path.join(PROJECT_ROOT, "storage")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if it does not exist and return it."""
    os.makedirs(path, exist_ok=True)
    return path
