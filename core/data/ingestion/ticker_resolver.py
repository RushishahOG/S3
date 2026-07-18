"""TickerResolver.

Converts a constituent's base symbol into a provider-specific ticker. The
default rule appends the NSE suffix ``.NS``; renamed companies or other
exchanges are handled through an external mapping file (YAML) so the downloader
never needs code changes when a symbol changes or a new provider/exchange is
added.

Mapping file shape (``config/ticker_mapping.yaml``)::

    overrides:
      "OLDNAME": "NEW.NS"
      "123XYZ": "ABC.BO"   # e.g. BSE alternative
"""

from __future__ import annotations

import os
from typing import Any

import yaml

from core.config.settings import settings
from core.utils.logging_config import get_logger
from core.utils.paths import ensure_dir

logger = get_logger(__name__)

DEFAULT_SUFFIX = ".NS"


class TickerResolver:
    def __init__(self, mapping_path: str | None = None) -> None:
        self.mapping_path = mapping_path or settings.ingestion.ticker_mapping_abs_path
        self._overrides: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.mapping_path):
            ensure_dir(os.path.dirname(self.mapping_path))
            # Create an empty mapping file on first run for easy editing.
            with open(self.mapping_path, "w", encoding="utf-8") as fh:
                yaml.safe_dump({"overrides": {}}, fh)
            logger.info("Created empty ticker mapping at %s", self.mapping_path)
            return
        with open(self.mapping_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        overrides = data.get("overrides", {}) or {}
        self._overrides = {str(k).upper(): str(v) for k, v in overrides.items()}
        logger.info("Loaded %d ticker overrides", len(self._overrides))

    def resolve(self, base_symbol: str) -> str:
        """Return the provider ticker for ``base_symbol``.

        If the symbol already carries a recognised exchange suffix it is
        returned unchanged (avoiding ``360ONE.NS`` -> ``360ONE.NS.NS``).
        """
        raw = str(base_symbol).strip().upper()
        if raw in self._overrides:
            return self._overrides[raw]
        if raw.endswith((".NS", ".BO", ".BSE", ".NSE")):
            return raw
        return f"{raw}{DEFAULT_SUFFIX}"

    def bare(self, base_symbol: str) -> str:
        """Return the symbol with any exchange suffix stripped.

        Used for fundamental actors (e.g. ``360ONE.NS`` -> ``360ONE``) that
        resolve the exchange themselves.
        """
        raw = str(base_symbol).strip().upper()
        if raw in self._overrides:
            raw = self._overrides[raw]
        for suf in (".NS", ".BO", ".BSE", ".NSE"):
            if raw.endswith(suf):
                return raw[: -len(suf)]
        return raw

    def reload(self) -> None:
        self._load()

    def as_dict(self) -> dict[str, Any]:
        return dict(self._overrides)
