"""Universe management (Application Service).

A generic, extensible manager that owns a registry of
:class:`BaseUniverseProvider` instances. Version 1 registers a single provider
- :class:`NIFTY500Universe` - so the entire application operates exclusively on
the NIFTY 500 constituents (plus the NIFTY 50 benchmark for Beta).

Multi-universe selection, custom CSV upload and index switching are intentionally
NOT exposed in V1; the abstraction remains so future universes plug in here
without touching downstream code.
"""

from __future__ import annotations

from core.config import settings
from core.data.universe.base_universe import BaseUniverseProvider, Universe
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

# Back-compat re-export so existing imports keep working.
__all__ = ["UniverseManager", "Universe", "BaseUniverseProvider"]


class UniverseManager:
    def __init__(self) -> None:
        self._providers: dict[str, BaseUniverseProvider] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        # Only the NIFTY 500 provider is registered in Version 1.
        from core.data.universe.nifty500 import NIFTY500Universe

        self.register(NIFTY500Universe())

    # -- registry ----------------------------------------------------------
    def register(self, provider: BaseUniverseProvider) -> None:
        self._providers[provider.name] = provider
        logger.debug("Registered universe provider %s", provider.name)

    def available_universes(self) -> list[str]:
        return sorted(self._providers)

    # -- access ------------------------------------------------------------
    def get_universe(self, name: str | None = None) -> Universe:
        name = name or settings.universe.default
        if name not in self._providers:
            raise KeyError(
                f"Universe provider {name!r} is not registered. "
                f"Available: {self.available_universes()}"
            )
        return self._providers[name].get_universe()

    def get_constituents(self, name: str | None = None):
        """Return the registered provider's constituent records (company + symbol)."""
        name = name or settings.universe.default
        if name not in self._providers:
            raise KeyError(
                f"Universe provider {name!r} is not registered. "
                f"Available: {self.available_universes()}"
            )
        return self._providers[name].get_constituents()

    def get_provider(self, name: str | None = None) -> BaseUniverseProvider:
        """Return the raw provider (e.g. to access universe-specific metadata)."""
        name = name or settings.universe.default
        if name not in self._providers:
            raise KeyError(
                f"Universe provider {name!r} is not registered. "
                f"Available: {self.available_universes()}"
            )
        return self._providers[name]

    def default_universe(self) -> Universe:
        """The universe the whole V1 platform operates on (NIFTY 500)."""
        return self.get_universe(settings.universe.default)

    @property
    def benchmark_ticker(self) -> str:
        return settings.universe.benchmark
