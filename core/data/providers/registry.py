"""Provider registry.

A tiny plugin registry so the Market Data Manager can resolve a provider by
configuration key. New providers self-register via :func:`register`.
"""

from __future__ import annotations

from core.data.providers.base_provider import BaseDataProvider, BaseFundamentalProvider
from core.utils.logging_config import get_logger

logger = get_logger(__name__)

_REGISTRY: dict[str, type[BaseDataProvider]] = {}


def register(key: str, provider_cls: type[BaseDataProvider]) -> None:
    """Register a provider class under ``key``."""
    if key in _REGISTRY and _REGISTRY[key] is not provider_cls:
        logger.warning("Overwriting provider registration for %s", key)
    _REGISTRY[key] = provider_cls
    logger.debug("Registered data provider %s -> %s", key, provider_cls.__name__)


def get_provider(key: str, *args, **kwargs) -> BaseDataProvider:
    """Instantiate a registered provider by key."""
    if key not in _REGISTRY:
        raise KeyError(
            f"Data provider {key!r} is not registered. "
            f"Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](*args, **kwargs)


def available_providers() -> list[str]:
    """Registered *price* providers (BaseDataProvider subclasses)."""
    return sorted(
        k for k, v in _REGISTRY.items() if issubclass(v, BaseDataProvider)
    )


def available_fundamental_providers() -> list[str]:
    """Registered *fundamental* providers (BaseFundamentalProvider subclasses)."""
    return sorted(
        k for k, v in _REGISTRY.items() if issubclass(v, BaseFundamentalProvider)
    )


def is_registered(key: str) -> bool:
    return key in _REGISTRY


# --- Built-in registration -------------------------------------------------
# Importing the concrete provider modules registers them automatically.
from core.data.providers.yahoo_finance import YahooFinanceProvider  # noqa: E402
from core.data.providers.apify_screener_provider import (  # noqa: E402, F401
    ApifyScreenerProvider,
)

register(YahooFinanceProvider.name, YahooFinanceProvider)
register(ApifyScreenerProvider.name, ApifyScreenerProvider)
