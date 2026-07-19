"""Shared application services.

Centralises singleton creation of the core engines so every Streamlit page
shares the same storage connection and managers.

We use lazy module-level singletons (not ``st.cache_resource``) so that a fresh
server start always builds objects from the *current* class definitions. This
avoids the stale-instance problem that ``cache_resource`` can cause when an
underlying class is edited but the cached factory function's source is
unchanged.
"""

from __future__ import annotations

from core.data.market_data_manager import MarketDataManager
from core.data.storage.storage_manager import StorageManager
from core.data.universe.universe_manager import UniverseManager

_storage: StorageManager | None = None
_mdm: MarketDataManager | None = None
_um: UniverseManager | None = None


def get_storage() -> StorageManager:
    global _storage
    if _storage is None:
        # The live app is read-only; ingestion/upserts run via separate scripts.
        # A read-only connection lets the backtest worker open its own concurrent
        # read connection without blocking on the app's file lock.
        _storage = StorageManager(read_only=True)
    return _storage


def get_market_data_manager() -> MarketDataManager:
    global _mdm
    if _mdm is None:
        _mdm = MarketDataManager(storage=get_storage())
    return _mdm


def get_universe_manager() -> UniverseManager:
    global _um
    if _um is None:
        _um = UniverseManager()
    return _um
