"""Tests for the Apify providers configuration (Screener single source of truth).

Validates dotenv loading / env interpolation and that the configured Screener
actor id resolves to a concrete value (no unresolved ``${...}`` placeholders).
The full Screener ingest + 16-factor Quality engine pipeline is covered by
``test_screener_integration.py``.
"""

from __future__ import annotations

import pytest

from core.config.providers_config import (
    _load_dotenv,
    get_provider_config,
    is_provider_enabled,
    providers_config,
)


def test_providers_config_loaded():
    apify = providers_config.get("apify", {})
    assert apify.get("enabled") is True
    actors = (apify.get("actors") or {})
    # The Screener actor is the single source of truth for fundamentals.
    scr_id = actors["screener"]["id"]
    _PLACEHOLDER = "${"
    assert scr_id and _PLACEHOLDER not in scr_id, f"screener actor id not set: {scr_id!r}"
    assert int(apify["batch_size"]) == 25
    assert int(apify["max_concurrency"]) == 3


def test_env_interpolation(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "SECRET123")
    # Reload the .env loader + config in isolation.
    import importlib

    import core.config.providers_config as pc

    pc._ENV_LOADED = False
    cfg = pc._load()
    assert cfg["apify"]["api_token"] == "SECRET123"


def test_is_provider_enabled(monkeypatch):
    monkeypatch.setenv("APIFY_API_TOKEN", "x")
    assert is_provider_enabled("apify") is True


def test_get_provider_config_returns_screener_id():
    cfg = get_provider_config("apify")
    assert cfg["actors"]["screener"]["id"]
