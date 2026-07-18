"""Provider configuration loader (external data vendors).

Reads ``config/providers.yaml`` and resolves ``${ENV_VAR}`` references from the
process environment. The project-root ``.env`` file is loaded into the
environment automatically so secrets (e.g. ``APIFY_API_TOKEN``) are NEVER
hardcoded in source code.

Access pattern::

    from core.config.providers_config import providers_config
    cfg = providers_config["apify"]            # dict-like
    token = providers_config["apify"]["api_token"]
    enabled = providers_config.get("apify", {}).get("enabled", False)

The structure is a plain (nested) ``dict`` so it stays trivially serialisable
and free of framework dependencies.
"""

from __future__ import annotations

import os
import re

import yaml

from core.utils.logging_config import get_logger
from core.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

_ENV_LOADED = False
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _load_dotenv() -> None:
    """Load project-root ``.env`` into ``os.environ`` (idempotent, no overwrite)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    dotenv_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _resolve_env(value: object) -> object:
    """Recursively replace ``${VAR}`` placeholders with environment values."""
    if isinstance(value, str):
        def _sub(match: re.Match) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return _ENV_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


def _load() -> dict:
    _load_dotenv()
    path = os.path.join(PROJECT_ROOT, "config", "providers.yaml")
    if not os.path.exists(path):
        logger.warning("providers.yaml not found at %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    resolved = _resolve_env(raw)
    return resolved.get("providers", {})


#: Module-level singleton. ``providers_config["apify"]["api_token"]`` etc.
providers_config: dict = _load()


def get_provider_config(name: str) -> dict:
    """Return the configuration dict for a named provider (empty if absent)."""
    return providers_config.get(name, {}) or {}


def is_provider_enabled(name: str) -> bool:
    return bool(get_provider_config(name).get("enabled", False))
