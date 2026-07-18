"""Thin Apify API client used by the fundamental providers.

Talks to the Apify "run synchronously and fetch dataset items" endpoint, which
is the simplest way to invoke an actor and stream its output as JSON without
polling a runId:

    POST https://api.apify.com/v2/acts/{actorId}/run-sync-get-dataset-items?token={token}

The client is intentionally tiny and vendor-agnostic: it knows nothing about
financial statements vs ratios. Each provider supplies the actor id and the
request payload; this module only performs the HTTP call, retries transient
failures, and optionally caches the raw response for later schema inspection.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from core.config.providers_config import get_provider_config
from core.utils.decorators import retry
from core.utils.logging_config import get_logger
from core.utils.paths import ensure_dir

logger = get_logger(__name__)

APIFY_API_BASE = "https://api.apify.com/v2/acts"


def _cache_dir() -> str:
    from core.config import settings

    return os.path.join(settings.storage.parquet_abs_dir, "..", "raw", "apify")


def _maybe_cache_raw(actor_id: str, payload: dict, items: Any) -> None:
    """Cache the raw actor response to disk when enabled in config."""
    cfg = get_provider_config("apify")
    if not cfg.get("cache_raw_response", False):
        return
    try:
        d = ensure_dir(_cache_dir())
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = actor_id.replace("/", "_")
        path = os.path.join(d, f"{safe}_{ts}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"payload": payload, "items": items}, fh, default=str)
        logger.debug("Cached raw Apify response to %s", path)
    except Exception as exc:  # pragma: no cover - best-effort cache
        logger.warning("Failed to cache raw Apify response: %s", exc)


@retry(max_attempts=3, backoff_seconds=5, max_sleep_per_attempt=20)
def run_actor(
    actor_id: str,
    payload: dict[str, Any],
    *,
    token: str,
    timeout_seconds: int = 120,
) -> list[dict]:
    """Run an Apify actor synchronously and return its dataset items.

    Parameters
    ----------
    actor_id : str
        The Apify actor id (from ``config/providers.yaml``).
    payload : dict
        The actor's input JSON (e.g. tickers / date range for financials).
    token : str
        Apify API token (resolved from ``${APIFY_API_TOKEN}`` at runtime).
    timeout_seconds : int
        HTTP timeout for the synchronous run.

    Returns
    -------
    list[dict]
        The actor's output dataset items (may be empty).
    """
    if not token:
        raise RuntimeError(
            "Apify API token is empty. Set APIFY_API_TOKEN in the project .env file."
        )

    url = f"{APIFY_API_BASE}/{actor_id}/run-sync-get-dataset-items"
    params = {"token": token}
    logger.info("Initializing Apify client for actor %s", actor_id)
    logger.info("Downloading financial/ratio data via Apify actor %s", actor_id)

    resp = requests.post(url, params=params, json=payload, timeout=timeout_seconds)
    if resp.status_code >= 400:
        # Surface the actor's error detail when present.
        detail = resp.text[:500]
        raise RuntimeError(f"Apify actor {actor_id} returned HTTP {resp.status_code}: {detail}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise RuntimeError(f"Apify actor {actor_id} returned non-JSON response") from exc

    # The run-sync-get-dataset-items endpoint returns either a bare list or an
    # envelope with an "items" / "data" key. Normalise both shapes.
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("items") or data.get("data") or []
    else:
        items = []

    _maybe_cache_raw(actor_id, payload, items)
    logger.info("Apify actor %s returned %d items", actor_id, len(items))
    return items
