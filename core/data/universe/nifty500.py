"""NIFTY 500 universe provider (the only universe implemented in Version 1).

Reads the official constituent file supplied by the competition (so ticker
lists are never hardcoded) and resolves each base symbol to its Yahoo Finance
ticker via the :class:`TickerResolver`.
"""

from __future__ import annotations

import os

from core.config import settings
from core.data.ingestion.constituents import Constituent, load_constituents
from core.data.ingestion.ticker_resolver import TickerResolver
from core.data.universe.base_universe import BaseUniverseProvider


class NIFTY500Universe(BaseUniverseProvider):
    name = "nifty500"
    description = "NSE NIFTY 500 index - the single Version 1 investment universe"
    benchmark_ticker = "NIFTY_500"

    def __init__(
        self,
        constituents_path: str | None = None,
        resolver: TickerResolver | None = None,
    ) -> None:
        self.constituents_path = constituents_path or settings.universe.constituents_abs_path
        self.resolver = resolver or TickerResolver()

    def get_constituents(self) -> list[Constituent]:
        """Return the official constituent records (company name + base symbol)."""
        if not os.path.exists(self.constituents_path):
            raise FileNotFoundError(
                f"NIFTY 500 constituent file not found at {self.constituents_path}."
            )
        return load_constituents(self.constituents_path)

    def get_tickers(self) -> list[str]:
        """Return resolved Yahoo Finance tickers for all constituents."""
        return [self.resolver.resolve(c.base_symbol) for c in self.get_constituents()]
