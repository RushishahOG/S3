"""Universe abstraction.

Defines the generic ``Universe`` value object and the ``BaseUniverseProvider``
contract. Version 1 ships exactly one provider - :class:`NIFTY500Universe` -
but the manager is built to register additional providers (NIFTY 50, Midcap
150, custom CSVs, ETFs, multi-asset) in later milestones without any change to
the rest of the platform.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Universe:
    """Immutable collection of tickers the platform operates on."""

    name: str
    tickers: list[str]
    description: str = ""
    source: str = "provider"

    def __len__(self) -> int:
        return len(self.tickers)

    def __iter__(self) -> Iterable[str]:
        return iter(self.tickers)


class BaseUniverseProvider(ABC):
    """Contract a universe provider must satisfy."""

    #: Unique registry key, e.g. ``"nifty500"``.
    name: str = "base"
    description: str = ""
    #: Benchmark ticker used for Beta / market-relative features.
    benchmark_ticker: str = "NIFTY_500"

    @abstractmethod
    def get_tickers(self) -> list[str]:
        """Return the ordered list of constituent tickers."""

    def get_universe(self) -> Universe:
        return Universe(self.name, self.get_tickers(), self.description, "provider")

    def is_available(self) -> bool:
        return True

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"

    @staticmethod
    def _read_tickers_csv(path: str) -> list[str]:
        df = pd.read_csv(path)
        if "ticker" not in df.columns:
            df = df.rename(columns={df.columns[0]: "ticker"})
        tickers = df["ticker"].dropna().astype(str).str.strip()
        tickers = tickers[tickers.str.lower() != "ticker"]
        return tickers.tolist()
