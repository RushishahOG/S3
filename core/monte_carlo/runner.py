"""Background runner for Monte Carlo simulations.

Runs the (potentially long) simulation in a daemon thread so the Streamlit UI
can poll progress and offer cancellation without freezing. Mirrors the pattern
used by the backtest executor in :mod:`app.pages.backtest.state`.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

from core.monte_carlo.engine import CancelSimulation, run_simulation
from core.monte_carlo.types import MCInput, SimulationConfig

_RUNNERS: dict[str, dict] = {}
_RUNNERS_LOCK = threading.Lock()


class MonteCarloRunner:
    """Executes a simulation off the main thread and reports progress."""

    def __init__(self, config: SimulationConfig, inp: MCInput) -> None:
        self.config = config
        self.inp = inp
        self.run_id = f"mc_{uuid.uuid4().hex[:8]}"
        self._bucket: dict[str, Any] = {
            "done": False,
            "error": None,
            "result": None,
            "completed": 0,
            "total": config.n_simulations,
            "start_time": None,
            "running": True,
            "cancel": False,
            "speed": 0.0,
            "eta": None,
        }
        with _RUNNERS_LOCK:
            _RUNNERS[self.run_id] = self._bucket

    def start(self) -> str:
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return self.run_id

    def _progress(self, completed: int, total: int) -> None:
        self._bucket["completed"] = completed
        self._bucket["total"] = total
        start = self._bucket["start_time"]
        if start is not None:
            elapsed = time.time() - start
            if elapsed > 0:
                speed = completed / elapsed
                self._bucket["speed"] = speed
                remaining = total - completed
                self._bucket["eta"] = remaining / speed if speed > 0 else None

    def _run(self) -> None:
        try:
            self._bucket["start_time"] = time.time()
            result = run_simulation(
                self.config,
                self.inp,
                progress_cb=self._progress,
                cancel_cb=lambda: self._bucket["cancel"],
            )
            if self._bucket["cancel"]:
                self._bucket["result"] = None
            else:
                self._bucket["result"] = result
        except CancelSimulation:
            self._bucket["result"] = None
        except Exception as exc:  # pragma: no cover - surface to UI
            import traceback

            self._bucket["error"] = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        finally:
            self._bucket["done"] = True
            self._bucket["running"] = False

    def request_cancel(self) -> None:
        self._bucket["cancel"] = True


def get_runner_bucket(run_id: str) -> dict | None:
    with _RUNNERS_LOCK:
        return _RUNNERS.get(run_id)


def clear_runner(run_id: str) -> None:
    with _RUNNERS_LOCK:
        _RUNNERS.pop(run_id, None)
