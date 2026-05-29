"""
Strategy router for v2 shard extraction.

The user picks one of ``"auto" | "exact" | "heuristic"``:

* ``"exact"`` -- :class:`LrslibExtractor` only; raises if it fails.
* ``"heuristic"`` -- :class:`RayShootingExtractor` only.
* ``"auto"`` -- try the exact strategy with a wall-clock deadline; if
  it exceeds ``timeout_seconds`` or raises, log a warning and fall
  back to the heuristic.

The timeout is enforced by a **cooperative deadline** passed into the
exact extractor: cell enumeration checks ``time.time()`` periodically
and raises :class:`ExtractionTimeout` once the deadline passes, so the
call returns promptly and control flows to the heuristic.

This deliberately avoids running the exact extractor in a worker
thread.  Python threads cannot be cancelled, and
``ThreadPoolExecutor.__exit__`` blocks on ``shutdown(wait=True)`` until
the (potentially 20-minute) exact run finishes — so a thread-based
timeout would *stall* the fallback rather than enable it.  Running
synchronously with a cooperative deadline keeps the fallback instant.
Per-call ``lrs`` subprocess timeouts (``LrslibExtractor.per_call_timeout``)
bound the only other long operation.
"""

from __future__ import annotations

import time
from typing import List, Literal, Optional

from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.logger import Logger
from .base import BaseExtractor, ShardMapping
from .cells import ExtractionTimeout
from .lrs_extractor import LrslibExtractor
from .ray_extractor import RayShootingExtractor


Strategy = Literal["auto", "exact", "heuristic"]


class ExtractionManager:
    """
    Routes between exact and heuristic extractors.

    :param strategy: ``"auto"``, ``"exact"`` or ``"heuristic"``.
    :param timeout_seconds: Wall-clock cap on the exact strategy in
        ``"auto"`` mode.  Ignored otherwise.
    :param exact: Optional pre-built :class:`LrslibExtractor`.  Built
        lazily on first use in ``"auto"``/``"exact"`` modes if omitted.
    :param heuristic: Optional pre-built :class:`RayShootingExtractor`.
    :param exact_unbounded_check: Forwarded to a lazily-built
        :class:`LrslibExtractor` (``"lp"`` or ``"lrs"``).
    :raises ValueError: If ``strategy`` is unknown.
    """

    def __init__(
        self,
        strategy: Strategy = "auto",
        *,
        timeout_seconds: float = 3600.0,
        exact: Optional[LrslibExtractor] = None,
        heuristic: Optional[RayShootingExtractor] = None,
        exact_unbounded_check: str = "lp",
        exact_num_workers: int = 1,
    ):
        if strategy not in ("auto", "exact", "heuristic"):
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected "
                "'auto', 'exact' or 'heuristic'"
            )
        self.strategy = strategy
        self._exact_unbounded_check = exact_unbounded_check
        self._exact_num_workers = exact_num_workers
        self.timeout_seconds = timeout_seconds
        self._exact = exact
        self._heuristic = heuristic

    def extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        """
        Run the configured strategy on the arrangement.

        :raises RuntimeError: If ``"exact"`` is selected and the exact
            extractor cannot complete.
        """
        if self.strategy == "heuristic":
            return self._get_heuristic().extract(hyperplanes)
        if self.strategy == "exact":
            return self._get_exact().extract(hyperplanes)
        return self._auto_extract(hyperplanes)

    def _auto_extract(self, hyperplanes: List[Hyperplane]) -> ShardMapping:
        """
        Try exact under a cooperative wall-clock deadline; fall back to
        heuristic on timeout or any failure.  Construction-time errors
        (e.g. missing ``lrs`` binary) also trigger the fallback.

        Runs synchronously: the exact extractor self-aborts at the
        deadline (raising :class:`ExtractionTimeout`), so there is no
        runaway background thread to wait on and the heuristic starts
        immediately.
        """
        try:
            exact = self._get_exact()
        except FileNotFoundError as exc:
            Logger(
                f"Exact extractor unavailable ({exc}); falling back to heuristic",
                level=Logger.Levels.warning,
            ).log()
            return self._get_heuristic().extract(hyperplanes)

        deadline = time.time() + self.timeout_seconds
        partial: ShardMapping = {}
        try:
            return exact.extract(hyperplanes, deadline=deadline)
        except ExtractionTimeout as exc:
            partial = exc.partial or {}
            Logger(
                f"Exact extractor hit the {self.timeout_seconds}s deadline "
                f"({exc}); salvaged {len(partial)} shards, topping up with "
                "the heuristic",
                level=Logger.Levels.warning,
            ).log()
        except Exception as exc:  # noqa: BLE001 - we intentionally fall back on anything
            Logger(
                f"Exact extractor raised {type(exc).__name__}: {exc}; "
                "falling back to heuristic",
                level=Logger.Levels.warning,
            ).log()

        # Union the heuristic's cells with whatever exact completed before
        # the deadline.  ``partial`` is spread last so exact's MILP points
        # (L1-minimal / near-origin) win over the heuristic's ray witness
        # for any cell both found.
        heuristic_result = self._get_heuristic().extract(hyperplanes)
        return {**heuristic_result, **partial}

    def _get_exact(self) -> LrslibExtractor:
        if self._exact is None:
            self._exact = LrslibExtractor(
                unbounded_check=self._exact_unbounded_check,
                num_workers=self._exact_num_workers,
            )
        return self._exact

    def _get_heuristic(self) -> RayShootingExtractor:
        if self._heuristic is None:
            self._heuristic = RayShootingExtractor()
        return self._heuristic
