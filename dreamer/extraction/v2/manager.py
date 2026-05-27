"""
Strategy router for v2 shard extraction.

The user picks one of ``"auto" | "exact" | "heuristic"``:

* ``"exact"`` -- :class:`LrslibExtractor` only; raises if it fails.
* ``"heuristic"`` -- :class:`RayShootingExtractor` only.
* ``"auto"`` -- try the exact strategy with a wall-clock timeout; if
  it exceeds ``timeout_seconds`` or raises, log a warning and fall
  back to the heuristic.

The timeout is enforced by running the exact extractor in a worker
thread.  That kind of timeout is *cooperative*: it cannot kill the
underlying ``lrs`` subprocess if it is already wedged.  Per-call
``lrs`` subprocess timeouts (``LrslibExtractor.per_call_timeout``) are
what actually guarantee progress; the wall-clock ``timeout_seconds``
is a global ceiling on top of that.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import List, Literal, Optional

from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.utils.logger import Logger
from .base import BaseExtractor, ShardMapping
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
    :raises ValueError: If ``strategy`` is unknown.
    """

    def __init__(
        self,
        strategy: Strategy = "auto",
        *,
        timeout_seconds: float = 3600.0,
        exact: Optional[LrslibExtractor] = None,
        heuristic: Optional[RayShootingExtractor] = None,
    ):
        if strategy not in ("auto", "exact", "heuristic"):
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected "
                "'auto', 'exact' or 'heuristic'"
            )
        self.strategy = strategy
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
        Try exact under a wall-clock cap; fall back to heuristic on
        failure or timeout.  Construction-time errors (e.g. missing
        ``lrs`` binary) also trigger the fallback.
        """
        try:
            exact = self._get_exact()
        except FileNotFoundError as exc:
            Logger(
                f"Exact extractor unavailable ({exc}); falling back to heuristic",
                level=Logger.Levels.warning,
            ).log()
            return self._get_heuristic().extract(hyperplanes)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(exact.extract, hyperplanes)
            try:
                return future.result(timeout=self.timeout_seconds)
            except FutureTimeout:
                Logger(
                    f"Exact extractor exceeded {self.timeout_seconds}s; "
                    "falling back to heuristic",
                    level=Logger.Levels.warning,
                ).log()
            except Exception as exc:  # noqa: BLE001 - we intentionally fall back on anything
                Logger(
                    f"Exact extractor raised {type(exc).__name__}: {exc}; "
                    "falling back to heuristic",
                    level=Logger.Levels.warning,
                ).log()
        return self._get_heuristic().extract(hyperplanes)

    def _get_exact(self) -> LrslibExtractor:
        if self._exact is None:
            self._exact = LrslibExtractor()
        return self._exact

    def _get_heuristic(self) -> RayShootingExtractor:
        if self._heuristic is None:
            self._heuristic = RayShootingExtractor()
        return self._heuristic
