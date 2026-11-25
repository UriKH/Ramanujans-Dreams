"""
Process start method for the package's pools, chosen per OS.

* **Linux** → ``forkserver``: children fork from a small, single-threaded
  server process.  Unlike plain ``fork`` it never forks the (multi-threaded)
  main process — no fork-with-threads deadlocks — and, unlike ``fork``, new
  workers can be started at any time, which the memory-adaptive pools need to
  grow.  Workers also start faster and lighter than with ``spawn`` (the server
  pre-imports the worker modules once and children share those pages).
* **macOS** → ``spawn`` (the platform default; ``fork`` is unsafe there with
  system frameworks, and ``spawn`` already starts workers dynamically).

Windows is not supported (``fpylll``).  ``system.POOL_START_METHOD`` overrides
the choice (``"auto"`` = per OS).
"""

from __future__ import annotations

import multiprocessing as mp
import sys

#: Modules the Linux fork server imports once, so every worker starts with them.
_FORKSERVER_PRELOAD = [
    "dreamer.utils.stage_pool",
    "dreamer.utils.storage.trajectory_attributes",
    "dreamer.search.methods.flatland.parallel_eval_worker",
]
_preload_set = False


def start_method() -> str:
    """The start method pools should use (see module docstring)."""
    from dreamer.configs import config
    choice = str(getattr(config.system, "POOL_START_METHOD", "auto") or "auto").lower()
    available = mp.get_all_start_methods()
    if choice != "auto":
        if choice not in available:
            raise ValueError(f"POOL_START_METHOD {choice!r} not available here: {available}")
        return choice
    if sys.platform.startswith("linux") and "forkserver" in available:
        return "forkserver"
    return "spawn"


def pool_context():
    """``multiprocessing`` context for the package's process pools.

    Inside a batch job it also makes the workers' numeric libraries
    single-threaded (see :func:`dreamer.utils.allocation.limit_worker_threads`):
    the pool is the parallelism, and PBS sets ``OMP_NUM_THREADS`` to the whole
    allocation.
    """
    global _preload_set
    from dreamer.utils.allocation import limit_worker_threads
    limit_worker_threads()
    method = start_method()
    ctx = mp.get_context(method)
    if method == "forkserver" and not _preload_set:
        try:
            ctx.set_forkserver_preload(_FORKSERVER_PRELOAD)
        except Exception:
            pass
        _preload_set = True
    return ctx


def start_parent_watchdog(interval: float = 1.0) -> None:
    """Exit this process when its parent process dies.

    A pool worker or manager whose parent is SIGKILLed (e.g. a shard process killed
    by the out-of-memory killer) is re-parented and would otherwise live on: it keeps
    the ``multiprocessing`` resource tracker's pipe open, so the run's main process
    waits for it at exit for ever.  A daemon thread polls ``os.getppid()`` every
    *interval* seconds and calls ``os._exit`` once the parent has changed (the
    original parent is gone).  Call it at the start of a child process (pool
    initializer, manager initializer).

    :param interval: Polling period in seconds.
    """
    import os
    import threading

    parent = os.getppid()

    def watch() -> None:
        import time
        while True:
            time.sleep(interval)
            if os.getppid() != parent:
                os._exit(1)

    threading.Thread(target=watch, name="parent-watchdog", daemon=True).start()
