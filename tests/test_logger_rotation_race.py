"""Log rotation when several RD runs start at once in the same working directory (R35).

Found 2026-11-26 (#23 F vs D check): two runs started together both saw ``run.log``, chose
the same archive index and ``os.replace``-d it; the second got ``FileNotFoundError`` and died
at start-up.  The same race could overwrite an archive (``os.replace`` onto ``run.log.<n>``).

Coverage:
  - the base log vanishes between the check and the rotation (another run rotated it):
    no error, the other run's archive is kept, a fresh log is started;
  - the chosen archive index is taken meanwhile: the existing archive is not overwritten;
  - real processes: 8 runs call ``start_run`` at the same moment, none fails, no archive lost.
"""

import multiprocessing as mp

from dreamer.configs.logging import logging_config
from dreamer.utils import logger as logger_mod
from dreamer.utils.logger import Logger


def _reset_logger_runtime_state():
    """Fresh logger state (same as ``tests/test_logger.py``)."""
    for handler in list(logger_mod.sys_logger.handlers):
        logger_mod.sys_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    Logger._instance = None
    Logger._file_handler = None
    Logger._force_overwrite_next_file_open = True
    Logger._last_generate_logs_value = bool(logging_config.GENERATE_LOGS)
    Logger.print_func = print


def _enable(path):
    logging_config.GENERATE_LOGS = True
    logging_config.LOG_FILENAME = str(path)
    _reset_logger_runtime_state()


def test_base_log_rotated_by_another_run_meanwhile(tmp_path, monkeypatch):
    import os.path
    log = tmp_path / "run.log"
    (tmp_path / "run.log.1").write_text("previous run\n")      # the other run archived it ...
    _enable(log)
    real_exists = os.path.exists
    stale = {"left": 1}

    def exists(path):                                        # ... after this run saw it still there
        if os.path.abspath(str(path)) == str(log) and stale["left"]:
            stale["left"] -= 1
            return True
        return real_exists(path)
    monkeypatch.setattr(os.path, "exists", exists)

    Logger.start_run()                                          # must not raise
    Logger("this run", Logger.Levels.info).log()
    assert (tmp_path / "run.log.1").read_text() == "previous run\n"
    assert "this run" in log.read_text()


def test_archive_index_taken_meanwhile_is_not_overwritten(tmp_path, monkeypatch):
    log = tmp_path / "run.log"
    log.write_text("previous run\n")
    _enable(log)

    def other_run_archives_at_1(base):
        (tmp_path / "run.log.1").write_text("other run's archive\n")
        return 1
    monkeypatch.setattr(Logger, "_next_run_index", classmethod(lambda cls, base: other_run_archives_at_1(base)))

    Logger.start_run()
    assert (tmp_path / "run.log.1").read_text() == "other run's archive\n"
    archived = [p.read_text() for p in tmp_path.glob("run.log.*")]
    assert "previous run\n" in archived


def _start(path, barrier, q):
    try:
        logging_config.GENERATE_LOGS = True
        logging_config.LOG_FILENAME = path
        barrier.wait(30)
        Logger.start_run()
        q.put("ok")
    except Exception as e:  # reported to the parent
        q.put(f"{type(e).__name__}: {e}")


def test_simultaneous_runs_in_one_folder(tmp_path):
    ctx = mp.get_context("spawn")
    for attempt in range(3):
        log = tmp_path / f"a{attempt}" / "run.log"
        log.parent.mkdir()
        log.write_text("previous run\n")
        n = 8
        barrier, q = ctx.Barrier(n), ctx.Queue()
        procs = [ctx.Process(target=_start, args=(str(log), barrier, q)) for _ in range(n)]
        for p in procs:
            p.start()
        results = [q.get(timeout=120) for _ in procs]
        for p in procs:
            p.join(30)
        assert results == ["ok"] * n, results
        texts = [p.read_text() for p in log.parent.glob("run.log*")]
        assert sum(t == "previous run\n" for t in texts) >= 1     # the previous log survives
