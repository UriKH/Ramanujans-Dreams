"""
Subprocess + text-format helpers for the lrslib ``lrs`` binary.

We deliberately avoid the C-API wrappers (``pyrs``, ``cdd``) because
they impose a C toolchain on end-users and crash on degenerate inputs.
Instead we drive the standalone ``lrs`` executable through stdin/stdout
using :mod:`subprocess`.

The lrs H-representation file format we emit looks like::

    cell_<id>
    H-representation
    begin
    <m> <d+1> integer
    <c_1> <a_1_1> <a_1_2> ... <a_1_d>
    ...
    <c_m> <a_m_1> <a_m_2> ... <a_m_d>
    end

Each row encodes the inequality ``c_i + a_i . x >= 0`` (lrs's native
convention).  The corresponding V-representation lines we parse are::

    V-representation
    begin
    <n> <d+1> rational
     1 <vertex coords>     <- bounded vertex (first column = 1)
     0 <ray direction>     <- unbounded ray  (first column = 0)
    end

The presence of any ``0 ...`` line identifies an unbounded cell.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np


_BEGIN_RE = re.compile(r"^\s*begin\s*$", re.IGNORECASE)
_END_RE = re.compile(r"^\s*end\s*$", re.IGNORECASE)
_HEADER_RE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(integer|rational|real)\s*$", re.IGNORECASE
)


def lrs_available(binary: str = "lrs") -> bool:
    """
    Return ``True`` iff a callable ``lrs`` binary is on ``PATH``.

    :param binary: Executable name to look up.  Defaults to ``"lrs"``.
    :return: Whether the binary is available.
    """
    return shutil.which(binary) is not None


def format_hrep(
    A: np.ndarray,
    c: np.ndarray,
    sign_vector: Sequence[int],
    *,
    name: str = "cell",
) -> str:
    """
    Build the lrs H-representation for a single cell of the arrangement.

    The cell is ``{ x : s_i * (A[i] . x + c[i]) >= 0 for all i }``,
    which we encode row-by-row as ``s_i * c_i + (s_i * A[i]) . x >= 0``.

    :param A: Hyperplane coefficient matrix, shape ``(N, D)``.
    :param c: Hyperplane constants, shape ``(N,)``.
    :param sign_vector: Length-``N`` sequence of ``+1`` / ``-1``.
    :param name: Free-form label written as the first line of the file.
    :return: Full file contents ready to hand to ``lrs``.
    :raises ValueError: If shapes are inconsistent or signs are bad.
    """
    A = np.asarray(A, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    s = np.asarray(sign_vector, dtype=np.int64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    if c.shape != (A.shape[0],):
        raise ValueError(f"c shape {c.shape} incompatible with A shape {A.shape}")
    if s.shape != (A.shape[0],):
        raise ValueError(f"sign_vector shape {s.shape} incompatible with A shape {A.shape}")
    if not np.all(np.isin(s, [-1, 1])):
        raise ValueError("sign_vector entries must be +1 or -1")

    n, d = A.shape
    rows: List[str] = []
    rows.append(name)
    rows.append("H-representation")
    rows.append("begin")
    rows.append(f"{n} {d + 1} integer")
    for i in range(n):
        coeffs = s[i] * A[i]
        const = int(s[i] * c[i])
        rows.append(" ".join([str(const), *map(str, coeffs.tolist())]))
    rows.append("end")
    return "\n".join(rows) + "\n"


@contextmanager
def hrep_tempfile(text: str) -> Iterator[Path]:
    """
    Write ``text`` to a tempfile and yield its path; clean up on exit.

    :param text: File contents (typically from :func:`format_hrep`).
    :return: Context yielding the tempfile path.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ine", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(text)
        path = Path(fh.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def run_lrs(hrep_text: str, *, binary: str = "lrs", timeout: float = 60.0) -> str:
    """
    Pipe an H-representation through ``lrs`` and return its raw stdout.

    :param hrep_text: H-representation in lrs format (see
        :func:`format_hrep`).
    :param binary: Path or name of the lrs binary.
    :param timeout: Subprocess timeout in seconds.
    :return: Raw stdout text emitted by ``lrs``.
    :raises FileNotFoundError: If the binary is not on ``PATH``.
    :raises subprocess.TimeoutExpired: On timeout.
    :raises subprocess.CalledProcessError: If ``lrs`` exits non-zero.
    """
    if not lrs_available(binary):
        raise FileNotFoundError(
            f"lrs binary not found on PATH (looked for {binary!r}). "
            "Install lrslib (e.g. `apt-get install lrslib` on WSL/Ubuntu)."
        )
    with hrep_tempfile(hrep_text) as path:
        completed = subprocess.run(
            [binary, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        )
    return completed.stdout


def parse_vrep_unbounded(vrep_text: str) -> bool:
    """
    Return ``True`` iff the V-representation contains at least one ray.

    A row in the V-representation starts with ``1`` for a vertex or
    ``0`` for a ray (unbounded direction).  Any ``0 ...`` row therefore
    proves the cell is unbounded.

    :param vrep_text: Raw stdout from :func:`run_lrs`.
    :return: ``True`` if any ray line is present in any ``begin``/``end``
        block.
    """
    in_block = False
    header_seen = False
    for raw in vrep_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _BEGIN_RE.match(line):
            in_block = True
            header_seen = False
            continue
        if _END_RE.match(line):
            in_block = False
            continue
        if not in_block:
            continue
        if not header_seen and _HEADER_RE.match(line):
            header_seen = True
            continue
        if line.startswith("*"):  # lrs comments
            continue
        first = line.split()[0]
        if first == "0":
            return True
    return False
