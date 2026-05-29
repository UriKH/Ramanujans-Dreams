"""
Atlas writer — emits DB-ready DTOs (CmfFamilyDTO, CmfDTO, ShardDTO) as
JSONL files alongside the existing pickle exports.

Used by:
  * Loading stage  — writes ``cmfs.jsonl`` + ``cmf_families.jsonl`` per constant.
  * Extraction stage — writes ``<cmf>__shards.jsonl`` per CMF.

Writes are idempotent: each file is keyed by the appropriate id field
(``cmf_id`` / ``family_id`` / ``shard_id``) and existing ids are skipped
on rerun.  This matches the trajectory-stage dedup pattern and means we
can safely re-invoke the pipeline without growing the files.
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Set

from ramanujantools.cmf import CMF, pFq

from dreamer.extraction.shard import Shard
from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.dtos import CmfDTO, CmfFamilyDTO, ShardDTO
from dreamer.utils.storage.trajectory_attributes import (
    _stable_id,
    derive_cmf_and_shard_ids,
)
from dreamer.utils.types import CMFData


# ---------------------------------------------------------------------------
# DTO builders
# ---------------------------------------------------------------------------

def _family_id_for(cmf: CMF) -> str:
    """Return a stable family identifier for *cmf*.

    For ``pFq`` instances we use ``"{p}F{q}"``; for everything else we fall
    back to the class name.  Keeps the family record compact while still
    distinguishing e.g. ``3F2`` from ``4F3``.
    """
    if isinstance(cmf, pFq):
        return f"{cmf.p}F{cmf.q}"
    return cmf.__class__.__name__


def _global_family_id_for(cmf: CMF) -> str:
    """Top-level family bucket — the class name (e.g. ``pFq``)."""
    return cmf.__class__.__name__


def build_cmf_family_dto(cmf: CMF) -> CmfFamilyDTO:
    """Construct a ``CmfFamilyDTO`` from a live CMF object.

    Matrix definitions are serialised via ``str(matrix)`` — a lossless,
    portable representation that round-trips through SymPy.
    """
    matrix_definitions = {
        str(sym): str(matrix) for sym, matrix in cmf.matrices.items()
    }
    return CmfFamilyDTO(
        family_id=_family_id_for(cmf),
        global_family_id=_global_family_id_for(cmf),
        matrix_definitions=matrix_definitions,
        dimensions=len(cmf.matrices),
    )


def build_cmf_dto(
    cmf_data: CMFData,
    constants: Iterable[Constant],
) -> CmfDTO:
    """Construct a ``CmfDTO`` from loaded ``CMFData``.

    The ``cmf_id`` is ``cmf_name`` to stay consistent with the rest of the
    pipeline (see :func:`derive_cmf_and_shard_ids`).  ``cmf_hyperplanes``
    is left empty here — hyperplanes are only known after extraction; a
    later patch (or a future extraction-side update) can fill them in.
    """
    shift = tuple(
        int(v) if isinstance(v, int) or (hasattr(v, "is_Integer") and v.is_Integer)
        else str(v)
        for v in cmf_data.shift.values()
    )
    return CmfDTO(
        cmf_id=cmf_data.cmf_name,
        family_id=_family_id_for(cmf_data.cmf),
        cmf_hyperplanes=[],
        coordinate_shift=shift,
        found_constants=[c.name for c in constants],
    )


def build_shard_dto(shard: Shard) -> ShardDTO:
    """Construct a ``ShardDTO`` from a live ``Shard`` instance.

    ``shard_id`` is the stable SHA-256 hash of the canonical inequality
    system (see :func:`derive_cmf_and_shard_ids`) — this stays
    deterministic across runs regardless of hyperplane enumeration order.

    ``shard_encoding`` is the **±1 sign vector** the shard was constructed
    with: ``encoding[i] = +1`` means the shard lies above hyperplane *i*,
    ``-1`` means below.  Paired one-to-one with the CMF's hyperplane list
    (stored in ``cmf_hyperplanes`` on the corresponding CmfDTO), which is
    the natural combinatorial label of a shard.
    """
    cmf_id, shard_id, _ = derive_cmf_and_shard_ids(shard)

    encoding = tuple(getattr(shard, "encoding", ()) or ())
    dimensionality = len(shard.symbols)

    interior_point = None
    if shard.start_coord is not None:
        try:
            interior_point = tuple(int(v) for v in shard.start_coord.values())
        except (TypeError, ValueError):
            interior_point = tuple(str(v) for v in shard.start_coord.values())

    return ShardDTO(
        shard_id=shard_id,
        cmf_id=cmf_id,
        shard_encoding=encoding,
        dimensionality=dimensionality,
        found_constants=[shard.const.name],
        interior_point=interior_point,
    )


# ---------------------------------------------------------------------------
# Idempotent JSONL append
# ---------------------------------------------------------------------------

def _load_existing_ids(path: str, id_field: str) -> Set[str]:
    """Return the set of *id_field* values already in the JSONL file.

    Returns an empty set when the file is absent or unreadable.  Malformed
    lines are skipped silently — consistent with ``load_seen_trajectories``.
    """
    seen: Set[str] = set()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                value = record.get(id_field)
                if value is not None:
                    seen.add(value)
    except FileNotFoundError:
        pass
    return seen


def append_dtos_jsonl(
    path: str,
    dtos: Iterable,
    id_field: str,
) -> int:
    """Append each DTO's JSON line to *path*, skipping ids already present.

    Parameters
    ----------
    path:
        Destination JSONL file.  Parent directory must exist.
    dtos:
        Iterable of dataclass DTOs that expose ``to_json_line()``.
    id_field:
        Field name to dedup on (e.g. ``"cmf_id"``, ``"shard_id"``).

    Returns
    -------
    Number of records actually appended (newcomers only).
    """
    existing = _load_existing_ids(path, id_field)
    written = 0
    with open(path, "a") as f:
        for dto in dtos:
            value = getattr(dto, id_field, None)
            if value is None or value in existing:
                continue
            f.write(dto.to_json_line() + "\n")
            existing.add(value)
            written += 1
    return written


# ---------------------------------------------------------------------------
# High-level helpers used by the stages
# ---------------------------------------------------------------------------

def write_cmf_records(
    root: str,
    const: Constant,
    cmf_data_list: List[CMFData],
) -> None:
    """Write CmfDTOs + CmfFamilyDTOs for *const* into ``<root>/<const>/``.

    Two files are produced (or extended) per constant directory:
      * ``cmfs.jsonl``        — one ``CmfDTO`` per CMF.
      * ``cmf_families.jsonl`` — one ``CmfFamilyDTO`` per distinct family.
    """
    safe_const = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
    const_path = os.path.join(root, safe_const)
    os.makedirs(const_path, exist_ok=True)

    cmf_dtos = [build_cmf_dto(data, [const]) for data in cmf_data_list]
    family_dtos = [build_cmf_family_dto(data.cmf) for data in cmf_data_list]

    append_dtos_jsonl(os.path.join(const_path, "cmfs.jsonl"), cmf_dtos, "cmf_id")
    append_dtos_jsonl(
        os.path.join(const_path, "cmf_families.jsonl"), family_dtos, "family_id"
    )


def update_cmf_hyperplanes(
    root: str,
    const: Constant,
    cmf_name: str,
    hyperplanes: Iterable,
) -> bool:
    """Fill in ``cmf_hyperplanes`` for an existing CmfDTO record.

    The loading stage writes the CMF record with an empty hyperplane list
    because hyperplanes are only known after extraction.  This helper
    rewrites the existing ``cmfs.jsonl`` line for ``cmf_name`` so the
    record is complete.

    Implementation: read all records, update the matching one in place,
    rewrite the file.  ``cmfs.jsonl`` is small (typically <10 records per
    constant) so this is cheap and keeps the file canonical — one row
    per CMF, no patch records to merge.

    Parameters
    ----------
    root:
        Same root used by :func:`write_cmf_records` (typically
        ``sys_config.EXPORT_CMFS``).
    const:
        Constant the CMF belongs to.
    cmf_name:
        Identifies the CmfDTO row to update (matches ``cmf_id``).
    hyperplanes:
        Iterable of ``Hyperplane`` objects; each is serialised via
        ``str(hp.expr)``.

    Returns
    -------
    ``True`` if a record was updated, ``False`` if no matching cmf_id was
    found or the file does not yet exist.
    """
    safe_const = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
    path = os.path.join(root, safe_const, "cmfs.jsonl")
    if not os.path.exists(path):
        return False

    serialized = [str(getattr(hp, "expr", hp)) for hp in hyperplanes]

    with open(path, "r") as f:
        lines = [ln for ln in (line.strip() for line in f) if ln]

    updated = False
    out_lines: List[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if record.get("cmf_id") == cmf_name:
            record["cmf_hyperplanes"] = serialized
            updated = True
        out_lines.append(json.dumps(record))

    if updated:
        with open(path, "w") as f:
            f.write("\n".join(out_lines) + "\n")
    return updated


def shard_records_path(root: str, const: Constant, cmf_name: str) -> str:
    """Return the ``<root>/<const>/<cmf>__shards.jsonl`` path for a CMF.

    Centralises the safe-name munging shared by the writer and reader so
    they can never drift apart.
    """
    safe_const = "".join(c for c in const.name if c.isalnum() or c in ("-", "_"))
    safe_cmf = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in str(cmf_name)
    ).strip("_") or "unknown"
    return os.path.join(root, safe_const, f"{safe_cmf}__shards.jsonl")


def write_shard_records(
    root: str,
    const: Constant,
    cmf_name: str,
    shards: Iterable[Shard],
) -> int:
    """Write ShardDTOs for one CMF into ``<root>/<const>/<cmf>__shards.jsonl``.

    Returns the number of new records appended (existing ids are skipped).
    """
    path = shard_records_path(root, const, cmf_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shard_dtos = [build_shard_dto(s) for s in shards]
    return append_dtos_jsonl(path, shard_dtos, "shard_id")


def read_shard_records(
    root: str,
    const: Constant,
    cmf_name: str,
) -> List[ShardDTO]:
    """Read the cached ShardDTOs for one CMF, or ``[]`` if none on disk.

    The inverse of :func:`write_shard_records`: lets the extraction stage
    reload previously-computed shards (encoding + interior point) and
    skip re-running the expensive enumeration.  Malformed lines are
    skipped rather than aborting the load.
    """
    path = shard_records_path(root, const, cmf_name)
    if not os.path.isfile(path):
        return []
    out: List[ShardDTO] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ShardDTO.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return out
