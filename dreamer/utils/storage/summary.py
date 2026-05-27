"""
Markdown summary writer.

Reads the JSONL outputs of a completed pipeline run — the per-shard
``<EXPORT_SEARCH_RESULTS>/<constant>/<shard_id>.jsonl`` trajectory store,
plus the optional ``<EXPORT_CMFS>/<constant>/cmfs.jsonl`` metadata file —
and renders a single ``summary.md`` describing what the run scanned,
what was found, and where the best trajectories live.

This-run filtering
==================

Past runs may leave orphan JSONL files in the search-results directory
(e.g. extraction sampling discovered a shard last time but not this
time — the file persists because the writer always appends).  Callers
that want the summary to describe **this run only** should pass
``this_run_shards`` (a mapping from constant name to the set of shard
ids the current pipeline knows about).  Files whose shard_id isn't in
that set are silently skipped; per-constant rows are likewise omitted
when no shards passed the filter.  Passing ``None`` (the default) keeps
the legacy behaviour of summarising every file on disk.

Designed to be safe to call after a partial / failed run: missing
directories, empty files, and non-finite deltas are all tolerated and
produce a report that still makes sense.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Set, Tuple

from dreamer.utils.multi_processing import load_seen_trajectories


# ---------------------------------------------------------------------------
# Per-shard / per-cmf aggregation
# ---------------------------------------------------------------------------

class _ShardStats:
    """Aggregate trajectory-level statistics for one shard."""

    __slots__ = (
        "shard_id", "cmf_id", "constant", "trajectories",
        "identified", "positive_delta", "best_delta", "best_trajectory_id",
        "best_start", "best_direction", "interior_point",
    )

    def __init__(self, shard_id: str, cmf_id: str, constant: str):
        self.shard_id = shard_id
        self.cmf_id = cmf_id
        self.constant = constant
        self.trajectories = 0
        self.identified = 0
        self.positive_delta = 0
        self.best_delta: Optional[float] = None
        self.best_trajectory_id: Optional[str] = None
        self.best_start: Optional[list] = None
        self.best_direction: Optional[list] = None
        # Representative integer point inside the shard, as returned by
        # the extractor (legacy lattice scan or v2 MILP).  Filled in from
        # ``ShardDTO.interior_point`` after the trajectory walk if the
        # ``EXPORT_CMFS`` sidecar is available.
        self.interior_point: Optional[Tuple[int, ...]] = None

    def add(self, record: dict) -> None:
        self.trajectories += 1
        if bool(record.get("identified")):
            self.identified += 1
        delta = _finite_float(record.get("delta_estimate"))
        if delta is not None:
            if delta > 0:
                self.positive_delta += 1
            if self.best_delta is None or delta > self.best_delta:
                self.best_delta = delta
                self.best_trajectory_id = record.get("trajectory_id")
                # ``start_point`` / ``direction`` may be absent on patch-only
                # trajectories (no base record ever written) — surface ``None``
                # in that case rather than crashing the summary.
                self.best_start = record.get("start_point")
                self.best_direction = record.get("direction")


def _finite_float(v) -> Optional[float]:
    """Return ``float(v)`` if it's a finite real number, else ``None``.

    The pipeline encodes "did not converge" as ``-inf``; treat that (and any
    ``NaN`` / parse failure) as "no data" so it doesn't poison max/sum stats.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ---------------------------------------------------------------------------
# CMF metadata sidecar
# ---------------------------------------------------------------------------

def _load_cmf_metadata(export_cmfs_root: Optional[str], constant_name: str) -> Dict[str, dict]:
    """Return ``{cmf_id: cmf_dict}`` from ``<root>/<safe_const>/cmfs.jsonl``.

    Empty when the file or directory is missing — the summary degrades to
    showing only the per-shard data with no family / shift annotations.
    The ``safe_const`` heuristic mirrors :func:`atlas_writer.write_shard_records`.
    """
    if not export_cmfs_root:
        return {}
    safe_const = "".join(c for c in constant_name if c.isalnum() or c in ("-", "_"))
    path = os.path.join(export_cmfs_root, safe_const, "cmfs.jsonl")
    if not os.path.isfile(path):
        return {}
    out: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            cmf_id = record.get("cmf_id")
            if cmf_id:
                out[cmf_id] = record
    return out


def _load_shard_metadata(
    export_cmfs_root: Optional[str], constant_name: str
) -> Dict[str, dict]:
    """Return ``{shard_id: shard_dict}`` from every ``*__shards.jsonl``
    under ``<root>/<safe_const>/``.

    Used to surface the per-shard ``interior_point`` (a witness integer
    coordinate from the extractor) so reviewers can sanity-check what
    starting point the search actually ran from.  Missing directory or
    files degrade silently.
    """
    if not export_cmfs_root:
        return {}
    safe_const = "".join(c for c in constant_name if c.isalnum() or c in ("-", "_"))
    const_dir = os.path.join(export_cmfs_root, safe_const)
    if not os.path.isdir(const_dir):
        return {}
    out: Dict[str, dict] = {}
    for fname in sorted(os.listdir(const_dir)):
        if not fname.endswith("__shards.jsonl"):
            continue
        path = os.path.join(const_dir, fname)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                shard_id = record.get("shard_id")
                if shard_id:
                    out[shard_id] = record
    return out


# ---------------------------------------------------------------------------
# Walk EXPORT_SEARCH_RESULTS and build the data model
# ---------------------------------------------------------------------------

def _collect_shard_stats(
    search_results_root: str,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> Dict[str, Dict[str, List[_ShardStats]]]:
    """Walk the search-results tree and aggregate per-shard stats.

    When ``this_run_shards`` is provided, it must map constant name to a
    set of shard ids that the current pipeline run actually scanned.  Any
    JSONL file whose shard id is *not* in that set is treated as an
    orphan from a previous run and silently dropped.  Constants absent
    from the map are also skipped (so an old "log-2" data dir won't leak
    into a "pi" summary).

    Returns ``{constant_name: {cmf_id: [shard_stats, ...]}}``.
    """
    out: Dict[str, Dict[str, List[_ShardStats]]] = defaultdict(lambda: defaultdict(list))
    if not os.path.isdir(search_results_root):
        return out

    for const_name in sorted(os.listdir(search_results_root)):
        const_dir = os.path.join(search_results_root, const_name)
        if not os.path.isdir(const_dir):
            continue
        if this_run_shards is not None and const_name not in this_run_shards:
            # No shards searched under this constant in the current run —
            # the whole subtree is stale.
            continue
        allowed = this_run_shards.get(const_name) if this_run_shards is not None else None
        for fname in sorted(os.listdir(const_dir)):
            if not fname.endswith(".jsonl"):
                continue
            shard_id = fname[: -len(".jsonl")]
            if allowed is not None and shard_id not in allowed:
                # Orphan from an earlier run — extraction sampling is
                # stochastic, so a shard discovered last time may not
                # appear this time.  Skip it cleanly.
                continue
            # ``shard_id`` has the structural form ``<cmf_id>__<hash>``; the
            # cmf id is everything before the trailing hash segment.  Falls
            # back to the whole string for legacy / unstructured ids.
            cmf_id = shard_id.rsplit("__", 1)[0] if "__" in shard_id else shard_id

            merged = load_seen_trajectories(os.path.join(const_dir, fname))
            if not merged:
                continue
            stats = _ShardStats(shard_id, cmf_id, const_name)
            for record in merged.values():
                stats.add(record)
            out[const_name][cmf_id].append(stats)
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _fmt_delta(value: Optional[float], *, precision: int = 4) -> str:
    """Render a δ value or ``"—"`` for missing data."""
    if value is None:
        return "—"
    return f"{value:.{precision}f}"


def _fmt_shard_label(shard_id: str, width: int = 18) -> str:
    """Trim the shard id to its trailing hash for tabular display."""
    tail = shard_id.rsplit("__", 1)[-1]
    return f"`…{tail[:width]}`"


def _fmt_traj_cell(s: _ShardStats) -> str:
    """Render the per-shard "best trajectory" cell.

    Carries both the start→direction (the math handle the user actually
    reads) and the trajectory id's trailing hash (the handle for
    ``examples/search_data.py show-trajectory``).
    """
    if not s.best_trajectory_id:
        return "—"
    tail = s.best_trajectory_id.rsplit("__", 1)[-1]
    start = list(s.best_start) if s.best_start is not None else "?"
    direction = list(s.best_direction) if s.best_direction is not None else "?"
    return f"`{start}` → `{direction}`  (`…{tail[:18]}`)"


def _render_overview(
    data: Dict[str, Dict[str, List[_ShardStats]]],
) -> Tuple[str, Optional[Tuple[str, _ShardStats]]]:
    """Render the top-level per-constant overview table.

    Returns ``(markdown, overall_best)`` where ``overall_best`` is
    ``(constant, shard_stats)`` for the best δ across the whole run, or
    ``None`` when no finite δ was recorded.
    """
    lines = [
        "## Run overview",
        "",
        "| Constant | CMFs | Shards | Trajectories | Identified | Positive δ | Best δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    overall_best: Optional[Tuple[str, _ShardStats]] = None
    for const_name in sorted(data.keys()):
        cmf_shards = data[const_name]
        n_cmfs = len(cmf_shards)
        all_shards = [s for shards in cmf_shards.values() for s in shards]
        n_shards = len(all_shards)
        n_traj = sum(s.trajectories for s in all_shards)
        n_ident = sum(s.identified for s in all_shards)
        n_pos = sum(s.positive_delta for s in all_shards)
        best_for_const = max(
            (s for s in all_shards if s.best_delta is not None),
            key=lambda s: s.best_delta,
            default=None,
        )
        best_str = _fmt_delta(best_for_const.best_delta) if best_for_const else "—"
        lines.append(
            f"| `{const_name}` | {n_cmfs} | {n_shards} | {n_traj} | "
            f"{n_ident} | {n_pos} | **{best_str}** |"
        )
        if best_for_const is not None and (
            overall_best is None
            or best_for_const.best_delta > overall_best[1].best_delta
        ):
            overall_best = (const_name, best_for_const)

    if overall_best is None:
        lines += ["", "_No converging trajectory recorded in this run._"]
    else:
        const, s = overall_best
        start = list(s.best_start) if s.best_start is not None else "?"
        direction = list(s.best_direction) if s.best_direction is not None else "?"
        lines += [
            "",
            f"**Overall best δ**: {_fmt_delta(s.best_delta)}  ",
            f"- constant: `{const}`  ",
            f"- start point: `{start}`  ",
            f"- direction: `{direction}`  ",
            f"- cmf: `{s.cmf_id}`  ",
            f"- shard: `{s.shard_id}`  ",
            f"- trajectory: `{s.best_trajectory_id}`",
        ]
    return "\n".join(lines), overall_best


def _fmt_interior_point(pt: Optional[Tuple[int, ...]]) -> str:
    """Format a shard's interior witness point for the markdown table."""
    if pt is None:
        return "—"
    return f"`{list(pt)}`"


def _render_per_constant_section(
    const_name: str,
    cmf_shards: Dict[str, List[_ShardStats]],
    cmf_metadata: Dict[str, dict],
) -> str:
    lines = [f"## {const_name}", ""]
    for cmf_id in sorted(cmf_shards.keys()):
        shards = sorted(
            cmf_shards[cmf_id],
            key=lambda s: (s.best_delta is None, -(s.best_delta or 0.0)),
        )
        meta = cmf_metadata.get(cmf_id, {})
        family = meta.get("family_id")
        shift = meta.get("coordinate_shift")
        n_traj = sum(s.trajectories for s in shards)
        n_ident = sum(s.identified for s in shards)
        n_pos = sum(s.positive_delta for s in shards)
        best_for_cmf = max(
            (s for s in shards if s.best_delta is not None),
            key=lambda s: s.best_delta,
            default=None,
        )
        best_str = _fmt_delta(best_for_cmf.best_delta) if best_for_cmf else "—"

        lines.append(f"### CMF `{cmf_id}`")
        bullets: List[str] = []
        if family:
            bullets.append(f"- Family: `{family}`")
        if shift is not None:
            bullets.append(f"- Coordinate shift: `{list(shift)}`")
        bullets.append(f"- Shards: **{len(shards)}**")
        bullets.append(
            f"- Trajectories: **{n_traj}**  "
            f"(identified: **{n_ident}**, positive δ: **{n_pos}**)"
        )
        bullets.append(f"- Best δ within CMF: **{best_str}**")
        lines += bullets + [""]

        lines += [
            "| Shard | Start point | Trajectories | Identified | Positive δ | Best δ | Best trajectory (start → direction) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for s in shards:
            lines.append(
                f"| {_fmt_shard_label(s.shard_id)} | "
                f"{_fmt_interior_point(s.interior_point)} | "
                f"{s.trajectories} | {s.identified} | {s.positive_delta} | "
                f"{_fmt_delta(s.best_delta)} | {_fmt_traj_cell(s)} |"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_summary_markdown(
    *,
    search_results_root: str,
    export_cmfs_root: Optional[str] = None,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> str:
    """Build the full ``summary.md`` content for a pipeline run.

    Parameters
    ----------
    search_results_root:
        Root directory containing per-constant subdirectories of per-shard
        JSONL files (``sys_config.EXPORT_SEARCH_RESULTS``).
    export_cmfs_root:
        Optional sidecar root where CMF DTOs live
        (``sys_config.EXPORT_CMFS``) — used only to annotate per-CMF rows
        with family / coordinate-shift info.  Missing files are tolerated.
    this_run_shards:
        Optional ``{constant_name: {shard_id, ...}}`` mapping describing
        which shards the *current* pipeline run actually touched.  When
        provided, any JSONL whose shard id is not in the set is treated
        as a stale leftover from a past run and dropped.  ``None``
        (default) summarises every file on disk.
    """
    data = _collect_shard_stats(search_results_root, this_run_shards)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    header = [
        "# Pipeline Summary",
        "",
        f"_Generated: {timestamp}_",
        "",
        f"Search-results root: `{search_results_root}`",
    ]
    if export_cmfs_root:
        header.append(f"CMF-metadata root: `{export_cmfs_root}`")
    if this_run_shards is not None:
        n_total = sum(len(v) for v in this_run_shards.values())
        header.append(
            f"Scope: this run only ({n_total} shard(s) across "
            f"{len(this_run_shards)} constant(s))"
        )
    header.append("")

    if not data:
        header += [
            "_No per-shard JSONL files were found — nothing to summarise._",
            "",
        ]
        return "\n".join(header)

    # Backfill each shard's interior witness point from the
    # ``<EXPORT_CMFS>/<const>/<cmf>__shards.jsonl`` sidecar so the
    # per-shard table can show where the search actually started.
    for const_name, cmf_shards in data.items():
        shard_meta = _load_shard_metadata(export_cmfs_root, const_name)
        if not shard_meta:
            continue
        for shards in cmf_shards.values():
            for s in shards:
                rec = shard_meta.get(s.shard_id)
                if not rec:
                    continue
                pt = rec.get("interior_point")
                if pt is not None:
                    try:
                        s.interior_point = tuple(int(v) for v in pt)
                    except (TypeError, ValueError):
                        s.interior_point = tuple(pt)

    overview_md, _overall_best = _render_overview(data)

    sections: List[str] = []
    for const_name in sorted(data.keys()):
        cmf_metadata = _load_cmf_metadata(export_cmfs_root, const_name)
        sections.append(
            _render_per_constant_section(const_name, data[const_name], cmf_metadata)
        )

    return "\n".join(header + [overview_md, "", "---", ""] + sections)


def write_summary(
    *,
    search_results_root: str,
    export_cmfs_root: Optional[str] = None,
    output_path: Optional[str] = None,
    this_run_shards: Optional[Mapping[str, Set[str]]] = None,
) -> Optional[str]:
    """Render and write ``summary.md`` for a pipeline run.

    The output defaults to ``<search_results_root>/summary.md``.  Returns
    the path written, or ``None`` when ``search_results_root`` itself is
    missing (caller should treat that as "nothing to summarise").

    See :func:`build_summary_markdown` for the meaning of
    ``this_run_shards``.
    """
    if not os.path.isdir(search_results_root):
        return None
    if output_path is None:
        output_path = os.path.join(search_results_root, "summary.md")
    content = build_summary_markdown(
        search_results_root=search_results_root,
        export_cmfs_root=export_cmfs_root,
        this_run_shards=this_run_shards,
    )
    with open(output_path, "w") as f:
        f.write(content)
    return output_path
