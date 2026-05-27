"""
Primitive CLI for browsing search-stage JSONL outputs.

Layout assumed:
    <root>/<constant_name>/<shard_id>.jsonl

where ``shard_id`` has the structural form ``<cmf_id>__<encoding_hash>`` and
each line is either a base ``TrajectoryDTO`` record or a patch record (the
loader merges them by ``trajectory_id``, last-write-wins on conflicts).

Three subcommands:

    list-shards   <cmf_id>
        List shards stored for the given CMF — every per-shard JSONL whose
        filename starts with ``<cmf_id>__``.  Prints ``<shard_id>  <encoding>``
        where ``encoding`` is reconstructed from the first record of the file
        (so the encoding column is empty for whole-space shards).

    list-trajectories <shard_id>
        List every trajectory in the shard with its start / direction /
        constant / trajectory_id.

    show-trajectory <trajectory_id>
        Dump every merged attribute for a single trajectory (base record +
        all patches folded in).

By default ``<root>`` is ``./search results`` (next to this script).
Override with ``--root <path>`` if your data lives elsewhere.

Usage examples (PowerShell)::

    python examples/search_data.py list-shards pFq_2_1_-1__0_0_0
    python examples/search_data.py list-trajectories pFq_2_1_-1__0_0_0__bbdd77c5aa993e6b
    python examples/search_data.py show-trajectory pFq_2_1_-1__0_0_0__bbdd77c5aa993e6b__6dc4cffa9a333318
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search results")


def _iter_shard_files(root: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(constant_name, jsonl_path)`` for every per-shard JSONL under *root*."""
    if not os.path.isdir(root):
        return
    for const_name in sorted(os.listdir(root)):
        const_dir = os.path.join(root, const_name)
        if not os.path.isdir(const_dir):
            continue
        for fname in sorted(os.listdir(const_dir)):
            if fname.endswith(".jsonl"):
                yield const_name, os.path.join(const_dir, fname)


def _merge_jsonl(path: str) -> Dict[str, dict]:
    """Read a per-shard JSONL and merge patches into base records.

    Same semantics as ``dreamer.utils.multi_processing.load_seen_trajectories``,
    duplicated locally so this script doesn't have to import the package.
    """
    merged: Dict[str, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = record.get("trajectory_id")
            if tid is None:
                continue
            if tid not in merged:
                merged[tid] = record
            else:
                existing_em = dict(merged[tid].get("extended_metrics") or {})
                new_em = dict(record.get("extended_metrics") or {})
                merged[tid].update(record)
                merged[tid]["extended_metrics"] = {**existing_em, **new_em}
    return merged


def _shard_id_from_filename(fname: str) -> str:
    """Strip the trailing ``.jsonl`` extension to recover the shard id."""
    return fname[:-len(".jsonl")] if fname.endswith(".jsonl") else fname


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_list_shards(args: argparse.Namespace) -> int:
    """List every shard JSONL whose filename starts with ``<cmf_id>__``."""
    matches: List[Tuple[str, str, str]] = []  # (constant, shard_id, sample_record_path)
    for const_name, jsonl_path in _iter_shard_files(args.root):
        fname = os.path.basename(jsonl_path)
        shard_id = _shard_id_from_filename(fname)
        if shard_id.startswith(f"{args.cmf_id}__"):
            matches.append((const_name, shard_id, jsonl_path))

    if not matches:
        print(f"No shards found for cmf_id={args.cmf_id!r} under {args.root}")
        return 1

    print(f"Shards for cmf_id={args.cmf_id} ({len(matches)} total):")
    print(f"{'constant':<16} {'shard_id':<60} {'records':>8}  encoding")
    print("-" * 110)
    for const_name, shard_id, path in matches:
        merged = _merge_jsonl(path)
        # Pull an encoding hint from the first record (when present).
        encoding = ""
        for r in merged.values():
            enc = r.get("shard_encoding")
            if enc is not None:
                encoding = str(enc)
                break
        print(f"{const_name:<16} {shard_id:<60} {len(merged):>8}  {encoding}")
    return 0


def cmd_list_trajectories(args: argparse.Namespace) -> int:
    """List trajectories inside a single shard JSONL."""
    target = args.shard_id
    found_path: Optional[str] = None
    found_const: Optional[str] = None
    for const_name, jsonl_path in _iter_shard_files(args.root):
        if _shard_id_from_filename(os.path.basename(jsonl_path)) == target:
            found_path = jsonl_path
            found_const = const_name
            break

    if found_path is None:
        print(f"No shard JSONL matching shard_id={target!r} under {args.root}")
        return 1

    merged = _merge_jsonl(found_path)
    if not merged:
        print(f"Shard {target} (under {found_const}) is empty.")
        return 0

    print(f"Shard {target}  (constant={found_const}, {len(merged)} trajectories)")
    print(f"{'trajectory_id':<86} {'identified':>10}  {'delta':>10}  start -> direction")
    print("-" * 160)
    for tid, r in merged.items():
        identified = r.get("identified", "?")
        delta = r.get("delta_estimate", "?")
        start = r.get("start_point")
        direction = r.get("direction")
        delta_str = (
            f"{delta:.4f}"
            if isinstance(delta, (int, float)) and delta == delta and abs(delta) < 1e30
            else str(delta)
        )
        print(
            f"{tid:<86} {str(identified):>10}  {delta_str:>10}  "
            f"{start} -> {direction}"
        )
    return 0


def cmd_show_trajectory(args: argparse.Namespace) -> int:
    """Show every merged field for one trajectory."""
    target = args.trajectory_id
    # Trajectory id structure: ``<shard_id>__<traj_hash>``.  Strip the last
    # ``__<hash>`` segment to find the shard_id and locate the JSONL directly
    # — much faster than scanning every shard.
    shard_id = target.rsplit("__", 1)[0] if "__" in target else None

    record: Optional[dict] = None
    constant_name: Optional[str] = None

    if shard_id:
        for const_name, jsonl_path in _iter_shard_files(args.root):
            if _shard_id_from_filename(os.path.basename(jsonl_path)) == shard_id:
                merged = _merge_jsonl(jsonl_path)
                if target in merged:
                    record = merged[target]
                    constant_name = const_name
                break  # shard id is unique; no need to keep scanning

    # Fallback: trajectory id might not follow the structural prefix (e.g.
    # legacy data) — scan every JSONL in last resort.
    if record is None:
        for const_name, jsonl_path in _iter_shard_files(args.root):
            merged = _merge_jsonl(jsonl_path)
            if target in merged:
                record = merged[target]
                constant_name = const_name
                break

    if record is None:
        print(f"Trajectory {target!r} not found under {args.root}")
        return 1

    print(f"Trajectory {target}  (constant={constant_name})")
    print("=" * 80)
    for key, value in record.items():
        if key == "extended_metrics":
            continue
        print(f"  {key:<22} {value}")

    em = record.get("extended_metrics") or {}
    if em:
        print()
        print("  extended_metrics:")
        for k in sorted(em.keys()):
            print(f"    {k:<32} {em[k]}")
    return 0


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="search_data.py",
        description=(
            "Primitive search over per-shard JSONL outputs.  "
            "Three subcommands: list-shards, list-trajectories, show-trajectory."
        ),
    )
    p.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=f"Search-results root (default: {DEFAULT_ROOT})",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ls = sub.add_parser("list-shards", help="List shards for a given cmf_id.")
    p_ls.add_argument("cmf_id", help="e.g. pFq_2_1_-1__0_0_0")
    p_ls.set_defaults(func=cmd_list_shards)

    p_lt = sub.add_parser(
        "list-trajectories", help="List trajectories in a given shard.",
    )
    p_lt.add_argument("shard_id", help="Shard id (e.g. pFq_2_1_-1__0_0_0__<hash>)")
    p_lt.set_defaults(func=cmd_list_trajectories)

    p_st = sub.add_parser(
        "show-trajectory", help="Show all attributes for a given trajectory.",
    )
    p_st.add_argument(
        "trajectory_id",
        help="Trajectory id (e.g. <shard_id>__<traj_hash>)",
    )
    p_st.set_defaults(func=cmd_show_trajectory)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
