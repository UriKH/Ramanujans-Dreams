"""
Primitive CLI for browsing search-stage JSONL outputs.

Layout assumed (flat directory — one file per shard):
    <root>/<shard_id>.jsonl

where ``shard_id`` has the structural form ``<cmf_id>__<encoding_hash>``
and each line is either a base ``TrajectoryDTO`` record or a patch record
(the loader merges them by ``trajectory_id``, last-write-wins on conflicts).

Per-constant attributes (``delta_estimate``, ``p_vector``, ``q_vector``,
``identified``) are stored as dicts keyed by constant name — one trajectory
covers all constants searched in that shard.

Three subcommands:

    list-shards   <cmf_id>
        List shards stored for the given CMF — every per-shard JSONL whose
        filename starts with ``<cmf_id>__``.  Prints
        ``<shard_id>  <records>  <constants>  encoding`` where
        ``constants`` is the union of constant names found in the JSONL.

    list-trajectories <shard_id>
        List every trajectory in the shard with its start / direction /
        constants / trajectory_id.

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
from typing import Dict, Iterator, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../examples/search results")


def _iter_shard_files(root: str) -> Iterator[str]:
    """Yield ``jsonl_path`` for every per-shard JSONL directly under *root*."""
    if not os.path.isdir(root):
        return
    for fname in sorted(os.listdir(root)):
        if fname.endswith(".jsonl"):
            yield os.path.join(root, fname)


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


def _constants_in_file(merged: Dict[str, dict]) -> Set[str]:
    """Collect constant names that appear in any record's ``delta_estimate`` dict."""
    consts: Set[str] = set()
    for r in merged.values():
        d = r.get("delta_estimate")
        if isinstance(d, dict):
            consts.update(d.keys())
    return consts


def _fmt_delta_dict(delta) -> str:
    """Render ``delta_estimate`` whether it is a dict or a legacy scalar."""
    if isinstance(delta, dict):
        parts = [f"{k}={v:.4f}" for k, v in delta.items() if isinstance(v, (int, float))]
        return "{" + ", ".join(parts) + "}" if parts else str(delta)
    if isinstance(delta, (int, float)):
        return f"{delta:.4f}"
    return str(delta)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_list_shards(args: argparse.Namespace) -> int:
    """List every shard JSONL whose filename starts with ``<cmf_id>__``."""
    matches: List[Tuple[str, str]] = []  # (shard_id, jsonl_path)
    for jsonl_path in _iter_shard_files(args.root):
        fname = os.path.basename(jsonl_path)
        shard_id = _shard_id_from_filename(fname)
        if shard_id.startswith(f"{args.cmf_id}__"):
            matches.append((shard_id, jsonl_path))

    if not matches:
        print(f"No shards found for cmf_id={args.cmf_id!r} under {args.root}")
        return 1

    print(f"Shards for cmf_id={args.cmf_id} ({len(matches)} total):")
    print(f"{'shard_id':<60} {'records':>8}  {'constants':<24}  encoding")
    print("-" * 130)
    for shard_id, path in matches:
        merged = _merge_jsonl(path)
        consts = _constants_in_file(merged)
        encoding = ""
        for r in merged.values():
            enc = r.get("shard_encoding")
            if enc is not None:
                encoding = str(enc)
                break
        print(
            f"{shard_id:<60} {len(merged):>8}  "
            f"{', '.join(sorted(consts)):<24}  {encoding}"
        )
    return 0


def cmd_list_trajectories(args: argparse.Namespace) -> int:
    """List trajectories inside a single shard JSONL."""
    target = args.shard_id
    found_path: Optional[str] = None
    for jsonl_path in _iter_shard_files(args.root):
        if _shard_id_from_filename(os.path.basename(jsonl_path)) == target:
            found_path = jsonl_path
            break

    if found_path is None:
        print(f"No shard JSONL matching shard_id={target!r} under {args.root}")
        return 1

    merged = _merge_jsonl(found_path)
    if not merged:
        print(f"Shard {target} is empty.")
        return 0

    consts = sorted(_constants_in_file(merged))
    print(f"Shard {target}  (constants={consts}, {len(merged)} trajectories)")
    print(
        f"{'trajectory_id':<86} {'identified':>22}  {'delta':>34}  start -> direction"
    )
    print("-" * 200)
    for tid, r in merged.items():
        identified = r.get("identified", "?")
        if isinstance(identified, dict):
            identified_str = "{" + ", ".join(f"{k}={v}" for k, v in identified.items()) + "}"
        else:
            identified_str = str(identified)
        delta = r.get("delta_estimate", "?")
        delta_str = _fmt_delta_dict(delta)
        start = r.get("start_point")
        direction = r.get("direction")
        print(
            f"{tid:<86} {identified_str:>22}  {delta_str:>34}  "
            f"{start} -> {direction}"
        )
    return 0


def cmd_show_trajectory(args: argparse.Namespace) -> int:
    """Show every merged field for one trajectory."""
    target = args.trajectory_id
    # Trajectory id structure: ``<shard_id>__<traj_hash>``.
    shard_id = target.rsplit("__", 1)[0] if "__" in target else None

    record: Optional[dict] = None

    if shard_id:
        for jsonl_path in _iter_shard_files(args.root):
            if _shard_id_from_filename(os.path.basename(jsonl_path)) == shard_id:
                merged = _merge_jsonl(jsonl_path)
                if target in merged:
                    record = merged[target]
                break

    # Fallback: scan every file.
    if record is None:
        for jsonl_path in _iter_shard_files(args.root):
            merged = _merge_jsonl(jsonl_path)
            if target in merged:
                record = merged[target]
                break

    if record is None:
        print(f"Trajectory {target!r} not found under {args.root}")
        return 1

    print(f"Trajectory {target}")
    print("=" * 80)
    for key, value in record.items():
        if key == "extended_metrics":
            continue
        # Pretty-print dict-valued per-constant fields.
        if isinstance(value, dict) and key in ("delta_estimate", "p_vector", "q_vector", "identified"):
            print(f"  {key:<22}")
            for cname, cval in value.items():
                print(f"    {cname:<20} {cval}")
        else:
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
            "Primitive search over per-shard JSONL outputs (flat directory).  "
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
