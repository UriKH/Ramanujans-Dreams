"""
Aesthetic δ-on-a-sphere projections from the **current JSONL pipeline**.

This is the pickle-free successor to ``graphs/shard delta sphere.py``.  The old
script unpickled ``Shard`` and ``DataManager`` objects and matched data to shards
by comparing start coordinates.  The system has since moved to a self-sustaining
JSONL layout (no pickle files):

  * Shards          — reconstructed from ``<EXPORT_CMFS>/<const>/<cmf>.json`` +
                      ``<EXPORT_CMFS>/<cmf>__shards.jsonl`` via
                      :func:`dreamer.utils.storage.atlas_writer.load_shards_from_export`.
  * Trajectories    — one file per shard at
                      ``<EXPORT_SEARCH_RESULTS>/<shard_id>.jsonl``; each line is a
                      ``TrajectoryDTO`` with a ``direction`` tuple and a
                      **per-constant** ``delta_estimate`` dict.

Matching data to a shard is therefore trivial now: the JSONL filename *is*
``<shard_id>.jsonl`` and ``derive_cmf_and_shard_ids(shard)`` yields that id — no
start-coordinate matching needed.

The sphere-rendering maths (rotate the best direction to the equator for a clean
local interpolation, ``griddata`` the δ field, trim to the shard's cone ``A·v ≤ 0``,
draw the bounding hyperplane circles) carries straight over from the old script.

Three options, exactly as requested:

  1. ``one_sphere_per_shard=True``  → an atlas (one sphere per shard, like the old
     ``generate_shard_atlas2``).  ``False`` → every shard drawn on a single shared
     sphere (they live in the same CMF coordinate frame).

  2. For a ``D > 3`` CMF, a :class:`ProjectionSpec` selects which 3 coordinates
     become the sphere's ``(x, y, z)`` and constrains the remaining coordinates to
     a *linear* function of those three — only trajectories lying in that 3-D
     subspace (within ``subspace_tol``) are drawn.  This realises the
     ``(x, y, z, f(x,y,z)) / (f(x,y,z), x, y, z) / …`` notation from the request.

  3. Overlay paths (e.g. gradient-ascent steps) are optional.  ``None`` →
     δ-projection only (like the old script's example output).  Otherwise a path
     is either pulled automatically from a second search-results directory or
     passed explicitly as a list of direction vectors per shard.

Run with the WSL conda env ``rama`` (matplotlib + scipy + the dreamer package).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from typing import Any, Callable

from dreamer.utils.constants.constant import Constant
from dreamer.utils.storage.atlas_writer import load_shards_from_export
from dreamer.utils.storage.trajectory_attributes import derive_cmf_and_shard_ids


# ===========================================================================
# Attribute extraction (what scalar to colour the sphere with)
# ===========================================================================
#
# A *value function* maps a merged trajectory record (a plain dict, exactly as
# stored in the JSONL) to the scalar to plot — or ``None`` to skip that
# trajectory.  δ is the default, but any Tier-1/Tier-2/Tier-3 attribute can be
# plotted instead, including string-formatted sympy expressions (e.g. the
# trajectory asymptotics) reduced to a number via substitution.

ValueFn = Callable[[Dict[str, Any]], Optional[float]]


def delta_value(constant: Constant) -> ValueFn:
    """Value function returning δ for *constant* (the default, classic behaviour).

    :param constant: the constant whose entry to read from ``delta_estimate``.
    :return: a :data:`ValueFn` extracting ``record["delta_estimate"][name]``.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        d = rec.get("delta_estimate") or {}
        v = d.get(constant.name)
        return None if v is None else float(v)
    return _fn


def field_value(key: str, transform: Optional[Callable[[Any], float]] = None) -> ValueFn:
    """Value function reading a top-level numeric field (e.g. ``limit_value``).

    :param key: the record key (e.g. ``"limit_value"``, ``"recurrence_order"``).
    :param transform: optional post-processor applied to the raw field value.
    :return: a :data:`ValueFn`.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        v = rec.get(key)
        if v is None:
            return None
        return float(transform(v)) if transform is not None else float(v)
    return _fn


def extended_metric_value(
    key: str,
    transform: Optional[Callable[[Any], float]] = None,
) -> ValueFn:
    """Value function reading a Tier-2/Tier-3 attribute from ``extended_metrics``.

    Background workers (and the post-process stage) populate the open
    ``extended_metrics`` dict — e.g. ``digits_per_step``, ``spectral_gap``,
    ``asymptotics``.  Use *transform* for non-numeric attributes (see
    :func:`sympy_attribute_value` for the string-sympy case).

    :param key: the metric name inside ``extended_metrics``.
    :param transform: optional callable turning the raw metric into a float.
    :return: a :data:`ValueFn`.
    """
    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        em = rec.get("extended_metrics") or {}
        if key not in em:
            return None
        raw = em[key]
        if raw is None:
            return None
        return float(transform(raw)) if transform is not None else float(raw)
    return _fn


def sympy_attribute_value(
    key: str,
    subs: Dict[str, float],
    *,
    in_extended_metrics: bool = True,
) -> ValueFn:
    """Value function reducing a *string-formatted sympy* attribute to a float.

    Some attributes (e.g. the trajectory **asymptotics**) are stored as a sympy
    expression serialised to a string.  This parses the string with
    ``sympy.sympify`` and substitutes *subs* (e.g. ``{"n": 1e6}``), returning the
    numeric ``evalf`` result.  Trajectories whose attribute is missing or does
    not evaluate to a finite real number are skipped.

    :param key: the attribute name (in ``extended_metrics`` by default, else a
        top-level field when ``in_extended_metrics=False``).
    :param subs: symbol→value substitutions applied before evaluation.
    :param in_extended_metrics: read from ``extended_metrics`` (``True``) vs the
        top-level record (``False``).
    :return: a :data:`ValueFn`.
    """
    import sympy as sp

    sym_subs = {sp.Symbol(k): v for k, v in subs.items()}

    def _fn(rec: Dict[str, Any]) -> Optional[float]:
        container = (rec.get("extended_metrics") or {}) if in_extended_metrics else rec
        raw = container.get(key)
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            expr = sp.sympify(raw)
            val = complex(expr.subs(sym_subs).evalf())
        except (sp.SympifyError, TypeError, ValueError):
            return None
        if abs(val.imag) > 1e-9:
            return None
        return float(val.real)
    return _fn


# ===========================================================================
# Projection spec (choose the 3-D subspace of a D-dimensional CMF)
# ===========================================================================

@dataclass
class ProjectionSpec:
    """Selects the 3-D subspace of a ``D``-dimensional CMF to draw on the sphere.

    The CMF's direction vectors live in ``D`` coordinates (one per CMF symbol,
    in ``shard.symbols`` order).  This spec picks **three** of them to be the
    sphere's ``(x, y, z)`` axes and constrains every remaining ("dependent")
    coordinate to a *linear* combination of those three.  A trajectory is kept
    only when each dependent coordinate matches its linear prediction (within a
    relative tolerance), i.e. the trajectory lies in the chosen subspace.

    :param axes: the three coordinate indices mapped to sphere ``(x, y, z)``.
    :param dependent: ``{coord_index: (a, b, c)}`` — the dependent coordinate at
        ``coord_index`` must equal ``a·x + b·y + c·z`` where ``(x, y, z)`` are
        the free coordinates selected by ``axes``.  Empty for ``D == 3``.
    """

    axes: Tuple[int, int, int] = (0, 1, 2)
    dependent: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)

    @classmethod
    def identity(cls, dim: int) -> "ProjectionSpec":
        """Trivial spec for a 3-D CMF: coords ``(0, 1, 2)`` → ``(x, y, z)``.

        :param dim: the CMF dimensionality (number of symbols).
        :raises ValueError: if ``dim < 3``.
        """
        if dim < 3:
            raise ValueError(f"Need at least 3 coordinates to project, got {dim}.")
        return cls(axes=(0, 1, 2), dependent={})

    @classmethod
    def from_layout(cls, layout: Sequence) -> "ProjectionSpec":
        """Build a spec from the ``(x, y, z, f(x,y,z))`` layout notation.

        ``layout`` has one entry per CMF coordinate (length ``D``):

          * ``"x"`` / ``"y"`` / ``"z"`` — this coordinate is a free sphere axis.
          * a 3-tuple ``(a, b, c)``      — this coordinate is dependent and equals
            ``a·x + b·y + c·z``.

        For example, a 4-D CMF drawn as ``(x, y, z, f)`` with ``f = x - y`` is
        ``ProjectionSpec.from_layout(["x", "y", "z", (1, -1, 0)])``; the
        ``(f, x, y, z)`` variant is ``[(1, -1, 0), "x", "y", "z"]``.

        :param layout: per-coordinate layout as described above.
        :raises ValueError: if the free axes are not exactly ``{x, y, z}``.
        """
        free: Dict[str, int] = {}
        dependent: Dict[int, Tuple[float, float, float]] = {}
        for idx, entry in enumerate(layout):
            if isinstance(entry, str) and entry.lower() in ("x", "y", "z"):
                free[entry.lower()] = idx
            else:
                coeffs = tuple(float(c) for c in entry)
                if len(coeffs) != 3:
                    raise ValueError(
                        f"Dependent coordinate {idx} needs 3 coefficients, got {entry!r}."
                    )
                dependent[idx] = coeffs  # type: ignore[assignment]
        if set(free) != {"x", "y", "z"}:
            raise ValueError(
                f"Layout must mark exactly one each of x, y, z; got {sorted(free)}."
            )
        return cls(axes=(free["x"], free["y"], free["z"]), dependent=dependent)

    def free_to_full(self, xyz: np.ndarray) -> np.ndarray:
        """Embed sphere points ``(x, y, z)`` back into full ``D``-dim space.

        Free axes receive ``x/y/z`` directly; dependent axes are filled with
        their linear prediction ``a·x + b·y + c·z``.  Used to test the
        reconstructed grid against the shard's constraint matrix ``A``.

        :param xyz: ``(N, 3)`` array of sphere points.
        :return: ``(N, D)`` array in the CMF coordinate frame.
        """
        n = len(self.axes) + len(self.dependent)
        full = np.zeros((xyz.shape[0], n), dtype=float)
        for slot, ax in enumerate(self.axes):
            full[:, ax] = xyz[:, slot]
        for idx, (a, b, c) in self.dependent.items():
            full[:, idx] = a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2]
        return full

    def project(self, directions: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
        """Select trajectories in the subspace and return their sphere points.

        :param directions: ``(N, D)`` raw direction vectors (CMF-symbol order).
        :param tol: relative tolerance for the dependent-coordinate constraint.
        :return: ``(unit_xyz, mask)`` where ``mask`` is the boolean row-filter of
            kept trajectories and ``unit_xyz`` is their ``(M, 3)`` unit-sphere
            projection (``M = mask.sum()``).
        """
        x = directions[:, self.axes[0]]
        y = directions[:, self.axes[1]]
        z = directions[:, self.axes[2]]

        scale = np.linalg.norm(directions, axis=1)
        scale[scale == 0] = 1.0

        mask = np.ones(directions.shape[0], dtype=bool)
        for idx, (a, b, c) in self.dependent.items():
            predicted = a * x + b * y + c * z
            mask &= np.abs(directions[:, idx] - predicted) <= tol * scale

        xyz = np.column_stack([x, y, z])[mask]
        norms = np.linalg.norm(xyz, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return xyz / norms, mask


# ===========================================================================
# Data loading (JSONL pipeline)
# ===========================================================================

def load_shards(export_cmfs: str, constant: Constant):
    """Reconstruct live ``Shard`` objects for *constant* from the JSONL export.

    Thin wrapper over :func:`load_shards_from_export` that returns just the
    list of shards for the single constant (or an empty list).

    :param export_cmfs: the ``EXPORT_CMFS`` directory (per-constant formatter
        JSONs + ``<cmf>__shards.jsonl`` files).
    :param constant: the constant whose shards to load.
    :return: list of reconstructed ``Shard`` objects.
    """
    by_const = load_shards_from_export(export_cmfs, [constant])
    shards = by_const.get(constant, [])
    print(f"Loaded {len(shards)} shards for constant {constant.name!r}.")
    return shards


def _merge_jsonl(path: Path) -> Dict[str, dict]:
    """Read a per-shard JSONL, folding patch lines into base records.

    Mirrors ``examples/search_data.py._merge_jsonl`` (last-write-wins on
    scalar fields, union on ``extended_metrics``) so the same merge semantics
    as the rest of the tooling are used.

    :param path: the ``<shard_id>.jsonl`` file.
    :return: ``{trajectory_id: merged_record}``.
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
                merged[tid].update(record)
    return merged


def load_shard_trajectories(
    shard,
    value_fn: "ValueFn",
    results_root: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a shard's trajectory directions + a scalar value from its JSONL file.

    The file is located by ``derive_cmf_and_shard_ids(shard)`` → ``<shard_id>``
    under *results_root* (the flat ``EXPORT_SEARCH_RESULTS`` layout).  The scalar
    plotted on the sphere is produced by *value_fn* (e.g. :func:`delta_value`,
    :func:`extended_metric_value`, :func:`sympy_attribute_value`); records for
    which it returns ``None`` / a non-finite number are skipped, so attributes
    that are absent or not-yet-computed on some trajectories are dropped cleanly.

    :param shard: a reconstructed ``Shard``.
    :param value_fn: callable ``(record_dict) -> Optional[float]`` extracting the
        scalar to colour the sphere with.
    :param results_root: the ``EXPORT_SEARCH_RESULTS`` directory.
    :return: ``(directions, values)`` — an ``(N, D)`` float array of raw
        directions and an ``(N,)`` array of values (empty arrays if none).
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return np.empty((0, len(shard.symbols))), np.empty((0,))

    merged = _merge_jsonl(path)
    directions: List[List[float]] = []
    values: List[float] = []
    for rec in merged.values():
        direction = rec.get("direction")
        if direction is None:
            continue
        try:
            value = value_fn(rec)
        except Exception:
            value = None
        if value is None or not np.isfinite(value):
            continue
        directions.append([float(v) for v in direction])
        values.append(float(value))

    if not directions:
        return np.empty((0, len(shard.symbols))), np.empty((0,))
    return np.asarray(directions, dtype=float), np.asarray(values, dtype=float)


def load_path_directions(
    shard,
    results_root: str,
) -> np.ndarray:
    """Load an *ordered* list of direction vectors for a shard's overlay path.

    Used for the "second results directory" overlay mode (e.g. a gradient-ascent
    run): the trajectories in that shard's JSONL are returned **in file order**,
    forming the polyline drawn on top of the δ field.

    :param shard: a reconstructed ``Shard``.
    :param results_root: the second ``EXPORT_SEARCH_RESULTS`` directory holding
        the path trajectories.
    :return: ``(K, D)`` array of raw direction vectors in traversal order
        (empty if the file is absent).
    """
    _, shard_id, _ = derive_cmf_and_shard_ids(shard)
    path = Path(results_root) / f"{shard_id}.jsonl"
    if not path.is_file():
        return np.empty((0, len(shard.symbols)))

    # Preserve first-seen order (insertion-ordered dict from the merge).
    merged = _merge_jsonl(path)
    dirs = [
        [float(v) for v in rec["direction"]]
        for rec in merged.values()
        if rec.get("direction") is not None
    ]
    return np.asarray(dirs, dtype=float) if dirs else np.empty((0, len(shard.symbols)))


# ===========================================================================
# Geometry helpers (carried over from the old script)
# ===========================================================================

def get_rotation_matrix(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Rotation aligning ``vec1`` onto ``vec2`` (Rodrigues' formula).

    Handles the parallel / antiparallel degenerate cases explicitly so the
    180° rotation preserves the right-hand rule.

    :param vec1: source vector.
    :param vec2: target vector.
    :return: a ``3×3`` rotation matrix ``R`` with ``R·vec1 ∥ vec2``.
    """
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    c = np.dot(a, b)
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        ortho = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))


def _camera_vector(elev: float, azim: float) -> np.ndarray:
    """Unit view direction for a matplotlib 3-D camera at ``(elev, azim)`` deg."""
    e, a = np.radians(elev), np.radians(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_sphere_horizon(ax, cam_elev: float, cam_azim: float) -> None:
    """Draw the silhouette circle of the sphere as seen from the camera.

    :param ax: a 3-D matplotlib axis.
    :param cam_elev: camera elevation in degrees.
    :param cam_azim: camera azimuth in degrees.
    """
    cam = _camera_vector(cam_elev, cam_azim)
    u = (np.array([-cam[1], cam[0], 0]) if abs(cam[0]) > 0.1 or abs(cam[1]) > 0.1
         else np.array([0, -cam[2], cam[1]]))
    u = u / np.linalg.norm(u)
    v = np.cross(cam, u)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        1.001 * (np.cos(t) * u[0] + np.sin(t) * v[0]),
        1.001 * (np.cos(t) * u[1] + np.sin(t) * v[1]),
        1.001 * (np.cos(t) * u[2] + np.sin(t) * v[2]),
        color="black", linewidth=1.5, alpha=0.8, zorder=2,
    )


def plot_hyperplanes_on_sphere(ax, shard, spec: ProjectionSpec,
                               cam_elev: float, cam_azim: float) -> None:
    """Draw the great circles where the shard's bounding hyperplanes meet the sphere.

    Each hyperplane normal (a row of ``shard.A``) is projected onto the three
    chosen coordinate axes (``spec.axes``); the camera-facing half of the
    resulting great circle is drawn.

    :param ax: a 3-D matplotlib axis.
    :param shard: the shard whose constraint matrix ``A`` supplies the normals.
    :param spec: the active projection spec (selects the 3 axes).
    :param cam_elev: camera elevation in degrees.
    :param cam_azim: camera azimuth in degrees.
    """
    if shard.A is None:
        return
    A = np.array(shard.A, dtype=float)
    cam = _camera_vector(cam_elev, cam_azim)
    theta = np.linspace(0, 2 * np.pi, 300)

    for row in A:
        n = row[list(spec.axes)]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-8:
            continue
        N = n / norm_n
        U = (np.array([-N[1], N[0], 0.0]) if abs(N[0]) > 0.1 or abs(N[1]) > 0.1
             else np.array([0.0, -N[2], N[1]]))
        U /= np.linalg.norm(U)
        V = np.cross(N, U)
        cx, cy, cz = (1.001 * (np.cos(theta) * U[i] + np.sin(theta) * V[i]) for i in range(3))
        pts = np.vstack([cx, cy, cz]).T
        hidden = pts.dot(cam) < -0.1
        cx[hidden], cy[hidden], cz[hidden] = np.nan, np.nan, np.nan
        ax.plot(cx, cy, cz, color="black", linewidth=1.5, alpha=0.85, zorder=5)


# ===========================================================================
# Core surface rendering
# ===========================================================================

def _best_camera(unit_xyz: np.ndarray, deltas: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Camera (elev, azim) pointing at the highest-δ direction.

    :param unit_xyz: ``(N, 3)`` unit-sphere points.
    :param deltas: ``(N,)`` δ values.
    :return: ``(elev_deg, azim_deg, v_best)``.
    """
    v_best = unit_xyz[int(np.nanargmax(deltas))]
    elev = float(np.degrees(np.arcsin(np.clip(v_best[2], -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(v_best[1], v_best[0])))
    return elev, azim, v_best


def render_shard_surface(
    ax,
    shard,
    unit_xyz: np.ndarray,
    deltas: np.ndarray,
    spec: ProjectionSpec,
    cmap,
    norm,
    *,
    grid_res: int = 400,
    cone_tol: float = 1e-4,
    knn_trim: float = 0.18,
) -> None:
    """Interpolate and draw one shard's δ field as a coloured patch on the sphere.

    Reproduces the old ``generate_shard_atlas2`` pipeline: rotate the best
    direction to the equator for a well-conditioned local ``griddata``, restore
    the surface to the global frame, trim to the shard cone (``A·v ≤ 0`` on the
    embedded full-dim points) and to a KNN radius around real samples, then
    ``plot_surface`` with per-face colours.

    :param ax: a 3-D matplotlib axis (already created).
    :param shard: the shard (supplies ``A`` for cone trimming).
    :param unit_xyz: ``(N, 3)`` unit-sphere projection of the kept trajectories.
    :param deltas: ``(N,)`` δ values aligned with ``unit_xyz``.
    :param spec: the active projection spec.
    :param cmap: a matplotlib colormap.
    :param norm: a matplotlib ``Normalize`` shared across all spheres.
    :param grid_res: interpolation grid resolution per axis.
    :param cone_tol: tolerance for the ``A·v ≤ 0`` cone-membership trim.
    :param knn_trim: drop grid points farther than this from any real sample.
    """
    _, _, v_best = _best_camera(unit_xyz, deltas)
    R = get_rotation_matrix(v_best, np.array([1.0, 0.0, 0.0]))
    rotated = unit_xyz @ R.T

    dt = np.arctan2(rotated[:, 1], rotated[:, 0])
    dp = np.arccos(np.clip(rotated[:, 2], -1, 1))
    gt, gp = np.mgrid[
        np.min(dt) - 0.1:np.max(dt) + 0.1:grid_res * 1j,
        np.min(dp) - 0.1:np.max(dp) + 0.1:grid_res * 1j,
    ]

    grid_delta = griddata((dt, dp), deltas, (gt, gp), method="linear")
    grid_delta = np.where(
        np.isnan(grid_delta),
        griddata((dt, dp), deltas, (gt, gp), method="nearest"),
        grid_delta,
    )

    gx = np.cos(gt) * np.sin(gp)
    gy = np.sin(gt) * np.sin(gp)
    gz = np.cos(gp)
    grid_pts = np.c_[gx.ravel(), gy.ravel(), gz.ravel()] @ R  # back to global frame

    # --- trim to the shard cone (embed sphere pts into full CMF dim first) ---
    if shard.A is not None:
        A = np.array(shard.A, dtype=float)
        A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
        full = spec.free_to_full(grid_pts)
        grid_delta.ravel()[np.any(full @ A_norm.T > cone_tol, axis=1)] = np.nan

    # --- trim to a neighbourhood of the real samples ---
    grid_delta.ravel()[cKDTree(unit_xyz).query(grid_pts)[0] > knn_trim] = np.nan

    colors = cmap(norm(grid_delta))
    colors[np.isnan(grid_delta), 3] = 0.0
    sx, sy, sz = (grid_pts[:, i].reshape(grid_res, grid_res) for i in range(3))
    ax.plot_surface(
        sx, sy, sz, facecolors=colors, shade=False, antialiased=True,
        rcount=grid_res, ccount=grid_res, zorder=5,
    )


def _draw_reference_grid(ax) -> None:
    """Draw a faint static lat/long wireframe (world-z up, 10° spacing)."""
    u = np.linspace(0, 2 * np.pi, 37)
    v = np.linspace(0, np.pi, 19)
    ax.plot_surface(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones_like(u), np.cos(v)),
        color="white", alpha=0.0, edgecolor="gray",
        linewidth=0.35, shade=False, zorder=1,
    )


def draw_overlay_path(ax, shard, path_dirs: np.ndarray, spec: ProjectionSpec,
                      tol: float, color: str = "black") -> None:
    """Project an ordered path of directions onto the sphere and draw it.

    Only path points that lie in the chosen subspace (per ``spec``) are drawn;
    they are placed at radius 1.01 so the line and markers sit just above the
    coloured δ surface.

    :param ax: a 3-D matplotlib axis.
    :param shard: the shard the path belongs to (unused for now, kept for API
        symmetry / future per-shard styling).
    :param path_dirs: ``(K, D)`` ordered direction vectors.
    :param spec: the active projection spec.
    :param tol: subspace membership tolerance.
    :param color: line / marker colour.
    """
    if path_dirs.size == 0:
        return
    unit_xyz, _ = spec.project(path_dirs, tol)
    if unit_xyz.shape[0] == 0:
        return
    pts = unit_xyz * 1.01
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=1.6,
            marker="o", markersize=3.0, zorder=10)
    # Mark the endpoint (the optimum the path climbed to).
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color=color, s=28,
               edgecolor="white", linewidth=0.6, zorder=11)


# ===========================================================================
# Top-level figures
# ===========================================================================

def _nice_step(value_range: float) -> float:
    """Pick a human-friendly colorbar tick step (~5 ticks) for a value range."""
    if value_range <= 0 or not np.isfinite(value_range):
        return 1.0
    raw = value_range / 5.0
    mag = 10 ** np.floor(np.log10(raw))
    for mult in (1, 2, 5, 10):
        if raw <= mult * mag:
            return mult * mag
    return 10 * mag


def _make_norm(values: np.ndarray, step: Optional[float]):
    """Build a shared ``Normalize`` and the tick step for the colorbar.

    :param values: all plotted scalars across every shard.
    :param step: explicit tick step; ``None`` → auto via :func:`_nice_step`.
    :return: ``(norm, vmin, vmax, step)`` with bounds snapped to ``step``.
    """
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if step is None:
        step = _nice_step(hi - lo)
    vmin = np.floor(lo / step) * step
    vmax = np.ceil(hi / step) * step
    if vmin == vmax:  # constant field — widen so the colorbar is valid
        vmin, vmax = vmin - step, vmax + step
    return plt.Normalize(vmin=vmin, vmax=vmax), vmin, vmax, step


def _add_colorbar(fig, cmap, norm, vmin: float, vmax: float,
                  step: float, label: str) -> None:
    """Add the shared vertical colorbar on the right of the figure.

    :param label: the colorbar axis label (e.g. the δ label or an attribute name).
    :param step: spacing between colorbar ticks.
    """
    cbar_ax = fig.add_axes([0.87, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax)
    ticks = np.arange(vmin, vmax + step * 0.05, step)
    cbar.set_ticks(ticks)
    decimals = max(0, int(np.ceil(-np.log10(step)))) if step < 1 else 1
    cbar.ax.set_yticklabels([f"{t:.{decimals}f}" for t in ticks], fontsize=13)
    cbar.set_label(label, fontsize=16, labelpad=15)


def generate_spheres(
    shards,
    constant: Constant,
    results_root: str,
    *,
    one_sphere_per_shard: bool = True,
    spec: Optional[ProjectionSpec] = None,
    subspace_tol: float = 1e-6,
    value_fn: Optional[ValueFn] = None,
    value_label: Optional[str] = None,
    value_step: Optional[float] = None,
    path_root: Optional[str] = None,
    explicit_paths: Optional[Dict[str, np.ndarray]] = None,
    draw_grid: bool = True,
    cmap_name: str = "coolwarm",
    show: bool = True,
):
    """Render the attribute-on-sphere projection(s) for a set of shards.

    By default the irrationality measure δ is drawn (classic behaviour).  Pass a
    *value_fn* to colour the sphere by any other trajectory attribute instead —
    e.g. ``field_value("limit_value")``, ``extended_metric_value("digits_per_step")``,
    or ``sympy_attribute_value("asymptotics", {"n": 1e6})`` for a string-formatted
    sympy attribute reduced to a number by substitution.

    :param shards: reconstructed ``Shard`` objects (e.g. from :func:`load_shards`).
    :param constant: the constant; used for the subspace default and for the
        default δ value function.
    :param results_root: ``EXPORT_SEARCH_RESULTS`` dir for the hedgehog data.
    :param one_sphere_per_shard: ``True`` → one sphere per shard (atlas, like the
        old script); ``False`` → all shards on a single shared sphere.
    :param spec: projection spec; defaults to the identity ``(0, 1, 2)`` spec for
        a 3-D CMF.  Required (non-identity) for ``D > 3``.
    :param subspace_tol: relative tolerance for the subspace membership test.
    :param value_fn: scalar extractor (see :data:`ValueFn`); ``None`` →
        :func:`delta_value` for *constant*.
    :param value_label: colorbar label; ``None`` → the δ label (or the value
        function's name when a custom one is supplied).
    :param value_step: explicit colorbar tick step; ``None`` → auto-chosen
        (δ keeps its conventional 0.2 step).
    :param path_root: optional second ``EXPORT_SEARCH_RESULTS`` dir whose
        trajectories are drawn as an ordered overlay path per shard.
    :param explicit_paths: optional ``{shard_id: (K, D) directions}`` overriding /
        supplementing ``path_root`` for specific shards.
    :param draw_grid: draw the faint reference lat/long wireframe.
    :param cmap_name: matplotlib colormap name.
    :param show: call ``plt.show()`` before returning.
    :return: the matplotlib ``Figure``.
    """
    explicit_paths = explicit_paths or {}
    cmap = plt.get_cmap(cmap_name)

    # Default to the classic δ field; keep its conventional label + 0.2 ticks.
    if value_fn is None:
        value_fn = delta_value(constant)
        if value_label is None:
            value_label = r"Irrationality Measure ($\delta$)"
        if value_step is None:
            value_step = 0.2
    elif value_label is None:
        value_label = "Trajectory attribute"

    # ---- gather per-shard (unit_xyz, values) using the projection spec -------
    per_shard = []
    for shard in shards:
        if spec is None:
            spec = ProjectionSpec.identity(len(shard.symbols))
        directions, values = load_shard_trajectories(shard, value_fn, results_root)
        if directions.shape[0] == 0:
            continue
        unit_xyz, mask = spec.project(directions, subspace_tol)
        if unit_xyz.shape[0] < 4:  # need a few points to interpolate
            continue
        per_shard.append((shard, unit_xyz, values[mask]))

    if not per_shard:
        raise RuntimeError("No shard had enough in-subspace trajectories to plot.")

    all_values = np.concatenate([d for _, _, d in per_shard])
    norm, vmin, vmax, value_step = _make_norm(all_values, value_step)

    def _resolve_path(shard) -> np.ndarray:
        _, shard_id, _ = derive_cmf_and_shard_ids(shard)
        if shard_id in explicit_paths:
            return np.asarray(explicit_paths[shard_id], dtype=float)
        if path_root is not None:
            return load_path_directions(shard, path_root)
        return np.empty((0, len(shard.symbols)))

    if one_sphere_per_shard:
        fig = _generate_atlas(per_shard, spec, cmap, norm, subspace_tol,
                              draw_grid, _resolve_path)
    else:
        fig = _generate_single(per_shard, spec, cmap, norm, subspace_tol,
                               draw_grid, _resolve_path)

    _add_colorbar(fig, cmap, norm, vmin, vmax, value_step, value_label)
    if show:
        plt.show()
    return fig


def _generate_atlas(per_shard, spec, cmap, norm, tol, draw_grid, resolve_path):
    """Render one sphere per shard in a single row (old ``generate_shard_atlas2``)."""
    n = len(per_shard)
    fig = plt.figure(figsize=(4.2 * n + 1.5, 4.2), dpi=300)
    for idx, (shard, unit_xyz, deltas) in enumerate(per_shard):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        ax.set_proj_type("persp")
        ax.set_box_aspect((1, 1, 1), zoom=1.32)

        render_shard_surface(ax, shard, unit_xyz, deltas, spec, cmap, norm)
        if draw_grid:
            _draw_reference_grid(ax)

        elev, azim, _ = _best_camera(unit_xyz, deltas)
        draw_sphere_horizon(ax, elev, azim)
        plot_hyperplanes_on_sphere(ax, shard, spec, elev, azim)
        draw_overlay_path(ax, shard, resolve_path(shard), spec, tol)

        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")

    plt.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.03, wspace=-0.15)
    return fig


def _generate_single(per_shard, spec, cmap, norm, tol, draw_grid, resolve_path):
    """Render every shard on one shared sphere (single global camera)."""
    fig = plt.figure(figsize=(6.0, 5.0), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 1), zoom=1.32)

    # Global camera = direction of the best δ across all shards.
    global_xyz = np.concatenate([u for _, u, _ in per_shard])
    global_delta = np.concatenate([d for _, _, d in per_shard])
    elev, azim, _ = _best_camera(global_xyz, global_delta)

    if draw_grid:
        _draw_reference_grid(ax)
    draw_sphere_horizon(ax, elev, azim)

    for shard, unit_xyz, deltas in per_shard:
        render_shard_surface(ax, shard, unit_xyz, deltas, spec, cmap, norm)
        plot_hyperplanes_on_sphere(ax, shard, spec, elev, azim)
        draw_overlay_path(ax, shard, resolve_path(shard), spec, tol)

    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    plt.subplots_adjust(left=0.01, right=0.85, top=0.97, bottom=0.03)
    return fig


# ===========================================================================
# Example entry point
# ===========================================================================

if __name__ == "__main__":
    # Paths mirror the on-disk example layout (see examples/CMFs + the flat
    # "examples/search results" dir).  Adjust to your run's export dirs.
    HERE = Path(__file__).resolve().parent
    EXPORT_CMFS = str(HERE / ".." / "examples" / "CMFs")
    SEARCH_RESULTS = str(HERE / ".." / "examples" / "search results")

    # The example data searches for log(2).  Constants are normally registered
    # by the loading stage; for a standalone run we register it directly.  Its
    # ``name`` must match the key used in the JSONL ``delta_estimate`` dicts.
    import sympy as sp
    const = Constant.registry.get("log-2") or Constant("log-2", sp.log(2))
    shards = load_shards(EXPORT_CMFS, const)

    # 3-D CMF → identity spec.  For a 4-D CMF you would pass e.g.
    #   spec=ProjectionSpec.from_layout(["x", "y", "z", (1, -1, 0)])
    generate_spheres(
        shards,
        const,
        SEARCH_RESULTS,
        one_sphere_per_shard=True,   # set False to overlay all shards on one sphere
        spec=None,                   # auto identity for D == 3
        path_root=None,              # set to a gradient-ascent results dir to overlay
        # --- colour by δ (default).  To colour by another attribute instead: ---
        #   value_fn=field_value("limit_value"), value_label="limit",
        #   value_fn=extended_metric_value("digits_per_step"), value_label="digits/step",
        #   value_fn=sympy_attribute_value("asymptotics", {"n": 1e6}),
        #       value_label="asymptotics(n=1e6)",
    )
