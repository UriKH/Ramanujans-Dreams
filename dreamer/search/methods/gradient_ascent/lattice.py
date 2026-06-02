"""
Lattice realization helpers for Gradient Ascent.

The optimizer operates on a *real-valued* direction; delta is continuous in the
direction's angle.  To evaluate delta we must realize a real direction as an
integer-coordinate trajectory.  :func:`snap_to_trajectory` returns the integer
flatland direction whose angle best matches a real direction, subject to an L2
length cap (so trajectories stay short / cheap) and shard-cone membership.

:func:`rotate_toward` produces the small angular perturbations used to estimate
the gradient by forward differences in angle space.
"""

from typing import Optional

import numpy as np

from dreamer.search.methods.flatland.geometry import FlatlandGeometry


def rotate_toward(d: np.ndarray, axis: int, angle: float) -> np.ndarray:
    """
    Rotate direction *d* by *angle* radians toward basis axis *axis*, preserving length.

    The rotation happens in the plane spanned by ``d`` and the unit basis vector
    ``e_axis``.  If that axis is (nearly) parallel to ``d`` the rotation is a
    no-op (the plane is degenerate) and ``d`` is returned unchanged.

    :param d: Real direction vector (length ``dim``).
    :param axis: Index of the coordinate basis vector to rotate toward.
    :param angle: Rotation angle in radians.
    :return: The rotated direction (same L2 norm as ``d``).
    """
    d = np.asarray(d, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm == 0.0:
        return d.copy()

    u = d / norm
    e = np.zeros_like(u)
    e[axis] = 1.0

    perp = e - np.dot(e, u) * u
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-12:
        return d.copy()  # axis parallel to d — no well-defined rotation plane.
    perp_unit = perp / perp_norm

    rotated_unit = np.cos(angle) * u + np.sin(angle) * perp_unit
    return rotated_unit * norm


def snap_to_trajectory(
    d: np.ndarray,
    geom: FlatlandGeometry,
    max_norm: float,
) -> Optional[np.ndarray]:
    """
    Realize a real direction *d* as the angle-best, length-capped, in-cone integer direction.

    Scans candidate integer directions obtained by scaling the unit direction to
    every integer length up to ``max_norm`` and rounding; keeps those with
    ``||z||_2 <= max_norm`` that lie inside the shard cone, and returns the one
    maximizing the cosine similarity to ``d`` (i.e. minimizing the angle).

    :param d: Real direction vector (length ``geom.d_flat``).
    :param geom: Flatland geometry providing the cone-membership test.
    :param max_norm: Maximum L2 norm of the returned integer direction.
    :return: The best integer flatland direction, or ``None`` if no in-cone,
        non-zero candidate exists within the length cap.
    """
    d = np.asarray(d, dtype=np.float64)
    d_norm = np.linalg.norm(d)
    if d_norm == 0.0:
        return None

    unit = d / d_norm

    # Candidate integer directions: round ``unit * L`` for every integer length
    # up to the cap, in one vectorised pass (trajectories are always integers).
    lengths = np.arange(1, int(np.floor(max_norm)) + 1, dtype=np.int64)
    candidates = np.rint(np.outer(lengths, unit)).astype(np.int64)  # (L, d)

    norms = np.linalg.norm(candidates, axis=1)
    keep = (norms > 0.0) & (norms <= max_norm)
    if not np.any(keep):
        return None
    candidates = candidates[keep]

    # Dedup identical rounded directions (small L collapse to the same vector).
    candidates = np.unique(candidates, axis=0)

    # Keep only in-cone candidates (single batched NumPy cone test).
    inside = geom.is_inside_many(candidates)
    if not np.any(inside):
        return None
    candidates = candidates[inside]

    # Pick the angle-best candidate: maximal cosine similarity to ``d``.
    cos = (candidates @ unit) / np.linalg.norm(candidates, axis=1)
    return candidates[int(np.argmax(cos))].astype(np.int64)
