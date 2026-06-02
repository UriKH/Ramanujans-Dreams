"""
Flatland geometry helper for Small Angle Search.

Isolates all of the "flatland" linear algebra so the search method itself stays
focused on the hill-climb logic.  Flatland is the lower-dimensional integer
lattice produced by :class:`HyperSpaceConditioner` from a shard's constraint
matrix ``A``: equality directions are collapsed and the remaining basis is
LLL/BKZ-reduced so that small integer steps correspond to small geometric angles.

A flatland coordinate vector ``z`` (length ``d_flat``) maps to a real-space
trajectory direction via ``v = Z_reduced @ z`` (length ``d_orig``).  Perturbation
and length-doubling happen on ``z``; attribute computation happens on ``v``.
"""

from typing import Iterator, List

import numpy as np
import sympy as sp
from ramanujantools import Position

from dreamer.extraction.samplers.conditioner import HyperSpaceConditioner
from dreamer.extraction.utils.fast_gcd import reduce_to_primitive
from dreamer.extraction.shard import Shard


class FlatlandGeometry:
    """Convert / check / perturb trajectory directions in flatland space."""

    def __init__(self, shard: Shard):
        """
        :param shard: The shard whose constraint geometry defines flatland space.
        """
        self.shard = shard
        self.symbols: List[sp.Symbol] = list(shard.symbols)
        d_orig = len(self.symbols)

        if shard.is_whole_space or shard.A is None:
            # No constraints: flatland is the full integer space, basis is identity.
            self.Z_reduced = np.eye(d_orig, dtype=np.int64)
            self.B_reduced = np.empty((0, d_orig))
        else:
            conditioner = HyperSpaceConditioner(np.asarray(shard.A, dtype=np.float64))
            self.Z_reduced, self.B_reduced, _ = conditioner.process()

        self.d_flat = self.Z_reduced.shape[1]

    # ------------------------------------------------------------------
    # Conversions
    # ------------------------------------------------------------------

    def to_real(self, z: np.ndarray) -> Position:
        """
        Map a flatland direction ``z`` to a real-space trajectory ``Position``.
        :param z: Integer flatland coordinate vector (length ``d_flat``).
        :return: Position over the shard's symbols (sympy Integers).
        """
        v = self.Z_reduced @ np.asarray(z, dtype=np.int64)
        return Position({sym: sp.Integer(int(val)) for sym, val in zip(self.symbols, v)})

    def to_flatland(self, v: Position) -> np.ndarray:
        """
        Recover the flatland coordinates of a real-space direction.

        ``v`` lies in the integer lattice spanned by the columns of
        ``Z_reduced``, so ``Z_reduced @ z = v`` has an exact integer solution,
        recovered via the least-squares pseudo-inverse and rounded.

        :param v: Real-space trajectory direction (Position over shard symbols).
        :return: Integer flatland coordinate vector (length ``d_flat``).
        """
        v_vec = np.array([float(v[sym]) for sym in self.symbols], dtype=np.float64)
        Z = self.Z_reduced.astype(np.float64)
        z = np.linalg.solve(Z.T @ Z, Z.T @ v_vec)
        return np.rint(z).astype(np.int64)

    # ------------------------------------------------------------------
    # Membership + perturbation
    # ------------------------------------------------------------------

    def is_inside(self, z: np.ndarray) -> bool:
        """
        :param z: Integer flatland direction.
        :return: True iff the real direction stays inside the shard cone.
        """
        return self.shard.is_valid_trajectory(self.to_real(z))

    def perturbations(self, z: np.ndarray) -> Iterator[np.ndarray]:
        """
        Yield the ``2 * d_flat`` neighbours of ``z`` (each coordinate ±1),
        each reduced to a primitive (GCD == 1) vector.  Zero vectors are skipped.
        :param z: Integer flatland direction to perturb.
        """
        z = np.asarray(z, dtype=np.int64)
        for i in range(self.d_flat):
            for sign in (1, -1):
                cand = z.copy()
                cand[i] += sign
                if not np.any(cand):
                    continue
                yield reduce_to_primitive(cand)
