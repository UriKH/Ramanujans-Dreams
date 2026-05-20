from __future__ import annotations

import hashlib
import json
import warnings
from typing import Optional, TYPE_CHECKING

import sympy as sp
from sympy.abc import n

from ramanujantools import LinearRecurrence, Matrix, Limit

if TYPE_CHECKING:
    from dreamer.utils.storage.dtos import TrajectoryDTO


# ---------------------------------------------------------------------------
# Module-level helpers — stable IDs and position conversion
# ---------------------------------------------------------------------------

def _stable_id(*parts: str, length: int = 16) -> str:
    """SHA-256 of pipe-joined parts, truncated to ``length`` hex chars.

    Deterministic across runs and processes (unlike Python's built-in ``hash``).
    """
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:length]


def _position_to_tuple(pos) -> tuple:
    """Convert a ramanujantools.Position (dict-like, may hold sympy Integers)
    to a plain tuple of JSON-serializable ints (or str fallback).
    """
    out = []
    for v in pos.values():
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            out.append(str(v))
    return tuple(out)


def _serialize_inequalities(shard) -> str:
    """Canonical string representation of the shard's ``Ax < b`` system.

    Converts each matrix/vector entry to a plain Python ``int`` before
    serialising — the numpy arrays may hold SymPy objects (e.g. ``NegativeOne``)
    that are not JSON-serialisable on their own.  The resulting JSON string is
    stable across Python sessions and independent of ``Shard.__str__``.
    Whole-space shards (``shard.A is None``) produce a fixed placeholder.

    Rows are sorted lexicographically (each row is [a1, ..., ak, b]) so that
    the canonical string is independent of the order in which the extractor
    enumerates hyperplanes between runs.
    """
    if shard.is_whole_space or shard.A is None or shard.b is None:
        return "whole_space"
    rows = sorted(
        [int(x) for x in row] + [int(shard.b[i])]
        for i, row in enumerate(shard.A.tolist())
    )
    return json.dumps(rows)


def derive_cmf_and_shard_ids(shard) -> tuple[str, str, str]:
    """Return ``(cmf_id, shard_id, shard_encoding_str)`` for *shard*.

    * ``cmf_id`` — the CMF name (unique per CMF in the current system).
    * ``shard_id`` — stable SHA-256 of ``(cmf_name, shard_encoding_str)``.
    * ``shard_encoding_str`` — canonical ``Ax < b`` string, also used as
      part of trajectory ids so the two levels stay consistent.
    """
    cmf_id = shard.cmf_name
    shard_encoding_str = _serialize_inequalities(shard)
    shard_id = _stable_id(cmf_id, shard_encoding_str)
    return cmf_id, shard_id, shard_encoding_str


# ---------------------------------------------------------------------------
# DTO factory
# ---------------------------------------------------------------------------

def build_trajectory_dto(
    handler: "TrajectoryAttributesHandler",
    *,
    cmf_id: str,
    shard_id: str,
    cmf_name: str,
    shard_encoding_str: str,
    start,
    direction,
) -> "TrajectoryDTO":
    """Build a ``TrajectoryDTO`` carrying Tier-1 attributes from a handler.

    The ``trajectory_id`` is deterministic: SHA-256 of
    ``(cmf_name, shard_encoding_str, start_tuple, direction_tuple)``.

    ``extended_metrics`` is left empty; background workers compute the
    asynchronous (Tier-2) attributes after the DTO is enqueued.

    Parameters
    ----------
    handler:
        A ``TrajectoryAttributesHandler`` constructed from the trajectory.
    cmf_id, shard_id:
        Identifiers for the parent CMF and shard.
    cmf_name, shard_encoding_str:
        Used together to build the deterministic trajectory id.
    start, direction:
        The ``ramanujantools.Position`` objects for this trajectory.
    """
    from dreamer.utils.storage.dtos import TrajectoryDTO  # lazy import avoids circular dep

    start_t = _position_to_tuple(start)
    dir_t = _position_to_tuple(direction)
    trajectory_id = _stable_id(cmf_name, shard_encoding_str, str(start_t), str(dir_t))
    return TrajectoryDTO(
        trajectory_id=trajectory_id,
        cmf_id=cmf_id,
        shard_id=shard_id,
        start_point=start_t,
        direction=dir_t,
        recurrence_relation=handler.formula_str(),
        recurrence_order=handler.order(),
        limit_value=float(handler.limit()),
        delta_estimate=float(handler.delta()),
        p_vector=handler.p_vector(),
        q_vector=handler.q_vector(),
    )


class TrajectoryAttributesHandler:
    """
    Lazy-computed container for a recurrence relation extracted from a
    CMF trajectory matrix.

    Nothing is computed at __init__. Each method computes on first call
    and caches the result.
    """

    # ------------------------------------------------------------------
    #  Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        traj_matrix: Matrix,
        walk_depth: int = 200,
    ):
        """
        Parameters
        ----------
        traj_matrix : ramanujantools.Matrix
            The symbolic d×d trajectory matrix M(n).
            Should already have z (and other free params) substituted
            if numeric computation is desired.
        walk_depth : int
            Default number of recurrence steps.
        """
        self._traj = traj_matrix
        self._depth = walk_depth
        self._cache: dict = {}

    @classmethod
    def from_cmf(
        cls,
        cmf,
        trajectory,
        start_point,
        z_value=None,
        walk_depth: int = 200,
    ) -> "TrajectoryAttributesHandler":
        """
        Build from a CMF, trajectory direction, and start point.
        """
        tmat = cmf.trajectory_matrix(trajectory, start_point)
        if z_value is not None:
            tmat = tmat.subs({sp.Symbol("z"): z_value})
        return cls(tmat, walk_depth=walk_depth)

    # ------------------------------------------------------------------
    #  Cache helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def clear_cache(self):
        self._cache.clear()

    def computed_attributes(self) -> list:
        return list(self._cache.keys())

    # ==================================================================
    #  TRAJECTORY MATRIX
    # ==================================================================

    def trajectory_matrix(self) -> Matrix:
        """The raw d×d symbolic trajectory matrix M(n)."""
        return self._traj

    def traj_size(self) -> int:
        """Dimension d of the trajectory matrix."""
        return self._traj.shape[0]

    def traj_rank(self, at_n: int = 5) -> int:
        """Rank of M(n) evaluated at a specific n (default 5)."""
        return self._get(f"traj_rank@{at_n}", lambda:
            self._traj.subs({n: at_n}).rank()
        )

    # ==================================================================
    #  LINEAR RECURRENCE  (the core object from ramanujantools)
    # ==================================================================

    def linear_recurrence(self) -> LinearRecurrence:
        """
        The LinearRecurrence object built from the trajectory matrix.

        This is the central object. It wraps the trajectory matrix by:
        1. Calling as_companion() to get the companion matrix
        2. Reading the last column (reversed) to get the relation coefficients
        3. Storing as relation = [a_0(n), a_1(n), ..., a_d(n)]
           where Σ a_i(n) · f(n-i) = 0

        All downstream attributes (recurrence_matrix, kamidelta, asymptotics)
        are methods on this object.
        """
        return self._get("linear_recurrence", lambda:
            LinearRecurrence(self._traj)
        )

    # ==================================================================
    #  COMPANION MATRIX
    # ==================================================================

    def companion(self) -> Matrix:
        """
        The companion (recurrence) matrix.

        Structure for d=2:
            [[0, c_2(n)],    ← right column: coeff of f(n-2)
             [1, c_1(n)]]    ← right column: coeff of f(n-1)

        All recurrence coefficients live in the LAST column.
        The 0s and 1s in the left columns are structural.
        """
        return self._get("companion", lambda:
            self.linear_recurrence().recurrence_matrix
        )

    # ==================================================================
    #  RECURRENCE FORMULA ATTRIBUTES
    # ==================================================================

    def order(self) -> int:
        """
        Order d of the recurrence.
        d=2 → f(n) depends on f(n-1) and f(n-2).
        """
        return self._get("order", lambda:
            self.linear_recurrence().order()
        )

    def relation(self) -> list:
        """
        Raw recurrence relation [a_0(n), a_1(n), ..., a_d(n)].

        These define:  Σ a_i(n) · f(n-i) = 0
        i.e.:  a_0(n)·f(n) + a_1(n)·f(n-1) + ... + a_d(n)·f(n-d) = 0

        Rearranging: f(n) = -[a_1(n)·f(n-1) + ... + a_d(n)·f(n-d)] / a_0(n)
        """
        return self._get("relation", lambda:
            self.linear_recurrence().relation
        )

    def recurrence_coeffs(self) -> list:
        """
        The 'friendly' coefficients [c_1(n), ..., c_d(n)] such that:
            f(n) = c_1(n)·f(n-1) + c_2(n)·f(n-2) + ... + c_d(n)·f(n-d)

        Derived from the last column of the companion matrix (bottom to top).
        """
        def compute():
            C = self.companion()
            d = self.order()
            return [sp.simplify(C[d - i, -1]) for i in range(1, d + 1)]
        return self._get("recurrence_coeffs", compute)

    def coeff_degrees(self) -> list:
        """
        Polynomial degrees of the relation coefficients.
        Uses LinearRecurrence.degrees() which returns degrees of [a_0, ..., a_d].
        """
        return self._get("coeff_degrees", lambda:
            self.linear_recurrence().degrees()
        )

    def formula_str(self) -> str:
        """
        Human-readable recurrence formula.
        Uses LinearRecurrence.__str__() which gives: Σ a_i(n)·p(n-i) = 0
        """
        return self._get("formula_str", lambda:
            str(self.linear_recurrence())
        )

    # ==================================================================
    #  EIGENVALUES
    # ==================================================================

    def sorted_eigenvalues(self) -> list:
        """
        Poincaré eigenvalues of the companion matrix, sorted by |λ| descending.
        Uses Matrix.sorted_eigenvals() which computes eigenvalues of the
        Poincaré characteristic polynomial (the asymptotic limit of the
        charpoly as n→∞).

        For a constant-coefficient recurrence, these are the actual
        characteristic roots. For polynomial coefficients, these are
        the leading-term roots that govern asymptotic growth.
        """
        return self._get("sorted_eigenvalues", lambda:
            self.companion().sorted_eigenvals()
        )

    def eigenvalue_errors(self) -> list:
        """
        log|λ₁/λᵢ| for i = 2, ..., d.
        These are the 'error terms' — the log-ratios between the dominant
        eigenvalue and each subdominant one. Used internally by kamidelta().

        From Matrix.errors():
            errors[0] = log|λ₁/λ₂|  (primary convergence rate)
            errors[1] = log|λ₁/λ₃|  (if d ≥ 3)
            etc.
        """
        return self._get("eigenvalue_errors", lambda:
            self.companion().errors()
        )

    def spectral_gap(self) -> Optional[float]:
        """
        |λ₁| − |λ₂| from the Poincaré eigenvalues.
        Large gap → fast convergence, small gap → slow/noisy.
        """
        def compute():
            eigs = self.sorted_eigenvalues()
            if len(eigs) >= 2:
                return float(abs(eigs[0]).evalf() - abs(eigs[1]).evalf())
            return None
        return self._get("spectral_gap", compute)

    # ==================================================================
    #  WALKING — LIMIT
    # ==================================================================

    def _limits(self, depths: list) -> list:
        """
        Internal: get Limit objects at specified depths.
        Uses the companion's built-in limit() which returns
        Limit objects with correct p/q extraction.

        Convention (from Limit class source):
            p = initial_values.row(0) * walk * final_projection.col(0)
              = e_0^T * W * e_{-1} = W[0, -1]
            q = initial_values.row(1) * walk * final_projection.col(1)
              = e_1^T * W * e_{-1} = W[1, -1]
        """
        return self.companion().limit({n: 1}, depths, {n: 0})

    def limit(self, depth: Optional[int] = None):
        """
        Numerical estimate of L = lim(n→∞) p_n/q_n.

        Returns an mpmath.mpf float.
        Uses Limit.as_float() for correct p/q extraction.
        """
        depth = depth or self._depth
        return self._get(f"limit_{depth}", lambda:
            self._limits([depth])[0].as_float()
        )

    def limit_rational(self, depth: Optional[int] = None):
        """
        Limit as an exact sympy Rational p/q.
        """
        depth = depth or self._depth
        return self._get(f"limit_rational_{depth}", lambda:
            self._limits([depth])[0].as_rational()
        )

    # ==================================================================
    #  DELTA — IRRATIONALITY MEASURE
    # ==================================================================

    def delta(self, depth: Optional[int] = None, L=None) -> float:
        """
        Irrationality measure δ at the given depth.
            |p/q − L| = 1 / q^(1+δ)

        If L is not given, estimates it from the walk at 2×depth.
        Uses Limit.delta(L) directly from ramanujantools.

        For any irrational L: δ ≥ 1 (Dirichlet's theorem).
        """
        depth = depth or self._depth
        def compute():
            if L is None:
                lim, lim_deep = self._limits([depth, 2 * depth])
                return lim.delta(lim_deep.as_float())
            else:
                lim = self._limits([depth])[0]
                return lim.delta(L)
        return self._get(f"delta_{depth}_{L}", compute)

    def delta_sequence(self, depth: Optional[int] = None, L=None) -> list:
        """
        δ values at every step from 1 to depth.
        Shows how the irrationality measure evolves with walk depth.

        Uses Limit.delta(L) at each step.
        """
        depth = depth or self._depth
        def compute():
            depths = list(range(1, depth + 1))
            if L is None:
                all_depths = depths + [2 * depth]
                limits = self._limits(all_depths)
                limit_val = limits[-1].as_float()
                limits = limits[:-1]
            else:
                limits = self._limits(depths)
                limit_val = L
            return [lim.delta(limit_val) for lim in limits]
        return self._get(f"delta_seq_{depth}_{L}", compute)

    def kamidelta(self, depth: int = 20) -> list:
        """
        BLIND irrationality measure prediction — NO knowledge of L needed.

        Uses the Kamidelta algorithm from ramanujantools:
            1. errors() computes log|λ₁/λᵢ| from Poincaré eigenvalues
            2. gcd_slope(depth) fits log(q̃_n) linearly
            3. kamidelta = -1 + error / slope

        Returns a list of predicted δ values (one per eigenvalue pair).
        For order-2 recurrences, this is a single-element list.
        """
        return self._get(f"kamidelta_{depth}", lambda:
            self.companion().kamidelta(depth)
        )

    def gcd_slope(self, depth: int = 20):
        """
        Linear fit slope of log(q̃_n) = log(q_n / gcd(p_n, q_n)).
        This measures how fast the reduced denominator grows.

        Used internally by kamidelta, but also useful on its own.
        """
        return self._get(f"gcd_slope_{depth}", lambda:
            self.companion().gcd_slope(depth)
        )

    # ==================================================================
    #  CONVERGENCE RATE
    # ==================================================================

    def precision_at(self, depth: Optional[int] = None) -> int:
        """
        Number of correct decimal digits at the given depth.
        Uses Limit.precision() which compares the last two walk steps.
        """
        depth = depth or self._depth
        return self._get(f"precision_{depth}", lambda:
            self._limits([depth])[0].precision()
        )

    def digits_per_step(self, max_depth: Optional[int] = None) -> list:
        """
        Δd(k) = precision(k) − precision(k-1) for each step k.
        Shows how many new digits each recurrence step contributes.

        Interpretation:
            roughly constant → exponential convergence  (~8 for Gosper N=29)
            growing with k   → factorial convergence    (super-exponential)
            shrinking with k → polynomial convergence   (slow)
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            depths = list(range(1, max_depth + 1))
            limits = self._limits(depths)
            precisions = [lim.precision() for lim in limits]
            return [
                (k + 1, precisions[k] - precisions[k - 1])
                for k in range(1, len(precisions))
                if precisions[k - 1] > 0
            ]
        return self._get(f"dps_{max_depth}", compute)

    def asymptotic_digits_per_step(self, max_depth: Optional[int] = None) -> Optional[float]:
        """
        Mean Δd in the tail (last 25%) of the digits-per-step trajectory.
        This is the stable long-run convergence rate.

        Expected:  Gosper N=29 → ≈8,  Chudnovsky N=41 → ≈14
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            dps = self.digits_per_step(max_depth)
            if not dps:
                return None
            tail = dps[max(0, int(0.75 * len(dps))):]
            if not tail:
                return None
            return sum(d for _, d in tail) / len(tail)
        return self._get(f"asymp_dps_{max_depth}", compute)

    def convergence_class(self, max_depth: Optional[int] = None) -> str:
        """
        Classify convergence by comparing early vs. late Δd values:
            'factorial'    — Δd grows (super-exponential)
            'exponential'  — Δd roughly constant
            'polynomial'   — Δd shrinks
            'unknown'      — not enough data
        """
        max_depth = max_depth or min(self._depth, 100)
        def compute():
            dps = self.digits_per_step(max_depth)
            if len(dps) < 8:
                return "unknown"
            vals = [d for _, d in dps]
            mid = len(vals) // 2
            early = sum(vals[:mid]) / mid
            late = sum(vals[mid:]) / len(vals[mid:])
            if early == 0:
                return "unknown"
            r = late / early
            if   r > 1.25: return "factorial"
            elif r < 0.80: return "polynomial"
            else:          return "exponential"
        return self._get(f"conv_class_{max_depth}", compute)

    # ==================================================================
    #  PENDING IMPLEMENTATIONS  (stubs — user will fill in later)
    # ==================================================================

    def p_vector(self) -> tuple:
        """Numerator projection vector p such that lim = p·walk / q·walk.

        TODO: implement using the LIReC / rational-identification logic
        currently in ``Searchable.compute_trajectory_data``.
        Returns an empty tuple until then; the DTO pipeline remains functional.
        """
        return ()

    def q_vector(self) -> tuple:
        """Denominator projection vector q (see ``p_vector``).

        TODO: implement alongside ``p_vector``.
        """
        return ()

    def identified(self) -> bool:
        """Whether this trajectory's limit was recognised as the target constant.

        TODO: implement via LIReC or a direct comparison to the constant value.
        Returns ``True`` for now so that the analysis percentage filter treats
        every trajectory as identified until real logic is added.
        """
        return True

    # ==================================================================
    #  ASYMPTOTICS  (Birkhoff-Trjitzinsky)
    # ==================================================================

    def asymptotics(self, precision=None) -> list:
        """
        Formal asymptotic basis for the recurrence solutions.

        Uses LinearRecurrence.asymptotics() which runs the Birkhoff-Trjitzinsky
        reduction algorithm to find the canonical fundamental matrix.

        Returns a list of sympy expressions — one per solution of the recurrence.
        The last column of the CFM (transposed) gives the asymptotic behavior
        of p_n and q_n.

        These encode the growth rates η (factorial), γ (exponential), β (polynomial)
        from the NeurIPS 2024 paper in symbolic form.
        """
        return self._get(f"asymptotics_{precision}", lambda:
            self.linear_recurrence().asymptotics(precision)
        )