from __future__ import annotations

import hashlib
import json
import warnings
from typing import List, Optional, TYPE_CHECKING, Tuple

import sympy as sp
from sympy.abc import n

from LIReC.db.access import db
from ramanujantools import LinearRecurrence, Matrix, Limit
from dreamer.utils.logger import Logger
from dreamer.utils.schemes.searchable import Searchable
from dreamer.configs import config

search_config = config.search

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


def _serialize_encoding(shard) -> str:
    """Canonical string form of the shard's ±1 sign vector.

    The extractor produces hyperplanes in a canonical sorted order, so
    ``shard.encoding[i]`` unambiguously refers to ``cmf.hyperplanes[i]``.
    Joining the ±1 values with commas gives a deterministic, compact
    label suitable for hashing into ``shard_id`` / ``trajectory_id``.
    Whole-space shards (no encoding) produce a fixed placeholder.
    """
    encoding = getattr(shard, "encoding", None)
    if not encoding:
        return "whole_space"
    return ",".join(str(int(s)) for s in encoding)


def derive_cmf_and_shard_ids(shard) -> tuple[str, str, str]:
    """Return ``(cmf_id, shard_id, shard_encoding_str)`` for *shard*.

    * ``cmf_id`` — the CMF name (unique per CMF in the current system).
    * ``shard_id`` — stable SHA-256 of ``(cmf_id, shard_encoding_str)``.
    * ``shard_encoding_str`` — canonical ±1 sign vector string (see
      :func:`_serialize_encoding`).  Also used as part of trajectory ids
      so the two levels stay consistent.
    """
    cmf_id = shard.cmf_name
    shard_encoding_str = _serialize_encoding(shard)
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
        p_vector=tuple(handler.p_vector()),
        q_vector=tuple(handler.q_vector()),
        identified=bool(handler.identified()),
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
        constant: Optional[sp.Expr] = None,
        walk_depth: int = 200,
        walk_type: int = 1,
        searchable: Optional[Searchable] = None,
    ):
        """
        Parameters
        ----------
        traj_matrix : ramanujantools.Matrix
            Symbolic d×d trajectory matrix M(n); free parameters (e.g. ``z``)
            must already be substituted for numeric computation.
        constant : sympy.Expr, optional
            The target constant this trajectory approximates (e.g. ``sp.pi``).
            Required for Tier-1 attributes (``delta``, ``limit``, p/q vectors);
            may be ``None`` in worker contexts that only need Tier-2/3 attrs.
        walk_depth : int
            Default number of recurrence steps for walks.
        walk_type : int
            ``1`` → walk uses ``M.inv().T`` (the dual recurrence);
            ``2`` → walk uses ``M`` directly.
        searchable : Searchable, optional
            The ``Searchable`` (typically the ``Shard``) this trajectory was
            sampled from.  When provided, its ``cache`` is consulted/updated
            for p/q vectors so repeated identification calls are avoided.
        """
        self._traj = traj_matrix
        self._constant = constant
        self._depth = walk_depth
        self._cache: dict = {}
        self._walk_type = walk_type # 1 for using inv().T in walk, 2 for direct walk
        self._searchable = searchable
        self._utility_cache: dict = {}  # separate cache for non-core attributes like p/q vectors
        self._identified = False

    @classmethod
    def from_cmf(
        cls,
        cmf,
        trajectory,
        start_point,
        constant: Optional[sp.Expr] = None,
        z_value=None,
        walk_depth: int = 200,
        walk_type: int = 1,
        searchable: Optional[Searchable] = None,
    ) -> "TrajectoryAttributesHandler":
        """Build a handler by computing ``cmf.trajectory_matrix(trajectory, start_point)``.

        ``z_value`` substitutes the free ``z`` symbol when given.  See
        ``__init__`` for the meaning of ``constant``, ``walk_*`` and
        ``searchable``.
        """
        tmat = cmf.trajectory_matrix(trajectory, start_point)
        if z_value is not None:
            tmat = tmat.subs({sp.Symbol("z"): z_value})
        return cls(tmat, constant, walk_depth, walk_type, searchable)

    # ------------------------------------------------------------------
    #  Cache helpers
    # ------------------------------------------------------------------

    def _get(self, key: str, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def _get_utility(self, key: str, fn):
        if key not in self._utility_cache:
            self._utility_cache[key] = fn()
        return self._utility_cache[key]

    def clear_cache(self):
        self._cache.clear()

    def clear_utility_cache(self):
        self._utility_cache.clear()

    def computed_attributes(self) -> list:
        return list(self._cache.keys())

    # ==================================================================
    #  TRAJECTORY MATRIX
    # ==================================================================

    def trajectory_matrix(self) -> Matrix:
        """Walk-ready d×d symbolic trajectory matrix.

        For ``walk_type=1`` returns ``M(n).inv().T`` (dual recurrence);
        for ``walk_type=2`` returns ``M(n)`` directly.  Cached.
        """
        return self._get_utility("trajectory_matrix", lambda: self._traj.inv().T if self.walk_type() == 1 else self._traj)

    def traj_size(self) -> int:
        """Dimension d of the trajectory matrix."""
        return self.trajectory_matrix().shape[0]

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
            LinearRecurrence(self.trajectory_matrix())
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
            self.trajectory_matrix().as_companion()
            # self.linear_recurrence().recurrence_matrix
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

    def constant(self) -> sp.Expr:
        """The target constant that this trajectory is approximating (e.g., π)."""
        return self._constant

    def walk_type(self) -> int:
        """Return the walk type (1 or 2) for this handler."""
        return self._walk_type

    def _effective_walk_values(self, depth: Optional[int] = None, walk_matrix: Optional[sp.Matrix] = None) -> list:
        """Return the column of the walked matrix that gets projected by p, q.

        Picks the first column with a non-zero top entry, prefers one with no
        zero entries (LIReC-friendly), and normalises by the top entry.
        Pass either ``depth`` (walks internally and caches) or a precomputed
        ``walk_matrix``.  Internal helper for limit / delta computations.
        """

        depth = depth or self._depth
        
        def compute():
            lirec_valid_col = None
            normalized_col = None
            walked = walk_matrix or self.trajectory_matrix().walk({n: 1}, depth, {n: 0})

            for col_ind in range(sp.shape(walked)[1]):
                if walked[0, col_ind].is_zero:
                    continue

                col = (walked / walked[0, col_ind]).col(col_ind)
                if normalized_col is None:
                    normalized_col = col

                if all([not v.is_zero for v in col]):
                    lirec_valid_col = col
                    break

            if normalized_col is None:
                Logger(
                    f'Could not normalize any walk matrix column. This was not supposed to happen'
                    'Skipping trajectory...', Logger.Levels.warning
                ).log()
                return None, None, None

            if lirec_valid_col is not None:
                normalized_col = lirec_valid_col
            
            return [item for item in normalized_col]
        
        if depth is None and walk_matrix is None:
            Logger(
                'No depth or walk matrix provided for effective walk values. This was not supposed to happen'
                'Skipping trajectory...', Logger.Levels.warning
            ).log()
            return None
        
        if depth is None:
            return compute()

        return self._get_utility(f"effective_walk_{depth}", compute)

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
        return self.trajectory_matrix().limit({n: 1}, depths, {n: 0})

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

    def delta(self, depth: Optional[int] = None) -> float:
        """
        Irrationality measure δ at the given depth.
            |p/q − L| = 1 / q^(1+δ)

        For any irrational L: δ ≥ 1 (Dirichlet's theorem).
        """
        depth = depth or self._depth
        def compute():
            converges, _ = self._convergence_sanity_check(depth)
            if not converges:
                return float('-inf')
            return self.delta_sequence(depth)[0]
        return self._get(f"delta_{depth}", compute)

    def delta_sequence(self, depth: Optional[int] = None) -> list:
        """
        δ values at every step from 1 to depth.
        Shows how the irrationality measure evolves with walk depth.

        Uses Limit.delta(L) at each step.
        """
        depth = depth or self._depth
        def compute():
            limits = self._limits(list(range(1, depth + 1)))
            p, q = self._pq_vector(depth)
            p = sp.Matrix(p).T
            q = sp.Matrix(q).T
            deltas = []
            high_res_constant = self.constant().evalf(search_config.CONSTANT_NO_DIGITS_HIGH_RES)

            for limit in limits:
                numerator = p.dot(self._effective_walk_values(None, limit))
                denom = q.dot(self._effective_walk_values(None, limit))
                estimated = sp.Abs(sp.Rational(numerator, denom))
                err = sp.Abs(estimated - high_res_constant)
                delta = -1 - sp.log(err) / sp.log(denom)

                if sp.Abs(denom) <= search_config.MIN_ESTIMATE_DENOMINATOR:
                    # probably didn't converge for some reason
                    deltas.append(float('-inf'))
                    continue

                # This part is not supposed to be reached at all, these are the final guardrails
                if delta == sp.oo or delta == sp.zoo:
                    deltas.append(float('-inf'))
                    continue

                deltas.append(float(delta.evalf(10)))
            return deltas
        
        return self._get(f"delta_seq_{depth}", compute)

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
            self.trajectory_matrix().gcd_slope(depth)
        )

    # ==================================================================
    #  CONVERGENCE RATE
    # ==================================================================

    def _convergence_sanity_check(self, depth: Optional[int] = None) -> Tuple[bool, List[Limit]]:
        """Check that the estimated limit stabilises across the depths configured
        in ``search.DEPTH_CONVERGENCE_THRESHOLD``.

        Returns ``(converges, limits)`` — ``converges`` is True iff successive
        estimates differ by less than ``search.LIMIT_DIFF_ERROR_BOUND``.
        """
        depth = depth or self._depth
        limits = self._limits([round(coef * depth) for coef in search_config.DEPTH_CONVERGENCE_THRESHOLD])
        floats = []
        p, q = self._pq_vector(depth)
        p = sp.Matrix(p).T
        q = sp.Matrix(q).T

        # extract estimated limit
        for limit in limits:
            walk_col = self._effective_walk_values(depth, limit.current)
            values = [item for item in walk_col]
            values_vec = sp.Matrix(values)
            numerator = p.dot(values_vec)
            denom = q.dot(values_vec)
            estimated = sp.Abs(sp.Rational(numerator, denom))
            floats.append(estimated)

        # check that the estimated limits are consistent (within error bound)
        diffs = [abs(floats[i] - floats[i-1]) for i in range(1, len(floats))]
        return all(diff < search_config.LIMIT_DIFF_ERROR_BOUND for diff in diffs), limits

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

    def _pq_vector(self, depth: Optional[int] = None) -> tuple:
        """Numerator and denominator projection vectors (p, q) such that constant = p·walk / q·walk."""
        depth = depth or self._depth

        def compute():
            # If searchable is provided, try to find a cached p/q pair that matches the effective walk values.
            
            low_res_constant = self.constant().evalf(search_config.CONSTANT_NO_DIGITS_LOW_RES)
            
            if self._searchable:
                def matcher(v):
                    v1, v2 = v
                    v1 = sp.Matrix(v1).T
                    v2 = sp.Matrix(v2).T
                    numerator = v1.dot(self._effective_walk_values(depth))
                    denom = v2.dot(self._effective_walk_values(depth))
                    err = sp.Abs(sp.Abs(sp.Rational(numerator, denom)) - low_res_constant)
                    return sp.N(err, 25) < search_config.CACHE_ACCEPTANCE_THRESHOLD

                if matched := self._searchable.cache.find(matcher):
                    return matched
            
            # Compute p, q using LIReC
            try:
                res = db.identify([low_res_constant] + self._effective_walk_values(depth)[1:])
            except Exception as e:
                Logger(f'Error while identifing constnat. LIReC failed with: "{e}"', Logger.Levels.exception).log()
                res = []

            # LIReC may also return an empty list when it cannot identify
            # the constant — fall back to the canonical p=e_0, q=e_1 vectors.
            if not res:
                d = len(self._effective_walk_values(depth))
                return ([1] + [0] * (d - 1), [0, 1] + [0] * (max(d - 2, 0)))

            # extract p, q from LIReC result
            res = res[0]
            res.include_isolated = 0
            estimated_expr = sp.nsimplify(str(res).rsplit(' ', 1)[0], rational=True)
            numerator, denom = sp.fraction(estimated_expr)
            p_dict = numerator.as_coefficients_dict()
            q_dict = denom.as_coefficients_dict()
            syms = sp.symbols(f'c:{self.traj_size()}')[1:]
            ext_syms = [1] + list(syms)
            # Coerce to native ints — projection coeffs are integers, and
            # sympy.Integer/One/Zero are not JSON-serialisable downstream.
            p = [int(p_dict[sym]) for sym in ext_syms]
            q = [int(q_dict[sym]) for sym in ext_syms]
            if self._searchable:
                self._searchable.cache.append((p, q))
            self._identified = True
            return p, q

        return self._get_utility("pq_vector", compute)

    def p_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector p such that p·walk gives the numerator sequence."""
        return self._pq_vector(depth or self._depth)[0]

    def q_vector(self, depth: Optional[int] = None) -> list:
        """Projection vector q such that q·walk gives the denominator sequence."""
        return self._pq_vector(depth or self._depth)[1]

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

    def identified(self) -> bool:
        """
        Whether the handler successfully identified a closed-form constant.

        This is a simple boolean check: did the p/q vector extraction yield
        a valid result that matches the effective walk values within the
        acceptance threshold?

        Note: this is not a guarantee of correctness, just a heuristic check.
        """
        return self._get("identified", lambda: self._identified)

    def companion_coboundary_rank(self) -> int:
        """
        Rank of the coboundary matrix of the companion.
        """
        return self._get("coboundary_rank", lambda:
            self.trajectory_matrix().companion_coboundary_matrix().rank()
        )