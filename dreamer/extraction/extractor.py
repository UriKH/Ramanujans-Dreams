from dreamer.configs import (
    sys_config,
    extraction_config
)
from dreamer.extraction.hyperplanes import Hyperplane
from dreamer.extraction.shard import Shard
from dreamer.utils.schemes.extraction_scheme import ExtractionScheme, ExtractionModScheme
from dreamer.utils.logger import Logger
from dreamer.utils.constants.constant import Constant
from dreamer.utils.schemes.searchable import Searchable
from dreamer.utils.storage.exporter import Exporter
from dreamer.utils.storage.formats import Formats
from dreamer.utils.storage.atlas_writer import (
    read_shard_records,
    update_cmf_hyperplanes,
    write_shard_records,
)
from dreamer.utils.ui.tqdm_config import SmartTQDM
from dreamer.configs import config
from dreamer.utils.types import CMFData
from .utils import initial_points as init_points
from .v2 import ExtractionManager, LrslibExtractor, RayShootingExtractor

import os.path
import sympy as sp
import numpy as np
import math
from collections import defaultdict
from functools import partial
from ramanujantools.cmf import pFq as rt_pFq
from ramanujantools import Position
from typing import Dict, List, Optional, Set, Tuple


class ShardExtractorMod(ExtractionModScheme):
    """
    Module for shard extraction
    """

    def __init__(self, cmf_data: Dict[Constant, List[CMFData]]):
        """
        Creates a shard extraction module
        :param cmf_data: A mapping from constants to a list of CMFs
        """
        super().__init__(
            cmf_data,
            name=self.__class__.__name__,
            desc='Shard extractor module',
            version='0.0.1'
        )

    def execute(self) -> Dict[Constant, List[Searchable]]:
        """
        Extract shards from CMFs
        :return: A mapping from constants to a list of shards
        """
        all_shards = defaultdict(list)

        consts_itr = iter(list(self.cmf_data.keys()))
        for const, cmf_data_list in SmartTQDM(
                self.cmf_data.items(), desc=f'Extracting shards for "{next(consts_itr).name}"',
                **sys_config.TQDM_CONFIG
        ):
            with Exporter.export_stream(
                    os.path.join(sys_config.PATH_TO_SEARCHABLES, const.name),
                    exists_ok=True,
                    clean_exists=True,
                    fmt=Formats(sys_config.EXPORT_SEARCHABLES_FORMAT),
            ) as export_stream:
                for i, cmd_data in enumerate(SmartTQDM(
                        cmf_data_list, desc=f'Computing shards',
                        **sys_config.TQDM_CONFIG)):
                    extractor = ShardExtractor(
                        const, cmd_data
                    )
                    shards = extractor.extract(call_number=i + 1)
                    all_shards[const] += shards
                    export_stream(shards, cmd_data.cmf_name)

                    # DB-ready ShardDTO records alongside the pickle export.
                    # Written idempotently per (const, cmf) so reruns don't
                    # grow the file.  Also backfills the CmfDTO row written
                    # by the loading stage with the hyperplanes we just
                    # computed.
                    if sys_config.EXPORT_CMFS:
                        write_shard_records(
                            sys_config.EXPORT_CMFS,
                            const,
                            cmd_data.cmf_name,
                            shards,
                        )
                        update_cmf_hyperplanes(
                            sys_config.EXPORT_CMFS,
                            const,
                            cmd_data.cmf_name,
                            extractor.hyperplanes,
                        )
        return all_shards


class ShardExtractor(ExtractionScheme):
    """
    Shard extractor is a representation of a shard finding method.
    """

    def __init__(self, const: Constant, cmf_data: CMFData):
        """
        Extracts the shards of a CMF
        :param const: Constant searched in this CMF
        :param cmf_data: CMF to extract shards from, more data for extraction and later usage
        """
        super().__init__(const, cmf_data)
        # Populated by extract(); read by ShardExtractorMod.execute() so it
        # can backfill the CmfDTO row with the hyperplanes used to derive
        # the shards.
        self.hyperplanes: Set[Hyperplane] = set()
        # self.pool = create_pool() if extraction_config.PARALLELIZE else None

    @property
    def symbols(self) -> List[sp.Symbol]:
        """
        :return: The CMF's symbols
        """
        return list(self.cmf_data.cmf.matrices.keys())

    def _extract_cmf_hps(self) -> List[Hyperplane]:
        """
        Compute the hyperplanes of the CMF - zeros of the characteristic polynomial of each matrix and the poles of each
         matrix entry.

        The result is **sorted by ``str(hp.expr)``** so the ordering is
        deterministic across runs: ``Hyperplane`` already canonicalises
        its expression in ``__post_init__`` (LCM denominators, GCD coeffs,
        leading-coefficient sign), so the sort key is stable.  A canonical
        order is what lets ``Shard.encoding`` (the ±1 sign vector) be a
        meaningful, hash-stable label for a shard.
        :return: A canonically-ordered list of filtered hyperplanes.
        """
        hps = set()
        symbols = list(self.cmf_data.cmf.matrices.keys())
        for s in symbols:
            if isinstance(self.cmf_data.cmf, rt_pFq):
                det = rt_pFq.determinant(self.cmf_data.cmf.p, self.cmf_data.cmf.q, self.cmf_data.cmf.z, s)
            else:
                det = self.cmf_data.cmf.matrices[s].det()
            zeros = sp.solve(det)
            zeros = [Hyperplane(lhs - rhs, symbols) for solution in zeros for lhs, rhs in solution.items()]
            hps.update(set(zeros))

            poles = set()
            for v in self.cmf_data.cmf.matrices[s].iter_values():
                if (den := v.as_numer_denom()[1]) == 1:
                    continue

                solutions = {
                    (sym, sol) for sym in den.free_symbols for sol in sp.solve(sp.simplify(den), sym)
                }
                for lhs, rhs in solutions:
                    poles.add(Hyperplane(lhs - rhs, symbols))
            hps.update(poles)

        # compute the relevant hyperplanes with respect to the shift, then
        # sort to give Shard.encoding[i] a stable meaning.
        filtered_hps = [
            hp for hp in hps
            if hp.apply_shift(self.cmf_data.shift).is_in_integer_shift()
        ]
        filtered_hps.sort(key=lambda hp: str(hp.expr))
        return filtered_hps

    def extract(self, call_number=None) -> List[Shard]:
        """
        Extracts the shards from the CMF.

        The discovery method is chosen by
        ``extraction_config.STRATEGY``:

        * ``"auto" | "exact" | "heuristic"`` -- delegate to the v2
          :class:`~dreamer.extraction.v2.ExtractionManager` (lrs + MILP
          with a ray-shooting fallback).  ``"auto"`` is the default and
          enables the wall-clock timeout protection.
        * ``"legacy"`` -- the original brute-force lattice scan in
          :mod:`dreamer.extraction.utils.initial_points` (kept verbatim
          for parity and benchmarking).

        Either path may be supplemented or fully driven by
        ``cmf_data.selected_points`` exactly as before.
        :return: The list of shards matching the CMF
        """
        # compute hyperplanes and prepare sample point
        hps = self._extract_cmf_hps()
        self.hyperplanes = hps

        if not hps:
            return [Shard.from_cmf_data(self.cmf_data, self.const, [], [])]

        symbols = list(hps)[0].symbols
        shard_encodings: Dict[Tuple[int, ...], Position] = dict()
        selected = [] if self.cmf_data.selected_points is None else self.cmf_data.selected_points

        if self.cmf_data.only_selected:
            if self.cmf_data.selected_points is None:
                raise ValueError('No start points were provided for extraction.')
        else:
            cached = None
            if config.extraction.LOAD_SHARD_CACHE:
                cached = self._load_cached_encodings(hps, symbols)
            if cached is not None:
                shard_encodings.update(cached)
                Logger(
                    f'Loaded {len(cached)} cached shards from shards.jsonl; '
                    'skipping extraction',
                    level=Logger.Levels.info,
                ).log()
            else:
                strategy = config.extraction.STRATEGY
                if strategy == 'legacy':
                    shard_encodings.update(self._discover_via_legacy(hps, symbols))
                elif strategy in ('auto', 'exact', 'heuristic'):
                    shard_encodings.update(
                        self._discover_via_v2(hps, symbols, strategy)
                    )
                else:
                    raise ValueError(
                        f"Unknown extraction strategy {strategy!r}; expected "
                        "'auto', 'exact', 'heuristic' or 'legacy'"
                    )

        if self.cmf_data.selected_points:
            points = [
                tuple(coord + shift for coord, shift in zip(p, self.cmf_data.shift.values()))
                for p in selected
            ]

            # validate shards using the sampled points
            for p in SmartTQDM(points, desc='Computing shard encodings', **sys_config.TQDM_CONFIG):
                enc = []
                point_dict = {sym: coord for sym, coord in zip(symbols, p)}
                for hp in hps:
                    res = hp.expr.subs(point_dict)
                    if res == 0:
                        break
                    enc.append(1 if res > 0 else -1)

                if len(enc) == len(hps):
                    shard_encodings[tuple(enc)] = Position(point_dict)

        Logger(
            f'In CMF no. {call_number}: found {len(hps)} hyperplanes and {len(shard_encodings)} shards ',
            level=Logger.Levels.info
        ).log()

        # Create shard objects.  The shift is identical for every shard,
        # so shift the hyperplanes ONCE here and reuse the result — this
        # avoids re-running the (sympy) per-hyperplane apply_shift inside
        # every Shard.__init__, which otherwise dominates this loop.
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps]
        shards = []
        for enc in SmartTQDM(shard_encodings.keys(), desc='Creating shard objects', **sys_config.TQDM_CONFIG):
            shards.append(Shard.from_cmf_data(
                self.cmf_data, self.const, shifted_hps, enc, shard_encodings[enc],
                hyperplanes_already_shifted=True,
            ))
        return shards

    def _load_cached_encodings(
        self, hps: List[Hyperplane], symbols: List[sp.Symbol]
    ) -> Optional[Dict[Tuple[int, ...], Position]]:
        """
        Load previously-computed shards from the ``<cmf>__shards.jsonl``
        cache so extraction can be skipped.

        Returns a mapping ``{sign-encoding: interior-point}`` rebuilt
        from the cached :class:`ShardDTO` records, or :data:`None` when
        there is no usable cache (no ``EXPORT_CMFS`` configured, missing
        / empty file, or a stale cache whose encodings no longer match
        the current hyperplane count).

        Hyperplanes are recomputed by the caller and passed in:
        ``_extract_cmf_hps`` returns them in a canonical, deterministic
        order, so ``encoding[i]`` still labels ``hps[i]`` exactly as it
        did when the cache was written.
        """
        if not sys_config.EXPORT_CMFS:
            return None

        dtos = read_shard_records(
            sys_config.EXPORT_CMFS, self.const, self.cmf_data.cmf_name
        )
        if not dtos:
            return None

        n = len(hps)
        out: Dict[Tuple[int, ...], Position] = {}
        for dto in dtos:
            enc = tuple(int(v) for v in dto.shard_encoding)
            if len(enc) != n:
                # Cache was written for a different hyperplane set — the
                # CMF or its hyperplanes changed.  Treat as stale and
                # force a fresh extraction rather than mis-aligning signs.
                Logger(
                    f'Ignoring stale shard cache (encoding length {len(enc)} '
                    f'!= {n} hyperplanes) for "{self.cmf_data.cmf_name}"',
                    level=Logger.Levels.warning,
                ).log()
                return None
            point = None
            if dto.interior_point is not None:
                point = Position(
                    {sym: int(v) for sym, v in zip(symbols, dto.interior_point)}
                )
            out[enc] = point
        return out

    def _discover_via_legacy(
        self, hps: List[Hyperplane], symbols: List[sp.Symbol]
    ) -> Dict[Tuple[int, ...], Position]:
        """
        Original brute-force lattice scan in
        :mod:`dreamer.extraction.utils.initial_points`.

        Preserved verbatim from the pre-v2 implementation so the
        ``legacy`` strategy remains a byte-for-byte fallback.
        """
        hps_list = list(hps)
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]
        A = np.array([hp.vectors[0] for hp in shifted_hps], dtype=np.int64)
        b = np.array([hp.vectors[1] for hp in shifted_hps], dtype=np.int64)
        S = config.extraction.INIT_POINT_MAX_COORD * 2 + 1
        cpus = cpus if (cpus := os.cpu_count()) else 1
        prefix_dims = max(min(int(round(math.log(cpus, S))), cpus - 1), 1)

        symmetries_func = None
        if issubclass(self.cmf_data.cmf.__class__, rt_pFq) and config.extraction.IGNORE_DUPLICATE_SEARCHABLES:
            symmetries_func = partial(init_points.filter_symmetrical_cones,
                                      p=self.cmf_data.cmf.p,
                                      q=self.cmf_data.cmf.q,
                                      shift=list(self.cmf_data.shift.values()))
        final_results = init_points.compute_mapping(
            self.cmf_data.cmf.dim(), S, A, b, prefix_dims, symmetries_func
        )
        unique_sigs = list(final_results.keys())
        decoded_vectors = init_points.decode_signatures(unique_sigs, len(hps))
        out: Dict[Tuple[int, ...], Position] = {}
        for i, sig in enumerate(unique_sigs):
            sign_vector = decoded_vectors[i]
            if 0 in sign_vector:
                continue
            actual_point = final_results[sig]
            out[tuple(sign_vector)] = Position(
                {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, actual_point)}
            )
        return out

    def _discover_via_v2(
        self,
        hps: List[Hyperplane],
        symbols: List[sp.Symbol],
        strategy: str,
    ) -> Dict[Tuple[int, ...], Position]:
        """
        Route through the v2 :class:`ExtractionManager`.

        The v2 module works on the *shifted* hyperplanes (so that the
        integer point it returns lives in the shifted lattice) and
        labels each shard by a ``+/-1`` sign tuple ordered identically
        to the input list -- matching how :class:`Shard.encoding` is
        interpreted downstream.  Integer witnesses are translated back
        to absolute coordinates by adding the shift.

        :param strategy: One of ``"auto" | "exact" | "heuristic"``.
        """
        if issubclass(self.cmf_data.cmf.__class__, rt_pFq) and config.extraction.IGNORE_DUPLICATE_SEARCHABLES:
            Logger(
                "IGNORE_DUPLICATE_SEARCHABLES=True is ignored under the "
                f"{strategy!r} strategy (v2 does not yet deduplicate pFq "
                "symmetric shards). Use STRATEGY='legacy' to keep the "
                "old dedup behaviour.",
                level=Logger.Levels.warning,
            ).log()

        hps_list = list(hps)
        shifted_hps = [hp.apply_shift(self.cmf_data.shift) for hp in hps_list]

        manager = ExtractionManager(
            strategy=strategy,
            timeout_seconds=config.extraction.STRATEGY_TIMEOUT_SECONDS,
            exact_unbounded_check=config.extraction.EXACT_UNBOUNDED_CHECK,
            exact_num_workers=config.extraction.EXACT_NUM_WORKERS,
        )
        mapping = manager.extract(shifted_hps)

        out: Dict[Tuple[int, ...], Position] = {}
        for sig, point in mapping.items():
            out[tuple(sig)] = Position(
                {sym: int(v) + self.cmf_data.shift[sym] for sym, v in zip(symbols, point)}
            )
        return out
