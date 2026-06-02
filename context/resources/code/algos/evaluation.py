from typing import List, Dict, Any
import multiprocessing
from ramanujantools import Position
from cmf_search.metrics import trajDelta, trajDeltaMulti


def _err(exc: BaseException):
    import traceback, sys
    traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)


def evaluate_trajectories(
    init: Position,
    trajectories: List[Position],
    ccmf,
    p: int,
    q: int,
    expr,
    true_const,
    depth: int,
    cores: int,
    walk_type: int = 1,
) -> List[Dict[str, Any]]:
    """
    Evaluate δ for a batch of trajectories using a multiprocessing pool.
    """
    if not trajectories:
        return []

    with multiprocessing.Pool(processes=cores) as pool:
        async_results = [
            pool.apply_async(
                trajDelta,
                args=(init, pos, ccmf, p, q, expr, true_const, depth,walk_type),
                error_callback=_err,
            )
            for pos in trajectories
        ]
        return [res.get() for res in async_results]




def evaluate_trajectories_multi(
    init: Position,
    trajectories: List[Position],
    ccmf,
    p: int,
    q: int,
    exprs,
    true_consts,
    names,
    depth: int,
    cores: int,
    walk_type: int = 1,
) -> List[Dict[str, Any]]:
    """
    Evaluate δ for a batch of trajectories using a multiprocessing pool.
    """
    if not trajectories:
        return []

    with multiprocessing.Pool(processes=cores) as pool:
        async_results = [
            pool.apply_async(
                trajDeltaMulti,
                args=(init, pos, ccmf, p, q, exprs, true_consts,names, depth,walk_type),
                error_callback=_err,
            )
            for pos in trajectories
        ]
        return [res.get() for res in async_results]
