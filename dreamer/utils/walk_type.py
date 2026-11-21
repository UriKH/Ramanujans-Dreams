"""Walk type resolution: which matrix a trajectory walk reads.

RD reads approximants from a walked matrix W_N = M(s) M(s+d) ... M(s+(N-1)d):

* **type 1** (``use_inv_t=True``, RD's default) — the inverse transpose W_N^{-T}
  (the dual recurrence). All of RD's mathematics is built on this walk type.
* **type 2** (``use_inv_t=False``) — W_N itself.

The walk type of a CMF is chosen in two places:

* the global knob ``search.DEFAULT_USES_INV_T`` (changes between runs);
* the per-CMF ``use_inv_t`` of a formatter (``pFq(..., use_inv_t=False)``), carried
  by ``CMFData`` and the CMF's shards.

Rule: an explicit per-CMF ``True`` / ``False`` overrides the global for that CMF;
``None`` (the default) follows the global **at the time of the walk** (it is not
frozen when the formatter or shard is built).
"""

from typing import Optional

from dreamer.configs import config


def resolve_walk_type(use_inv_t: Optional[bool]) -> int:
    """Walk type (1 or 2) for a per-CMF ``use_inv_t`` setting.

    :param use_inv_t: ``True`` → type 1, ``False`` → type 2, ``None`` → the
        current global ``search.DEFAULT_USES_INV_T``.
    :return: ``1`` (walk ``W^{-T}``) or ``2`` (walk ``W``).
    """
    if use_inv_t is None:
        use_inv_t = config.search.DEFAULT_USES_INV_T
    return 1 if use_inv_t else 2


def searchable_walk_type(searchable) -> int:
    """Walk type of a searchable (shard), or of the global setting when it has none.

    :param searchable: A :class:`~dreamer.utils.schemes.searchable.Searchable`
        (its ``use_inv_t`` attribute), any object without that attribute, or ``None``.
    :return: ``1`` or ``2`` (see :func:`resolve_walk_type`).
    """
    return resolve_walk_type(getattr(searchable, "use_inv_t", None))
