"""
Deprecation warnings for the legacy (pre-DTO) search / δ path.

The legacy path — ``Searchable.calc_delta`` / ``compute_trajectory_data``, ``SerialSearcher.search``,
``GeneticSearchMethod`` / ``GeneticSearchMod`` and the ``DataManager``-based analyzer — is superseded by
``TrajectoryAttributesHandler`` and the DTO stage modules, and does not receive their fixes (e.g. the
adaptive LIReC precision for slow-converging walks).  It is kept until dreamer v2 removes it.
"""
import warnings

#: When deprecated code is removed.
REMOVAL = "dreamer v2"


def warn_deprecated(name: str, replacement: str) -> None:
    """
    Emit a ``DeprecationWarning`` for a legacy entry point, attributed to its caller.
    :param name: The deprecated function or class, as users call it.
    :param replacement: What to use instead.
    :return: None.
    """
    warnings.warn(
        f"{name} is deprecated and will be removed in {REMOVAL}: it is the legacy δ path, which lacks "
        f"the current identification and precision fixes. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
