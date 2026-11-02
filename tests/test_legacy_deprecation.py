"""The legacy (pre-DTO) δ path warns that it is deprecated.

It lacks the current identification and precision fixes (e.g. the adaptive LIReC precision), so a
caller must be told.  Each entry point warns before doing any work; the calls below use placeholder
arguments, so whatever happens after the warning (an error, an empty result) is irrelevant here.
"""
import warnings
from types import SimpleNamespace

import pytest

from dreamer.analysis.analysis_methods.serial_scan_analyzer import Analyzer
from dreamer.search.methods.genetic import GeneticSearchMethod
from dreamer.search.methods.hedgehog_scan import SerialSearcher
from dreamer.search.searchers.genetic_mod import GeneticSearchMod
from dreamer.utils.schemes.searchable import Searchable


def _deprecation_warnings(call):
    """
    Run *call* and collect the ``DeprecationWarning``s it emits; any exception after them is ignored.
    :param call: Zero-argument callable.
    :return: The recorded deprecation warnings.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            call()
        except Exception:
            pass
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


@pytest.mark.parametrize("name, call", [
    ("Searchable.calc_delta",
     lambda: Searchable.calc_delta(SimpleNamespace(dim=2), None, None, None, None, 1.0)),
    ("Searchable.compute_trajectory_data",
     lambda: Searchable.compute_trajectory_data(SimpleNamespace(), None, None)),
    ("SerialSearcher.search", lambda: SerialSearcher.search(SimpleNamespace())),
    ("GeneticSearchMethod", lambda: GeneticSearchMethod(None, None)),
    ("GeneticSearchMod", lambda: GeneticSearchMod([])),
    ("analysis_methods.serial_scan_analyzer.Analyzer", lambda: Analyzer(None, [])),
])
def test_legacy_entry_point_warns(name, call):
    found = _deprecation_warnings(call)
    assert any(str(w.message).startswith(f"{name} is deprecated") for w in found), [str(w.message) for w in found]


def test_warning_points_at_the_caller():
    found = _deprecation_warnings(lambda: Analyzer(None, []))
    assert found and found[0].filename == __file__


def test_sample_pairs_is_not_deprecated():
    """``SerialSearcher.sample_pairs`` is the live part of the hedgehog searcher (analysis uses it)."""
    found = _deprecation_warnings(lambda: SerialSearcher.sample_pairs(SimpleNamespace()))
    assert not found
