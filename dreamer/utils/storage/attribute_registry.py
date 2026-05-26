"""
Central registry of trajectory attributes that can be computed by the pipeline.

A single dictionary maps an attribute name (the public, config-facing string) to
a function that, given a fully constructed ``TrajectoryAttributesHandler``,
returns a JSON-serialisable value.

The pipeline stages — analysis, search-stage producer, and search-stage
background workers — read their own ordered list of attribute names from
config and call ``compute_attribute(handler, name)`` for each.  Adding a new
attribute therefore takes one line in this file plus, if needed, a new method
on the handler.

The serialiser layer is intentionally tight:
    handler -> raw value -> JSON-safe value (str / float / list / None / ...)
This keeps SymPy / mpmath / numpy objects out of the JSONL files.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from dreamer.utils.storage.trajectory_attributes import TrajectoryAttributesHandler


# ---------------------------------------------------------------------------
# Small helpers that normalise raw handler outputs to JSON-safe forms
# ---------------------------------------------------------------------------

def _opt_float(value) -> float | None:
    """Return ``float(value)`` or ``None`` when the value is None."""
    return None if value is None else float(value)


def _list_of_str(values) -> list[str]:
    """Stringify each element — used for symbolic objects (eigenvalues, asymptotics)."""
    return [str(v) for v in values]


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------

AttributeComputer = Callable[["TrajectoryAttributesHandler"], Any]

ATTRIBUTE_REGISTRY: Dict[str, AttributeComputer] = {
    # Tier-1 — core scalars (always computed, main thread, free from the walk).
    "delta":             lambda h: float(h.delta()),
    "limit":             lambda h: float(h.limit()),
    "order":             lambda h: int(h.order()),
    "formula":           lambda h: h.formula_str(),

    # Tier-2 — heavier numerical / spectral attributes (background workers).
    "eigenvalues":       lambda h: _list_of_str(h.sorted_eigenvalues()),
    "spectral_gap":      lambda h: _opt_float(h.spectral_gap()),
    "gcd_slope":         lambda h: _opt_float(h.gcd_slope()),
    "convergence_class": lambda h: h.convergence_class(),

    # Tier-3 — symbolic / expensive attributes (post-process stage, deferred).
    "asymptotics":       lambda h: _list_of_str(h.asymptotics()),
    "kamidelta":         lambda h: _list_of_str(h.kamidelta()),
}


def compute_attribute(handler: "TrajectoryAttributesHandler", name: str) -> Any:
    """Compute a single registered attribute on *handler*.

    Raises ``KeyError`` if *name* is not registered — the misspelled-config
    case should fail loudly rather than silently skip an attribute.
    """
    if name not in ATTRIBUTE_REGISTRY:
        raise KeyError(
            f"Unknown attribute '{name}'. Registered: {sorted(ATTRIBUTE_REGISTRY)}"
        )
    return ATTRIBUTE_REGISTRY[name](handler)


def compute_attributes(
    handler: "TrajectoryAttributesHandler",
    names,
    on_error: str = "store",
) -> Dict[str, Any]:
    """Compute many attributes and collect them into a dict keyed by name.

    Parameters
    ----------
    handler:
        The handler to compute against.
    names:
        Iterable of attribute names (typically a config tuple).
    on_error:
        ``'store'`` — record the exception message under ``<name>_error`` and
        continue (default; lets a single misbehaving attribute not poison the
        whole pipeline).  ``'raise'`` — re-raise the first exception.

    Returns
    -------
    dict mapping each requested name to its computed value (or an
    ``<name>_error`` key when on_error='store' and a failure occurred).
    """
    out: Dict[str, Any] = {}
    for name in names:
        try:
            out[name] = compute_attribute(handler, name)
        except Exception as exc:  # pragma: no cover — behaviour driven by tests
            if on_error == "raise":
                raise
            out[f"{name}_error"] = str(exc)
    return out


def register_attribute(name: str, fn: AttributeComputer) -> None:
    """Register a custom attribute at runtime (e.g. for experiments).

    Overwrites any existing entry under the same name.
    """
    ATTRIBUTE_REGISTRY[name] = fn
