"""Weight profiles the Rashi agent can A/B test.

rashi_correlation_engine scores with three module-level tables:
  PRIORITY_WEIGHTS, LAYER_WEIGHTS, ANCHOR_WEIGHTS.
The engine reads them at call time, so a profile is applied by swapping those tables in place
ONLY for the duration of one engine call and restoring them afterwards (the same strategy the
disease agent uses on its own tables).

* The baseline is an immutable snapshot (read-only mappings): nothing here can change it.
* Every engine run goes through PROFILE_LOCK, so threads never see another thread's weights.
  This lock covers only the Rashi tables; the disease agents have their own lock and tables.
* The restore runs in `finally`, from a snapshot taken inside the lock, so an engine error
  cannot leave swapped weights behind.

"nakshatra_heavy" and "anchor_heavy" are experiments (nakshatra layer 1.25 -> 1.875, anchors x1.5),
not tuned values: they are only tried when AGENTIC_RASHI_MULTI_PROFILE=true.
"""
import threading
from contextlib import contextmanager
from types import MappingProxyType
from typing import Dict, Mapping

import rashi_correlation_engine as rce
from agentic_disease_first.profiles import _scaled   # same helper the disease profiles use

PROFILE_LOCK = threading.RLock()

WEIGHT_TABLES = (
    "PRIORITY_WEIGHTS",
    "LAYER_WEIGHTS",
    "ANCHOR_WEIGHTS",
)

NAKSHATRA_HEAVY_LAYER = 1.875   # baseline 1.25 x 1.5
ANCHOR_HEAVY_FACTOR = 1.5


def _snapshot() -> Dict[str, Dict[str, float]]:
    return {name: dict(getattr(rce, name)) for name in WEIGHT_TABLES}


def _freeze(tables: Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    return MappingProxyType({name: MappingProxyType(dict(table)) for name, table in tables.items()})


with PROFILE_LOCK:
    _BASE = _freeze(_snapshot())

PROFILES: Mapping[str, Mapping[str, Mapping[str, float]]] = MappingProxyType({
    # Today's weights. Never swapped: the live tables are used as they are.
    "clinical": _BASE,
    # Louder nakshatra (disease-style) layer
    "nakshatra_heavy": _freeze({
        **_BASE,
        "LAYER_WEIGHTS": {**_BASE["LAYER_WEIGHTS"], "nakshatra": NAKSHATRA_HEAVY_LAYER},
    }),
    # Louder P6 / P7 anchors
    "anchor_heavy": _freeze({
        **_BASE,
        "ANCHOR_WEIGHTS": _scaled(_BASE["ANCHOR_WEIGHTS"], ANCHOR_HEAVY_FACTOR),
    }),
})


@contextmanager
def weights_context(weights: Mapping[str, Mapping[str, float]] | None):
    """Run a block under the lock; swap the engine tables when `weights` is given, then restore."""
    with PROFILE_LOCK:
        if not weights:
            yield
            return
        saved = _snapshot()
        try:
            for name in WEIGHT_TABLES:
                table = getattr(rce, name)
                table.clear()
                table.update(weights[name])
            yield
        finally:
            for name in WEIGHT_TABLES:
                table = getattr(rce, name)
                table.clear()
                table.update(saved[name])


def profile_context(name: str):
    """Lock + (for non-clinical profiles) temporarily swap the weights."""
    return weights_context(None if name == "clinical" or name not in PROFILES else PROFILES[name])


def top_terms(result, n=3):
    """Top-N group labels, used to compare how stable the profiles are."""
    return [
        g["label"]
        for g in (result.get("top3") or [])[:n]
    ]
