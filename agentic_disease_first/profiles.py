"""Alternative weight profiles the agent can A/B test.

disease_first_logic scores with three module-level tables:
  SOURCE_WEIGHTS, PLANET_LAYER_WEIGHTS, RASHI_LAYER_WEIGHTS.
disease_compare_logic imports the very same dict objects, so swapping them in place changes
both engines. A profile is applied by swapping those tables ONLY for the duration of one
engine call and restoring them afterwards.

The swap changes shared module state, so every engine run goes through PROFILE_LOCK (the
agentic_disease_compare folder uses the same lock). Threads therefore never see another
thread's weights. Under the GIL this costs no throughput.

"planet_heavy" and "rashi_heavy" are experiments (today's weights x1.5 on one layer), not
tuned values: they are only tried when AGENTIC_MULTI_PROFILE=true.
"""
import threading
from contextlib import contextmanager
from typing import Dict

import disease_first_logic as dfl

PROFILE_LOCK = threading.RLock()

WEIGHT_TABLES = ("SOURCE_WEIGHTS", "PLANET_LAYER_WEIGHTS", "RASHI_LAYER_WEIGHTS")


def _scaled(table: Dict[str, float], factor: float) -> Dict[str, float]:
    return {key: round(value * factor, 3) for key, value in table.items()}


def _snapshot() -> Dict[str, Dict[str, float]]:
    return {name: dict(getattr(dfl, name)) for name in WEIGHT_TABLES}


_BASE = _snapshot()

PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    # Today's weights. Never swapped: the live tables are used as they are.
    "clinical": _BASE,
    # Louder planet-organ layer
    "planet_heavy": {**_BASE, "PLANET_LAYER_WEIGHTS": _scaled(_BASE["PLANET_LAYER_WEIGHTS"], 1.5)},
    # Louder rashi-organ layer
    "rashi_heavy": {**_BASE, "RASHI_LAYER_WEIGHTS": _scaled(_BASE["RASHI_LAYER_WEIGHTS"], 1.5)},
}


@contextmanager
def weights_context(weights: Dict[str, Dict[str, float]] | None):
    """Run a block under the lock; swap the shared tables when `weights` is given, then restore."""
    with PROFILE_LOCK:
        if not weights:
            yield
            return
        saved = _snapshot()
        try:
            for name in WEIGHT_TABLES:
                table = getattr(dfl, name)
                table.clear()
                table.update(weights[name])
            yield
        finally:
            for name in WEIGHT_TABLES:
                table = getattr(dfl, name)
                table.clear()
                table.update(saved[name])


def profile_context(name: str):
    """Lock + (for non-clinical profiles) temporarily swap the weights."""
    return weights_context(None if name == "clinical" or name not in PROFILES else PROFILES[name])


def top_terms(result: Dict, n: int = 3):
    """Top-N disease names, used to compare how stable the profiles are."""
    return [d["disease"] for d in (result.get("top_diseases") or [])[:n]]
