"""Alternative weight profiles the agent can A/B test.

disease_compare_logic scores with the same three tables as disease_first_logic
(SOURCE_WEIGHTS, PLANET_LAYER_WEIGHTS, RASHI_LAYER_WEIGHTS): it imports the very same dict
objects. A profile swaps those tables in place ONLY for one engine call and restores them
afterwards. That also changes disease_first_logic while it runs, so both agents share ONE
lock (PROFILE_LOCK, defined in agentic_disease_first.profiles).

The close-match rules (similarity cut-offs in _is_very_close_disease) are hard-coded in the
engine and are not part of any profile.

"planet_heavy" and "rashi_heavy" are experiments (today's weights x1.5 on one layer), not
tuned values: they are only tried when AGENTIC_MULTI_PROFILE=true.
"""
from typing import Dict

import disease_compare_logic as dcl
import disease_first_logic as dfl
from agentic_disease_first.profiles import PROFILE_LOCK, WEIGHT_TABLES, weights_context  # noqa: F401

# The swap only works while both engines read the same dict objects.
for _table in WEIGHT_TABLES:
    if getattr(dcl, _table) is not getattr(dfl, _table):
        raise RuntimeError(
            f"disease_compare_logic.{_table} is no longer the same object as "
            f"disease_first_logic.{_table}; the agentic profiles cannot swap it."
        )


def _scaled(table: Dict[str, float], factor: float) -> Dict[str, float]:
    return {key: round(value * factor, 3) for key, value in table.items()}


_BASE = {name: dict(getattr(dfl, name)) for name in WEIGHT_TABLES}

PROFILES: Dict[str, Dict[str, Dict[str, float]]] = {
    # Today's weights. Never swapped: the live tables are used as they are.
    "clinical": _BASE,
    # Louder planet-organ layer
    "planet_heavy": {**_BASE, "PLANET_LAYER_WEIGHTS": _scaled(_BASE["PLANET_LAYER_WEIGHTS"], 1.5)},
    # Louder rashi-organ layer
    "rashi_heavy": {**_BASE, "RASHI_LAYER_WEIGHTS": _scaled(_BASE["RASHI_LAYER_WEIGHTS"], 1.5)},
}


def profile_context(name: str):
    """Lock + (for non-clinical profiles) temporarily swap the weights."""
    return weights_context(None if name == "clinical" or name not in PROFILES else PROFILES[name])


def top_terms(result: Dict, n: int = 3):
    """Top-N disease names, used to compare how stable the profiles are."""
    return [d["disease"] for d in (result.get("top_diseases") or [])[:n]]
