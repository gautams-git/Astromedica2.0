"""Alternative weight profiles the agent can A/B test.

Each profile is applied by monkey-patching SOURCE_WEIGHTS / HL_MULTIPLIERS
ONLY for the duration of one call — restored afterwards.
"""
from contextlib import contextmanager
from typing import Dict

import hitlist_engine as he


PROFILES: Dict[str, Dict] = {
    # Current defaults from your file
    "clinical": {
        "SOURCE_WEIGHTS": dict(he.SOURCE_WEIGHTS),
        "HL_MULTIPLIERS": dict(he.HL_MULTIPLIERS),
    },
    # Bolder planet emphasis — surfaces single-source signals
    "planet_first": {
        "SOURCE_WEIGHTS": {
            **he.SOURCE_WEIGHTS,
            "P1": 10.0,
            "P2": 5.0,
            "P3": 3.0,
        },
        "HL_MULTIPLIERS": {
            **he.HL_MULTIPLIERS,
            "HL1": 2.5,
            "HL2": 4.0,
            "HL3": 5.0,
        },
    },
    # Flatter — favors cross-source agreement
    "consensus_first": {
        "SOURCE_WEIGHTS": {
            **he.SOURCE_WEIGHTS,
            "P1": 5.0,
            "P2": 4.0,
            "P3": 3.5,
            "P4": 3.0,
            "P5": 2.5,
        },
        "HL_MULTIPLIERS": {
            "HL1": 1.5,
            "HL2": 5.0,
            "HL3": 5.0,
        },
    },
}


@contextmanager
def profile_context(name: str):
    """Temporarily swap weights; restore on exit."""
    if name not in PROFILES:
        yield
        return
    saved_src = dict(he.SOURCE_WEIGHTS)
    saved_hl = dict(he.HL_MULTIPLIERS)
    try:
        he.SOURCE_WEIGHTS.clear()
        he.SOURCE_WEIGHTS.update(PROFILES[name]["SOURCE_WEIGHTS"])
        he.HL_MULTIPLIERS.clear()
        he.HL_MULTIPLIERS.update(PROFILES[name]["HL_MULTIPLIERS"])
        yield
    finally:
        he.SOURCE_WEIGHTS.clear()
        he.SOURCE_WEIGHTS.update(saved_src)
        he.HL_MULTIPLIERS.clear()
        he.HL_MULTIPLIERS.update(saved_hl)


def top_terms(result: Dict, n: int = 4):
    """Extract top-N group labels for stability comparison."""
    return [g["label"] for g in result.get("all_groups", [])[:n]]