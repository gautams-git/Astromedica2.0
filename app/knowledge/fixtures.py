"""Load the current in-repo domain tables. These are the seed source, not a production fallback."""

from __future__ import annotations

import json
import os
from typing import Any


def _records(value: Any, mode: str) -> dict:
    if mode == "blob":
        return {"_blob": value}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (list, tuple, set)):
        return {"_blob": list(value)}
    return {"_blob": value}


def load_fixture_records() -> dict[str, dict]:
    import knowledge_base
    import parhora_table
    import rashi_truth_normalization
    import rashi_truth_review
    import second_ascendant
    import symptom_mapper
    import term_to_systems
    import zone_engine
    from combine_output import FOOD_TABLE, LIFESTYLE_TABLE, WELLNESS_TABLE

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    disease_path = os.path.join(root, "data", "disease_table.json")
    with open(disease_path, encoding="utf-8") as handle:
        disease_table = json.load(handle)

    from rl_enhancer.bootstrap_data import (
        BOOTSTRAP_BRANCH_HINTS,
        BOOTSTRAP_DISEASE_MAP,
        BOOTSTRAP_ORGAN_ALIASES,
        BOOTSTRAP_ORGAN_BRANCH_FAMILIES,
        BOOTSTRAP_ORGAN_MAP,
        BOOTSTRAP_STOP_WORDS,
        BOOTSTRAP_SYMPTOM_LIBRARY,
    )

    knowledge = {
        "rashi_organs": _records(knowledge_base.RASHI_ORGANS, "map"),
        "rashi_degrees": _records(knowledge_base.RASHI_DEGREES, "map"),
        "planet_diseases": _records(knowledge_base.PLANET_DISEASES, "map"),
        "nakshatra_lords": _records(knowledge_base.NAKSHATRA_LORDS, "map"),
        "nakshatra_diseases": _records(knowledge_base.NAKSHATRA_DISEASES, "map"),
        "planet_dosha": _records(knowledge_base.PLANET_DOSHA, "map"),
        "dosha_symptoms": _records(knowledge_base.DOSHA_SYMPTOMS, "map"),
        "planet_friends": _records(knowledge_base.PLANET_FRIENDS, "map"),
        "gender_exclusive_terms": _records(knowledge_base.GENDER_EXCLUSIVE_TERMS, "map"),
        "all_organs": _records(rashi_truth_review.ALL_ORGANS, "blob"),
        "nakshatra_disease_to_superset_organs": _records(rashi_truth_review.NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS, "map"),
        "planet_term_to_rashi_organs": _records(rashi_truth_normalization.PLANET_TERM_TO_RASHI_ORGANS, "map"),
        "term_to_systems": _records(term_to_systems.TERM_TO_SYSTEMS, "map"),
        "system_display_names": _records(term_to_systems.SYSTEM_DISPLAY_NAMES, "map"),
        "merge_groups": _records(term_to_systems.MERGE_GROUPS, "map"),
        "term_symptom_rules": _records(symptom_mapper.TERM_SYMPTOM_RULES, "blob"),
        "symptom_fallback": _records(symptom_mapper.FALLBACK, "map"),
        "sign_zones": _records(zone_engine.SIGN_ZONES, "map"),
        "zone_sign_lords": _records(zone_engine.SIGN_LORDS, "map"),
        "zodiac_signs": _records(second_ascendant.ZODIAC_SIGNS, "blob"),
        "nakshatra_order": _records(second_ascendant.NAKSHATRAS, "blob"),
        "second_ascendant_sign_lords": _records(second_ascendant.SIGN_LORDS, "map"),
        "recommendation_food": _records(FOOD_TABLE, "map"),
        "recommendation_lifestyle": _records(LIFESTYLE_TABLE, "map"),
        "recommendation_wellness": _records(WELLNESS_TABLE, "map"),
        "parhora_table": _records(parhora_table.PARHORA_TABLE, "blob"),
        "disease_table_file": _records(disease_table, "blob"),
    }
    rl = {
        "symptom_library": _records(BOOTSTRAP_SYMPTOM_LIBRARY, "map"),
        "organ_aliases": _records(BOOTSTRAP_ORGAN_ALIASES, "map"),
        "branch_hints": _records(BOOTSTRAP_BRANCH_HINTS, "map"),
        "organ_map": _records(BOOTSTRAP_ORGAN_MAP, "map"),
        "disease_map": _records(BOOTSTRAP_DISEASE_MAP, "blob"),
        "stop_words": _records(sorted(BOOTSTRAP_STOP_WORDS), "blob"),
        "organ_branch_families": _records(BOOTSTRAP_ORGAN_BRANCH_FAMILIES, "map"),
    }
    return {"knowledge": knowledge, "rl_config": rl}
