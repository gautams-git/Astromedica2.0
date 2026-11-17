"""Copy Firestore-backed reference tables into the modules the engines already import.

The engines keep their current names. When a dataset is published, the existing
dict or list object is updated in place so earlier `from module import TABLE`
bindings see the same contents. Scoring code is not touched.
"""

from __future__ import annotations

import importlib

_BINDINGS = (
    ("rashi_organs", "knowledge_base", "RASHI_ORGANS"),
    ("rashi_degrees", "knowledge_base", "RASHI_DEGREES"),
    ("planet_diseases", "knowledge_base", "PLANET_DISEASES"),
    ("nakshatra_lords", "knowledge_base", "NAKSHATRA_LORDS"),
    ("nakshatra_diseases", "knowledge_base", "NAKSHATRA_DISEASES"),
    ("planet_dosha", "knowledge_base", "PLANET_DOSHA"),
    ("dosha_symptoms", "knowledge_base", "DOSHA_SYMPTOMS"),
    ("planet_friends", "knowledge_base", "PLANET_FRIENDS"),
    ("gender_exclusive_terms", "knowledge_base", "GENDER_EXCLUSIVE_TERMS"),
    ("all_organs", "rashi_truth_review", "ALL_ORGANS"),
    ("nakshatra_disease_to_superset_organs", "rashi_truth_review", "NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS"),
    ("planet_term_to_rashi_organs", "rashi_truth_normalization", "PLANET_TERM_TO_RASHI_ORGANS"),
    ("term_to_systems", "term_to_systems", "TERM_TO_SYSTEMS"),
    ("system_display_names", "term_to_systems", "SYSTEM_DISPLAY_NAMES"),
    ("merge_groups", "term_to_systems", "MERGE_GROUPS"),
    ("term_symptom_rules", "symptom_mapper", "TERM_SYMPTOM_RULES"),
    ("symptom_fallback", "symptom_mapper", "FALLBACK"),
    ("sign_zones", "zone_engine", "SIGN_ZONES"),
    ("zone_sign_lords", "zone_engine", "SIGN_LORDS"),
    ("zodiac_signs", "second_ascendant", "ZODIAC_SIGNS"),
    ("nakshatra_order", "second_ascendant", "NAKSHATRAS"),
    ("second_ascendant_sign_lords", "second_ascendant", "SIGN_LORDS"),
)


def _restore_degrees(records: dict) -> dict:
    restored = {}
    for key, value in records.items():
        if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
            restored[key] = (value[0], value[1])
        else:
            restored[key] = value
    return restored


def materialise(dataset_id: str, records: dict):
    if dataset_id == "rashi_degrees":
        return _restore_degrees(records)
    if "_blob" in records and len(records) == 1:
        return records["_blob"]
    return dict(records)


def publish(service) -> list[str]:
    updated = []
    for dataset_id, module_name, attr in _BINDINGS:
        records = service.require(dataset_id)
        module = importlib.import_module(module_name)
        target = getattr(module, attr)
        value = materialise(dataset_id, records)
        if isinstance(target, dict) and isinstance(value, dict):
            target.clear()
            target.update(value)
        elif isinstance(target, list) and isinstance(value, list):
            target[:] = value
        else:
            raise TypeError(f"{module_name}.{attr} cannot accept dataset {dataset_id}")
        updated.append(dataset_id)
    return updated
