"""Which embedded tables are shared domain data, and which are RL configuration.

Algorithm weights, scoring multipliers, and prompt text are not datasets.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_name: str
    description: str
    category: str  # knowledge | rl_config | reference
    record_mode: str  # map | blob
    source_files: tuple[str, ...]
    status: str = "active"
    required: bool = True


KNOWLEDGE_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("rashi_organs", "Rashi organs", "Sign to body-part reference used by the engines.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("rashi_degrees", "Rashi degree spans", "Sidereal degree spans for rashis.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("planet_diseases", "Planet diseases", "Planet to disease-term reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("nakshatra_lords", "Nakshatra lords", "Nakshatra lord reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("nakshatra_diseases", "Nakshatra diseases", "Pada-aware nakshatra disease reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("planet_dosha", "Planet dosha", "Planet to dosha reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("dosha_symptoms", "Dosha symptoms", "Dosha symptom reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("planet_friends", "Planet friends", "Natural friendship reference.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("gender_exclusive_terms", "Gender terms", "Terms filtered by gender. This is reference data, not a score.", "knowledge", "map", ("knowledge_base.py",)),
    DatasetSpec("all_organs", "Organ superset", "Canonical organ names used by the rashi truth review.", "knowledge", "blob", ("rashi_truth_review.py",)),
    DatasetSpec("nakshatra_disease_to_superset_organs", "Disease organ map", "Disease term to organ superset.", "knowledge", "map", ("rashi_truth_review.py",)),
    DatasetSpec("planet_term_to_rashi_organs", "Planet term organs", "Planet term to rashi organs.", "knowledge", "map", ("rashi_truth_normalization.py",)),
    DatasetSpec("term_to_systems", "Term to systems", "Medical term to body-system map.", "knowledge", "map", ("term_to_systems.py",)),
    DatasetSpec("system_display_names", "System display names", "Display names for body systems.", "knowledge", "map", ("term_to_systems.py",)),
    DatasetSpec("merge_groups", "System merge groups", "System names that the hitlist treats as one group.", "knowledge", "map", ("term_to_systems.py",)),
    DatasetSpec("term_symptom_rules", "Symptom question rules", "Question-bank rules used by the classic chat.", "knowledge", "blob", ("symptom_mapper.py",)),
    DatasetSpec("symptom_fallback", "Symptom fallback questions", "Fallback questions when a term has no specific rule.", "knowledge", "map", ("symptom_mapper.py",)),
    DatasetSpec("sign_zones", "Sign zones", "South-Indian zone numbers for signs.", "knowledge", "map", ("zone_engine.py",)),
    DatasetSpec("zone_sign_lords", "Zone sign lords", "Sign lords used by the zone engine.", "knowledge", "map", ("zone_engine.py",)),
    DatasetSpec("zodiac_signs", "Zodiac signs", "Sign order used by the second-ascendant calculation.", "knowledge", "blob", ("second_ascendant.py",)),
    DatasetSpec("nakshatra_order", "Nakshatra order", "Nakshatra order used by the second-ascendant calculation.", "knowledge", "blob", ("second_ascendant.py",)),
    DatasetSpec("second_ascendant_sign_lords", "Second ascendant lords", "Sign lords paired with the second-ascendant calculation.", "knowledge", "map", ("second_ascendant.py",)),
    DatasetSpec("recommendation_food", "Food recommendations", "Post-chat food table.", "knowledge", "map", ("combine_output.py",), required=False),
    DatasetSpec("recommendation_lifestyle", "Lifestyle recommendations", "Post-chat lifestyle table.", "knowledge", "map", ("combine_output.py",), required=False),
    DatasetSpec("recommendation_wellness", "Wellness recommendations", "Post-chat wellness table.", "knowledge", "map", ("combine_output.py",), required=False),
    DatasetSpec("parhora_table", "Parhora table", "Orphan parhora reference. Not on the request path.", "reference", "blob", ("parhora_table.py",), status="reference", required=False),
    DatasetSpec("disease_table_file", "Disease table file", "JSON snapshot of tables that also live in knowledge_base.py.", "reference", "blob", ("data/disease_table.json",), status="reference", required=False),
)

RL_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("symptom_library", "RL symptom library", "Disease to symptom phrases learned and seeded for RL.", "rl_config", "map", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("organ_aliases", "RL organ aliases", "Alias to canonical organ for the RL parser.", "rl_config", "map", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("branch_hints", "RL branch hints", "Disease to question branch. Hints never override chat safety.", "rl_config", "map", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("organ_map", "RL organ map", "Chat organ normalisation map.", "rl_config", "map", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("disease_map", "RL disease map", "Chat disease patterns.", "rl_config", "blob", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("stop_words", "RL stop words", "Tokens ignored by the RL parser.", "rl_config", "blob", ("rl_enhancer/bootstrap_data.py",)),
    DatasetSpec("organ_branch_families", "RL organ branch families", "Organ to branch family.", "rl_config", "map", ("rl_enhancer/bootstrap_data.py",)),
)


def spec_by_id(dataset_id: str) -> DatasetSpec | None:
    for spec in (*KNOWLEDGE_SPECS, *RL_SPECS):
        if spec.dataset_id == dataset_id:
            return spec
    return None
