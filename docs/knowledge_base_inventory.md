# Knowledge base inventory

Domain reference data is registered in `app/knowledge/registry.py`. Algorithm weights, score multipliers, prompt text, and gender-filter *functions* are not datasets. The gender term list is a dataset because it is reference data. The filter functions stay in `knowledge_base.py`.

## Shared knowledge (`knowledge_base/{datasetId}`)

| Dataset | Source | Consumers | Required at calculation time |
| --- | --- | --- | --- |
| `rashi_organs` | `knowledge_base.RASHI_ORGANS` | engines, rules, diagnose | yes |
| `rashi_degrees` | `knowledge_base.RASHI_DEGREES` | engines | yes |
| `planet_diseases` | `knowledge_base.PLANET_DISEASES` | engines, rules | yes |
| `nakshatra_lords` | `knowledge_base.NAKSHATRA_LORDS` | engines | yes |
| `nakshatra_diseases` | `knowledge_base.NAKSHATRA_DISEASES` | `get_nakshatra_diseases` | yes |
| `planet_dosha` | `knowledge_base.PLANET_DOSHA` | engines | yes |
| `dosha_symptoms` | `knowledge_base.DOSHA_SYMPTOMS` | engines | yes |
| `planet_friends` | `knowledge_base.PLANET_FRIENDS` | engines | yes |
| `gender_exclusive_terms` | `knowledge_base.GENDER_EXCLUSIVE_TERMS` | gender filters | yes |
| `all_organs` | `rashi_truth_review.ALL_ORGANS` | rashi and organ engines | yes |
| `nakshatra_disease_to_superset_organs` | `rashi_truth_review` | rashi truth | yes |
| `planet_term_to_rashi_organs` | `rashi_truth_normalization` | rashi truth | yes |
| `term_to_systems` | `term_to_systems.py` | hitlist | yes |
| `system_display_names` | `term_to_systems.py` | hitlist | yes |
| `merge_groups` | `term_to_systems.py` | hitlist | yes |
| `term_symptom_rules` | `symptom_mapper.TERM_SYMPTOM_RULES` | classic chat | yes |
| `symptom_fallback` | `symptom_mapper.FALLBACK` | classic chat | yes |
| `sign_zones` | `zone_engine.SIGN_ZONES` | zone engine | yes |
| `zone_sign_lords` | `zone_engine.SIGN_LORDS` | zone engine | yes |
| `zodiac_signs` | `second_ascendant.ZODIAC_SIGNS` | second ascendant | yes |
| `nakshatra_order` | `second_ascendant.NAKSHATRAS` | second ascendant | yes |
| `second_ascendant_sign_lords` | `second_ascendant.SIGN_LORDS` | second ascendant | yes |
| `recommendation_food` | `combine_output.FOOD_TABLE` | final report | no |
| `recommendation_lifestyle` | `combine_output.LIFESTYLE_TABLE` | final report | no |
| `recommendation_wellness` | `combine_output.WELLNESS_TABLE` | final report | no |
| `parhora_table` | `parhora_table.py` | no runtime importer found | no, status `reference` |
| `disease_table_file` | `data/disease_table.json` | `disease_diagnose.py` also has its own snapshot | no, status `reference` |

`parsed_constellation.json` and `constellation_dump.csv` are generated extracts. They are not runtime sources. The spreadsheet `Manoj-Lord-Organs-Rashi-Nakshatra-Disease.xlsx` is the input to `parse_excel.py`, not a request-path dataset.

Publishing a Firestore dataset updates the module object in place (`dict.clear` / `update`, or list slice assignment). `rashi_degrees` lists are turned back into pairs so existing comparisons keep seeing tuples. A test seeds the current tables, publishes them, and checks `RASHI_ORGANS`, `RASHI_DEGREES`, and `get_nakshatra_diseases` against the pre-publish copies.

## RL configuration (`rl_config/{datasetId}`)

These stay out of `knowledge_base`:

| Dataset | Bootstrap object | Current Firestore collection |
| --- | --- | --- |
| `symptom_library` | `BOOTSTRAP_SYMPTOM_LIBRARY` | `symptom_library` |
| `organ_aliases` | `BOOTSTRAP_ORGAN_ALIASES` | `organ_aliases` |
| `branch_hints` | `BOOTSTRAP_BRANCH_HINTS` | `branch_hints` |
| `organ_map` | `BOOTSTRAP_ORGAN_MAP` | `chat_maps/organ_map` |
| `disease_map` | `BOOTSTRAP_DISEASE_MAP` | `chat_maps/disease_map` |
| `stop_words` | `BOOTSTRAP_STOP_WORDS` | `stop_words` |
| `organ_branch_families` | `BOOTSTRAP_ORGAN_BRANCH_FAMILIES` | `organ_branch_families` |

The live RL process still reads the existing collections through `rl_enhancer.config_loader`. The new path is the migration target and the strict-mode failure mode. Counts were not taken from production. The bootstrap sizes in code are the seed, not a census of the live database.

## Not datasets

`SOURCE_WEIGHTS`, Shadbala schedules, hitlist multipliers, and the safety prompt are algorithm or product logic. They were left in their modules.

## Cache

`KnowledgeBaseService` keeps a process cache. A repeat read inside `KB_CACHE_TTL_SECONDS` (default 60) does not list the records again. A changed `dataset_version` reloads them. A missing dataset raises `MissingDatasetError`. A Firestore error raises `FirestoreUnavailableError`. Neither returns an empty table.
