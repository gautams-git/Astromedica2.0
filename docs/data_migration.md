# Data migration

Production data was not read or written by this change. The command refuses `--apply` against project `astrohealth-509609` unless `--allow-production-project` is present. Do not pass that flag until a backup exists and the dry-run manifest has been reviewed.

## Command

```bash
# Dry-run. Requires the emulator or FIRESTORE_BACKEND=firestore.
python -m migrations.migrate_firestore --manifest /tmp/astromedica-manifest.json

# Copy missing target documents. Existing targets are skipped.
python -m migrations.migrate_firestore --apply --manifest /tmp/astromedica-manifest.json
```

Startup does not migrate. There is no delete step.

## Mapping

| Source | Target | Rule |
| --- | --- | --- |
| Embedded domain tables in `knowledge_base.py`, `rashi_truth_review.py`, `rashi_truth_normalization.py`, `term_to_systems.py`, `symptom_mapper.py`, `zone_engine.py`, `second_ascendant.py`, `combine_output.py`, `parhora_table.py`, `data/disease_table.json` | `knowledge_base/{datasetId}/records/{recordId}` | Create if absent. Skip if the checksum matches. Skip and report `drift` if the checksum differs. |
| `rl_enhancer/bootstrap_data.py` and the current RL collections | `rl_config/{datasetId}/records/{recordId}` | Same create-if-absent rule. Not copied into `knowledge_base`. |
| `users/{id}/identity_vault/{patientId}` | `patients/{patientId}/private/identity` | Copy when the target is missing. Source remains. |
| `users/{id}/chart_sources/{patientId}` | `patients/{patientId}/private/source` | Copy when the target is missing. |
| `users/{id}/algorithm_results/*` grouped by `input_id` | `patients/{patientId}/runs/{inputId}/algorithms/{algorithm}` | Create the run only when that run id is missing. |
| `chat_logs/{doc}` | `patients/{patientId}/chats/legacy_{doc}/turns/{sequence}` | Copied only for users who already own that patient. Unowned logs stay unassigned. |

User documents are not replaced. `charts_used` and `email` already stored on a user are kept.

## Reconciliation

The apply report includes `created` counts and `reconciled.checksums_matched` for knowledge datasets. A dataset counts as matched only when the stored checksum equals the canonical JSON checksum of the fixture records. Treat `drift_skip` as a stop: inspect the document before any later update.

Chat logs without an owning `patients/{patientId}` document are counted in `legacy_chat_logs` and are not invented under a new user.

## Backups

Export the database before a production apply:

```bash
gcloud firestore export gs://BUCKET/astromedica-pre-migration
```

The bucket is not created by this repository. Keep the export until the manifest checksums match and the application has been checked against a non-production copy.

## Tests

`tests/test_data_plane.py` loads a memory store, checks that dry-run does not write, applies once, checks the legacy identity document is still present, checks the copied hitlist output, and applies again with zero new knowledge, identity, and run documents. It also checks the production-project guard. It does not open a network connection.
