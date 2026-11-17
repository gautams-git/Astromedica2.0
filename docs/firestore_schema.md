# Firestore schema

Project: `astrohealth-509609`. The API uses the Admin SDK. `firestore.rules` denies every client read and write.

Schema version on new documents is `2`.

## Path correction

`knowledge_base/datasets/{datasetId}/records/{recordId}` has five segments, so the last segment is a collection, not a document. The stored layout is:

- `knowledge_base/{datasetId}` dataset metadata
- `knowledge_base/{datasetId}/records/{recordId}` one record

`rl_config/{datasetId}/records/{recordId}` is already a valid four-segment path. Dataset metadata lives on `rl_config/{datasetId}`.

## Account and patient

`users/{internalUserId}`

`internal_user_id`, `clerk_user_id`, `email`, `role`, `charts_used`, `active_patient_id`, `created_at`, `updated_at`, `schema_version`

`clerk_index/{clerkUserId}` maps that Clerk id to one internal id. Creation runs in a transaction. A second create returns the existing id. A different internal id for the same Clerk id is rejected.

`users/{internalUserId}/patients/{patientId}`

`patient_id`, `created_at`, `updated_at`, `algo_version`, `hidden`, `top_organs`, `last_chat_at`, `chats_count`, `latest_run_id`, `schema_version`, `owner_internal_user_id`

`users/{internalUserId}/patients/{patientId}/private/identity`

Name, birth date, birth time, birth place, latitude, longitude, and gender. When `IDENTITY_ENCRYPTION_KEY` is set, those fields are Fernet ciphertext and are not stored in clear text.

`users/{internalUserId}/patients/{patientId}/private/source`

Prokerala chart, Prokerala dasha, raw source, provider, version, retrieval time, and input hash. Nested lists are stored as canonical JSON. Values past 900 KiB are split into `parts/{field_0000}` documents. Parts are reassembled on read. Nothing is truncated.

## Runs

`users/{internalUserId}/patients/{patientId}/runs/{runId}`

`run_id`, `run_number`, `algorithm_version`, `created_at`, `status`, `input_reference`, `input_hash`, combined result, `result_summary`, `execution_metadata`, `schema_version`

`users/{internalUserId}/patients/{patientId}/runs/{runId}/algorithms/{algorithmId}`

`algorithm_id`, `algorithm_version`, `status`, `result` or split JSON, `created_at`, `execution_metadata`, `schema_version`, `result_sha256`

Current algorithm ids written by the chart pipeline include `rule1`, `dasha`, `complete_analysis`, `disease_filter`, `hitlist`, `rashi_correlation`, `organ_truth_correlation`, `dasha_chat_priority`, `nakshatra_diseases`, `planet_diseases`, `most_probable`, `top_diseases`, `diagnosis`, `zone_analysis`, `second_ascendant`, `imp_rashi_distance`, `birth_current_distance`, `health_forecast`, and `combined_algo`. The repository accepts any id the pipeline produces.

## Chat and RL output

`users/{internalUserId}/patients/{patientId}/chats/{chatId}`

`chat_id`, `status`, `completed`, `created_at`, `updated_at`, `completed_at`, `turn_count`, `findings`, `cluster_scores`, `associated_run_id`, `chat_algorithm_version`, `schema_version`

Turns are not an array on that document.

`.../chats/{chatId}/turns/{turnId}`

`turn_id`, `sequence_number`, `role`, `content`, `question_type`, `cluster_rank`, `branch_id`, `answer_result`, `timestamp`, `metadata`, `schema_version`

`.../chats/{chatId}/rl_outputs/{rlRunId}`

`rl_run_id`, `model_version`, `input_chat_reference`, `output`, `status`, `created_at`, `completed_at`, `execution_metadata`, `schema_version`

`chat_session_index/{sessionId}` stores only `internal_user_id`, `patient_id`, and `chat_id` so a worker can find a session without scanning patients.

## Audit

`users/{internalUserId}/patients/{patientId}/audit/{eventId}`

`event_id`, `event_type`, `actor`, `timestamp`, `source`, `metadata`, `schema_version`

Identity fields and any metadata key containing `secret`, `token`, or `name` are dropped. Events used by the new path: `algorithm_run`, `chat_started`, `chat_completed`, `rl_completed`. `chart_created` and `chart_corrected` use the same append method.

## Shared data and operations

`knowledge_base/{datasetId}` and `records/{recordId}` hold shared domain tables. `rl_config/{datasetId}` holds RL configuration. RL writes do not update `knowledge_base`.

`rl_models/{modelVersion}` stores `artifact_uri`, `artifact_checksum`, `training_dataset_version`, `training_timestamp`, `evaluation_metrics`, `status`, and `created_at`. `artifact_uri` must be a `gs://` URI. Weights stay in Cloud Storage or the existing weights volume.

`kb_chunks/{chunkId}` is reserved for later RAG chunks. There is no vector index.

`job_records/{jobId}` copies status from the in-memory RL queue.

`system_config/{configId}` rejects keys that look like secrets.

## Legacy collections kept in place

`users/{id}/inputs`, `chart_inputs`, `prokerala_results`, `algorithm_results`, `combined_results`, `chart_sources`, `identity_vault`, `account`, `tester_charts`, `admin_audit`, plus top-level `symptom_library`, `organ_aliases`, `branch_hints`, `chat_maps`, `stop_words`, `organ_branch_families`, and `chat_logs`.

Migration copies. It does not delete these.
