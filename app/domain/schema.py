"""Target document fields. Schema version 2 is the Firestore-first model."""

from __future__ import annotations

SCHEMA_VERSION = 2

# Stay under Firestore's 1 MiB document limit with room for indexes and encoding.
SOFT_DOCUMENT_BYTES = 900_000
PART_BYTES = 700_000

USER_FIELDS = (
    "internal_user_id",
    "clerk_user_id",
    "email",
    "role",
    "charts_used",
    "active_patient_id",
    "created_at",
    "updated_at",
    "schema_version",
)

PATIENT_FIELDS = (
    "patient_id",
    "created_at",
    "updated_at",
    "algo_version",
    "hidden",
    "top_organs",
    "last_chat_at",
    "chats_count",
    "latest_run_id",
    "schema_version",
)

CHAT_MODES = ("classic", "agentic_rules", "model_shadow", "model_assisted")
RL_HINT_MODES = ("off", "shadow", "enabled")

# The literal path knowledge_base/datasets/{datasetId}/records/{recordId}
# has an odd number of segments, so it is not a legal Firestore document path.
# Dataset metadata lives on the dataset document and records live underneath it.
KNOWLEDGE_COLLECTION = "knowledge_base"
RL_CONFIG_COLLECTION = "rl_config"
RL_MODELS_COLLECTION = "rl_models"
KB_CHUNKS_COLLECTION = "kb_chunks"
JOB_RECORDS_COLLECTION = "job_records"
SYSTEM_CONFIG_COLLECTION = "system_config"
CHAT_SESSION_INDEX = "chat_session_index"
CLERK_INDEX = "clerk_index"
USERS = "users"
