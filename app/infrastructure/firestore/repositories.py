"""Patient, run, chat, audit, knowledge, and job repositories."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from app.core.clock import utc_now
from app.core.config import load_data_settings
from app.core.errors import DocumentTooLargeError, ForgedIdentityError, OwnershipError
from app.core.security import assert_owner, require_document_id
from app.domain.schema import (
    CHAT_SESSION_INDEX,
    CLERK_INDEX,
    JOB_RECORDS_COLLECTION,
    KB_CHUNKS_COLLECTION,
    KNOWLEDGE_COLLECTION,
    RL_CONFIG_COLLECTION,
    RL_MODELS_COLLECTION,
    SCHEMA_VERSION,
    SOFT_DOCUMENT_BYTES,
    SYSTEM_CONFIG_COLLECTION,
    USERS,
)
from app.infrastructure.firestore.codec import (
    byte_length,
    canonical_json,
    decode_result,
    encode_result,
    sha256_text,
    split_text,
)
from app.infrastructure.firestore.crypto import decrypt_mapping, encrypt_mapping, encryption_active

_IDENTITY_KEYS = (
    "name",
    "date_of_birth",
    "birth_time",
    "birth_place",
    "latitude",
    "longitude",
    "gender",
    "father_name",
    "mother_name",
)


def _json_safe(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _record_id(key: str) -> str:
    if key and len(key.encode("utf-8")) < 400 and "/" not in key and key not in {".", ".."}:
        return key
    return "h_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


class DataPlane:
    """All durable writes go through this object. Algorithms do not."""

    def __init__(self, store: Any, *, encryption_key: str | None = None) -> None:
        self.store = store
        settings = load_data_settings()
        self.encryption_key = encryption_key if encryption_key is not None else settings.identity_encryption_key
        self.encryption_required = settings.identity_encryption_required

    # ── users ──────────────────────────────────────────────────────────

    def ensure_user(
        self,
        clerk_user_id: str,
        *,
        internal_user_id: str | None = None,
        email: str | None = None,
        role: str = "patient",
    ) -> dict:
        clerk_user_id = require_document_id(clerk_user_id, label="clerk user id")
        if internal_user_id:
            internal_user_id = require_document_id(internal_user_id, label="internal user id")

        def build(index: dict | None, user: dict | None):
            now = utc_now()
            if index and index.get("internal_user_id"):
                existing_id = index["internal_user_id"]
                if internal_user_id and internal_user_id != existing_id:
                    raise ForgedIdentityError("clerk id is already mapped to a different internal user")
                base = user or {
                    "internal_user_id": existing_id,
                    "clerk_user_id": clerk_user_id,
                    "created_at": index.get("created_at") or now,
                    "role": role or "patient",
                    "charts_used": 0,
                    "active_patient_id": None,
                    "schema_version": SCHEMA_VERSION,
                }
                merged = dict(base)
                merged["internal_user_id"] = existing_id
                merged["clerk_user_id"] = clerk_user_id
                if email and not merged.get("email"):
                    merged["email"] = email
                merged.setdefault("role", role or "patient")
                merged.setdefault("charts_used", 0)
                merged.setdefault("schema_version", SCHEMA_VERSION)
                merged["updated_at"] = now
                return None, merged, dict(merged)
            new_id = internal_user_id or uuid.uuid4().hex
            record = {
                "internal_user_id": new_id,
                "clerk_user_id": clerk_user_id,
                "email": email or None,
                "role": role or "patient",
                "charts_used": 0,
                "active_patient_id": None,
                "created_at": now,
                "updated_at": now,
                "schema_version": SCHEMA_VERSION,
            }
            record = {key: value for key, value in record.items() if value is not None}
            index_doc = {"internal_user_id": new_id, "created_at": now, "schema_version": SCHEMA_VERSION}
            return index_doc, record, dict(record)

        return self.store.update_user_link(clerk_user_id, build)

    def get_user(self, actor_id: str) -> dict | None:
        actor_id = require_document_id(actor_id, label="internal user id")
        return self.store.get((USERS, actor_id))

    def resolve_clerk_user(self, clerk_user_id: str) -> dict | None:
        clerk_user_id = require_document_id(clerk_user_id, label="clerk user id")
        index = self.store.get((CLERK_INDEX, clerk_user_id))
        if not index:
            return None
        return self.store.get((USERS, index["internal_user_id"]))

    # ── patients ───────────────────────────────────────────────────────

    def _patient_path(self, owner_id: str, patient_id: str) -> tuple[str, ...]:
        return (USERS, owner_id, "patients", patient_id)

    def upsert_patient(self, actor_id: str, patient_id: str, fields: dict | None = None) -> dict:
        assert_owner(actor_id, actor_id)
        patient_id = require_document_id(patient_id, label="patient id")
        path = self._patient_path(actor_id, patient_id)
        current = self.store.get(path) or {}
        now = utc_now()
        document = {
            "patient_id": patient_id,
            "owner_internal_user_id": actor_id,
            "created_at": current.get("created_at") or now,
            "updated_at": now,
            "algo_version": current.get("algo_version"),
            "hidden": current.get("hidden", False),
            "top_organs": current.get("top_organs") or [],
            "last_chat_at": current.get("last_chat_at"),
            "chats_count": current.get("chats_count", 0),
            "latest_run_id": current.get("latest_run_id"),
            "schema_version": SCHEMA_VERSION,
        }
        for key, value in (fields or {}).items():
            if key in {"patient_id", "owner_internal_user_id", "created_at"}:
                continue
            document[key] = value
        document = {key: value for key, value in document.items() if value is not None}
        self.store.set(path, document)
        return document

    def get_patient(self, actor_id: str, owner_id: str, patient_id: str) -> dict | None:
        assert_owner(actor_id, owner_id)
        patient_id = require_document_id(patient_id, label="patient id")
        return self.store.get(self._patient_path(owner_id, patient_id))

    def save_identity(self, actor_id: str, patient_id: str, identity: dict) -> dict:
        self.upsert_patient(actor_id, patient_id)
        payload = {key: identity.get(key) for key in _IDENTITY_KEYS if identity.get(key) not in (None, "")}
        payload["patient_id"] = patient_id
        document = {
            "patient_id": patient_id,
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
        }
        if encryption_active(self.encryption_key):
            document["encrypted"] = True
            document["key_version"] = "v1"
            document["ciphertext"] = encrypt_mapping(payload, self.encryption_key)
        elif self.encryption_required:
            raise OwnershipError("identity encryption key is required")
        else:
            document["encrypted"] = False
            document.update(payload)
        path = self._patient_path(actor_id, patient_id) + ("private", "identity")
        self.store.set(path, document)
        return {"patient_id": patient_id, "encrypted": document["encrypted"]}

    def read_identity(self, actor_id: str, owner_id: str, patient_id: str) -> dict | None:
        assert_owner(actor_id, owner_id)
        document = self.store.get(self._patient_path(owner_id, patient_id) + ("private", "identity"))
        if not document:
            return None
        if document.get("encrypted"):
            if not self.encryption_key:
                raise OwnershipError("identity encryption key is required")
            return decrypt_mapping(document["ciphertext"], self.encryption_key)
        return {key: document.get(key) for key in ("patient_id", *_IDENTITY_KEYS) if key in document}

    def save_source(self, actor_id: str, patient_id: str, source: dict) -> dict:
        self.upsert_patient(actor_id, patient_id)
        body = {
            "patient_id": patient_id,
            "source_provider": source.get("source_provider") or "prokerala",
            "source_version": source.get("source_version") or "v2",
            "retrieved_at": source.get("retrieved_at") or utc_now(),
            "input_hash": source.get("input_hash") or sha256_text(canonical_json(source.get("raw_source") or {})),
            "schema_version": SCHEMA_VERSION,
            "prokerala_chart": source.get("prokerala_chart"),
            "prokerala_dasha": source.get("prokerala_dasha"),
            "raw_source": source.get("raw_source"),
        }
        encoded = self._store_large(
            self._patient_path(actor_id, patient_id) + ("private", "source"),
            body,
            blob_fields=("prokerala_chart", "prokerala_dasha", "raw_source"),
        )
        return {"patient_id": patient_id, "input_hash": body["input_hash"], "encoding": encoded}

    def save_run(
        self,
        actor_id: str,
        patient_id: str,
        run_id: str,
        *,
        algorithm_version: str,
        status: str,
        input_reference: str,
        input_hash: str,
        combined_result: Any = None,
        result_summary: dict | None = None,
        execution_metadata: dict | None = None,
        run_number: int | None = None,
    ) -> dict:
        self.upsert_patient(actor_id, patient_id)
        run_id = require_document_id(run_id, label="run id")
        path = self._patient_path(actor_id, patient_id) + ("runs", run_id)
        current = self.store.get(path) or {}
        if run_number is None:
            run_number = int(current.get("run_number") or 0) or self._next_run_number(actor_id, patient_id)
        document = {
            "run_id": run_id,
            "run_number": run_number,
            "algorithm_version": algorithm_version,
            "created_at": current.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "status": status,
            "input_reference": input_reference,
            "input_hash": input_hash,
            "result_summary": result_summary or {},
            "execution_metadata": execution_metadata or {},
            "schema_version": SCHEMA_VERSION,
            "patient_id": patient_id,
        }
        if combined_result is not None:
            encoded = encode_result(combined_result)
            if encoded.get("result_encoding") == "json" and len(encoded["result_json"].encode("utf-8")) > SOFT_DOCUMENT_BYTES:
                parts = split_text(encoded["result_json"])
                self._write_parts(path, "combined", parts)
                document["combined_encoding"] = "split"
                document["combined_parts"] = len(parts)
            else:
                document["combined_encoding"] = encoded["result_encoding"]
                if encoded["result_encoding"] == "json":
                    document["combined_json"] = encoded["result_json"]
                else:
                    document["combined_result"] = encoded["result"]
        self._reject_if_huge(document)
        self.store.set(path, document)
        self.upsert_patient(actor_id, patient_id, {"latest_run_id": run_id})
        return document

    def save_algorithm_output(
        self,
        actor_id: str,
        patient_id: str,
        run_id: str,
        algorithm_id: str,
        result: Any,
        *,
        algorithm_version: str,
        status: str = "complete",
        execution_metadata: dict | None = None,
    ) -> dict:
        self.get_run_or_raise(actor_id, patient_id, run_id)
        algorithm_id = require_document_id(algorithm_id, label="algorithm id")
        path = self._patient_path(actor_id, patient_id) + ("runs", run_id, "algorithms", algorithm_id)
        encoded = encode_result(result)
        document = {
            "algorithm_id": algorithm_id,
            "algorithm_version": algorithm_version,
            "status": status,
            "created_at": utc_now(),
            "execution_metadata": execution_metadata or {},
            "schema_version": SCHEMA_VERSION,
            "result_encoding": encoded["result_encoding"],
        }
        if encoded["result_encoding"] == "json" and len(encoded["result_json"].encode("utf-8")) > SOFT_DOCUMENT_BYTES:
            parts = split_text(encoded["result_json"])
            self._write_parts(path, "result", parts)
            document["result_encoding"] = "split"
            document["result_parts"] = len(parts)
            document["result_sha256"] = sha256_text(encoded["result_json"])
        elif encoded["result_encoding"] == "json":
            document["result_json"] = encoded["result_json"]
            document["result_sha256"] = sha256_text(encoded["result_json"])
        else:
            document["result"] = encoded["result"]
            document["result_sha256"] = sha256_text(canonical_json(result))
        self._reject_if_huge(document)
        self.store.set(path, document)
        return document

    def read_algorithm_output(self, actor_id: str, owner_id: str, patient_id: str, run_id: str, algorithm_id: str) -> Any:
        assert_owner(actor_id, owner_id)
        path = self._patient_path(owner_id, patient_id) + ("runs", run_id, "algorithms", algorithm_id)
        document = self.store.get(path)
        if not document:
            return None
        parts = None
        if document.get("result_encoding") == "split":
            parts = self._read_parts(path, "result", int(document.get("result_parts") or 0))
        wrapped = {
            "result_encoding": document.get("result_encoding"),
            "result": document.get("result"),
            "result_json": document.get("result_json"),
        }
        return decode_result(wrapped, parts)

    def get_run_or_raise(self, actor_id: str, patient_id: str, run_id: str) -> dict:
        document = self.store.get(self._patient_path(actor_id, patient_id) + ("runs", run_id))
        if not document:
            raise OwnershipError("run does not exist for this patient")
        return document

    def _next_run_number(self, actor_id: str, patient_id: str) -> int:
        count = 0
        cursor = None
        while True:
            rows, cursor = self.store.list_docs(
                self._patient_path(actor_id, patient_id) + ("runs",),
                limit=100,
                start_after=cursor,
            )
            count += len(rows)
            if not cursor:
                break
        return count + 1

    # ── chat ───────────────────────────────────────────────────────────

    def open_chat(self, actor_id: str, patient_id: str, chat_id: str, fields: dict | None = None) -> dict:
        self.upsert_patient(actor_id, patient_id)
        chat_id = require_document_id(chat_id, label="chat id")
        path = self._patient_path(actor_id, patient_id) + ("chats", chat_id)
        current = self.store.get(path) or {}
        now = utc_now()
        document = {
            "chat_id": chat_id,
            "patient_id": patient_id,
            "owner_internal_user_id": actor_id,
            "status": current.get("status") or "active",
            "completed": current.get("completed", False),
            "created_at": current.get("created_at") or now,
            "updated_at": now,
            "completed_at": current.get("completed_at"),
            "turn_count": current.get("turn_count", 0),
            "findings": current.get("findings") or {},
            "cluster_scores": current.get("cluster_scores") or {},
            "associated_run_id": current.get("associated_run_id"),
            "chat_algorithm_version": current.get("chat_algorithm_version") or "classic",
            "schema_version": SCHEMA_VERSION,
        }
        for key, value in (fields or {}).items():
            if value is not None:
                document[key] = value
        self.store.set(path, document)
        patient = self.get_patient(actor_id, actor_id, patient_id) or {}
        if not current:
            self.upsert_patient(actor_id, patient_id, {"chats_count": int(patient.get("chats_count") or 0) + 1})
        self.store.set(
            (CHAT_SESSION_INDEX, chat_id),
            {
                "chat_id": chat_id,
                "internal_user_id": actor_id,
                "patient_id": patient_id,
                "schema_version": SCHEMA_VERSION,
            },
        )
        return document

    def append_turn(self, actor_id: str, patient_id: str, chat_id: str, turn: dict) -> dict:
        chat = self.open_chat(actor_id, patient_id, chat_id)
        sequence = int(chat.get("turn_count") or 0) + 1
        turn_id = require_document_id(str(turn.get("turn_id") or f"{sequence:06d}"), label="turn id")
        path = self._patient_path(actor_id, patient_id) + ("chats", chat_id, "turns", turn_id)
        if self.store.get(path):
            return self.store.get(path) or {}
        document = {
            "turn_id": turn_id,
            "sequence_number": sequence,
            "role": turn.get("role") or "user",
            "content": turn.get("content") or "",
            "question_type": turn.get("question_type"),
            "cluster_rank": turn.get("cluster_rank"),
            "branch_id": turn.get("branch_id"),
            "answer_result": turn.get("answer_result"),
            "timestamp": turn.get("timestamp") or utc_now(),
            "validated": True,
            "metadata": turn.get("metadata") or {},
            "schema_version": SCHEMA_VERSION,
        }
        document = {key: value for key, value in document.items() if value is not None}
        self._reject_if_huge(document)
        self.store.set(path, document)
        self.open_chat(
            actor_id,
            patient_id,
            chat_id,
            {"turn_count": sequence, "last_chat_at": document["timestamp"], "status": "active"},
        )
        self.upsert_patient(actor_id, patient_id, {"last_chat_at": document["timestamp"]})
        return document

    def list_turns(self, actor_id: str, owner_id: str, patient_id: str, chat_id: str) -> list[dict]:
        assert_owner(actor_id, owner_id)
        rows = self._list_all(self._patient_path(owner_id, patient_id) + ("chats", chat_id, "turns"))
        rows.sort(key=lambda item: (item.get("sequence_number") or 0, item.get("timestamp") or ""))
        return rows

    def complete_chat(self, actor_id: str, patient_id: str, chat_id: str, findings: dict | None = None) -> dict:
        now = utc_now()
        return self.open_chat(
            actor_id,
            patient_id,
            chat_id,
            {
                "status": "completed",
                "completed": True,
                "completed_at": now,
                "findings": findings or {},
            },
        )

    def save_rl_output(self, actor_id: str, patient_id: str, chat_id: str, rl_run_id: str, output: Any, **fields) -> dict:
        self.open_chat(actor_id, patient_id, chat_id)
        rl_run_id = require_document_id(rl_run_id, label="rl run id")
        path = self._patient_path(actor_id, patient_id) + ("chats", chat_id, "rl_outputs", rl_run_id)
        encoded = encode_result(output)
        document = {
            "rl_run_id": rl_run_id,
            "model_version": fields.get("model_version") or "unknown",
            "input_chat_reference": f"users/{actor_id}/patients/{patient_id}/chats/{chat_id}",
            "status": fields.get("status") or "complete",
            "created_at": utc_now(),
            "completed_at": utc_now(),
            "execution_metadata": fields.get("execution_metadata") or {},
            "schema_version": SCHEMA_VERSION,
            "result_encoding": encoded["result_encoding"],
        }
        if encoded["result_encoding"] == "json" and len(encoded["result_json"].encode("utf-8")) > SOFT_DOCUMENT_BYTES:
            parts = split_text(encoded["result_json"])
            self._write_parts(path, "output", parts)
            document["result_encoding"] = "split"
            document["result_parts"] = len(parts)
        elif encoded["result_encoding"] == "json":
            document["output_json"] = encoded["result_json"]
        else:
            document["output"] = encoded["result"]
        self.store.set(path, document)
        return document

    def read_rl_output(self, actor_id: str, owner_id: str, patient_id: str, chat_id: str, rl_run_id: str) -> Any:
        assert_owner(actor_id, owner_id)
        path = self._patient_path(owner_id, patient_id) + ("chats", chat_id, "rl_outputs", rl_run_id)
        document = self.store.get(path)
        if not document:
            return None
        parts = None
        if document.get("result_encoding") == "split":
            parts = self._read_parts(path, "output", int(document.get("result_parts") or 0))
            return decode_result({"result_encoding": "split"}, parts)
        if document.get("result_encoding") == "json":
            return decode_result({"result_encoding": "json", "result_json": document.get("output_json")})
        return document.get("output")

    # ── audit ──────────────────────────────────────────────────────────

    def append_audit(self, actor_id: str, patient_id: str, event_type: str, metadata: dict | None = None, *, source: str = "api") -> dict:
        self.upsert_patient(actor_id, patient_id)
        event_id = uuid.uuid4().hex
        safe_meta = dict(metadata or {})
        for key in list(safe_meta):
            if key in _IDENTITY_KEYS or "name" in key or "token" in key or "secret" in key:
                safe_meta.pop(key)
        document = {
            "event_id": event_id,
            "event_type": event_type,
            "actor": actor_id,
            "timestamp": utc_now(),
            "source": source,
            "metadata": safe_meta,
            "schema_version": SCHEMA_VERSION,
        }
        self.store.set(self._patient_path(actor_id, patient_id) + ("audit", event_id), document)
        return document

    # ── knowledge, rl config, jobs, chunks ─────────────────────────────

    def write_dataset_if_absent(self, dataset_id: str, meta: dict, records: dict[str, Any]) -> str:
        dataset_id = require_document_id(dataset_id, label="dataset id")
        meta_path = (KNOWLEDGE_COLLECTION, dataset_id)
        existing = self.store.get(meta_path)
        checksum = sha256_text(canonical_json(records))
        if existing:
            if existing.get("checksum") == checksum:
                return "unchanged"
            return "drift"
        now = utc_now()
        document = {
            "dataset_id": dataset_id,
            "dataset_name": meta.get("dataset_name") or dataset_id,
            "description": meta.get("description") or "",
            "schema_version": SCHEMA_VERSION,
            "dataset_version": checksum[:16],
            "status": meta.get("status") or "active",
            "source_files": meta.get("source_files") or [],
            "record_count": len(records),
            "created_at": now,
            "updated_at": now,
            "checksum": checksum,
            "migration_status": "seeded",
            "record_mode": meta.get("record_mode") or "map",
            "category": meta.get("category") or "knowledge",
        }
        self.store.set(meta_path, document)
        batch = []
        for key, value in records.items():
            record_id = _record_id(str(key))
            batch.append(
                (
                    meta_path + ("records", record_id),
                    {
                        "record_id": record_id,
                        "record_key": str(key),
                        "dataset_id": dataset_id,
                        "data": _json_safe(value),
                        "source": meta.get("source_files", ["repository"])[0] if meta.get("source_files") else "repository",
                        "version": checksum[:16],
                        "created_at": now,
                        "updated_at": now,
                        "schema_version": SCHEMA_VERSION,
                    },
                )
            )
        self.store.batch_set(batch)
        return "created"

    def read_dataset(self, dataset_id: str) -> tuple[dict | None, dict[str, Any]]:
        meta = self.store.get((KNOWLEDGE_COLLECTION, dataset_id))
        if not meta:
            return None, {}
        rows = self._list_all((KNOWLEDGE_COLLECTION, dataset_id, "records"))
        records = {}
        for row in rows:
            key = row.get("record_key") or row.get("record_id")
            records[key] = row.get("data")
        return meta, records

    def write_rl_config_if_absent(self, dataset_id: str, records: dict[str, Any], description: str) -> str:
        """RL configuration is not written into the shared knowledge base."""
        dataset_id = require_document_id(dataset_id, label="rl dataset id")
        path = (RL_CONFIG_COLLECTION, dataset_id)
        existing = self.store.get(path)
        checksum = sha256_text(canonical_json(records))
        if existing:
            return "unchanged" if existing.get("checksum") == checksum else "drift"
        now = utc_now()
        self.store.set(
            path,
            {
                "dataset_id": dataset_id,
                "description": description,
                "checksum": checksum,
                "record_count": len(records),
                "schema_version": SCHEMA_VERSION,
                "created_at": now,
                "updated_at": now,
                "status": "active",
            },
        )
        batch = []
        for key, value in records.items():
            record_id = _record_id(str(key))
            batch.append(
                (
                    path + ("records", record_id),
                    {
                        "record_id": record_id,
                        "record_key": str(key),
                        "data": _json_safe(value),
                        "schema_version": SCHEMA_VERSION,
                        "created_at": now,
                        "updated_at": now,
                    },
                )
            )
        self.store.batch_set(batch)
        return "created"

    def save_model_metadata(self, model_version: str, fields: dict) -> dict:
        model_version = require_document_id(model_version, label="model version")
        document = {
            "model_version": model_version,
            "artifact_uri": fields.get("artifact_uri") or "",
            "artifact_checksum": fields.get("artifact_checksum") or "",
            "training_dataset_version": fields.get("training_dataset_version") or "",
            "training_timestamp": fields.get("training_timestamp") or utc_now(),
            "evaluation_metrics": fields.get("evaluation_metrics") or {},
            "status": fields.get("status") or "registered",
            "created_at": utc_now(),
            "schema_version": SCHEMA_VERSION,
        }
        if document["artifact_uri"].startswith("gs://") or not document["artifact_uri"]:
            self.store.set((RL_MODELS_COLLECTION, model_version), document)
            return document
        raise DocumentTooLargeError("model artifacts must be a gs:// URI or empty, not inline weights")

    def save_chunk(self, chunk_id: str, fields: dict) -> dict:
        chunk_id = require_document_id(chunk_id, label="chunk id")
        document = {
            "chunk_id": chunk_id,
            "source_dataset": fields.get("source_dataset") or "",
            "source_record_id": fields.get("source_record_id") or "",
            "content": fields.get("content") or "",
            "metadata": fields.get("metadata") or {},
            "version": fields.get("version") or "1",
            "created_at": utc_now(),
            "schema_version": SCHEMA_VERSION,
        }
        self._reject_if_huge(document)
        self.store.set((KB_CHUNKS_COLLECTION, chunk_id), document)
        return document

    def save_job(self, job_id: str, fields: dict) -> dict:
        job_id = require_document_id(job_id, label="job id")
        current = self.store.get((JOB_RECORDS_COLLECTION, job_id)) or {}
        document = {
            "job_id": job_id,
            "status": fields.get("status") or current.get("status") or "queued",
            "retry_count": int(fields.get("retry_count") if fields.get("retry_count") is not None else current.get("retry_count") or 0),
            "execution_metadata": fields.get("execution_metadata") or {},
            "created_at": current.get("created_at") or utc_now(),
            "updated_at": utc_now(),
            "schema_version": SCHEMA_VERSION,
            "patient_id": fields.get("patient_id") or current.get("patient_id"),
            "message": str(fields.get("message") or "")[:500],
        }
        document = {key: value for key, value in document.items() if value is not None}
        self.store.set((JOB_RECORDS_COLLECTION, job_id), document)
        return document

    def save_system_config(self, config_id: str, fields: dict) -> dict:
        config_id = require_document_id(config_id, label="config id")
        for key in fields:
            lowered = key.lower()
            if "secret" in lowered or "password" in lowered or "token" in lowered or "key" in lowered:
                raise ForgedIdentityError("system_config cannot store secrets")
        document = {
            "config_id": config_id,
            "values": fields,
            "updated_at": utc_now(),
            "schema_version": SCHEMA_VERSION,
        }
        self.store.set((SYSTEM_CONFIG_COLLECTION, config_id), document)
        return document

    def delete_patient_tree(self, actor_id: str, patient_id: str) -> bool:
        """Explicit deletion. Migration never calls this."""
        assert_owner(actor_id, actor_id)
        patient_id = require_document_id(patient_id, label="patient id")
        return self.store.delete(self._patient_path(actor_id, patient_id))

    def session_index(self, chat_id: str) -> dict | None:
        chat_id = require_document_id(chat_id, label="chat id")
        return self.store.get((CHAT_SESSION_INDEX, chat_id))

    # ── helpers ────────────────────────────────────────────────────────

    def _list_all(self, collection_path: tuple[str, ...]) -> list[dict]:
        rows: list[dict] = []
        cursor = None
        while True:
            page, cursor = self.store.list_docs(collection_path, limit=200, start_after=cursor)
            rows.extend(document for _doc_id, document in page)
            if not cursor:
                return rows

    def _write_parts(self, doc_path: tuple[str, ...], field: str, parts: list[str]) -> None:
        batch = []
        for index, part in enumerate(parts):
            batch.append((doc_path + ("parts", f"{field}_{index:04d}"), {"seq": index, "field": field, "data": part, "schema_version": SCHEMA_VERSION}))
        self.store.batch_set(batch)

    def _read_parts(self, doc_path: tuple[str, ...], field: str, count: int) -> list[str]:
        parts = []
        for index in range(count):
            document = self.store.get(doc_path + ("parts", f"{field}_{index:04d}")) or {}
            parts.append(document.get("data") or "")
        return parts

    def _store_large(self, path: tuple[str, ...], body: dict, blob_fields: tuple[str, ...]) -> str:
        inline = dict(body)
        encoding = "native"
        for field in blob_fields:
            value = inline.get(field)
            if value is None:
                continue
            encoded = encode_result(value)
            if encoded["result_encoding"] == "json" and len(encoded["result_json"].encode("utf-8")) > SOFT_DOCUMENT_BYTES:
                parts = split_text(encoded["result_json"])
                self._write_parts(path, field, parts)
                inline[field] = None
                inline[f"{field}_encoding"] = "split"
                inline[f"{field}_parts"] = len(parts)
                encoding = "split"
            elif encoded["result_encoding"] == "json":
                inline[field] = None
                inline[f"{field}_json"] = encoded["result_json"]
                inline[f"{field}_encoding"] = "json"
                encoding = "json"
            else:
                inline[field] = encoded["result"]
                inline[f"{field}_encoding"] = "native"
        inline = {key: value for key, value in inline.items() if value is not None}
        self._reject_if_huge(inline)
        self.store.set(path, inline)
        return encoding

    def _reject_if_huge(self, document: dict) -> None:
        if byte_length(document) > SOFT_DOCUMENT_BYTES:
            raise DocumentTooLargeError("document exceeds the Firestore size budget and was not truncated")
