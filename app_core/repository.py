"""User-scoped persistence.

Firestore is the authoritative store when configured. Tests without an emulator
use the in-memory implementation of the same interface. The frontend never
receives Firestore credentials.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from app_core.identifiers import SCHEMA_VERSION, new_internal_user_id
from app_core.settings import Settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strip_none(document: dict) -> dict:
    return {key: value for key, value in document.items() if value is not None}


class MemoryRepository:
    def __init__(self):
        self._lock = Lock()
        self.users_by_clerk: dict[str, dict] = {}
        self.users: dict[str, dict] = {}
        self.docs: dict[tuple, dict] = {}

    def ping(self) -> bool:
        return True

    def get_or_create_user(self, clerk_user_id: str) -> dict:
        with self._lock:
            existing = self.users_by_clerk.get(clerk_user_id)
            if existing:
                existing["updated_at"] = _now()
                return dict(existing)
            internal_id = new_internal_user_id()
            record = {
                "internal_user_id": internal_id,
                "clerk_user_id": clerk_user_id,
                "created_at": _now(),
                "updated_at": _now(),
                "schema_version": SCHEMA_VERSION,
            }
            self.users_by_clerk[clerk_user_id] = record
            self.users[internal_id] = record
            return dict(record)

    def _path(self, internal_user_id: str, collection: str, doc_id: str) -> tuple:
        return (internal_user_id, collection, doc_id)

    def save(self, internal_user_id: str, collection: str, doc_id: str, data: dict) -> dict:
        stored = _strip_none(dict(data))
        stored.setdefault("created_at", _now())
        stored["updated_at"] = _now()
        stored["schema_version"] = SCHEMA_VERSION
        stored["owner_internal_user_id"] = internal_user_id
        with self._lock:
            key = self._path(internal_user_id, collection, doc_id)
            current = self.docs.get(key, {})
            if current.get("created_at"):
                stored["created_at"] = current["created_at"]
            self.docs[key] = stored
            return dict(stored)

    def get(self, internal_user_id: str, collection: str, doc_id: str) -> dict | None:
        with self._lock:
            found = self.docs.get(self._path(internal_user_id, collection, doc_id))
            return dict(found) if found else None

    def owns_patient(self, internal_user_id: str, patient_id: str) -> bool:
        record = self.get(internal_user_id, "patients", patient_id)
        return bool(record and record.get("owner_internal_user_id") == internal_user_id)

    def bind_patient(self, internal_user_id: str, patient_id: str, input_id: str) -> dict:
        return self.save(
            internal_user_id,
            "patients",
            patient_id,
            {
                "patient_id": patient_id,
                "input_id": input_id,
                "source": "chart",
                "owner_internal_user_id": internal_user_id,
            },
        )


    def list_records(self, internal_user_id: str) -> dict:
        with self._lock:
            owned = [
                dict(doc)
                for (owner, _collection, _doc_id), doc in self.docs.items()
                if owner == internal_user_id
            ]
        return {
            "inputs": [doc for doc in owned if doc.get("record_type") == "input"],
            "results": [doc for doc in owned if doc.get("record_type") == "combined"],
        }


class FirestoreRepository(MemoryRepository):
    """Firestore-backed repository. Admin credentials stay on the server."""

    def __init__(self, settings: Settings):
        super().__init__()
        if settings.firestore_emulator_host:
            os.environ["FIRESTORE_EMULATOR_HOST"] = settings.firestore_emulator_host
        from google.auth.credentials import AnonymousCredentials
        from google.cloud import firestore

        from google.api_core.retry import Retry, if_transient_error

        kwargs = {"project": settings.firestore_project_id}
        if settings.firestore_emulator_host:
            kwargs["credentials"] = AnonymousCredentials()
        self._client = firestore.Client(**kwargs)
        # Give up after ~10 s. The library default retries for 5 minutes, so an
        # expired Google login or a network problem froze every request.
        self._call = {"retry": Retry(predicate=if_transient_error, timeout=10.0), "timeout": 10.0}

    def ping(self) -> bool:
        try:
            list(self._client.collections(**self._call))
            return True
        except Exception:
            return False

    def get_or_create_user(self, clerk_user_id: str) -> dict:
        index = self._client.collection("clerk_index").document(clerk_user_id)
        snapshot = index.get(**self._call)
        if snapshot.exists:
            internal_id = snapshot.to_dict()["internal_user_id"]
            user_ref = self._client.collection("users").document(internal_id)
            user_ref.set({"updated_at": _now()}, merge=True, **self._call)
            data = user_ref.get(**self._call).to_dict() or {}
            data["internal_user_id"] = internal_id
            return data

        internal_id = new_internal_user_id()
        record = {
            "internal_user_id": internal_id,
            "clerk_user_id": clerk_user_id,
            "created_at": _now(),
            "updated_at": _now(),
            "schema_version": SCHEMA_VERSION,
        }
        self._client.collection("users").document(internal_id).set(record, **self._call)
        index.set({"internal_user_id": internal_id, "created_at": record["created_at"]}, **self._call)
        return record

    def _user_collection(self, internal_user_id: str, collection: str):
        return self._client.collection("users").document(internal_user_id).collection(collection)

    def save(self, internal_user_id: str, collection: str, doc_id: str, data: dict) -> dict:
        stored = _strip_none(dict(data))
        ref =self._user_collection(internal_user_id, collection).document(doc_id)
        existing = ref.get(**self._call)
        created_at = _now()
        if existing.exists:
            created_at = (existing.to_dict() or {}).get("created_at", created_at)
        stored["created_at"] = created_at
        stored["updated_at"] = _now()
        stored["schema_version"] = SCHEMA_VERSION
        stored["owner_internal_user_id"] = internal_user_id
        ref.set(stored, **self._call)
        return stored

    def get(self, internal_user_id: str, collection: str, doc_id: str) -> dict | None:
        snapshot = self._user_collection(internal_user_id, collection).document(doc_id).get(**self._call)
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        if data.get("owner_internal_user_id") != internal_user_id:
            return None
        return data

    def list_records(self, internal_user_id: str) -> dict:
        inputs = [
            doc.to_dict()
            for doc in self._user_collection(internal_user_id, "inputs").stream(**self._call)
        ]
        results = [
            doc.to_dict()
            for doc in self._user_collection(internal_user_id, "combined_results").stream(**self._call)
        ]
        return {"inputs": inputs, "results": results}


def build_repository(settings: Settings):
    if settings.is_test and not settings.firestore_emulator_host:
        return MemoryRepository()
    if settings.firestore_emulator_host or settings.is_production or os.environ.get("FIRESTORE_BACKEND") == "firestore":
        return FirestoreRepository(settings)
    if settings.app_env == "development" and not settings.firestore_emulator_host:
        return MemoryRepository()
    return FirestoreRepository(settings)