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

    # ── Admin page (admin_routes.py): list and delete ────────────────────
    def list_users(self) -> list[dict]:
        with self._lock:
            return [dict(user) for user in self.users.values()]

    def list_docs(self, internal_user_id: str, collection: str, patient_id: str | None = None) -> list[tuple]:
        """(doc_id, data) pairs of one user collection, optionally only one patient's."""
        with self._lock:
            return [
                (doc_id, dict(doc))
                for (owner, coll, doc_id), doc in self.docs.items()
                if owner == internal_user_id and coll == collection
                and (patient_id is None or doc.get("patient_id") == patient_id)
            ]

    def delete(self, internal_user_id: str, collection: str, doc_id: str) -> bool:
        with self._lock:
            return self.docs.pop(self._path(internal_user_id, collection, doc_id), None) is not None

    def delete_user(self, internal_user_id: str, clerk_user_id: str | None) -> int:
        """Removes the user record, its Clerk link and every document it owns.
        The Clerk login itself is untouched: signing in again starts fresh."""
        with self._lock:
            keys = [key for key in self.docs if key[0] == internal_user_id]
            for key in keys:
                del self.docs[key]
            self.users.pop(internal_user_id, None)
            if clerk_user_id:
                self.users_by_clerk.pop(clerk_user_id, None)
            return len(keys)


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

    # ── Admin page (admin_routes.py): list and delete ────────────────────
    def list_users(self) -> list[dict]:
        users = []
        for snap in self._client.collection("users").stream(**self._call):
            data = snap.to_dict() or {}
            data["internal_user_id"] = snap.id
            users.append(data)
        return users

    def list_docs(self, internal_user_id: str, collection: str, patient_id: str | None = None) -> list[tuple]:
        from google.cloud import firestore
        query = self._user_collection(internal_user_id, collection)
        if patient_id is not None:
            query = query.where(filter=firestore.FieldFilter("patient_id", "==", patient_id))
        return [(snap.id, snap.to_dict() or {}) for snap in query.stream(**self._call)]

    def delete(self, internal_user_id: str, collection: str, doc_id: str) -> bool:
        """Deletes the document and all of its subcollections (Firestore's own
        delete leaves subcollections behind)."""
        ref = self._user_collection(internal_user_id, collection).document(doc_id)
        existed = ref.get(**self._call).exists
        self._client.recursive_delete(ref)
        return existed

    def delete_user(self, internal_user_id: str, clerk_user_id: str | None) -> int:
        count = self._client.recursive_delete(self._client.collection("users").document(internal_user_id))
        if clerk_user_id:
            self._client.collection("clerk_index").document(clerk_user_id).delete(**self._call)
        return count


def build_repository(settings: Settings):
    if settings.is_test and not settings.firestore_emulator_host:
        return MemoryRepository()
    if settings.firestore_emulator_host or settings.is_production or os.environ.get("FIRESTORE_BACKEND") == "firestore":
        return FirestoreRepository(settings)
    if settings.app_env == "development" and not settings.firestore_emulator_host:
        return MemoryRepository()
    return FirestoreRepository(settings)