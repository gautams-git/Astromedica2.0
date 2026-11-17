"""Firestore-backed document store. Server-side Admin SDK only."""

from __future__ import annotations

from typing import Any, Callable

from app.core.errors import FirestoreUnavailableError


class FirestoreDocumentStore:
    def __init__(self, client: Any, call: dict | None = None) -> None:
        self.client = client
        self.call = call or {}
        self.reads = 0
        self.writes = 0

    def _ref(self, path: tuple[str, ...]):
        if len(path) < 2 or len(path) % 2 != 0:
            raise FirestoreUnavailableError(f"not a document path: {path}")
        ref = self.client.collection(path[0]).document(path[1])
        index = 2
        while index < len(path):
            ref = ref.collection(path[index]).document(path[index + 1])
            index += 2
        return ref

    def _collection(self, collection_path: tuple[str, ...]):
        if len(collection_path) % 2 != 1:
            raise FirestoreUnavailableError(f"not a collection path: {collection_path}")
        if len(collection_path) == 1:
            return self.client.collection(collection_path[0])
        parent = self._ref(collection_path[:-1])
        return parent.collection(collection_path[-1])

    def _guard(self, fn):
        try:
            return fn()
        except FirestoreUnavailableError:
            raise
        except Exception as exc:
            raise FirestoreUnavailableError(str(exc)[:300]) from exc

    def get(self, path: tuple[str, ...]) -> dict | None:
        def read():
            self.reads += 1
            snapshot = self._ref(path).get(**self.call)
            if not snapshot.exists:
                return None
            data = snapshot.to_dict() or {}
            data.setdefault("_id", snapshot.id)
            return data

        return self._guard(read)

    def set(self, path: tuple[str, ...], data: dict, merge: bool = False) -> None:
        def write():
            self.writes += 1
            stored = {key: value for key, value in data.items() if key != "_id"}
            self._ref(path).set(stored, merge=merge, **self.call)

        self._guard(write)

    def delete(self, path: tuple[str, ...]) -> bool:
        def remove():
            ref = self._ref(path)
            existed = ref.get(**self.call).exists
            self.client.recursive_delete(ref)
            self.writes += 1
            return existed

        return self._guard(remove)

    def list_docs(
        self,
        collection_path: tuple[str, ...],
        *,
        limit: int = 100,
        start_after: str | None = None,
    ) -> tuple[list[tuple[str, dict]], str | None]:
        def read():
            self.reads += 1
            from google.cloud.firestore_v1.field_path import FieldPath

            query = self._collection(collection_path).order_by(FieldPath.document_id()).limit(max(1, limit))
            if start_after:
                snapshot = self._collection(collection_path).document(start_after).get(**self.call)
                if snapshot.exists:
                    query = query.start_after(snapshot)
            rows = []
            last = None
            for snap in query.stream(**self.call):
                data = snap.to_dict() or {}
                data["_id"] = snap.id
                rows.append((snap.id, data))
                last = snap.id
            next_cursor = last if len(rows) == max(1, limit) else None
            return rows, next_cursor

        return self._guard(read)

    def batch_set(self, items: list[tuple[tuple[str, ...], dict]]) -> None:
        def write():
            batch = self.client.batch()
            pending = 0
            for path, data in items:
                stored = {key: value for key, value in data.items() if key != "_id"}
                batch.set(self._ref(path), stored)
                pending += 1
                self.writes += 1
                if pending == 400:
                    batch.commit(**self.call)
                    batch = self.client.batch()
                    pending = 0
            if pending:
                batch.commit(**self.call)

        self._guard(write)

    def update_user_link(
        self,
        clerk_user_id: str,
        build: Callable[[dict | None, dict | None], tuple],
    ) -> Any:
        from google.cloud import firestore

        index_ref = self.client.collection("clerk_index").document(clerk_user_id)

        @firestore.transactional
        def runner(transaction):
            index_snap = index_ref.get(transaction=transaction)
            index = index_snap.to_dict() if index_snap.exists else None
            user = None
            user_ref = None
            if index and index.get("internal_user_id"):
                user_ref = self.client.collection("users").document(index["internal_user_id"])
                user_snap = user_ref.get(transaction=transaction)
                user = user_snap.to_dict() if user_snap.exists else None
            new_index, new_user, result = build(dict(index) if index else None, dict(user) if user else None)
            if new_user is not None:
                target = user_ref or self.client.collection("users").document(result["internal_user_id"])
                transaction.set(target, {key: value for key, value in new_user.items() if key != "_id"})
            if new_index is not None:
                transaction.set(index_ref, new_index)
            return result

        def run():
            self.writes += 1
            return runner(self.client.transaction(max_attempts=5))

        return self._guard(run)
