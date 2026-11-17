"""In-process document store used by tests and local development."""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable

from app.core.clock import utc_now


class MemoryDocumentStore:
    def __init__(self) -> None:
        self._lock = RLock()
        self.docs: dict[tuple[str, ...], dict] = {}
        self.reads = 0
        self.writes = 0

    def get(self, path: tuple[str, ...]) -> dict | None:
        with self._lock:
            self.reads += 1
            found = self.docs.get(tuple(path))
            return dict(found) if found else None

    def set(self, path: tuple[str, ...], data: dict, merge: bool = False) -> None:
        with self._lock:
            self.writes += 1
            key = tuple(path)
            if merge and key in self.docs:
                current = dict(self.docs[key])
                current.update(data)
                self.docs[key] = current
            else:
                self.docs[key] = dict(data)

    def delete(self, path: tuple[str, ...]) -> bool:
        with self._lock:
            prefix = tuple(path)
            keys = [key for key in self.docs if key[: len(prefix)] == prefix]
            for key in keys:
                del self.docs[key]
            return bool(keys)

    def list_docs(
        self,
        collection_path: tuple[str, ...],
        *,
        limit: int = 100,
        start_after: str | None = None,
    ) -> tuple[list[tuple[str, dict]], str | None]:
        prefix = tuple(collection_path)
        depth = len(prefix) + 1
        with self._lock:
            self.reads += 1
            ids = sorted(
                {
                    key[len(prefix)]
                    for key in self.docs
                    if len(key) >= depth and key[: len(prefix)] == prefix
                }
            )
        if start_after:
            ids = [item for item in ids if item > start_after]
        page = ids[: max(1, limit)]
        next_cursor = page[-1] if len(ids) > len(page) else None
        rows = []
        for doc_id in page:
            document = self.get(prefix + (doc_id,)) or {}
            rows.append((doc_id, document))
        return rows, next_cursor

    def batch_set(self, items: list[tuple[tuple[str, ...], dict]]) -> None:
        for path, data in items:
            self.set(path, data)

    def update_user_link(
        self,
        clerk_user_id: str,
        build: Callable[[dict | None, dict | None], tuple],
    ) -> Any:
        with self._lock:
            index = self.docs.get(("clerk_index", clerk_user_id))
            index = dict(index) if index else None
            user = None
            if index and index.get("internal_user_id"):
                found = self.docs.get(("users", index["internal_user_id"]))
                user = dict(found) if found else None
            new_index, new_user, result = build(index, user)
            if new_user is not None:
                self.docs[("users", result["internal_user_id"])] = dict(new_user)
                self.writes += 1
            if new_index is not None:
                self.docs[("clerk_index", clerk_user_id)] = dict(new_index)
                self.writes += 1
            return result

    def touch(self) -> str:
        return utc_now()
