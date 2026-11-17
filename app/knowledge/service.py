"""Central knowledge-base access. Production Firestore misses raise; they do not return {}."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import DataSettings, load_data_settings
from app.core.errors import FirestoreUnavailableError, MissingDatasetError
from app.knowledge.registry import KNOWLEDGE_SPECS, spec_by_id

_LOG = logging.getLogger("astromedica.knowledge")


class KnowledgeBaseService:
    def __init__(self, plane, settings: DataSettings | None = None) -> None:
        self.plane = plane
        self.settings = settings or load_data_settings()
        self._cache: dict[str, dict] = {}
        self._checked_at: dict[str, float] = {}
        self._versions: dict[str, str] = {}
        self.hits = 0
        self.misses = 0
        self.last_latency_ms = 0.0
        self._fixtures: dict | None = None

    def stats(self) -> dict:
        return {
            "source": self.settings.kb_source,
            "strict": self.settings.kb_strict,
            "hits": self.hits,
            "misses": self.misses,
            "cached": sorted(self._cache),
            "last_latency_ms": round(self.last_latency_ms, 3),
        }

    def invalidate(self, dataset_id: str | None = None) -> None:
        if dataset_id is None:
            self._cache.clear()
            self._checked_at.clear()
            return
        self._cache.pop(dataset_id, None)
        self._checked_at.pop(dataset_id, None)

    def require(self, dataset_id: str) -> dict:
        spec = spec_by_id(dataset_id)
        if spec is None or spec.category == "rl_config":
            raise MissingDatasetError(f"{dataset_id} is not a shared knowledge dataset")
        started = time.perf_counter()
        try:
            if self.settings.kb_source == "fixtures":
                if self.settings.kb_strict or self.settings.is_production:
                    raise MissingDatasetError("fixture knowledge is disabled when KB_STRICT or production is set")
                return self._from_fixtures(dataset_id)
            return self._from_store(dataset_id)
        except MissingDatasetError:
            raise
        except FirestoreUnavailableError:
            raise
        except Exception as exc:
            if self.settings.kb_source != "firestore":
                raise
            raise FirestoreUnavailableError(f"knowledge base read failed for {dataset_id}") from exc
        finally:
            self.last_latency_ms = (time.perf_counter() - started) * 1000

    def require_record(self, dataset_id: str, record_key: str) -> Any:
        records = self.require(dataset_id)
        if record_key not in records:
            raise MissingDatasetError(f"{dataset_id} has no record {record_key}")
        return records[record_key]

    def warm_required(self) -> list[str]:
        loaded = []
        for spec in KNOWLEDGE_SPECS:
            if spec.required and spec.category == "knowledge":
                self.require(spec.dataset_id)
                loaded.append(spec.dataset_id)
        return loaded

    def _from_fixtures(self, dataset_id: str) -> dict:
        now = time.time()
        cached = self._cache.get(dataset_id)
        if cached is not None and now - self._checked_at.get(dataset_id, 0) < self.settings.kb_cache_ttl_seconds:
            self.hits += 1
            return cached
        self.misses += 1
        if self._fixtures is None:
            from app.knowledge.fixtures import load_fixture_records

            self._fixtures = load_fixture_records()
        records = self._fixtures["knowledge"].get(dataset_id)
        if not records:
            raise MissingDatasetError(f"fixture dataset {dataset_id} is missing")
        self._cache[dataset_id] = records
        self._checked_at[dataset_id] = now
        _LOG.info("knowledge_cache_miss dataset=%s source=fixtures", dataset_id)
        return records

    def _from_store(self, dataset_id: str) -> dict:
        now = time.time()
        cached = self._cache.get(dataset_id)
        fresh = cached is not None and now - self._checked_at.get(dataset_id, 0) < self.settings.kb_cache_ttl_seconds
        if fresh:
            self.hits += 1
            return cached
        meta, records = self.plane.read_dataset(dataset_id)
        if not meta or not records:
            self.invalidate(dataset_id)
            raise MissingDatasetError(f"knowledge dataset {dataset_id} is not in Firestore")
        version = meta.get("dataset_version")
        if cached is not None and self._cache_version.get(dataset_id) == version:
            self.hits += 1
            self._checked_at[dataset_id] = now
            return cached
        self.misses += 1
        self._cache[dataset_id] = records
        self._cache_version[dataset_id] = version
        self._checked_at[dataset_id] = now
        _LOG.info(
            "knowledge_cache_miss dataset=%s source=firestore version=%s records=%s latency_ms=%.2f",
            dataset_id,
            version,
            len(records),
            self.last_latency_ms,
        )
        return records

    @property
    def _cache_version(self) -> dict:
        if not hasattr(self, "_versions"):
            self._versions = {}
        return self._versions
