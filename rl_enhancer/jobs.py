"""Thread-safe in-memory job store."""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Job:
    id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobStore:
    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()

    def create(self, input_path: Optional[str] = None) -> Job:
        with self._lock:
            job = Job(id=uuid.uuid4().hex[:12], input_path=input_path)
            self._jobs[job.id] = job
            self._order.append(job.id)
            return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for k, v in fields.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            job.updated_at = time.time()
            return job

    def next_queued(self) -> Optional[Job]:
        with self._lock:
            for jid in self._order:
                job = self._jobs.get(jid)
                if job and job.status == "queued":
                    return job
            return None

    def list(self, limit: int = 50) -> List[Job]:
        with self._lock:
            items = [self._jobs[j] for j in reversed(self._order)
                     if j in self._jobs]
            return items[:limit]

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            if job.status in ("queued", "running"):
                job.status = "cancelled"
                job.message = "Cancelled by user"
                job.updated_at = time.time()
            return job


STORE = JobStore()