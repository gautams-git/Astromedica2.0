"""Thread-safe in-memory job store.

The worker still consumes this queue. job_records is a durable status copy,
not a second queue.
"""
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


def _mirror_job(job: Job) -> None:
    try:
        from app.chat.sessions import session_backend
        plane = session_backend()
        if plane is None:
            return
        plane.save_job(job.id, {
            "status": job.status,
            "message": job.message,
            "patient_id": None,
            "execution_metadata": {"progress": job.progress, "error": job.error},
        })
    except Exception:
        return


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
        _mirror_job(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **fields: Any) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            for key, value in fields.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            job.updated_at = time.time()
        _mirror_job(job)
        return job

    def next_queued(self) -> Optional[Job]:
        with self._lock:
            for job_id in self._order:
                job = self._jobs.get(job_id)
                if job and job.status == "queued":
                    return job
            return None

    def list(self, limit: int = 50) -> List[Job]:
        with self._lock:
            items = [self._jobs[job_id] for job_id in reversed(self._order) if job_id in self._jobs]
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
        _mirror_job(job)
        return job


STORE = JobStore()
