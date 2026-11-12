"""Per-chart memory of earlier runs."""
import hashlib
import json
import os
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

MAX_KEYS = 500          # oldest charts are dropped first
MAX_RUNS_PER_KEY = 20


class MemoryStore:
    """In-process memory, shared by every agent call in this worker.

    Set AGENT_MEMORY_PATH to also keep it in a JSON file. It is per worker process: with
    several gunicorn workers each worker has its own.
    """

    _shared: Optional["MemoryStore"] = None
    _shared_lock = threading.Lock()

    @classmethod
    def shared(cls) -> "MemoryStore":
        with cls._shared_lock:
            if cls._shared is None:
                cls._shared = cls()
            return cls._shared

    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("AGENT_MEMORY_PATH")
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
        if self.path and os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._store = OrderedDict(json.load(f))

    @staticmethod
    def chart_key(processed: Dict, user_id: Optional[str], algorithm: str = "combined_algo") -> str:
        """The key follows the chart itself (patient + planets + ascendant), not the algorithm output."""
        payload = json.dumps(
            {"patient": processed.get("patient_id"), "planets": processed.get("planets"),
             "ascendant": processed.get("ascendant"), "user": user_id, "algorithm": algorithm},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def remember(self, key: str, entry: Dict[str, Any]):
        with self._lock:
            runs = self._store.setdefault(key, [])
            runs.append(entry)
            del runs[:-MAX_RUNS_PER_KEY]
            self._store.move_to_end(key)
            while len(self._store) > MAX_KEYS:
                self._store.popitem(last=False)
            self._flush()

    def recall(self, key: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._store.get(key, []))

    def _flush(self):
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump(self._store, f, default=str)

    def trend(self, key: str) -> Dict[str, Any]:
        """Consensus-score change (%) per organ versus the previous run."""
        history = self.recall(key)
        if len(history) < 2:
            return {"previous_runs": len(history), "deltas": {}}
        prev = history[-2].get("top_scores", {})
        curr = history[-1].get("top_scores", {})
        deltas = {}
        for label, score in curr.items():
            if label in prev and prev[label]:
                deltas[label] = round((score - prev[label]) / prev[label] * 100, 1)
        return {"previous_runs": len(history), "deltas": deltas}
