"""Session memory for organ-truth agent."""
import hashlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MemoryStore:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.getenv("AGENT_ORGAN_MEMORY_PATH")
        self._store: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if self.path and os.path.exists(self.path):
            with open(self.path, "r") as f:
                self._store = defaultdict(list, json.load(f))

    @staticmethod
    def chart_key(chart_data: Dict, user_id: Optional[str]) -> str:
        payload = json.dumps(
            {"chart": chart_data, "user": user_id},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def remember(self, key: str, entry: Dict[str, Any]):
        self._store[key].append(entry)
        self._flush()

    def recall(self, key: str) -> List[Dict[str, Any]]:
        return list(self._store.get(key, []))

    def _flush(self):
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump(self._store, f, default=str)

    def trend(self, key: str) -> Dict[str, Any]:
        history = self.recall(key)
        if len(history) < 2:
            return {"previous_runs": len(history), "deltas": {}}
        prev = history[-2].get("top_organs", {})
        curr = history[-1].get("top_organs", {})
        deltas = {}
        for organ, score in curr.items():
            if organ in prev and prev[organ]:
                deltas[organ] = round(
                    (score - prev[organ]) / prev[organ] * 100, 1
                )
        return {"previous_runs": len(history), "deltas": deltas}