"""Process-local rate limits.

This is sufficient for the single-VM deployment. It does not coordinate across
multiple processes or virtual machines. Each Gunicorn worker keeps its own window.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from threading import Lock

from app_core.errors import RateLimitError

_LIMITS = {
    "chart": (6, 300),
    "chat": (30, 300),
    "read": (120, 60),
}


class RateLimiter:
    def __init__(self):
        self._events: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def check(self, identity: str, bucket: str) -> None:
        limit, window = _LIMITS[bucket]
        now = time.monotonic()
        key = f"{bucket}:{identity}"
        with self._lock:
            queue = self._events[key]
            while queue and now - queue[0] > window:
                queue.popleft()
            if len(queue) >= limit:
                raise RateLimitError(retry_after=window)
            queue.append(now)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
