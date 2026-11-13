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
    "unlock": (10, 300),   # Dev view password tries (server.py /dev-mode/unlock)
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
                # Seconds until the oldest counted request leaves the window,
                # so the page can show a real countdown.
                wait = max(1, int(window - (now - queue[0])) + 1)
                error = RateLimitError(retry_after=wait)
                error.bucket, error.limit, error.window = bucket, limit, window
                raise error
            queue.append(now)

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
