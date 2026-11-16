"""Memory of earlier Rashi runs per user / chart.

Same store as the disease agent, but a separate instance: sharing one would let Rashi runs push
disease runs out of the MAX_KEYS window, and two stores writing one JSON file would overwrite
each other. Keys also carry the algorithm name, so they can never collide.
"""
import os
from threading import Lock
from typing import Optional

from agentic_disease_first.memory import MemoryStore

ALGORITHM = "rashi_correlation"


def _rashi_path() -> Optional[str]:
    """AGENT_MEMORY_PATH_RASHI, else AGENT_MEMORY_PATH with a _rashi suffix, else in-process only."""
    explicit = os.getenv("AGENT_MEMORY_PATH_RASHI")
    if explicit:
        return explicit
    base = os.getenv("AGENT_MEMORY_PATH")
    if not base:
        return None
    root, ext = os.path.splitext(base)
    return f"{root}_rashi{ext or '.json'}"


class RashiMemoryStore(MemoryStore):
    # Own singleton slots: the inherited shared() would otherwise return the disease store.
    _shared: Optional["RashiMemoryStore"] = None
    _shared_lock = Lock()

    def __init__(self, path: Optional[str] = None):
        super().__init__(path or _rashi_path())


__all__ = ["ALGORITHM", "MemoryStore", "RashiMemoryStore"]
