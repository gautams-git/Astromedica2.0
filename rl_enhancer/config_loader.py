"""
In-memory cache of the four configurable data structures.

- Loaded from Firestore on app start.
- Refreshed every 30 minutes by a background thread.
- Falls back to bootstrap values if Firestore is unreachable.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Tuple, Set

# config_loader.py — replace the bootstrap_data import with this
from .bootstrap_data import (
    BOOTSTRAP_SYMPTOM_LIBRARY, BOOTSTRAP_ORGAN_ALIASES,
    BOOTSTRAP_BRANCH_HINTS, BOOTSTRAP_ORGAN_MAP, BOOTSTRAP_DISEASE_MAP,
    BOOTSTRAP_STOP_WORDS, BOOTSTRAP_ORGAN_BRANCH_FAMILIES,
)

logger = logging.getLogger("config_loader")

REFRESH_SECONDS = 30 * 60   # 30 minutes

_LOCK = threading.RLock()

# Live dictionaries exposed to the rest of the app.
SYMPTOM_LIBRARY: Dict[str, List[str]] = dict(BOOTSTRAP_SYMPTOM_LIBRARY)
ORGAN_ALIASES: Dict[str, str] = dict(BOOTSTRAP_ORGAN_ALIASES)
BRANCH_HINTS: Dict[str, str] = dict(BOOTSTRAP_BRANCH_HINTS)
ORGAN_MAP: Dict[str, str] = dict(BOOTSTRAP_ORGAN_MAP)
DISEASE_MAP: List[Tuple[Any, str]] = []
# config_loader.py — add these two below the existing cache declarations
STOP_WORDS: Set[str] = set(BOOTSTRAP_STOP_WORDS)
ORGAN_BRANCH_FAMILIES: Dict[str, List[str]] = dict(BOOTSTRAP_ORGAN_BRANCH_FAMILIES)

_last_loaded: float = 0.0
_source: str = "bootstrap"


def _compile_bootstrap_disease_map():
    import re
    return [(re.compile(e["pattern"], re.I), e["canonical"])
            for e in BOOTSTRAP_DISEASE_MAP]


# Initialise disease_map from bootstrap right away
DISEASE_MAP = _compile_bootstrap_disease_map()


def reload_all() -> Dict[str, Any]:
    """Reload everything from Firestore. Falls back to bootstrap on failure."""
    # config_loader.py — inside reload_all(), after the global line
    global SYMPTOM_LIBRARY, ORGAN_ALIASES, BRANCH_HINTS, ORGAN_MAP, DISEASE_MAP
    global STOP_WORDS, ORGAN_BRANCH_FAMILIES
    global _last_loaded, _source

    try:
        from . import config_db
    except ImportError as e:
        logger.warning("config_db unavailable: %s", e)
        with _LOCK:
            _last_loaded = time.time()
            _source = "bootstrap"
        return _status()

    try:
        config_db.seed_if_empty()

        new_symptoms = config_db.load_symptom_library()
        new_aliases = config_db.load_organ_aliases()
        new_hints = config_db.load_branch_hints()
        new_chat_maps = config_db.load_chat_maps()
        # config_loader.py — inside the try-block, after the existing loads
        new_stop = config_db.load_stop_words()
        new_families = config_db.load_organ_branch_families()

        with _LOCK:
            SYMPTOM_LIBRARY = new_symptoms or dict(BOOTSTRAP_SYMPTOM_LIBRARY)
            ORGAN_ALIASES = new_aliases or dict(BOOTSTRAP_ORGAN_ALIASES)
            BRANCH_HINTS = new_hints or dict(BOOTSTRAP_BRANCH_HINTS)
            ORGAN_MAP = new_chat_maps["organ_map"] or dict(BOOTSTRAP_ORGAN_MAP)
            DISEASE_MAP = new_chat_maps["disease_map"] or _compile_bootstrap_disease_map()
            # config_loader.py — inside the with _LOCK: block, after the existing assignments
            STOP_WORDS = new_stop or set(BOOTSTRAP_STOP_WORDS)
            ORGAN_BRANCH_FAMILIES = (new_families
                                     or dict(BOOTSTRAP_ORGAN_BRANCH_FAMILIES))
            _last_loaded = time.time()
            _source = "firestore"

            # config_loader.py — inside reload_all(), update the log message
            logger.info(
                "config_loader reloaded (symptoms=%d, aliases=%d, hints=%d, "
                "organ_map=%d, disease_map=%d, stop_words=%d, families=%d)",
                len(SYMPTOM_LIBRARY), len(ORGAN_ALIASES),
                len(BRANCH_HINTS), len(ORGAN_MAP), len(DISEASE_MAP),
                len(STOP_WORDS), len(ORGAN_BRANCH_FAMILIES),
            )
    except Exception as e:
        logger.warning("config_loader: reload failed, keeping current values: %s", e)
        with _LOCK:
            _last_loaded = time.time()
            _source = "bootstrap"

    return _status()


def _status() -> Dict[str, Any]:
    with _LOCK:
        return {
            "source": _source,
            "last_loaded": _last_loaded,
            "symptom_library_size": len(SYMPTOM_LIBRARY),
            "organ_aliases_size": len(ORGAN_ALIASES),
            "branch_hints_size": len(BRANCH_HINTS),
            "organ_map_size": len(ORGAN_MAP),
            "disease_map_size": len(DISEASE_MAP),
            "stop_words_size": len(STOP_WORDS),
            "organ_branch_families_size": len(ORGAN_BRANCH_FAMILIES),
        }



def status() -> Dict[str, Any]:
    return _status()


# ---------------------------------------------------------------------------
# Background refresher
# ---------------------------------------------------------------------------

_thread: threading.Thread | None = None
_stop = threading.Event()


def _loop(interval_seconds: int = REFRESH_SECONDS) -> None:
    logger.info("config_loader refresher started (every %ds)", interval_seconds)
    while not _stop.wait(interval_seconds):
        logger.info("config_loader: periodic reload")
        reload_all()


def start_refresher(interval_seconds: int = REFRESH_SECONDS) -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_loop, args=(interval_seconds,),
        name="config-refresher", daemon=True,
    )
    _thread.start()


def stop_refresher() -> None:
    _stop.set()

# ---------------------------------------------------------------------------
# Chat-log pass-through (per-patient, fetched live)
# ---------------------------------------------------------------------------

def get_chat_logs_for_patient(patient_id: str,
                              limit: int = 50) -> List[Dict[str, Any]]:
    """
    Return all chat_log documents for a patient from Firestore.
    Falls back to an empty list if Firestore is unavailable.
    """
    if not patient_id:
        return []
    try:
        from . import config_db
        return config_db.load_chat_logs_for_patient(patient_id, limit=limit)
    except Exception as e:
        logger.warning("Could not load chat logs for %s: %s", patient_id, e)
        return []


def upsert_chat_log(doc: Dict[str, Any]) -> None:
    """Store a single chat-log document (patient_id + session_id key)."""
    from . import config_db
    config_db.upsert_chat_log(doc)


# config_loader.py — append

def refresh_symptom_library_after_write() -> Dict[str, Any]:
    """
    Reload the symptom library from Firestore, swap the in-memory copy, and
    invalidate the rarity table in enhancer.py so it rebuilds on next use.
    """
    global SYMPTOM_LIBRARY

    try:
        from . import config_db
        new_library = config_db.load_symptom_library()
    except Exception as e:
        logger.warning("Could not reload symptom library: %s", e)
        return {"ok": False, "error": str(e)}

    with _LOCK:
        if new_library:
            SYMPTOM_LIBRARY = new_library

    try:
        from . import enhancer
        if hasattr(enhancer, "_DF_CACHE"):
            enhancer._DF_CACHE = None  # type: ignore[attr-defined]
    except Exception:
        pass

    return {"ok": True, "symptom_library_size": len(SYMPTOM_LIBRARY)}