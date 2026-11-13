"""
Google Firestore client for the medical enhancer's configuration and chat logs.

Collections:
  symptom_library        doc id = disease slug            { disease, symptoms }
  organ_aliases          doc id = alias slug              { alias, canonical }
  branch_hints           doc id = disease slug            { disease, branch }
  chat_maps              doc id = organ_map | disease_map { kind, map|list }
  stop_words             doc id = word                    { word }
  organ_branch_families  doc id = organ slug              { organ, branches }
  chat_logs              doc id = {patient_id}__{started_at} (URL-safe)

Env vars:
  FIRESTORE_PROJECT_ID   optional, defaults to ADC project
  GOOGLE_APPLICATION_CREDENTIALS   path to a service-account JSON

Abhishek's Firestore version. Changes made when merging into our app are
marked "Patched:" — import paths, the two missing loaders (stop_words,
organ_branch_families) and their seeding, a 10 s limit on every Firestore
call, index-free per-patient chat loading, and delete_chat_log().
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Set

from google.api_core.retry import Retry, if_transient_error
from google.cloud import firestore

from .bootstrap_data import (
    BOOTSTRAP_SYMPTOM_LIBRARY, BOOTSTRAP_ORGAN_ALIASES,
    BOOTSTRAP_BRANCH_HINTS, BOOTSTRAP_ORGAN_MAP, BOOTSTRAP_DISEASE_MAP,
    BOOTSTRAP_STOP_WORDS, BOOTSTRAP_ORGAN_BRANCH_FAMILIES,
)

logger = logging.getLogger("config_db")

# Collection names
COLL_SYMPTOM_LIBRARY = "symptom_library"
COLL_ORGAN_ALIASES = "organ_aliases"
COLL_BRANCH_HINTS = "branch_hints"
COLL_CHAT_MAPS = "chat_maps"
COLL_CHAT_LOGS = "chat_logs"
COLL_STOP_WORDS = "stop_words"
COLL_ORGAN_BRANCH_FAMILIES = "organ_branch_families"

# Patched: give up after ~10 s. The library default retries for 5 minutes,
# which froze RL saves and training whenever the Google login had expired.
CALL = {"retry": Retry(predicate=if_transient_error, timeout=10.0), "timeout": 10.0}

_client_lock = threading.RLock()
_client: Optional[firestore.Client] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def project_id() -> Optional[str]:
    # Patched: read when the client is made, so a value from .env counts.
    return os.getenv("FIRESTORE_PROJECT_ID") or None


def _get_client() -> firestore.Client:
    global _client
    with _client_lock:
        if _client is None:
            _client = firestore.Client(project=project_id())
        return _client


def get_db():
    """Kept for backward compatibility — returns the Firestore client."""
    return _get_client()


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(s: str) -> str:
    s = (s or "").strip().lower()
    s = _SLUG_RE.sub("_", s).strip("_")
    return s or "unknown"


def _chat_log_doc_id(patient_id: str, started_at: str) -> str:
    return f"{_slug(patient_id)}__{_slug(started_at)}"


def _is_empty(client: firestore.Client, collection: str) -> bool:
    return next(client.collection(collection).limit(1).stream(**CALL), None) is None


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_if_empty() -> None:
    """Insert the bootstrap values if their collections are empty."""
    try:
        client = _get_client()

        if _is_empty(client, COLL_SYMPTOM_LIBRARY):
            batch = client.batch()
            count = 0
            for disease, symptoms in BOOTSTRAP_SYMPTOM_LIBRARY.items():
                ref = client.collection(COLL_SYMPTOM_LIBRARY).document(_slug(disease))
                batch.set(ref, {"disease": disease.lower(),
                                "symptoms": list(symptoms)})
                count += 1
                if count % 400 == 0:
                    batch.commit(**CALL)
                    batch = client.batch()
            batch.commit(**CALL)
            logger.info("Seeded %d symptom_library entries", count)

        if _is_empty(client, COLL_ORGAN_ALIASES):
            batch = client.batch()
            count = 0
            for alias, canonical in BOOTSTRAP_ORGAN_ALIASES.items():
                ref = client.collection(COLL_ORGAN_ALIASES).document(_slug(alias))
                batch.set(ref, {"alias": alias.lower(), "canonical": canonical})
                count += 1
            batch.commit(**CALL)
            logger.info("Seeded %d organ_aliases entries", count)

        if _is_empty(client, COLL_BRANCH_HINTS):
            batch = client.batch()
            count = 0
            for disease, branch in BOOTSTRAP_BRANCH_HINTS.items():
                ref = client.collection(COLL_BRANCH_HINTS).document(_slug(disease))
                batch.set(ref, {"disease": disease.lower(), "branch": branch})
                count += 1
            batch.commit(**CALL)
            logger.info("Seeded %d branch_hints entries", count)

        organ_map_doc = client.collection(COLL_CHAT_MAPS).document("organ_map")
        disease_map_doc = client.collection(COLL_CHAT_MAPS).document("disease_map")
        if not organ_map_doc.get(**CALL).exists:
            organ_map_doc.set({"kind": "organ_map", "map": BOOTSTRAP_ORGAN_MAP}, **CALL)
            logger.info("Seeded chat_maps/organ_map")
        if not disease_map_doc.get(**CALL).exists:
            disease_map_doc.set({"kind": "disease_map",
                                 "list": BOOTSTRAP_DISEASE_MAP}, **CALL)
            logger.info("Seeded chat_maps/disease_map")

        # Patched: these two were never seeded in the Firestore version.
        if _is_empty(client, COLL_STOP_WORDS):
            batch = client.batch()
            for word in BOOTSTRAP_STOP_WORDS:
                batch.set(client.collection(COLL_STOP_WORDS).document(_slug(word)),
                          {"word": word.lower()})
            batch.commit(**CALL)
            logger.info("Seeded %d stop_words entries", len(BOOTSTRAP_STOP_WORDS))

        if _is_empty(client, COLL_ORGAN_BRANCH_FAMILIES):
            batch = client.batch()
            for organ, branches in BOOTSTRAP_ORGAN_BRANCH_FAMILIES.items():
                batch.set(client.collection(COLL_ORGAN_BRANCH_FAMILIES).document(_slug(organ)),
                          {"organ": organ, "branches": list(branches)})
            batch.commit(**CALL)
            logger.info("Seeded %d organ_branch_families entries",
                        len(BOOTSTRAP_ORGAN_BRANCH_FAMILIES))

        # Create chat_logs collection marker doc (optional, harmless).
        # Patched: "__meta__" is a reserved Firestore id (400 error every reload).
        marker = client.collection(COLL_CHAT_LOGS).document("meta_marker")
        if not marker.get(**CALL).exists:
            marker.set({"kind": "meta", "created_at": firestore.SERVER_TIMESTAMP}, **CALL)

    except Exception as e:
        logger.warning("Firestore seeding skipped: %s", e)


# ---------------------------------------------------------------------------
# Loaders (used by config_loader.reload_all)
# ---------------------------------------------------------------------------

def load_symptom_library() -> Dict[str, List[str]]:
    client = _get_client()
    out: Dict[str, List[str]] = {}
    for doc in client.collection(COLL_SYMPTOM_LIBRARY).stream(**CALL):
        d = doc.to_dict() or {}
        disease = (d.get("disease") or doc.id).strip().lower()
        symptoms = d.get("symptoms") or []
        if disease and isinstance(symptoms, list):
            out[disease] = [str(s) for s in symptoms]
    return out


def load_organ_aliases() -> Dict[str, str]:
    client = _get_client()
    out: Dict[str, str] = {}
    for doc in client.collection(COLL_ORGAN_ALIASES).stream(**CALL):
        d = doc.to_dict() or {}
        alias = (d.get("alias") or doc.id).strip().lower()
        canonical = d.get("canonical")
        if alias and canonical:
            out[alias] = str(canonical)
    return out


def load_branch_hints() -> Dict[str, str]:
    client = _get_client()
    out: Dict[str, str] = {}
    for doc in client.collection(COLL_BRANCH_HINTS).stream(**CALL):
        d = doc.to_dict() or {}
        disease = (d.get("disease") or doc.id).strip().lower()
        branch = d.get("branch")
        if disease and branch:
            out[disease] = str(branch).lower()
    return out


def load_chat_maps() -> Dict[str, Any]:
    import re as _re
    client = _get_client()
    organ_map: Dict[str, str] = {}
    disease_list: List[Dict[str, str]] = []

    organ_doc = client.collection(COLL_CHAT_MAPS).document("organ_map").get(**CALL)
    if organ_doc.exists:
        for k, v in ((organ_doc.to_dict() or {}).get("map") or {}).items():
            organ_map[str(k).lower()] = str(v)

    disease_doc = client.collection(COLL_CHAT_MAPS).document("disease_map").get(**CALL)
    if disease_doc.exists:
        for entry in ((disease_doc.to_dict() or {}).get("list") or []):
            pat = entry.get("pattern")
            canonical = entry.get("canonical")
            if pat and canonical:
                disease_list.append({"pattern": str(pat),
                                     "canonical": str(canonical)})

    compiled = [(_re.compile(e["pattern"], _re.I), e["canonical"])
                for e in disease_list]
    return {"organ_map": organ_map, "disease_map": compiled}


# Patched: config_loader.reload_all() calls these two; without them every
# reload failed and fell back to the built-in values.
def load_stop_words() -> Set[str]:
    client = _get_client()
    out: Set[str] = set()
    for doc in client.collection(COLL_STOP_WORDS).stream(**CALL):
        w = ((doc.to_dict() or {}).get("word") or "").strip().lower()
        if w:
            out.add(w)
    return out


def load_organ_branch_families() -> Dict[str, List[str]]:
    client = _get_client()
    out: Dict[str, List[str]] = {}
    for doc in client.collection(COLL_ORGAN_BRANCH_FAMILIES).stream(**CALL):
        d = doc.to_dict() or {}
        organ = (d.get("organ") or "").strip()
        branches = d.get("branches") or []
        if organ and isinstance(branches, list):
            out[organ] = [str(b).lower() for b in branches]
    return out


# ---------------------------------------------------------------------------
# Append-only symptom library write
# ---------------------------------------------------------------------------

def append_symptoms_to_library(additions: Dict[str, List[str]]) -> Dict[str, int]:
    """
    Append-only merge using Firestore transactions.

    Firestore's ArrayUnion gives us the exact semantics we need:
      - never removes items
      - never reorders existing items
      - adds only the items not already present
    """
    client = _get_client()

    inserted = 0
    appended = 0
    added = 0
    skipped = 0

    for disease, new_items in additions.items():
        d_key = (disease or "").strip().lower()
        if not d_key:
            continue

        incoming: List[str] = []
        seen: Set[str] = set()
        for s in new_items or []:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            k = t.lower()
            if k in seen:
                continue
            seen.add(k)
            incoming.append(t)

        if not incoming:
            skipped += 1
            continue

        doc_ref = client.collection(COLL_SYMPTOM_LIBRARY).document(_slug(d_key))
        doc = doc_ref.get(**CALL)

        if not doc.exists:
            doc_ref.set({"disease": d_key, "symptoms": incoming}, **CALL)
            inserted += 1
            added += len(incoming)
            continue

        existing = (doc.to_dict() or {}).get("symptoms") or []
        existing_lower = {s.lower() for s in existing if isinstance(s, str)}
        fresh = [s for s in incoming if s.lower() not in existing_lower]

        if not fresh:
            skipped += 1
            continue

        doc_ref.update({"symptoms": firestore.ArrayUnion(fresh)}, **CALL)
        appended += 1
        added += len(fresh)

    return {
        "diseases_inserted": inserted,
        "diseases_appended": appended,
        "symptoms_added": added,
        "diseases_skipped": skipped,
    }


def write_symptom_library(library: Dict[str, List[str]]) -> int:
    """Backward-compatible wrapper. Append-only by delegation."""
    stats = append_symptoms_to_library(library)
    return stats.get("symptoms_added", 0)


# ---------------------------------------------------------------------------
# Chat logs
# ---------------------------------------------------------------------------

def ensure_chat_log_indexes() -> None:
    """Firestore has no manual indexes; kept for API parity. No-op."""
    return None


def upsert_chat_log(doc: Dict[str, Any]) -> None:
    """
    Insert or update a chat log keyed on (patient_id, started_at).
    Every new started_at creates a new document.
    """
    patient_id = str(doc.get("patient_id") or "unknown")
    started_at = str(doc.get("started_at") or "").strip()
    if not started_at:
        from datetime import datetime, timezone
        started_at = datetime.now(timezone.utc).isoformat()

    client = _get_client()
    doc_id = _chat_log_doc_id(patient_id, started_at)
    payload = {**doc, "patient_id": patient_id, "started_at": started_at}
    client.collection(COLL_CHAT_LOGS).document(doc_id).set(payload, merge=True, **CALL)


def _patient_chat_docs(patient_id: str):
    # Patched: equality filter only, sorted here. where() + order_by() needs a
    # composite index; without it the query failed and RL saw 0 past chats.
    client = _get_client()
    return list(client.collection(COLL_CHAT_LOGS)
                .where(filter=firestore.FieldFilter("patient_id", "==", patient_id))
                .stream(**CALL))


def load_chat_logs_for_patient(patient_id: str,
                               limit: int = 50) -> List[Dict[str, Any]]:
    if not patient_id:
        return []
    docs: List[Dict[str, Any]] = []
    for d in _patient_chat_docs(patient_id):
        payload = d.to_dict() or {}
        payload.pop("_cached_at", None)
        docs.append(payload)
    docs.sort(key=lambda p: str(p.get("started_at") or ""))
    return docs[:limit]


def delete_chat_log(patient_id: str, session_id: str) -> int:
    """Patched: delete one patient's chat by session_id (used to remove test
    chats so V2 doesn't learn from them). Returns how many docs were deleted."""
    deleted = 0
    for d in _patient_chat_docs(patient_id):
        if (d.to_dict() or {}).get("session_id") == session_id:
            d.reference.delete(**CALL)
            deleted += 1
    return deleted


def chat_logs_total() -> int:
    """Approximate; Firestore has no cheap total-count on large sets."""
    try:
        client = _get_client()
        return sum(1 for d in client.collection(COLL_CHAT_LOGS).limit(1000).stream(**CALL)
                   if (d.to_dict() or {}).get("kind") != "meta")
    except Exception:
        return 0


def chat_logs_distinct_patients() -> int:
    try:
        client = _get_client()
        seen = set()
        for d in client.collection(COLL_CHAT_LOGS).limit(1000).stream(**CALL):
            pid = (d.to_dict() or {}).get("patient_id")
            if pid:
                seen.add(pid)
        return len(seen)
    except Exception:
        return 0


def load_all_chat_logs(limit: int = 10_000) -> List[Dict[str, Any]]:
    """
    Return every chat-log document, up to `limit`.

    Firestore has no cursor-based 'find' — we stream the collection and
    strip the bookkeeping fields before returning plain dicts.
    """
    client = _get_client()
    docs: List[Dict[str, Any]] = []
    try:
        for doc in client.collection(COLL_CHAT_LOGS).limit(limit).stream(**CALL):
            payload = doc.to_dict() or {}
            # Skip the metadata marker doc seeded on first launch
            if payload.get("kind") == "meta":
                continue
            payload.pop("_cached_at", None)
            docs.append(payload)
    except Exception as e:
        logger.warning("load_all_chat_logs failed: %s", e)
    return docs
