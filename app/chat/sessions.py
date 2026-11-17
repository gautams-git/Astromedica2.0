"""Durable chat sessions so a second worker or a restart can resume."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import load_data_settings
from app.core.errors import OwnershipError
from app.domain.schema import SOFT_DOCUMENT_BYTES
from app.infrastructure.firestore.codec import byte_length

_LOG = logging.getLogger("astromedica.chat")
_backend = None


def configure_session_backend(plane) -> None:
    global _backend
    _backend = plane


def session_backend():
    if _backend is not None:
        return _backend
    if not load_data_settings().chat_durable:
        return None
    try:
        from app_core.runtime import get_services

        return getattr(get_services().repository, "data_plane", None)
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return None


def snapshot_session(session: dict) -> dict:
    kept = {}
    for key, value in session.items():
        if key.startswith("_"):
            continue
        encoded = _jsonable(value)
        if encoded is not None:
            kept[key] = encoded
    return kept


def persist_session(actor_id: str | None, session_id: str, session: dict) -> None:
    plane = session_backend()
    if plane is None or not actor_id or not session.get("patient_id"):
        return
    try:
        fields = {
            "status": "completed" if session.get("completed") else "active",
            "completed": bool(session.get("completed")),
            "cluster_scores": session.get("cluster_scores") or {},
            "associated_run_id": session.get("associated_run_id"),
            "chat_algorithm_version": session.get("chat_mode") or "classic",
            "session_snapshot": snapshot_session(session),
        }
        if byte_length(fields) > SOFT_DOCUMENT_BYTES:
            fields.pop("session_snapshot", None)
            fields["snapshot_status"] = "omitted_over_size"
        plane.open_chat(actor_id, str(session["patient_id"]), session_id, fields)
    except Exception as exc:
        _LOG.warning("chat session persist failed: %s", type(exc).__name__)


def recover_session(actor_id: str, session_id: str) -> dict | None:
    plane = session_backend()
    if plane is None:
        return None
    index = plane.session_index(session_id)
    if not index or index.get("internal_user_id") != actor_id:
        return None
    if not plane.get_patient(actor_id, actor_id, index["patient_id"]):
        raise OwnershipError("session owner does not own the patient")
    chat = plane.store.get(("users", actor_id, "patients", index["patient_id"], "chats", session_id))
    if not chat:
        return None
    snapshot = dict(chat.get("session_snapshot") or {})
    if not snapshot.get("messages"):
        turns = plane.list_turns(actor_id, actor_id, index["patient_id"], session_id)
        snapshot["messages"] = [{"role": turn.get("role"), "content": turn.get("content")} for turn in turns]
        snapshot["turn_count"] = len(snapshot["messages"])
    snapshot.setdefault("system_prompt", "")
    snapshot["patient_id"] = index["patient_id"]
    snapshot["_internal_user_id"] = actor_id
    return snapshot
