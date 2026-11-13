"""
db_layer.py — the single choke point for all database access.

No other module should read or write a database directly. Every function
below is called by the rest of the app.

Main DB = the colleague's repository in app_core/ (Firestore, or in-memory
for local testing — see app_auth.py). Everything is stored per user:
  users/{internal_user_id}/inputs, chart_inputs, prokerala_results,
      algorithm_results, combined_results, patients, identity_vault

Still on local JSON stubs in _db_stub/ (# TODO(DB)):
  users      — only the old email/password login (auth.py), replaced by Clerk
  kb_chunks  — RAG vectors; Firestore is not a vector store, still undecided
"""

from __future__ import annotations

import json
import os
import uuid

_STUB_DIR = os.path.join(os.path.dirname(__file__), "_db_stub")
_USERS_FILE = os.path.join(_STUB_DIR, "users.json")
_IDENTITY_VAULT_FILE = os.path.join(_STUB_DIR, "identity_vault.json")
_KB_CHUNKS_FILE = os.path.join(_STUB_DIR, "kb_chunks.json")


def _ensure_stub_dir() -> None:
    os.makedirs(_STUB_DIR, exist_ok=True)


def _read_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data) -> None:
    _ensure_stub_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# USERS — per-user auth
# ══════════════════════════════════════════════════════════════

def get_user_by_email(email: str) -> dict | None:
    """# TODO(DB): SELECT * FROM users WHERE email = %s"""
    users = _read_json(_USERS_FILE, {})
    return users.get(email.strip().lower())


def create_user(email: str, password_hash: str, role: str = "patient") -> str:
    """# TODO(DB): INSERT INTO users (id, email, password_hash, role) VALUES (...)"""
    email = email.strip().lower()
    users = _read_json(_USERS_FILE, {})
    if email in users:
        raise ValueError(f"user already exists: {email}")
    user_id = str(uuid.uuid4())
    users[email] = {
        "id": user_id,
        "email": email,
        "password_hash": password_hash,
        "role": role,
    }
    _write_json(_USERS_FILE, users)
    return user_id


# ══════════════════════════════════════════════════════════════
# IDENTITY VAULT — de-identification layer's PII store
# ══════════════════════════════════════════════════════════════

def _current_user_id() -> str | None:
    """The logged-in user's id, set on g.user by app_auth.login_required."""
    from flask import g, has_request_context
    if not has_request_context():
        return None
    return (getattr(g, "user", None) or {}).get("internal_user_id")


def _repository():
    from app_core.runtime import get_services
    return get_services().repository


def save_identity_vault(
    patient_id: str,
    name: str,
    dob: str,
    birth_place: str,
    father_name: str | None = None,
    mother_name: str | None = None,
) -> None:
    record = {
        "patient_id": patient_id,
        "name": name,
        "dob": dob,
        "birth_place": birth_place,
        "father_name": father_name,
        "mother_name": mother_name,
    }
    user_id = _current_user_id()
    if user_id:
        _repository().save(user_id, "identity_vault", patient_id, record)
        return
    # Outside a request (scripts) there is no user to file it under.
    vault = _read_json(_IDENTITY_VAULT_FILE, {})
    vault[patient_id] = record
    _write_json(_IDENTITY_VAULT_FILE, vault)


def get_identity_vault(patient_id: str) -> dict | None:
    """Only finds the current user's own patients."""
    user_id = _current_user_id()
    if user_id:
        return _repository().get(user_id, "identity_vault", patient_id)
    vault = _read_json(_IDENTITY_VAULT_FILE, {})
    return vault.get(patient_id)


# ══════════════════════════════════════════════════════════════
# CHART GENERATIONS — birth input, Prokerala data, algorithm outputs
# ══════════════════════════════════════════════════════════════

def save_chart_generation(patient_id: str, chart_input: dict, source: str, results: dict) -> dict:
    """Saves one /generate-chart run under the current user and marks the
    patient as theirs (that's what the auth ownership check reads)."""
    from app_core.runtime import get_services
    user_id = _current_user_id()
    if not user_id:
        raise RuntimeError("save_chart_generation needs a logged-in request")
    return get_services().persistence.save_generation(user_id, chart_input, patient_id, source, results)


# ══════════════════════════════════════════════════════════════
# KB CHUNKS — RAG retrieval corpus
# ══════════════════════════════════════════════════════════════

def insert_kb_chunk(text: str, embedding: list[float], metadata: dict | None = None) -> str:
    """# TODO(DB): INSERT INTO kb_chunks (id, text, embedding, metadata) VALUES (...)"""
    chunks = _read_json(_KB_CHUNKS_FILE, [])
    chunk_id = str(uuid.uuid4())
    chunks.append({
        "id": chunk_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {},
    })
    _write_json(_KB_CHUNKS_FILE, chunks)
    return chunk_id


def query_similar_chunks(embedding: list[float], top_k: int = 5) -> list[dict]:
    """# TODO(DB): SELECT * FROM kb_chunks ORDER BY embedding <=> %s LIMIT %s
    Stub returns [] until real chunks exist and a real vector index runs
    the similarity search — there is no in-memory cosine-similarity
    fallback here on purpose, so callers see an honest "no results" state
    rather than a fake ranking over stub data.
    """
    return []
