"""
deidentification.py — splits a patient record in two before anything
AI-facing (RAG retrieval, the two LLMs, the RL Driver) ever sees it.

deidentify() takes the full record, writes the identity fields (name, dob,
birth_place, parent names) to db_layer's Identity Vault, and returns only
{token, planets, houses, priority_ladder, combined_scores} — the shape the
client's architecture spec requires for everything downstream.

reidentify() is the only way back to a real identity, and it requires both
a patient_id (not just a token — see note on generate_token below) and an
authorized role. Nothing in the AI layer is expected to ever call it.
"""

from __future__ import annotations

import hashlib
import hmac

import db_layer
from config import DEIDENTIFICATION_SECRET

# Roles allowed to look up a real identity from the vault. Kept here so
# it's one obvious place to review, not scattered across callers.
_REIDENTIFY_ALLOWED_ROLES = {"admin", "doctor"}


def generate_token(patient_id: str) -> str:
    """
    Opaque per-patient identifier for the AI layer: HMAC-SHA256(patient_id).

    This is one-way by design — HMAC is not encryption, so a token cannot
    be reversed back into a patient_id. Its job is only to let the AI/RAG/RL
    layers refer to "this patient" consistently across calls without ever
    knowing who that patient actually is. Real re-identification always
    goes through reidentify(), using the patient_id from an authorized
    session — never by trying to invert this token.
    """
    return hmac.new(
        DEIDENTIFICATION_SECRET.encode("utf-8"),
        patient_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def deidentify(
    patient_id: str,
    name: str,
    dob: str,
    birth_place: str,
    planets: list,
    houses: list,
    priority_ladder: dict,
    combined_scores: list,
    father_name: str | None = None,
    mother_name: str | None = None,
) -> dict:
    """
    Splits a patient record into identity (written to the vault, not
    returned) and the de-identified payload (returned) that everything
    downstream — RAG, LLM 1, LLM 2, the RL Driver — is allowed to see.
    """
    db_layer.save_identity_vault(
        patient_id=patient_id,
        name=name,
        dob=dob,
        birth_place=birth_place,
        father_name=father_name,
        mother_name=mother_name,
    )

    token = generate_token(patient_id)
    print(f"[De-ID] name/dob/birth_place stripped -> token={token[:8]}...{token[-4:]}")
    print("[De-ID] forwarding: token, planets[], houses[], priority_ladder, combined_scores")

    return {
        "token": token,
        "planets": planets,
        "houses": houses,
        "priority_ladder": priority_ladder,
        "combined_scores": combined_scores,
    }


def reidentify(patient_id: str, requesting_role: str) -> dict | None:
    """
    Looks up the real identity for patient_id — only if requesting_role is
    authorized. Raises PermissionError otherwise, rather than silently
    returning None, so an unauthorized call is loud and easy to catch in
    testing rather than quietly failing.
    """
    if requesting_role not in _REIDENTIFY_ALLOWED_ROLES:
        raise PermissionError(
            f"role '{requesting_role}' is not authorized to reidentify patients"
        )
    return db_layer.get_identity_vault(patient_id)
