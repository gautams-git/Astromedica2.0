"""Explicit patient deletion. Not used by migration."""

from __future__ import annotations


def delete_patient(plane, internal_user_id: str, patient_id: str) -> bool:
    """Remove the patient document and its nested subcollections.

    Callers that also store Prokerala files on disk or model artifacts in
    Cloud Storage have to delete those separately. This function does not
    delete another user's copy of a shared patient id.
    """
    return bool(plane.delete_patient_tree(internal_user_id, patient_id))
