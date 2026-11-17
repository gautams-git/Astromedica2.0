"""Identity checks that do not trust a browser-supplied user id."""

from __future__ import annotations

from app.core.errors import ForgedIdentityError, OwnershipError

_FORBIDDEN = set("/\\.#$[]\x00")


def require_document_id(value: str, *, label: str) -> str:
    if not isinstance(value, str):
        raise ForgedIdentityError(f"{label} is missing")
    cleaned = value.strip()
    if not cleaned or len(cleaned) > 700 or cleaned in {".", ".."}:
        raise ForgedIdentityError(f"{label} is not a usable document id")
    if any(char in _FORBIDDEN for char in cleaned):
        raise ForgedIdentityError(f"{label} contains a forbidden character")
    return cleaned


def assert_owner(actor_internal_user_id: str, owner_internal_user_id: str) -> None:
    """Reject a request whose actor is not the record owner.

    Admin bypass belongs in the admin routes, not in the patient repository.
    """
    actor = require_document_id(actor_internal_user_id, label="actor")
    owner = require_document_id(owner_internal_user_id, label="owner")
    if actor != owner:
        raise OwnershipError("patient data is isolated to the owning account")
