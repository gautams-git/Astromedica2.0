"""Field encryption for private identity documents.

The key comes from IDENTITY_ENCRYPTION_KEY or DEIDENTIFICATION_SECRET.
It is never written to Firestore.
"""

from __future__ import annotations

import base64
import hashlib
import json

from app.core.errors import DataPlaneError

_PREFIX = "enc:v1:"


def encryption_active(key: str) -> bool:
    return bool(key)


def _fernet(key: str):
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise DataPlaneError("cryptography is required to encrypt identity fields") from exc
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def encrypt_mapping(payload: dict, key: str) -> str:
    token = _fernet(key).encrypt(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
    return _PREFIX + token.decode("ascii")


def decrypt_mapping(ciphertext: str, key: str) -> dict:
    if not ciphertext.startswith(_PREFIX):
        raise DataPlaneError("identity ciphertext is not in the expected format")
    raw = _fernet(key).decrypt(ciphertext[len(_PREFIX) :].encode("ascii"))
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise DataPlaneError("identity ciphertext did not contain an object")
    return data
