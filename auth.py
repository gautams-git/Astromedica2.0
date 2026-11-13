"""
auth.py — real per-user authentication, replacing the single shared
SITE_PASSWORD gate.

Password hashing via werkzeug.security — already a Flask dependency
(werkzeug is what Flask itself is built on), so this needs no new package.
Uses PBKDF2 under the hood, not the crude single shared password this
replaces.

Nothing here talks to storage directly — register_user() and
authenticate() both go through db_layer.py's users table (still a stub
today), same "one choke point" rule as everything else.
"""

from __future__ import annotations

from werkzeug.security import check_password_hash, generate_password_hash

import db_layer

VALID_ROLES = {"patient", "doctor", "admin"}


def register_user(email: str, password: str, role: str = "patient") -> str:
    """
    Hashes the password before it ever reaches db_layer — the stored
    users.json never contains a plaintext password, matching what a real
    users table would enforce too.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"invalid role '{role}' — must be one of {VALID_ROLES}")
    password_hash = generate_password_hash(password)
    return db_layer.create_user(email, password_hash, role)


def authenticate(email: str, password: str) -> dict | None:
    """
    Returns the user record if credentials are valid, None otherwise.
    Deliberately returns the same None for "no such user" and "wrong
    password" — the caller should show one generic error either way, so a
    failed login doesn't reveal whether an email address has an account.
    """
    user = db_layer.get_user_by_email(email)
    if not user:
        return None
    if not check_password_hash(user["password_hash"], password):
        return None
    return user
