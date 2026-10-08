"""
seed_users.py — one-off script to create test accounts, since there's no
self-registration flow (out of scope for this pass — the spec's RBAC is
Patient/Doctor/Admin, not public sign-up).

Run directly: python seed_users.py
Safe to re-run — skips any account that already exists rather than erroring.
"""

import auth

TEST_ACCOUNTS = [
    ("doctor@astromedica.test", "ChangeMe123!", "doctor"),
    ("admin@astromedica.test", "ChangeMe123!", "admin"),
]

if __name__ == "__main__":
    for email, password, role in TEST_ACCOUNTS:
        try:
            user_id = auth.register_user(email, password, role)
            print(f"created {role}: {email} (id={user_id})")
        except ValueError:
            print(f"already exists, skipped: {email}")
