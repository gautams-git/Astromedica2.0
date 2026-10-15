"""
app_auth.py — glue between our Flask routes and the colleague's auth + DB
code in app_core/ (Clerk login + Firestore repository), copied in unchanged
apart from removing debug prints.

Switch: AUTH_ENABLED environment variable (default off).
  off -> no login; every request runs as one local user ("local-dev") so the
         DB saves can still be tested. No patient-ownership checks.
  on  -> Clerk token required on protected routes (401 without it), and a
         user can only open patients they generated themselves (404 otherwise).
         Needs CLERK_PUBLISHABLE_KEY + CLERK_SECRET_KEY in the environment.

Main DB: app_core picks the store from the environment —
  default (APP_ENV=development)  -> in-memory, lost on every restart
  FIRESTORE_EMULATOR_HOST set    -> local Firestore emulator
  APP_ENV=production / FIRESTORE_BACKEND=firestore -> real Firestore
"""

from __future__ import annotations

import os
from functools import wraps

# His settings default to port 8000; ours runs on 5000. Only a default —
# anything set in the environment wins.
os.environ.setdefault("FRONTEND_URL", "http://localhost:5000")
os.environ.setdefault("API_URL", "http://localhost:5000")

from flask import g, jsonify

from app_core.errors import (
    AuthError,
    AuthorizationError,
    RateLimitError,
    RequestValidationError,
    ServiceUnavailableError,
)
from app_core.http import enforce_patient_access, require_auth
from app_core.repository import MemoryRepository
from app_core.runtime import get_services
from app_core.validation import require_patient_id

from google.api_core.exceptions import GoogleAPIError
from google.auth.exceptions import GoogleAuthError

# Google errors from the main DB (Firestore down, expired Google login...).
DB_ERRORS = (GoogleAPIError, GoogleAuthError)

# Routes that wrap their body in `except Exception` must re-raise these first,
# otherwise a 401/404 from the checks below (or a DB outage) turns into a
# generic 500 instead of the handlers in register_auth_errors.
AUTH_ERRORS = (AuthError, AuthorizationError, RateLimitError, RequestValidationError,
               ServiceUnavailableError) + DB_ERRORS


def db_error_hint(exc: Exception) -> str:
    """One console line saying what went wrong with the main DB and how to fix it."""
    text = str(exc)
    if "Reauthentication is needed" in text or "invalid_grant" in text:
        return "Google login expired - run `gcloud auth application-default login`, then restart the server"
    if "Getting metadata from plugin failed" in text or isinstance(exc, GoogleAuthError):
        # Any other failure to get a Google login token for this PC.
        return ("Google login on this PC is not working - run `gcloud auth application-default login`, "
                "then restart the server")
    if "default credentials were not found" in text.lower():
        return "no Google login on this PC - run `gcloud auth application-default login`"
    if "PERMISSION_DENIED" in text or "403" in text:
        return "this Google account has no access to the Firestore project (needs Cloud Datastore User)"
    return f"{type(exc).__name__}: {text[:200]}"

AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
LOCAL_DEV_USER = "local-dev"

# Admins see every patient. Clerk user ids (user_...), comma-separated, from
# .env. Each signed-in user's id is printed once in the console, so copy it
# from there. Everyone else is a patient and sees only their own charts.
ADMIN_CLERK_USER_IDS = {
    item.strip() for item in os.environ.get("ADMIN_CLERK_USER_IDS", "").split(",") if item.strip()
}
_announced_users: set[str] = set()


def is_admin() -> bool:
    if not AUTH_ENABLED:
        return True  # login off = local testing, sees everything as before
    principal = getattr(g, "principal", None)
    return bool(principal and principal.clerk_user_id in ADMIN_CLERK_USER_IDS)


def _announce_user(func):
    """Prints each signed-in user's Clerk id and role once per server run."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        clerk_id = g.principal.clerk_user_id
        if clerk_id not in _announced_users:
            _announced_users.add(clerk_id)
            role = "admin" if is_admin() else "patient (add this id to ADMIN_CLERK_USER_IDS in .env for admin)"
            print(f"[Auth] Signed in: {clerk_id} -> {role}")
        return func(*args, **kwargs)
    return wrapper


def login_required(rate_limit: str | None = None):
    """Route decorator. Sets g.user = {"internal_user_id": ...} either way."""
    def decorator(func):
        if AUTH_ENABLED:
            return require_auth(rate_limit=rate_limit)(_announce_user(func))

        @wraps(func)
        def wrapper(*args, **kwargs):
            g.user = get_services().repository.get_or_create_user(LOCAL_DEV_USER)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _hidden_from_user(patient_id: str) -> bool:
    # A chart replaced by the user's one correction (chart_access) stays for
    # admins but is no longer the user's to open.
    doc = get_services().repository.get(g.user["internal_user_id"], "patients", patient_id)
    return bool(doc and doc.get("hidden"))


def check_patient_access(patient_id: str) -> None:
    """Rejects a malformed id always; rejects someone else's (or a hidden)
    patient for non-admins when auth is on."""
    if AUTH_ENABLED and not is_admin():
        enforce_patient_access(patient_id)
        if _hidden_from_user(patient_id):
            raise AuthorizationError()
    else:
        require_patient_id(patient_id)


def user_owns_patient(patient_id: str) -> bool:
    if is_admin():
        return True
    doc = get_services().repository.get(g.user["internal_user_id"], "patients", patient_id)
    return bool(doc and doc.get("owner_internal_user_id") == g.user["internal_user_id"]
                and not doc.get("hidden"))


def register_auth_errors(app) -> None:
    """Only the error handlers his decorators need — not his register_http(),
    whose catch-all handler would hide our routes' own error messages."""

    @app.errorhandler(AuthError)
    def _auth_error(_exc):
        return jsonify({"success": False, "error": "Authentication is required."}), 401

    @app.errorhandler(AuthorizationError)
    def _not_yours(_exc):
        return jsonify({"success": False, "error": "Not found."}), 404

    @app.errorhandler(RateLimitError)
    def _limited(exc):
        user = (getattr(g, "user", None) or {}).get("internal_user_id", "?")
        if getattr(exc, "bucket", None):
            print(f"[RateLimit] user {user[:8]}... hit the '{exc.bucket}' limit "
                  f"({exc.limit} per {exc.window // 60} min) - retry in {exc.retry_after}s")
        response = jsonify({"success": False, "code": "rate_limited", "error": "Too many requests.",
                            "retry_after": exc.retry_after})
        response.status_code = 429
        response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.errorhandler(RequestValidationError)
    def _invalid(exc):
        return jsonify({"success": False, "error": exc.message}), 400

    @app.errorhandler(ServiceUnavailableError)
    def _unavailable(_exc):
        return jsonify({"success": False, "error": "The service is not ready."}), 503

    def _db_unavailable(exc):
        print(f"[DB] [FAIL] Firestore not reachable: {db_error_hint(exc)}")
        return jsonify({"success": False,
                        "error": "The database is unavailable right now. Please try again in a minute."}), 503

    for error_type in DB_ERRORS:
        app.register_error_handler(error_type, _db_unavailable)


def main_db_label() -> str:
    """Short name for log lines: 'Firestore' or the in-memory test store."""
    return "the in-memory test DB" if type(get_services().repository) is MemoryRepository else "Firestore"


def describe_main_db() -> str:
    # FirestoreRepository subclasses MemoryRepository, so compare the exact type.
    if type(get_services().repository) is MemoryRepository:
        return "in-memory (testing only, cleared on every restart)"
    settings = get_services().settings
    if settings.firestore_emulator_host:
        return f"Firestore emulator at {settings.firestore_emulator_host}"
    return f"Firestore project {settings.firestore_project_id}"


def print_startup_status() -> None:
    settings = get_services().settings
    if AUTH_ENABLED:
        state = "ON" if settings.auth_configured else "ON but Clerk keys are missing -> protected routes return 503"
        print(f"[Auth] Clerk login: {state}")
        print(f"[Auth] Admins (see all patients): {len(ADMIN_CLERK_USER_IDS)} in ADMIN_CLERK_USER_IDS")
    else:
        print(f"[Auth] Clerk login: OFF (set AUTH_ENABLED=true to turn on) -> requests run as '{LOCAL_DEV_USER}'")
    print(f"[DB] Main DB: {describe_main_db()}")
    repo = get_services().repository
    if type(repo) is not MemoryRepository:
        # Catch an expired Google login now, not on the first patient request.
        try:
            list(repo._client.collections(**repo._call))
            print("[DB] Firestore connection OK")
        except Exception as e:
            print(f"[DB] [FAIL] Firestore not reachable: {db_error_hint(e)}")
