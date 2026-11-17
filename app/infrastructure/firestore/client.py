"""One Firestore client for the process. Credentials come from ADC or the emulator."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

from app.core.config import DataSettings, load_data_settings
from app.core.errors import FirestoreUnavailableError

_lock = Lock()
_client: Any = None
_client_key: tuple | None = None


def get_shared_client(settings: DataSettings | None = None) -> Any | None:
    """Return the shared client, or None when Firestore is not configured."""
    settings = settings or load_data_settings()
    if not settings.firestore_enabled and not settings.emulator_host:
        return None
    if not settings.project_id and not settings.emulator_host:
        raise FirestoreUnavailableError("FIRESTORE_PROJECT_ID is required")
    global _client, _client_key
    key = (settings.project_id, settings.emulator_host)
    with _lock:
        if _client is not None and _client_key == key:
            return _client
        if settings.emulator_host:
            os.environ["FIRESTORE_EMULATOR_HOST"] = settings.emulator_host
        try:
            from google.cloud import firestore
        except ImportError as exc:
            raise FirestoreUnavailableError("google-cloud-firestore is not installed") from exc
        kwargs: dict[str, Any] = {"project": settings.project_id or "astromedica-local"}
        if settings.emulator_host:
            from google.auth.credentials import AnonymousCredentials

            kwargs["credentials"] = AnonymousCredentials()
        _client = firestore.Client(**kwargs)
        _client_key = key
        return _client


def reset_shared_client() -> None:
    global _client, _client_key
    with _lock:
        _client = None
        _client_key = None


def call_options() -> dict:
    from google.api_core.retry import Retry, if_transient_error

    return {"retry": Retry(predicate=if_transient_error, deadline=10.0), "timeout": 10.0}
