"""
gemini_client.py — how the app reaches Gemini: through Vertex AI (Google
Cloud's AI service), logged in with Application Default Credentials (ADC).

Ameya's organisation policy blocks API keys, so there is no GEMINI_API_KEY.
ADC is the same Google login Firestore uses:
  - on a PC:  `gcloud auth application-default login` (expires about daily)
  - on the VM: its service account (astromedica-backend-runner), never expires;
    it needs the "Vertex AI User" role.

Settings (.env, all optional):
  GEMINI_LOCATION  region for Gemini, default asia-south1 (Mumbai)
  GEMINI_PROJECT   Cloud project; default FIRESTORE_PROJECT_ID, then the ADC project

Used by server.py (chat), llm1_search.py and rag_pipeline.py (embeddings).
"""

from __future__ import annotations

import asyncio
import os
import threading

import google.auth
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
LOGIN_HINT = ("Google login for Gemini is not working - run "
              "`gcloud auth application-default login`, then restart the server")

_lock = threading.Lock()
_creds = None
_adc_project: str | None = None


class GeminiAuthError(RuntimeError):
    """ADC login missing or expired. The message says how to fix it."""


def location() -> str:
    return os.environ.get("GEMINI_LOCATION", "").strip() or "asia-south1"


def project() -> str:
    explicit = (os.environ.get("GEMINI_PROJECT", "").strip()
                or os.environ.get("FIRESTORE_PROJECT_ID", "").strip())
    if explicit:
        return explicit
    _credentials()
    return _adc_project or ""


def _credentials():
    global _creds, _adc_project
    with _lock:
        if _creds is None:
            try:
                _creds, _adc_project = google.auth.default(scopes=_SCOPES)
            except GoogleAuthError as e:
                raise GeminiAuthError(f"{LOGIN_HINT} ({e})") from e
        return _creds


def _fresh_token() -> str:
    creds = _credentials()
    with _lock:
        if not creds.valid:
            try:
                creds.refresh(Request())
            except GoogleAuthError as e:
                raise GeminiAuthError(f"{LOGIN_HINT} ({str(e)[:150]})") from e
        return creds.token


async def auth_headers() -> dict:
    # Refreshing the token is a blocking HTTP call (about once an hour).
    token = await asyncio.to_thread(_fresh_token)
    return {"Authorization": f"Bearer {token}", "x-goog-user-project": project()}


def _model_url(model: str, method: str) -> str:
    loc = location()
    host = "aiplatform.googleapis.com" if loc == "global" else f"{loc}-aiplatform.googleapis.com"
    return (f"https://{host}/v1/projects/{project()}/locations/{loc}"
            f"/publishers/google/models/{model}:{method}")


def generate_url(model: str) -> str:
    return _model_url(model, "generateContent")


def embed_url(model: str) -> str:
    return _model_url(model, "predict")


def auth_failure_message(status_code: int, body: str) -> str:
    """Console line for a 401/403 from Vertex AI."""
    if status_code == 401:
        return f"Gemini (Vertex AI) refused the login: {LOGIN_HINT}"
    if "aiplatform.googleapis.com" in body and ("disabled" in body or "not been used" in body):
        return (f"Vertex AI API is not enabled in project {project()} - run "
                f"`gcloud services enable aiplatform.googleapis.com --project {project()}`")
    return (f"This Google account has no permission for Gemini in project {project()} "
            f"(needs the 'Vertex AI User' role). Original error: {body[:200]}")


def print_startup_status(model: str) -> None:
    """One line at server start: is Gemini reachable with this login?"""
    try:
        _fresh_token()
        print(f"[Gemini] Vertex AI via ADC OK - project {project()}, region {location()}, model {model}")
    except GeminiAuthError as e:
        print(f"[Gemini] [FAIL] {e}")
