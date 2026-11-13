"""Environment-backed settings. Secrets come only from the environment."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def issuer_from_publishable_key(publishable_key: str) -> str:
    """Clerk publishable keys embed the Frontend API host used as the token issuer."""
    if not publishable_key:
        return ""
    parts = publishable_key.split("_")
    if len(parts) < 3:
        return ""
    encoded = parts[-1]
    padded = encoded + ("=" * (-len(encoded) % 4))
    try:
        decoded = base64.b64decode(padded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""
    host = decoded.strip().rstrip("$")
    if not host or " " in host or "/" in host:
        return ""
    return f"https://{host}"


@dataclass(frozen=True)
class Settings:
    app_env: str
    frontend_url: str
    api_url: str
    allowed_origins: list[str]
    authorized_parties: list[str]
    clerk_publishable_key: str
    clerk_secret_key: str
    clerk_jwt_key: str
    clerk_audience: str
    clerk_issuer: str
    firestore_project_id: str
    firestore_emulator_host: str
    prokerala_client_id: str
    prokerala_client_secret: str
    gemini_api_key: str
    gemini_model: str
    trust_cloudflare: bool
    trust_proxy: bool
    max_body_bytes: int
    port: int

    @property
    def is_test(self) -> bool:
        return self.app_env == "test"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def auth_configured(self) -> bool:
        has_verifier = bool(self.clerk_jwt_key or self.clerk_secret_key)
        return has_verifier and bool(self.clerk_issuer) and bool(self.authorized_parties)

    @property
    def using_firestore_emulator(self) -> bool:
        return bool(self.firestore_emulator_host)


def load_settings() -> Settings:
    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    frontend_url = os.environ.get("FRONTEND_URL", "http://localhost:8000").rstrip("/")
    api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
    origins = _csv(os.environ.get("ALLOWED_ORIGINS", frontend_url))
    parties = _csv(os.environ.get("CLERK_AUTHORIZED_PARTIES", frontend_url))
    publishable = os.environ.get("CLERK_PUBLISHABLE_KEY", "").strip()
    issuer = os.environ.get("CLERK_ISSUER", "").strip() or issuer_from_publishable_key(publishable)
    audience = os.environ.get("CLERK_AUDIENCE", "").strip()
    return Settings(
        app_env=app_env,
        frontend_url=frontend_url,
        api_url=api_url,
        allowed_origins=origins,
        authorized_parties=parties,
        clerk_publishable_key=publishable,
        clerk_secret_key=os.environ.get("CLERK_SECRET_KEY", "").strip(),
        clerk_jwt_key=os.environ.get("CLERK_JWT_KEY", "").strip(),
        clerk_audience=audience,
        clerk_issuer=issuer,
        firestore_project_id=os.environ.get("FIRESTORE_PROJECT_ID", "astromedica-local").strip(),
        firestore_emulator_host=os.environ.get("FIRESTORE_EMULATOR_HOST", "").strip(),
        prokerala_client_id=os.environ.get("PROKERALA_CLIENT_ID", "").strip(),
        prokerala_client_secret=os.environ.get("PROKERALA_CLIENT_SECRET", "").strip(),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        gemini_model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        trust_cloudflare=os.environ.get("TRUST_CLOUDFLARE", "false").strip().lower() in {"1", "true", "yes"},
        trust_proxy=os.environ.get("TRUST_PROXY", "false").strip().lower() in {"1", "true", "yes"},
        max_body_bytes=int(os.environ.get("MAX_BODY_BYTES", "262144")),
        port=int(os.environ.get("PORT", "8000")),
    )
