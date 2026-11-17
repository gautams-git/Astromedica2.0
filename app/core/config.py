"""Data-plane settings. Secrets stay in the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DataSettings:
    app_env: str
    project_id: str
    emulator_host: str
    kb_source: str
    kb_strict: bool
    kb_cache_ttl_seconds: float
    chat_mode: str
    rl_hint_mode: str
    chat_durable: bool
    rl_config_strict: bool
    identity_encryption_key: str
    identity_encryption_required: bool
    model_provider: str

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def firestore_enabled(self) -> bool:
        backend = os.environ.get("FIRESTORE_BACKEND", "").strip().lower()
        if self.emulator_host or backend == "firestore":
            return True
        return self.is_production


def load_data_settings() -> DataSettings:
    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    project = os.environ.get("FIRESTORE_PROJECT_ID", "").strip() or os.environ.get(
        "GOOGLE_CLOUD_PROJECT", ""
    ).strip()
    if not project:
        project = "astromedica-local" if app_env != "production" else ""
    chat_mode = os.environ.get("CHAT_MODE", "classic").strip().lower() or "classic"
    rl_hint = os.environ.get("RL_HINT_MODE", "off").strip().lower() or "off"
    provider = os.environ.get("MODEL_PROVIDER", "vertex").strip().lower() or "vertex"
    return DataSettings(
        app_env=app_env,
        project_id=project,
        emulator_host=os.environ.get("FIRESTORE_EMULATOR_HOST", "").strip(),
        kb_source=os.environ.get("KB_SOURCE", "fixtures").strip().lower() or "fixtures",
        kb_strict=_flag("KB_STRICT"),
        kb_cache_ttl_seconds=float(os.environ.get("KB_CACHE_TTL_SECONDS", "60")),
        chat_mode=chat_mode,
        rl_hint_mode=rl_hint,
        chat_durable=_flag("CHAT_DURABLE", "true"),
        rl_config_strict=_flag("RL_CONFIG_STRICT"),
        identity_encryption_key=os.environ.get("IDENTITY_ENCRYPTION_KEY", "").strip()
        or os.environ.get("DEIDENTIFICATION_SECRET", "").strip(),
        identity_encryption_required=_flag("IDENTITY_ENCRYPTION_REQUIRED"),
        model_provider=provider,
    )
