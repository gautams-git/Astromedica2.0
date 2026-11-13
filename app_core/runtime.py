"""Process-wide service graph."""

from __future__ import annotations

from app_core.auth import ClerkAuthenticator
from app_core.persistence import ChartPersistence
from app_core.ratelimit import RateLimiter
from app_core.repository import build_repository
from app_core.settings import Settings, load_settings

_services = None


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.authenticator = ClerkAuthenticator(settings)
        self.repository = build_repository(settings)
        self.persistence = ChartPersistence(self.repository)
        self.limiter = RateLimiter()
        self.session_owners: dict[str, dict] = {}

    def ready(self) -> tuple[bool, dict]:
        auth_ok = self.settings.auth_configured
        if self.settings.is_test:
            store_ok = self.repository.ping()
        elif self.settings.is_production:
            store_ok = self.repository.ping()
        else:
            store_ok = bool(self.settings.firestore_emulator_host) and self.repository.ping()
        return auth_ok and store_ok, {
            "authentication": "ok" if auth_ok else "not_configured",
            "datastore": "ok" if store_ok else "not_ready",
        }


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services(load_settings())
    return _services


def reset_services(settings: Settings | None = None) -> Services:
    global _services
    _services = Services(settings or load_settings())
    return _services
