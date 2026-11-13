"""Clerk session-token verification.

The Clerk SDK authenticates the request (signature, expiry, not-before,
audience, and authorized party). This module then requires the issuer and
subject claims so the application identity is taken only from a verified token.
"""

from __future__ import annotations

from dataclasses import dataclass

from clerk_backend_api.security import AuthenticateRequestOptions, authenticate_request

from app_core.errors import AuthError, ServiceUnavailableError
from app_core.settings import Settings


@dataclass(frozen=True)
class Principal:
    clerk_user_id: str
    session_id: str
    authorized_party: str


class ClerkAuthenticator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def authenticate(self, flask_request) -> Principal:
        if not self.settings.auth_configured:
            raise ServiceUnavailableError("authentication_not_configured")

        options = AuthenticateRequestOptions(
            secret_key=self.settings.clerk_secret_key or None,
            jwt_key=self.settings.clerk_jwt_key or None,
            audience=self.settings.clerk_audience or None,
            authorized_parties=list(self.settings.authorized_parties),
            accepts_token=["session_token"],
        )
        state = authenticate_request(flask_request, options)
        if not state.is_signed_in or not state.payload:
            raise AuthError("unauthenticated")

        payload = state.payload
        issuer = payload.get("iss")
        subject = payload.get("sub")
        if not issuer or issuer != self.settings.clerk_issuer:
            raise AuthError("unauthenticated")
        if not isinstance(subject, str) or not subject.strip():
            raise AuthError("unauthenticated")
        session_id = payload.get("sid") or ""
        azp = payload.get("azp") or ""
        return Principal(clerk_user_id=subject.strip(), session_id=str(session_id), authorized_party=str(azp))
