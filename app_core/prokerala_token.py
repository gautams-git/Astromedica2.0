"""Process-local Prokerala OAuth token cache.

A single Compute Engine VM does not need a shared cache. The lock prevents
concurrent refreshes inside one process. Each Gunicorn worker keeps its own
token. Tokens are refreshed five minutes before expiry and are never logged.
"""

from __future__ import annotations

import asyncio
import time

import httpx

from app_core.errors import UpstreamError

_TOKEN_URL = "https://api.prokerala.com/token"
_REFRESH_SKEW_SECONDS = 300


class ProkeralaTokenCache:
    def __init__(self):
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    def clear(self) -> None:
        self._token = None
        self._expires_at = 0.0

    async def get_token(self, client_id: str, client_secret: str) -> str:
        if not client_id or not client_secret:
            raise UpstreamError("prokerala_not_configured")
        async with self._lock:
            if self._token and time.time() < self._expires_at:
                return self._token
            last_status = None
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=15.0) as client:
                        response = await client.post(
                            _TOKEN_URL,
                            data={
                                "grant_type": "client_credentials",
                                "client_id": client_id,
                                "client_secret": client_secret,
                            },
                        )
                except httpx.HTTPError as exc:
                    last_status = type(exc).__name__
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                if response.status_code == 200:
                    payload = response.json()
                    self._token = payload["access_token"]
                    expires_in = int(payload.get("expires_in", 3600))
                    self._expires_at = time.time() + max(60, expires_in - _REFRESH_SKEW_SECONDS)
                    return self._token
                last_status = response.status_code
                if response.status_code in {429, 500, 502, 503, 504}:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                self.clear()
                raise UpstreamError("prokerala_auth")
            raise UpstreamError(f"prokerala_auth_{last_status}")


token_cache = ProkeralaTokenCache()


async def get_prokerala_token(client_id: str, client_secret: str) -> str:
    return await token_cache.get_token(client_id, client_secret)


async def prokerala_get(client: httpx.AsyncClient, url: str, token: str, params: dict) -> httpx.Response:
    """Retry only idempotent GETs that failed for a transient reason."""
    response = None
    for attempt in range(3):
        response = await client.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
        if response.status_code not in {429, 500, 502, 503, 504}:
            return response
        await asyncio.sleep(0.2 * (attempt + 1))
    return response