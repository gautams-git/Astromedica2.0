"""HTTP middleware: identity, headers, and safe errors."""

from __future__ import annotations

import time
import uuid
from functools import wraps

from flask import g, jsonify, request
from werkzeug.exceptions import HTTPException

from app_core.errors import (
    AuthError,
    AuthorizationError,
    RateLimitError,
    RequestValidationError,
    ServiceUnavailableError,
    UpstreamError,
)
from app_core.logging_config import log
from app_core.runtime import get_services
from app_core.validation import require_patient_id


def client_ip(settings) -> str:
    if settings.trust_cloudflare:
        cf_ip = request.headers.get("CF-Connecting-IP", "")
        if cf_ip:
            return cf_ip.split(",")[0].strip()
    if settings.trust_proxy:
        forwarded = request.headers.get("X-Forwarded-For", "")
        parts = [item.strip() for item in forwarded.split(",") if item.strip()]
        if parts:
            return parts[0]
    return request.remote_addr or ""


def register_http(app) -> None:
    settings = get_services().settings
    app.config["MAX_CONTENT_LENGTH"] = settings.max_body_bytes
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

    @app.before_request
    def begin_request():
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        g.started = time.perf_counter()

    @app.after_request
    def finish_request(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/static") or request.path in {"/", "/results", "/health"}:
            response.headers["Cache-Control"] = "no-store"
        duration_ms = int((time.perf_counter() - getattr(g, "started", time.perf_counter())) * 1000)
        user = getattr(g, "user", None) or {}
        log.info(
            "request",
            extra={
                "request_id": getattr(g, "request_id", ""),
                "route": request.path,
                "status": response.status_code,
                "duration_ms": duration_ms,
                "internal_user_id": user.get("internal_user_id", ""),
            },
        )
        return response

    @app.errorhandler(AuthError)
    def auth_error(_exc):
        return jsonify({"success": False, "error": "Authentication is required."}), 401

    @app.errorhandler(AuthorizationError)
    def hidden(_exc):
        return jsonify({"success": False, "error": "Not found."}), 404

    @app.errorhandler(RateLimitError)
    def limited(exc: RateLimitError):
        response = jsonify({"success": False, "error": "Too many requests."})
        response.status_code = 429
        response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.errorhandler(RequestValidationError)
    def invalid(exc: RequestValidationError):
        return jsonify({"success": False, "error": exc.message}), 400

    @app.errorhandler(UpstreamError)
    def upstream(_exc: UpstreamError):
        return jsonify({"success": False, "error": "The chart service is unavailable."}), 502

    @app.errorhandler(ServiceUnavailableError)
    def unavailable(_exc):
        return jsonify({"success": False, "error": "The service is not ready."}), 503

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"success": False, "error": "Request is too large."}), 413

    @app.errorhandler(400)
    def bad_request(_exc):
        return jsonify({"success": False, "error": "The request could not be understood."}), 400

    @app.errorhandler(Exception)
    def unexpected(_exc):
        if isinstance(_exc, HTTPException):
            return _exc
        log.error(
            "unhandled_error",
            extra={"request_id": getattr(g, "request_id", ""), "route": request.path, "error_type": type(_exc).__name__},
        )
        return jsonify({"success": False, "error": "The request could not be completed."}), 500


def require_auth(fn=None, *, rate_limit: str | None = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            services = get_services()
            principal = services.authenticator.authenticate(request)
            user = services.repository.get_or_create_user(principal.clerk_user_id)
            g.principal = principal
            g.user = user
            if rate_limit:
                services.limiter.check(user["internal_user_id"], rate_limit)
            return func(*args, **kwargs)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def enforce_patient_access(patient_id: str):
    require_patient_id(patient_id)
    services = get_services()
    if not services.repository.owns_patient(g.user["internal_user_id"], patient_id):
        services.repository.save(
            g.user["internal_user_id"],
            "audit_events",
            uuid.uuid4().hex,
            {"record_type": "audit", "action": "patient_access_denied", "source": "api"},
        )
        raise AuthorizationError()


def safe_failure():
    return jsonify({"success": False, "error": "The request could not be completed."}), 500
