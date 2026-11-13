"""Safe client errors. Details stay in server logs."""

from __future__ import annotations


class AuthError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class AuthorizationError(Exception):
    """Resource is missing or not visible to the caller. Respond as not found."""


class RateLimitError(Exception):
    def __init__(self, retry_after: int = 60):
        super().__init__("rate_limited")
        self.retry_after = retry_after


class RequestValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UpstreamError(Exception):
    def __init__(self, operation: str):
        super().__init__(operation)
        self.operation = operation


class ServiceUnavailableError(Exception):
    def __init__(self, check: str):
        super().__init__(check)
        self.check = check
