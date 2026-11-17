"""Encode values so Firestore can store them without dropping fields."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from app.domain.schema import PART_BYTES, SOFT_DOCUMENT_BYTES


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"unsupported value type {type(value).__name__}")


def _unsafe(value: Any) -> bool:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, (tuple, set)):
        return True
    if isinstance(value, list):
        return any(isinstance(item, (list, tuple)) or _unsafe(item) for item in value)
    if isinstance(value, dict):
        return any((not isinstance(key, str)) or _unsafe(item) for key, item in value.items())
    return False


def encode_result(value: Any) -> dict:
    """Return fields that preserve `value` exactly.

    Native maps are stored directly when Firestore accepts them. Everything
    else is canonical JSON. Callers split the JSON when it is too large.
    """
    if value is None:
        return {"result_encoding": "native", "result": None}
    if not _unsafe(value):
        encoded = canonical_json(value)
        if len(encoded.encode("utf-8")) <= SOFT_DOCUMENT_BYTES:
            return {"result_encoding": "native", "result": json.loads(encoded)}
    text = canonical_json(value)
    return {"result_encoding": "json", "result_json": text, "result": None}


def decode_result(document: dict, parts: list[str] | None = None) -> Any:
    encoding = document.get("result_encoding") or "native"
    if encoding == "split":
        text = "".join(parts or [])
        return json.loads(text) if text else None
    if encoding == "json":
        return json.loads(document.get("result_json") or "null")
    return document.get("result")


def split_text(text: str, part_bytes: int = PART_BYTES) -> list[str]:
    raw = text.encode("utf-8")
    if len(raw) <= part_bytes:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(raw):
        end = min(start + part_bytes, len(raw))
        while end > start and (raw[end - 1] & 0xC0) == 0x80:
            end -= 1
        if end == start:
            end = min(start + part_bytes, len(raw))
        parts.append(raw[start:end].decode("utf-8"))
        start = end
    return parts


def byte_length(document: dict) -> int:
    return len(canonical_json(document).encode("utf-8"))
