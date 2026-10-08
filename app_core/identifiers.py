"""Stable document identifiers. Caller-supplied user ids are never used."""

from __future__ import annotations

import hashlib
import uuid


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "legacy-rules-1"


def new_internal_user_id() -> str:
    return uuid.uuid4().hex


def birth_key(name: str, dob: str, birth_time: str, lat: float, lng: float) -> str:
    raw = f"{name.lower().strip()}|{dob}|{birth_time}|{round(float(lat), 4)}|{round(float(lng), 4)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def input_id_for(internal_user_id: str, birth: str) -> str:
    return hashlib.sha256(f"{internal_user_id}|{birth}".encode("utf-8")).hexdigest()[:32]


def algorithm_result_id(input_id: str, algorithm_name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in algorithm_name)
    return f"{input_id}_{safe}"[:700]
