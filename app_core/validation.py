"""Request validation for chart and chat inputs."""

from __future__ import annotations

import re
from datetime import datetime

from app_core.errors import RequestValidationError

_PATIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_TIME = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def require_patient_id(patient_id: str) -> str:
    if not isinstance(patient_id, str) or not _PATIENT_ID.fullmatch(patient_id):
        raise RequestValidationError("patient_id is invalid")
    return patient_id


def require_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise RequestValidationError("session_id is invalid")
    return session_id


def parse_chart_input(body: dict) -> dict:
    if not isinstance(body, dict):
        raise RequestValidationError("A JSON object is required")

    name = str(body.get("name", "")).strip()
    dob = str(body.get("dob", "")).strip()
    birth_time = str(body.get("birth_time", "08:00")).strip()[:5]
    birth_place = str(body.get("birth_place", "")).strip()
    father_name = str(body.get("father_name", "")).strip()
    mother_name = str(body.get("mother_name", "")).strip()
    gender = str(body.get("gender", "Unknown")).strip() or "Unknown"

    if not name or len(name) > 120:
        raise RequestValidationError("name is required")
    try:
        datetime.strptime(dob, "%Y-%m-%d")
    except ValueError as exc:
        raise RequestValidationError("dob must be YYYY-MM-DD") from exc
    if not _TIME.fullmatch(birth_time):
        raise RequestValidationError("birth_time must be HH:MM")
    try:
        lat = float(body.get("lat"))
        lng = float(body.get("lng"))
    except (TypeError, ValueError) as exc:
        raise RequestValidationError("lat and lng are required") from exc
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180) or (lat == 0 and lng == 0):
        raise RequestValidationError("lat and lng are out of range")
    if gender not in {"Male", "Female", "Unknown"}:
        raise RequestValidationError("gender is invalid")
    if len(birth_place) > 200 or len(father_name) > 120 or len(mother_name) > 120:
        raise RequestValidationError("A text field is too long")

    return {
        "name": name,
        "dob": dob,
        "birth_time": birth_time,
        "lat": lat,
        "lng": lng,
        "birth_place": birth_place,
        "father_name": father_name,
        "mother_name": mother_name,
        "gender": gender,
    }


def parse_chat_message(message: str) -> str:
    text = (message or "").strip()
    if not text:
        raise RequestValidationError("message is required")
    if len(text) > 4000:
        raise RequestValidationError("message is too long")
    return text
