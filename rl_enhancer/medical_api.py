"""
Fetch clinical knowledge (symptom lists) from free public APIs.

Sources:
  1. MedlinePlus Web Service (NIH)
  2. Wikipedia REST API

Returns plain symptom phrases, cleaned and deduplicated, ready to merge
into the symptom library.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger("medical_api")

CACHE_DIR = Path(__file__).resolve().parent / "data" / "api_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = 60 * 60 * 24 * 14  # two weeks

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
WIKIPEDIA_URL = "https://en.wikipedia.org/api/rest_v1/page/summary"

USER_AGENT = "MedicalEnhancer/1.0 (research)"
REQUEST_TIMEOUT = 12.0
RETRIES = 2
RATE_LIMIT_SECONDS = 0.4

_last_request_at = 0.0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return safe[:80] or "empty"


def _cache_path(text: str) -> Path:
    return CACHE_DIR / f"{_cache_key(text)}.json"


def _read_cache(text: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(text)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - data.get("_cached_at", 0) > CACHE_TTL_SECONDS:
            return None
        return data.get("payload")
    except Exception:
        return None


def _write_cache(text: str, payload: Dict[str, Any]) -> None:
    try:
        _cache_path(text).write_text(
            json.dumps({"_cached_at": time.time(), "payload": payload},
                       indent=2),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Cache write failed for %s: %s", text, e)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _http_get(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    global _last_request_at
    if requests is None:
        return None

    elapsed = time.time() - _last_request_at
    wait = RATE_LIMIT_SECONDS - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.time()

    headers = {"User-Agent": USER_AGENT}
    for attempt in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers,
                             timeout=REQUEST_TIMEOUT)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:
            logger.debug("HTTP attempt %d failed for %s: %s",
                         attempt + 1, url, e)
            time.sleep(0.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_REF_RE = re.compile(r"\[\d+\]")
_WHITESPACE_RE = re.compile(r"\s+")

_SYMPTOM_HINTS = (
    "pain", "ache", "fever", "cough", "fatigue", "nausea", "vomiting",
    "shortness of breath", "dyspnea", "dizziness", "swelling", "edema",
    "rash", "headache", "bleeding", "weakness", "weight loss", "chills",
    "sweating", "palpitations", "diarrhea", "constipation", "abdominal",
    "chest", "joint", "muscle", "vision", "confusion", "numbness",
    "tingling", "itching", "jaundice", "urination", "thirst", "polyuria",
    "polydipsia", "seizure", "tremor", "syncope", "hematuria",
    "proteinuria", "pallor", "tachycardia", "hypotension", "hypertension",
    "cramp", "stiffness", "bloating", "reflux", "wheeze", "dysarthria",
    "hemoptysis", "hemiplegia", "paresthesia", "tinnitus",
)


def _clean_sentence(s: str) -> str:
    s = _REF_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    s = s.strip(" ,;:.-")
    if not s:
        return ""
    if len(s) < 15 or len(s) > 220:
        return ""
    return s[0].upper() + s[1:]


def _looks_like_symptom(s: str) -> bool:
    low = s.lower()
    if not low:
        return False
    if len(low.split()) > 25:
        return False
    if low.endswith("."):
        low = low[:-1]
    return any(hint in low for hint in _SYMPTOM_HINTS)


def _extract_symptoms_from_text(text: str) -> List[str]:
    if not text:
        return []
    cleaned: List[str] = []
    seen: Set[str] = set()
    for raw in _SENTENCE_SPLIT.split(text):
        c = _clean_sentence(raw)
        if not c or not _looks_like_symptom(c):
            continue
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(c)
        if len(cleaned) >= 8:
            break
    return cleaned


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def _fetch_medlineplus(disease: str) -> List[str]:
    cached = _read_cache(f"mlp::{disease}")
    if cached is not None:
        return list(cached.get("symptoms") or [])

    r = _http_get(MEDLINEPLUS_URL, params={
        "db": "healthTopics", "term": disease, "retmax": 5, "rettype": "brief",
    })
    if r is None:
        _write_cache(f"mlp::{disease}", {"symptoms": []})
        return []

    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.text)
    except Exception as e:
        logger.debug("MedlinePlus parse error for %s: %s", disease, e)
        _write_cache(f"mlp::{disease}", {"symptoms": []})
        return []

    snippets: List[str] = []
    for doc in root.findall(".//document")[:3]:
        for content in doc.iter("content"):
            t = (content.text or "").strip()
            if t:
                snippets.append(t)

    symptoms = _extract_symptoms_from_text(" ".join(snippets))
    _write_cache(f"mlp::{disease}", {"symptoms": symptoms})
    return symptoms


def _fetch_wikipedia(disease: str) -> List[str]:
    cached = _read_cache(f"wiki::{disease}")
    if cached is not None:
        return list(cached.get("symptoms") or [])

    title = urllib.parse.quote(disease.replace(" ", "_"))
    r = _http_get(f"{WIKIPEDIA_URL}/{title}")
    if r is None:
        _write_cache(f"wiki::{disease}", {"symptoms": []})
        return []

    try:
        extract = r.json().get("extract") or ""
    except Exception:
        _write_cache(f"wiki::{disease}", {"symptoms": []})
        return []

    symptoms = _extract_symptoms_from_text(extract)
    _write_cache(f"wiki::{disease}", {"symptoms": symptoms})
    return symptoms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_symptoms_for_disease(disease: str) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for src in (_fetch_medlineplus, _fetch_wikipedia):
        try:
            items = src(disease)
        except Exception as e:
            logger.debug("Source %s failed for %s: %s",
                         src.__name__, disease, e)
            continue
        for s in items:
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out