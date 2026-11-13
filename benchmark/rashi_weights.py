"""Constrained weight proposal for Rashi Correlation only.

The proposal is applied through agentic_rashi_correlation.profiles.weights_context.
It cannot add keys, change formulas, or leave the 50%–150% band around the
frozen clinical baseline. Invalid proposals are rejected and not executed.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

from agentic_rashi_correlation.profiles import WEIGHT_TABLES, _BASE

MODEL_MOCK = "benchmark-mock-rashi-weights-v1"
PROMPT_VERSION = "rashi-weight-proposal-v1"


def baseline_tables() -> dict[str, dict[str, float]]:
    return {name: {key: float(value) for key, value in _BASE[name].items()} for name in WEIGHT_TABLES}


def _vertex_ready() -> bool:
    """True when Application Default Credentials can reach Vertex AI.

    This deployment does not use GEMINI_API_KEY. Gemini is called with the
    same Google login Firestore uses (gemini_client.py).
    """
    try:
        import google.auth
        from google.auth.exceptions import GoogleAuthError

        google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return True
    except (GoogleAuthError, ImportError, OSError):
        return False


def resolve_mode() -> str:
    explicit = os.getenv("BENCHMARK_RASHI_WEIGHT_MODE", "").strip().lower()
    if explicit in {"live", "mock", "off"}:
        return explicit
    if os.getenv("GEMINI_API_KEY") or _vertex_ready():
        return "live"
    # No Vertex login and no key. Reports label this as a scripted selector.
    return "mock"


def _extract_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model response did not contain a JSON object")
    payload = json.loads(text[start:end + 1])
    if not isinstance(payload, dict):
        raise ValueError("model response JSON was not an object")
    return payload


TABLE_ALIASES = {
    "priority_weights": "PRIORITY_WEIGHTS",
    "layer_weights": "LAYER_WEIGHTS",
    "anchor_weights": "ANCHOR_WEIGHTS",
    "PRIORITY_WEIGHTS": "PRIORITY_WEIGHTS",
    "LAYER_WEIGHTS": "LAYER_WEIGHTS",
    "ANCHOR_WEIGHTS": "ANCHOR_WEIGHTS",
}


def validate_proposal(proposal: dict | None, baseline: dict[str, dict[str, float]] | None = None) -> dict[str, Any]:
    baseline = baseline or baseline_tables()
    errors = []
    merged = {name: dict(table) for name, table in baseline.items()}
    proposal = proposal or {}
    allowed = set(WEIGHT_TABLES)
    for raw_name, updates in proposal.items():
        if raw_name in {"justification", "model"}:
            continue
        table_name = TABLE_ALIASES.get(raw_name)
        if table_name not in allowed:
            errors.append(f"unexpected parameter table: {raw_name}")
            continue
        if not isinstance(updates, dict):
            errors.append(f"{table_name} must be an object of weights")
            continue
        for key, value in updates.items():
            if key not in baseline[table_name]:
                errors.append(f"unexpected parameter {table_name}.{key}")
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors.append(f"invalid value {table_name}.{key}")
                continue
            if not math.isfinite(number) or number <= 0:
                errors.append(f"invalid value {table_name}.{key}={value}")
                continue
            low = baseline[table_name][key] * 0.5
            high = baseline[table_name][key] * 1.5
            if number < low - 1e-9 or number > high + 1e-9:
                errors.append(f"out of range {table_name}.{key}={number} (allowed {low:.4f}..{high:.4f})")
                continue
            merged[table_name][key] = number

    for name in WEIGHT_TABLES:
        base_sum = sum(baseline[name].values()) or 1.0
        new_sum = sum(merged[name].values())
        if not (0.5 * base_sum <= new_sum <= 1.5 * base_sum):
            errors.append(f"{name} sum {new_sum:.4f} is outside 50%–150% of the baseline sum")

    changed = []
    for name in WEIGHT_TABLES:
        for key, value in merged[name].items():
            if abs(value - baseline[name][key]) > 1e-9:
                changed.append({
                    "table": name,
                    "key": key,
                    "baseline": baseline[name][key],
                    "proposed": value,
                })
    return {
        "valid": not errors,
        "errors": errors,
        "weights": merged if not errors else None,
        "changed": changed,
        "justification": str(proposal.get("justification") or ""),
    }


def mock_proposal() -> dict[str, Any]:
    base = baseline_tables()
    return {
        "priority_weights": {},
        "layer_weights": {"nakshatra": round(base["LAYER_WEIGHTS"]["nakshatra"] * 1.2, 4)},
        "anchor_weights": {"P6": round(base["ANCHOR_WEIGHTS"]["P6"] * 1.2, 4)},
        "justification": (
            "Scripted selector raises the nakshatra layer and the P6 anchor by 20%, "
            "inside the published 50%–150% band. This is not a live model response."
        ),
    }


def propose_weights(chart_summary: dict | None = None) -> dict[str, Any]:
    """Return a validated proposal. Invalid or unavailable proposals are not weights to execute."""
    mode = resolve_mode()
    baseline = baseline_tables()
    record = {
        "mode": mode,
        "prompt_version": PROMPT_VERSION,
        "model_identifier": None,
        "transport": None,
        "live_model": False,
        "generation": {"temperature": 0.0, "seed": None, "deterministic_mock": mode == "mock"},
        "baseline": baseline,
        "raw_proposal": None,
        "validation": None,
    }
    if mode == "off":
        record["validation"] = validate_proposal({}, baseline)
        record["validation"]["note"] = "Weight proposal is switched off. Clinical weights stay in place."
        record["model_identifier"] = "none"
        return record

    if mode == "mock":
        proposal = mock_proposal()
        record["model_identifier"] = MODEL_MOCK
        record["transport"] = "scripted"
        record["raw_proposal"] = proposal
        record["validation"] = validate_proposal(proposal, baseline)
        return record

    record["live_model"] = True
    try:
        prompt = {
            "task": "Propose only weight overrides for the existing Rashi Correlation tables.",
            "defaults": baseline,
            "constraints": "Each supplied value must stay between 50% and 150% of its default. Do not add keys. Do not change formulas.",
            "chart_summary": chart_summary or {},
            "response_schema": {
                "priority_weights": {},
                "layer_weights": {},
                "anchor_weights": {},
                "justification": "short text",
            },
        }
        prompt_text = "Return JSON only.\n" + json.dumps(prompt, default=str)
        if os.getenv("GEMINI_API_KEY"):
            text, model_name, temperature = _call_api_key_model(prompt_text)
            record["transport"] = "gemini_api_key"
        else:
            text, model_name, temperature = _call_vertex(prompt_text)
            record["transport"] = "vertex_adc"
        record["model_identifier"] = model_name
        record["generation"]["temperature"] = temperature
        proposal = _extract_json(str(text))
        record["raw_proposal"] = proposal
        record["validation"] = validate_proposal(proposal, baseline)
    except Exception as exc:
        record["validation"] = {
            "valid": False,
            "errors": [f"{type(exc).__name__}: {exc}"],
            "weights": None,
            "changed": [],
            "justification": "",
        }
    return record


def _call_api_key_model(prompt_text: str) -> tuple[str, str, float]:
    from agentic_rashi_correlation.config import LLMConfig, get_llm

    config = LLMConfig()
    llm = get_llm(config)
    if not config.is_live:
        raise RuntimeError("live mode was requested but no model client is configured")
    response = llm.invoke(prompt_text)
    text = response.content if hasattr(response, "content") else str(response)
    if isinstance(text, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
    return str(text), config.model, config.temperature


def _call_vertex(prompt_text: str) -> tuple[str, str, float]:
    """Ask Gemini on Vertex AI with Application Default Credentials. No API key."""
    import httpx

    import gemini_client
    from config import GEMINI_MODEL

    url = gemini_client.generate_url(GEMINI_MODEL)
    headers = {
        "Authorization": f"Bearer {gemini_client._fresh_token()}",
        "x-goog-user-project": gemini_client.project(),
    }
    generation = {
        "temperature": 0.0,
        "maxOutputTokens": 1024,
        "responseMimeType": "application/json",
    }
    if "2.5" in GEMINI_MODEL:
        generation["thinkingConfig"] = {"thinkingBudget": 256}
    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "Return one JSON object of weight overrides for the existing Rashi tables. "
                    "Use only keys that already exist. Keep every value between 50% and 150% of its default."
                ),
            }],
        },
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": generation,
    }
    response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
    if response.status_code in (401, 403):
        raise RuntimeError(gemini_client.auth_failure_message(response.status_code, response.text))
    if response.status_code != 200:
        raise RuntimeError(f"Vertex AI error {response.status_code}: {response.text[:300]}")
    data = response.json()
    candidate = (data.get("candidates") or [{}])[0]
    parts = ((candidate.get("content") or {}).get("parts") or [])
    texts = [part.get("text", "") for part in parts if part.get("text") and not part.get("thought")]
    if not texts:
        raise RuntimeError("Vertex AI returned no JSON text")
    return "".join(texts), f"vertex:{GEMINI_MODEL}", 0.0
