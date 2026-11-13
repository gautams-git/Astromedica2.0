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


def resolve_mode() -> str:
    explicit = os.getenv("BENCHMARK_RASHI_WEIGHT_MODE", "").strip().lower()
    if explicit in {"live", "mock", "off"}:
        return explicit
    if os.getenv("GEMINI_API_KEY"):
        return "live"
    # Local demonstration when no model key is configured. Reports label this
    # as a scripted selector, not as a live model.
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
        record["raw_proposal"] = proposal
        record["validation"] = validate_proposal(proposal, baseline)
        return record

    record["live_model"] = True
    try:
        from agentic_rashi_correlation.config import LLMConfig, get_llm
        config = LLMConfig()
        record["model_identifier"] = config.model
        record["generation"]["temperature"] = config.temperature
        llm = get_llm(config)
        if not config.is_live:
            record["live_model"] = False
            record["validation"] = {
                "valid": False,
                "errors": ["live mode was requested but no model client is configured"],
                "weights": None,
                "changed": [],
                "justification": "",
            }
            return record
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
        response = llm.invoke(
            "Return JSON only.\n" + json.dumps(prompt, default=str)
        )
        text = response.content if hasattr(response, "content") else str(response)
        if isinstance(text, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
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
