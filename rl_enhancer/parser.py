"""Accept the wrapper-object input shape."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

MAX_ORGANS = 20
MAX_DISEASES_PER_ORGAN = 20


class ParseError(ValueError):
    pass


def _parse_conditions_list(data: Any) -> List[Tuple[str, List[str]]]:
    if not isinstance(data, list):
        raise ParseError(
            "conditions must be a JSON array of {Organ: [Diseases]} objects."
        )
    if not data:
        raise ParseError("conditions array is empty.")
    if len(data) > MAX_ORGANS:
        raise ParseError(f"Too many organs ({len(data)}). Maximum {MAX_ORGANS}.")

    result: List[Tuple[str, List[str]]] = []
    seen_organs: set[str] = set()

    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ParseError(f"conditions[{i}]: expected an object.")
        if len(entry) != 1:
            raise ParseError(
                f"conditions[{i}]: must have exactly one organ key."
            )

        organ, diseases = next(iter(entry.items()))
        organ = str(organ).strip()
        if not organ:
            raise ParseError(f"conditions[{i}]: organ name is empty.")
        if not isinstance(diseases, list):
            raise ParseError(
                f"conditions[{i}] ('{organ}'): value must be an array."
            )
        if not diseases:
            raise ParseError(f"conditions[{i}] ('{organ}'): array is empty.")
        if len(diseases) > MAX_DISEASES_PER_ORGAN:
            raise ParseError(
                f"conditions[{i}] ('{organ}'): too many diseases."
            )

        clean: List[str] = []
        seen_d: set[str] = set()
        for j, d in enumerate(diseases):
            if not isinstance(d, str):
                raise ParseError(
                    f"conditions[{i}] ('{organ}') disease {j}: "
                    f"must be a string."
                )
            ds = d.strip()
            if not ds:
                raise ParseError(
                    f"conditions[{i}] ('{organ}') disease {j}: empty."
                )
            key = ds.lower()
            if key in seen_d:
                continue
            seen_d.add(key)
            clean.append(ds)

        if not clean:
            raise ParseError(
                f"conditions[{i}] ('{organ}'): no valid disease names."
            )

        organ_key = organ.lower()
        if organ_key in seen_organs:
            for idx, (o, existing) in enumerate(result):
                if o.lower() == organ_key:
                    existing_lower = {x.lower() for x in existing}
                    result[idx] = (
                        o,
                        existing + [d for d in clean
                                    if d.lower() not in existing_lower],
                    )
                    break
        else:
            seen_organs.add(organ_key)
            result.append((organ, clean))

    return result


def to_preview(parsed: List[Tuple[str, List[str]]]) -> List[dict]:
    return [{"organ": o, "diseases": d} for o, d in parsed]

# parser.py — REPLACE parse_input with this version

def parse_input(raw: bytes | str) -> Tuple[List[Tuple[str, List[str]]],
                                            Dict[str, Any]]:
    """Return (parsed_conditions, wrapper_metadata)."""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ParseError(f"Payload is not valid UTF-8: {e}")
    else:
        text = raw

    text = text.strip()
    if not text:
        raise ParseError("Payload is empty.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ParseError(
            f"Invalid JSON at line {e.lineno} column {e.colno}: {e.msg}"
        )

    if not isinstance(data, dict):
        raise ParseError(
            "Top level must be a JSON object containing a 'conditions' key."
        )

    if "conditions" not in data:
        raise ParseError('Top-level object must contain a "conditions" key.')

    parsed = _parse_conditions_list(data["conditions"])
    wrapper: Dict[str, Any] = {
        "patient_id":   data.get("patient_id"),
        "patient_name": data.get("patient_name"),
        "age":          data.get("age"),
        "sex":          data.get("sex"),
        # Inline chat_log takes priority over DB lookup in main.py
        "chat_log":     data.get("chat_log"),
    }
    return parsed, wrapper