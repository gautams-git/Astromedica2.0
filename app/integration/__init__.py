"""Hooks the existing Flask flow into the new repositories.

Failures are logged. Classic chart generation still returns when the legacy
save succeeded, and the error is visible in the log. Strict knowledge checks
are separate and do refuse calculation.
"""

from __future__ import annotations

import logging

from app.core.clock import utc_now
from app.infrastructure.firestore.codec import canonical_json, sha256_text

_LOG = logging.getLogger("astromedica.mirror")


def _plane(repository):
    return getattr(repository, "data_plane", None)


def mirror_generation(repository, internal_user_id: str, patient_id: str, chart_input: dict, source: str, response: dict, input_id: str) -> None:
    plane = _plane(repository)
    if plane is None:
        return
    try:
        identity = {
            "name": chart_input.get("name"),
            "date_of_birth": chart_input.get("dob"),
            "birth_time": chart_input.get("birth_time"),
            "birth_place": chart_input.get("birth_place"),
            "latitude": chart_input.get("lat"),
            "longitude": chart_input.get("lng"),
            "gender": chart_input.get("gender"),
            "father_name": chart_input.get("father_name"),
            "mother_name": chart_input.get("mother_name"),
        }
        plane.save_identity(internal_user_id, patient_id, identity)
        raw = response.get("prokerala_result") or response.get("prokerala_results")
        if raw is not None:
            plane.save_source(
                internal_user_id,
                patient_id,
                {
                    "source_provider": "prokerala",
                    "source_version": source,
                    "retrieved_at": utc_now(),
                    "raw_source": raw,
                    "prokerala_chart": raw.get("raw_chart") if isinstance(raw, dict) else raw,
                    "prokerala_dasha": raw.get("raw_dasha") if isinstance(raw, dict) else None,
                    "input_hash": sha256_text(canonical_json(raw)),
                },
            )
        summary = {}
        combined = response.get("combined_algo")
        if isinstance(combined, dict):
            summary = {"top_organs": combined.get("top_organs") or combined.get("top3") or []}
        plane.save_run(
            internal_user_id,
            patient_id,
            input_id,
            algorithm_version=str(response.get("algo_version") or "legacy-rules-1"),
            status="complete",
            input_reference=input_id,
            input_hash=sha256_text(canonical_json(chart_input)),
            combined_result=combined,
            result_summary=summary,
            execution_metadata={"source": source},
        )
        skip = {
            "input",
            "input_id",
            "raw_response",
            "prokerala_result",
            "prokerala_results",
            "planetary_positions",
            "combined_algo",
        }
        for name, value in response.items():
            if name in skip or not isinstance(value, (dict, list)):
                continue
            plane.save_algorithm_output(
                internal_user_id,
                patient_id,
                input_id,
                str(name),
                value,
                algorithm_version=str(response.get("algo_version") or "legacy-rules-1"),
            )
        if summary.get("top_organs"):
            plane.upsert_patient(internal_user_id, patient_id, {"top_organs": summary["top_organs"], "latest_run_id": input_id})
        plane.append_audit(internal_user_id, patient_id, "algorithm_run", {"run_id": input_id, "source": source})
    except Exception as exc:
        _LOG.error("generation mirror failed patient=%s error=%s", patient_id, type(exc).__name__)
        raise


def mirror_chart_source(repository, internal_user_id: str, patient_id: str, raw_chart: dict, raw_dasha: dict, profile: dict) -> None:
    plane = _plane(repository)
    if plane is None:
        return
    plane.save_source(
        internal_user_id,
        patient_id,
        {
            "prokerala_chart": raw_chart,
            "prokerala_dasha": raw_dasha,
            "raw_source": {"chart": raw_chart, "dasha": raw_dasha, "profile_keys": sorted(profile or {})},
            "source_provider": "prokerala",
            "source_version": "chart_source",
        },
    )


def mirror_identity(repository, internal_user_id: str, patient_id: str, record: dict) -> None:
    plane = _plane(repository)
    if plane is None or not internal_user_id:
        return
    plane.save_identity(
        internal_user_id,
        patient_id,
        {
            "name": record.get("name"),
            "date_of_birth": record.get("dob"),
            "birth_place": record.get("birth_place"),
            "father_name": record.get("father_name"),
            "mother_name": record.get("mother_name"),
        },
    )


def mirror_turn(repository, internal_user_id: str, patient_id: str, chat_id: str, turn: dict) -> None:
    plane = _plane(repository)
    if plane is None or not internal_user_id:
        return
    plane.append_turn(internal_user_id, patient_id, chat_id, turn)


def mirror_rl_output(repository, internal_user_id: str, patient_id: str, chat_id: str, rl_run_id: str, output, model_version: str) -> None:
    plane = _plane(repository)
    if plane is None or not internal_user_id:
        return
    plane.save_rl_output(
        internal_user_id,
        patient_id,
        chat_id,
        rl_run_id,
        output,
        model_version=model_version,
        status="complete",
    )
    plane.append_audit(internal_user_id, patient_id, "rl_completed", {"rl_run_id": rl_run_id, "chat_id": chat_id})
