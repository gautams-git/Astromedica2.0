"""Pre-warm the RL hint for the next turn. Read-only on the patient folder."""
import concurrent.futures as cf
import threading
from typing import Any, Dict

from agentic_chat import config as cfg

_POOL = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rl-hint")
_FUTURES: Dict[str, cf.Future] = {}
_LOCK = threading.Lock()


def start_prewarm(session_id: str, session: Dict[str, Any],
                  patient_id: str) -> None:
    if not cfg.RL_HINT:
        return
    with _LOCK:
        _FUTURES[session_id] = _POOL.submit(_compute, session, patient_id)


def take_hint(session_id: str) -> Dict[str, Any]:
    with _LOCK:
        fut = _FUTURES.pop(session_id, None)
    if fut is None:
        return {}
    try:
        return fut.result(timeout=cfg.RL_HINT_WAIT)
    except Exception:
        return {}


def _compute(session: Dict[str, Any], patient_id: str) -> Dict[str, Any]:
    from rl_enhancer.chat_log_parser import parse_chat_log_list
    from rl_enhancer.enhancer import (
        canonical_organ, canonical_disease,
        symptoms_for_disease, _evidence_for,
    )
    from combine_output import _current_session_log

    log = _load_chat_log(patient_id)
    summary = parse_chat_log_list([_current_session_log(log)])

    out: Dict[str, Any] = {}
    for organ, diseases in _conditions_from_plan(session):
        for disease in diseases:
            d = canonical_disease(disease)
            o = canonical_organ(organ)
            endorsed, denied, _ = _evidence_for(d, o, summary)
            out[d] = {
                "endorsed": len(endorsed),
                "denied": len(denied),
                "library_symptoms": symptoms_for_disease(
                    d, organ=o, chat_summary=summary
                )[:6],
            }
    return out


def _load_chat_log(patient_id: str) -> Dict[str, Any]:
    import json, os
    path = f"patients/{patient_id}/chat_log.json"
    if not os.path.exists(path):
        return {"turns": []}
    with open(path) as f:
        return json.load(f)


def _conditions_from_plan(session: Dict[str, Any]):
    plan = session.get("chat_plan", {}) or {}
    for cluster in plan.get("clusters", []) or []:
        organ = cluster.get("organ", "")
        diseases = [b.get("label", "") for b in cluster.get("branches", [])]
        if organ and diseases:
            yield organ, diseases