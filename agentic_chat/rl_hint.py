"""Pre-warm the RL hint for the next turn. Read-only on the patient folder."""
import concurrent.futures as cf
import json
import os
import threading
from typing import Any, Dict

from agentic_chat import config as cfg

_POOL = cf.ThreadPoolExecutor(max_workers=2, thread_name_prefix="rl-hint")
_FUTURES: Dict[str, cf.Future] = {}
_LOCK = threading.Lock()
_MAX_PENDING = 200      # hints of abandoned chats are dropped, oldest first


def hint_key(label) -> str:
    """The key a branch label has in the hint. Used on both sides (building the hint and
    reading it), so the two can never disagree about case or spaces."""
    return str(label or "").strip().lower()


def start_prewarm(session_id: str, session: Dict[str, Any],
                  patient_id: str) -> None:
    if not cfg.RL_HINT:
        return
    with _LOCK:
        _FUTURES.pop(session_id, None)              # re-insert: the newest goes last
        _FUTURES[session_id] = _POOL.submit(_compute, session, patient_id)
        while len(_FUTURES) > _MAX_PENDING:
            _FUTURES.pop(next(iter(_FUTURES)))


def take_hint(session_id: str) -> Dict[str, Any]:
    with _LOCK:
        fut = _FUTURES.pop(session_id, None)
    if fut is None:
        return {}
    try:
        return fut.result(timeout=cfg.RL_HINT_WAIT)
    except Exception:
        return {}


def drop_hint(session_id: str) -> None:
    """The chat is over: forget any hint still being prepared for it."""
    with _LOCK:
        _FUTURES.pop(session_id, None)


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
            try:
                d = canonical_disease(disease)
                o = canonical_organ(organ)
                endorsed, denied, _ = _evidence_for(d, o, summary)
                # keyed by the plan's own label, so the planner finds it again
                out[hint_key(disease)] = {
                    "endorsed": len(endorsed),
                    "denied": len(denied),
                    "library_symptoms": symptoms_for_disease(
                        d, organ=o, chat_summary=summary
                    )[:6],
                }
            except Exception:           # one disease failing must not lose the others
                continue
    return out


def _load_chat_log(patient_id: str) -> Dict[str, Any]:
    path = f"patients/{patient_id}/chat_log.json"
    if not os.path.exists(path):
        return {"turns": []}
    try:
        with open(path) as f:
            return json.load(f)
    except (ValueError, OSError):       # the chat may be writing the file right now
        return {"turns": []}


def _conditions_from_plan(session: Dict[str, Any]):
    plan = session.get("chat_plan", {}) or {}
    for cluster in plan.get("clusters", []) or []:
        organ = cluster.get("system") or cluster.get("organ", "")     # the plan names it "system"
        diseases = [b.get("label", "") for b in cluster.get("branches", [])]
        if organ and diseases:
            yield organ, diseases
