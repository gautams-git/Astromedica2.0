"""Thin readers of the deterministic inputs the agent pulls from."""
from typing import Any, Dict

from chat_module import (
    cluster_by_rank,
    branch_by_id,
    update_cluster_support,
    save_chat_turn,
    fallback_question_meta,
    should_finish_early,
    build_final_text,
)
from question_generator import _bad_question, _banned_names, _plan_input
from symptom_mapper import classify_answer
from term_to_systems import TERM_TO_SYSTEMS


def rules_label(message: str) -> str:
    return classify_answer(message)


def known_term(message: str) -> bool:
    lower = message.lower()
    return any(term.lower() in lower for term in TERM_TO_SYSTEMS.keys())


def apply_score(st, session: Dict[str, Any]) -> None:
    meta = session.get("last_question_meta", {}) or {}
    rank = meta.get("cluster_rank")
    if not rank:
        return
    score = (session.get("cluster_scores", {}) or {}).get(str(rank))
    if not score:
        return
    update_cluster_support(
        score,
        cluster_by_rank(session, rank),
        meta.get("branch_id", ""),
        st.label,
        st.user_message,
        meta.get("question_type", ""),
    )


def planner_meta(turn: int, session: Dict[str, Any]) -> Dict[str, Any]:
    return fallback_question_meta(turn, session) or {}


def should_finish(turn: int, session: Dict[str, Any]) -> bool:
    from agentic_chat import config as cfg
    if turn >= cfg.MAX_TURNS:
        return True
    return bool(should_finish_early({**session, "turn_count": turn}))


_QUESTION_KEY = {
    "screening": "screening_questions",
    "drilldown": "drilldown_questions",
    "severity": "severity_questions",
    "red_flag": "red_flag_questions",
}


def bank_question(session: Dict[str, Any], meta: Dict[str, Any]) -> str:
    cluster = cluster_by_rank(session, meta.get("cluster_rank"))
    branch = branch_by_id(cluster, meta.get("branch_id", "")) or {}
    key = _QUESTION_KEY.get(meta.get("question_type"))
    if not key:
        return ""
    pool = branch.get(key) or cluster.get(key) or []
    asked = {m["content"] for m in session["messages"] if m["role"] == "assistant"}
    for q in pool:
        if q not in asked:
            return q
    return pool[0] if pool else ""


def question_is_bad(text: str, session: Dict[str, Any]) -> str:
    banned = _banned_names(_plan_input(session["chat_plan"]))
    return _bad_question(text, banned)


def persist_turn(st, session: Dict[str, Any]) -> Dict[str, Any]:
    session["last_question_meta"] = {**st.meta, "question_text": st.reply}
    session["messages"].append({"role": "user", "content": st.user_message})
    save_chat_turn(
        st.patient_id, st.turn, "user", st.user_message,
        {
            "session_id": st.session_id,
            "answer_result": st.label,
            "rule_label": st.rule_label,
            "model_label": st.extracted.get("label"),
            "answered_cluster_rank": st.meta.get("cluster_rank"),
            "answered_branch_id": st.meta.get("branch_id"),
            "answered_question_type": st.meta.get("question_type", ""),
        },
    )
    session["messages"].append({"role": "assistant", "content": st.reply})
    session["turn_count"] += 1
    save_chat_turn(
        st.patient_id, session["turn_count"], "assistant", st.reply,
        {
            **session["last_question_meta"],
            "session_id": st.session_id,
            "cluster_scores": session.get("cluster_scores", {}),
            "trace": st.trace,
        },
    )
    if st.is_final:
        _finish_chat(st, session)

    return {
        "response": st.reply,
        "turn": session["turn_count"],
        "is_final": st.is_final,
        "patient_id": st.patient_id,
        "debug_context": session.get("debug_context", {}),
    }


def _finish_chat(st, session: Dict[str, Any]) -> None:
    session["completed"] = True
    try:
        from chat_module import sync_rl_report, queue_rl
        sync_rl_report(st.patient_id)
        if not session.get("is_tester"):
            queue_rl(st.patient_id)
    except Exception as exc:
        st.trace["finish_error"] = type(exc).__name__


def final_text(session: Dict[str, Any]) -> str:
    return build_final_text(session) or ""