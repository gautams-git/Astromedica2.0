"""Thin readers of the deterministic inputs the agent pulls from."""
from typing import Any, Dict

from chat_module import (
    cluster_by_rank,
    branch_by_id,
    update_cluster_support,
    save_chat_turn,
    build_backend_reasoning,
    fallback_question_meta,
    should_finish_early,
    build_final_text,
)
from question_generator import _bad_question, _banned_names, _plan_input
from symptom_mapper import classify_answer
from term_to_systems import TERM_TO_SYSTEMS


def rules_label(message: str) -> str:
    return classify_answer(message)


_TERM_PATTERNS = None


def _term_patterns():
    """One compiled pattern per known term, matching whole words only (so "head" does not
    match inside "ahead", nor "ear" inside "year")."""
    global _TERM_PATTERNS
    if _TERM_PATTERNS is None:
        import re
        _TERM_PATTERNS = [
            re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])")
            for term in TERM_TO_SYSTEMS.keys()
        ]
    return _TERM_PATTERNS


def known_term(message: str) -> bool:
    lower = message.lower()
    return any(pattern.search(lower) for pattern in _term_patterns())


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


GENERIC_QUESTION = "Is there anything else about how you have been feeling that you would like to tell me?"
_END_META = {"cluster_rank": 0, "question_type": "final"}


def _asked(session: Dict[str, Any]) -> set:
    """Every question already put to the patient: the assistant messages, plus the bank text
    behind them (a reworded question no longer matches its bank text)."""
    asked = {m["content"] for m in session.get("messages", []) if m.get("role") == "assistant"}
    return asked | set(session.get("asked_questions", []))


def _first_unasked(pool, asked: set) -> str:
    return next((q for q in (pool or []) if q and q not in asked), "")


def bank_question(session: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """The next question not yet asked, of the type (and branch) the planner chose.
    "" when that type has no question list (e.g. "context") or all of its questions were asked.
    It never repeats a question."""
    cluster = cluster_by_rank(session, meta.get("cluster_rank")) or {}
    branch = branch_by_id(cluster, meta.get("branch_id", "")) or {}
    key = _QUESTION_KEY.get(meta.get("question_type"))
    if not key:
        return ""
    asked = _asked(session)
    return _first_unasked(branch.get(key), asked) or _first_unasked(cluster.get(key), asked)


def next_question(session: Dict[str, Any], meta: Dict[str, Any]):
    """(text, meta) of the question to ask. The planner's choice when the bank has it; otherwise
    the next unasked question of another type in the same cluster, then in the other clusters;
    then one general closing question; and when nothing is left, ("", final meta) to end the chat.
    The returned meta is the one the question really belongs to, so the answer is credited to it."""
    text = bank_question(session, meta)
    if text:
        return text, meta

    asked = _asked(session)
    # "context" has no question list: fall back to the follow-up types. Red-flag questions are
    # never used as filler.
    types = ("severity", "drilldown", "screening") if meta.get("question_type") == "context" \
        else ("drilldown", "severity", "screening")
    clusters = (session.get("chat_plan", {}) or {}).get("clusters", []) or []
    current = meta.get("cluster_rank")
    ordered = sorted(clusters, key=lambda c: (c.get("rank") != current, c.get("rank") or 0))
    for cluster in ordered:
        branches = cluster.get("branches", []) or []
        if cluster.get("rank") == current:       # the branch the planner chose comes first
            branches = sorted(branches, key=lambda b: b.get("id") != meta.get("branch_id"))
        for qtype in types:
            key = _QUESTION_KEY[qtype]
            for branch in branches:
                text = _first_unasked(branch.get(key), asked)
                if text:
                    return text, {"cluster_rank": cluster.get("rank"), "question_type": qtype,
                                  "branch_id": branch.get("id", "")}
            text = _first_unasked(cluster.get(key), asked)
            if text:
                return text, {"cluster_rank": cluster.get("rank"), "question_type": qtype,
                              "branch_id": (branches[0].get("id", "") if branches else "")}

    if GENERIC_QUESTION not in asked:
        # cluster_rank 0: the answer is not credited to any cluster or branch
        return GENERIC_QUESTION, {"cluster_rank": 0, "question_type": "context", "branch_id": ""}
    return "", dict(_END_META)


def question_is_bad(text: str, session: Dict[str, Any]) -> str:
    banned = _banned_names(_plan_input(session["chat_plan"]))
    return _bad_question(text, banned)


def _update_debug_context(st, session: Dict[str, Any]) -> None:
    """Keep the entries the dev panel shows up to date, as the classic path does."""
    try:
        debug = session.setdefault("debug_context", {})
        debug["latest_answer_result"] = st.label
        debug["latest_question_explanation"] = build_backend_reasoning(
            session, st.turn, session["last_question_meta"], answer_label=st.label)
        debug["cluster_scores"] = session.get("cluster_scores", {})
    except Exception as exc:        # the dev panel must never break a chat
        print(f"[AgenticChat] could not update the dev panel info: {type(exc).__name__}: {exc}")


def persist_turn(st, session: Dict[str, Any]) -> Dict[str, Any]:
    session["last_question_meta"] = {**st.meta, "question_text": st.reply}
    _update_debug_context(st, session)
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
    if st.bank:
        session.setdefault("asked_questions", []).append(st.bank)
    session["turn_count"] += 1
    try:
        save_chat_turn(
            st.patient_id, session["turn_count"], "assistant", st.reply,
            {
                **session["last_question_meta"],
                "session_id": st.session_id,
                "cluster_scores": session.get("cluster_scores", {}),
                "backend_reasoning": session.get("debug_context", {}).get("latest_question_explanation"),
                "trace": st.trace,
            },
        )
    except Exception:
        # run_turn puts the session back and the classic path saves this answer again,
        # so take it out of the log now.
        _undo_user_turn(st)
        raise
    if st.is_final:
        _finish_chat(st, session)

    return {
        "response": st.reply,
        "turn": session["turn_count"],
        "is_final": st.is_final,
        "patient_id": st.patient_id,
        "debug_context": session.get("debug_context", {}),
    }


def _undo_user_turn(st) -> None:
    """Best effort: remove the user turn this call just wrote to chat_log.json."""
    import json
    log_file = f"patients/{st.patient_id}/chat_log.json"
    try:
        with open(log_file) as f:
            log = json.load(f)
        turns = log.get("turns") or []
        if turns and turns[-1].get("role") == "user" and turns[-1].get("turn") == st.turn:
            turns.pop()
            with open(log_file, "w") as f:
                json.dump(log, f, indent=2)
    except Exception as exc:
        print(f"[AgenticChat] could not undo the saved answer for patient {st.patient_id}: "
              f"{type(exc).__name__}: {exc}")


def _finish_chat(st, session: Dict[str, Any]) -> None:
    """The closing steps of the classic path (chat_module.send_message): mark the chat complete
    in chat_log.json, refresh For RL Report.json, and queue the RL job (not for tester chats)."""
    import json
    from datetime import datetime
    from chat_module import sync_rl_report, queue_rl

    session["completed"] = True
    log_file = f"patients/{st.patient_id}/chat_log.json"
    try:
        with open(log_file) as f:
            log = json.load(f)
        log["completed"] = True
        log["completed_at"] = datetime.now().isoformat()
        log["final_assessment"] = st.reply
        log["hitlist_cluster_scores"] = session.get("cluster_scores", {})
        log["chat_plan_source"] = session.get("chat_plan_source", "hitlist_v1")
        with open(log_file, "w") as f:
            json.dump(log, f, indent=2)
    except Exception as exc:       # without the log there is nothing to sync or queue
        print(f"[AgenticChat] could not mark the chat complete for patient {st.patient_id}: "
              f"{type(exc).__name__}: {exc}")
        st.trace["finish_error"] = type(exc).__name__
        return

    sync_rl_report(st.patient_id, log)          # it handles its own errors
    if session.get("skip_rl"):                  # set by /start-consultation for tester chats
        print(f"[RL] Patient {st.patient_id}: tester chat - RL not queued")
    else:
        queue_rl(st.patient_id)                 # it handles its own errors
    print(f"[AgenticChat] Chat completed - log saved for patient {st.patient_id}")


def final_text(session: Dict[str, Any]) -> str:
    return build_final_text(session) or ""