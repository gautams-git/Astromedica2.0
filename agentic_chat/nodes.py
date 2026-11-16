"""The six nodes. Each is a function; each has one job."""
import asyncio
from typing import Any, Dict

from agentic_chat import config as cfg
from agentic_chat import inputs
from agentic_chat.rag_reader import fetch as rag_fetch, examples_from_hits
from agentic_chat.rl_hint import drop_hint, hint_key, start_prewarm, take_hint
from agentic_chat.state import TurnState


LABELS = ("yes_strong", "yes_but", "no_but", "no_strong")


# ── 1. understand ───────────────────────────────────────
def _valid_extract(d: Dict[str, Any]) -> Dict[str, Any]:
    if d.get("label") not in LABELS:
        raise ValueError("bad label")
    return {
        "label": d["label"],
        "symptoms": [str(s)[:60] for s in d.get("symptoms", [])][:6],
        "duration": str(d.get("duration", ""))[:40],
        "severity": d.get("severity", "none"),
        "red_flag": bool(d.get("red_flag", False)),
    }


async def understand(st: TurnState, session: Dict[str, Any],
                     provider, retriever=None) -> None:
    st.rule_label = inputs.rules_label(st.user_message)
    st.label = st.rule_label

    if cfg.CHAT_UNDERSTAND == "rules":
        return

    last = session.get("last_question_meta", {}) or {}
    question = last.get("question_text", "")

    rag_examples = ""
    if cfg.RAG_MODE != "off" and retriever is None:
        st.trace["rag_error"] = "no_retriever"      # RAG is switched on but nothing can be searched
    if (
        cfg.RAG_MODE != "off"
        and retriever is not None
        and not inputs.known_term(st.user_message)
    ):
        rag = await rag_fetch(st.user_message, {"session": st.session_id}, retriever)
        if "hits" in rag:
            rag_examples = examples_from_hits(rag["hits"])
            st.trace["rag_hits"] = len(rag["hits"])
        elif "error" in rag:
            st.trace["rag_error"] = rag["error"]

    try:
        raw = await asyncio.wait_for(
            provider.extract(question=question, answer=st.user_message,
                             rag_examples=rag_examples),
            timeout=cfg.LLM_TIMEOUT,
        )
        st.extracted = _valid_extract(raw)
    except Exception as exc:
        st.trace["understand_error"] = type(exc).__name__
        return

    st.trace["labels"] = {"rule": st.rule_label, "model": st.extracted["label"]}
    if cfg.CHAT_UNDERSTAND == "llm":
        st.label = st.extracted["label"]


# ── 2. score ────────────────────────────────────────────
def score(st: TurnState, session: Dict[str, Any]) -> None:
    inputs.apply_score(st, session)
    if st.extracted.get("red_flag"):
        session["red_flag_seen"] = True     # ask ONE red-flag question next (plan() clears it)
        session["red_flag_ever"] = True     # the final message tells the patient to see a doctor


# ── 3. plan ─────────────────────────────────────────────
def plan(st: TurnState, session: Dict[str, Any]) -> None:
    if inputs.should_finish(st.turn, session):
        st.meta = {"cluster_rank": 0, "question_type": "final"}
        return

    meta = inputs.planner_meta(st.turn, session)

    # pop: the red-flag question is asked once, not on every later turn
    if session.pop("red_flag_seen", False) and meta.get("cluster_rank"):
        red = {**meta, "question_type": "red_flag"}
        if inputs.bank_question(session, red):          # only if a red-flag question is left to ask
            st.meta = red
            return

    hint = take_hint(st.session_id)
    if hint:
        st.hint = hint
        st.trace["hint_ready"] = True
        if not cfg.RL_HINT_SHADOW:
            meta = _apply_hint(session, meta, hint, st.trace)
        else:
            st.trace["hint_shadow"] = _preview_hint(session, meta, hint)

    st.meta = meta


def _apply_hint(session, meta, hint, trace) -> Dict[str, Any]:
    if meta.get("question_type") not in ("screening", "drilldown"):
        return meta
    cluster = inputs.cluster_by_rank(session, meta["cluster_rank"])
    states = session["cluster_scores"][str(meta["cluster_rank"])].get("branches", {})
    pool = [
        b for b in cluster.get("branches", [])
        if states.get(b["id"], {}).get("status") != "rejected"
    ]
    if not pool:
        return meta

    def answered(b):
        h = hint.get(hint_key(b.get("label")), {})
        return h.get("endorsed", 0) + h.get("denied", 0)

    pick = (
        min(pool, key=answered)
        if meta["question_type"] == "screening"
        else max(pool, key=answered)
    )
    trace["hint_changed_branch"] = pick["id"] != meta.get("branch_id")
    return {**meta, "branch_id": pick["id"], "hint_used": True}


def _preview_hint(session, meta, hint) -> Dict[str, Any]:
    try:
        return {"would_pick": _apply_hint(session, meta, hint, {})}
    except Exception:
        return {}


# ── 4. word ─────────────────────────────────────────────
async def word(st: TurnState, session: Dict[str, Any],
               provider) -> None:
    if st.meta.get("question_type") == "final":
        return
    # The planner's choice if the bank has a question for it; otherwise the next unasked one.
    # st.meta becomes the meta the question really belongs to (or "final" if nothing is left).
    bank, st.meta = inputs.next_question(session, st.meta)
    st.bank = bank
    if st.meta.get("question_type") == "final":
        st.trace["no_question_left"] = True
        return
    text = bank
    if cfg.CHAT_PHRASE_LLM and bank:
        try:
            text = await asyncio.wait_for(
                provider.phrase(
                    bank=bank,
                    previous_answer=st.user_message,
                    symptoms=st.extracted.get("symptoms", []),
                ),
                timeout=cfg.LLM_TIMEOUT,
            )
        except Exception as exc:
            st.trace["phrase_error"] = type(exc).__name__
    st.reply = (text or "").strip() or bank


# ── 5. guard ────────────────────────────────────────────
def guard(st: TurnState, session: Dict[str, Any]) -> None:
    if st.meta.get("question_type") == "final":
        return
    # A reworded question that fails the checks is replaced by the bank text it was based on.
    reason = inputs.question_is_bad(st.reply, session)
    if reason and st.bank and st.reply != st.bank:
        st.trace["guard"] = reason
        st.reply = st.bank
    # Never send a blank message: with nothing to ask, end the chat properly.
    if not st.reply.strip():
        st.trace["empty_reply"] = True
        st.meta = {"cluster_rank": 0, "question_type": "final"}


# ── 6. persist ──────────────────────────────────────────
def persist(st: TurnState, session: Dict[str, Any]) -> Dict[str, Any]:
    if st.meta.get("question_type") == "final":
        st.reply = inputs.final_text(session) or "Thank you. Your consultation is complete."
        st.reply += cfg.DISCLAIMER
        st.is_final = True
        drop_hint(st.session_id)                # the chat is over: forget any hint still being prepared
    return inputs.persist_turn(st, session)


# ── 7. pre-warm (does not block the reply) ──────────────
def prewarm(st: TurnState, session: Dict[str, Any]) -> None:
    if not st.is_final:
        start_prewarm(st.session_id, session, st.patient_id)