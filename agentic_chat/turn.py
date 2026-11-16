"""The pipeline. This is what a turn looks like in the final architecture."""
import copy
from typing import Any, Dict

from agentic_chat import config as cfg
from agentic_chat import nodes
from agentic_chat.llm_provider import ChatLLM
from agentic_chat.state import TurnState

# The session fields a turn changes. If the turn fails part-way they are put back, so the
# classic fallback in chat_module.send_message starts from the state before this answer.
# Without this the fallback would score the same answer a second time.
_TURN_FIELDS = ("cluster_scores", "messages", "last_question_meta", "turn_count",
                "red_flag_seen", "red_flag_ever", "asked_questions", "completed")
_MISSING = object()


def _snapshot(session: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (copy.deepcopy(session[k]) if k in session else _MISSING) for k in _TURN_FIELDS}


def _restore(session: Dict[str, Any], snap: Dict[str, Any]) -> None:
    for key, value in snap.items():
        if value is _MISSING:
            session.pop(key, None)
        else:
            session[key] = value


async def run_turn(session: Dict[str, Any],
                   session_id: str,
                   user_message: str,
                   provider: ChatLLM,
                   retriever=None) -> Dict[str, Any]:
    """Run one agentic turn. The patient waits only for this function.
    If it raises, the session is exactly as it was before the turn."""
    if cfg.CHAT_MODE != "agentic":
        raise RuntimeError("run_turn called while CHAT_MODE != agentic")

    st = TurnState(
        session_id=session_id,
        patient_id=session["patient_id"],
        user_message=user_message,
        turn=session["turn_count"] + 1,
    )

    snap = _snapshot(session)
    try:
        await nodes.understand(st, session, provider, retriever=retriever)   # 1
        nodes.score(st, session)                                             # 2
        nodes.plan(st, session)                                              # 3
        await nodes.word(st, session, provider)                              # 4
        nodes.guard(st, session)                                             # 5
        result = nodes.persist(st, session)                                  # 6
    except Exception:
        _restore(session, snap)
        raise

    try:
        nodes.prewarm(st, session)                                           # 7 (non-blocking)
    except Exception as exc:        # the reply is already saved: never fail the turn for this
        print(f"[AgenticChat] pre-warm failed ({type(exc).__name__}: {exc})")
    return result
