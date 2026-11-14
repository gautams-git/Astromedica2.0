"""The pipeline. This is what a turn looks like in the final architecture."""
from typing import Any, Dict

from agentic_chat import config as cfg
from agentic_chat import nodes
from agentic_chat.llm_provider import ChatLLM
from agentic_chat.state import TurnState


async def run_turn(session: Dict[str, Any],
                   session_id: str,
                   user_message: str,
                   provider: ChatLLM,
                   retriever=None) -> Dict[str, Any]:
    """Run one agentic turn. The patient waits only for this function."""
    if cfg.CHAT_MODE != "agentic":
        raise RuntimeError("run_turn called while CHAT_MODE != agentic")

    st = TurnState(
        session_id=session_id,
        patient_id=session["patient_id"],
        user_message=user_message,
        turn=session["turn_count"] + 1,
    )

    await nodes.understand(st, session, provider, retriever=retriever)   # 1
    nodes.score(st, session)                                             # 2
    nodes.plan(st, session)                                              # 3
    await nodes.word(st, session, provider)                              # 4
    nodes.guard(st, session)                                             # 5
    result = nodes.persist(st, session)                                  # 6
    nodes.prewarm(st, session)                                           # 7 (non-blocking)
    return result