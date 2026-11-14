"""Agentic chat — the chat agent is the application.

Every input (model, RAG, RL, algorithms, KB, bank, plan) is a bounded read.
The only write the agent performs is the chat log.
"""
from agentic_chat.turn import run_turn
from agentic_chat.state import TurnState
from agentic_chat.config import (
    CHAT_MODE, CHAT_LLM, CHAT_UNDERSTAND, CHAT_PHRASE_LLM,
    RAG_MODE, RL_HINT, RL_HINT_SHADOW,
)

__all__ = [
    "run_turn", "TurnState",
    "CHAT_MODE", "CHAT_LLM", "CHAT_UNDERSTAND", "CHAT_PHRASE_LLM",
    "RAG_MODE", "RL_HINT", "RL_HINT_SHADOW",
]