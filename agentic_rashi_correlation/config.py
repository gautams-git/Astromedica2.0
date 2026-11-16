"""Runtime configuration for the Rashi Correlation agent.

The LLM settings and client are the disease agent's (same GEMINI_API_KEY / LLM_* variables),
so there is one place to configure the model. Only the agent switches are Rashi-specific.
"""
import os
from dataclasses import dataclass, field

from agentic_disease_first.config import LLMConfig, get_llm


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class RashiAgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    confidence_threshold: float = 0.7
    max_replan_attempts: int = 2
    # Off by default: only the "clinical" profile (today's weights) runs, so the result is
    # identical to calling the engine directly. Separate from AGENTIC_MULTI_PROFILE so the
    # disease agents are not affected.
    enable_multi_profile: bool = field(default_factory=lambda: _flag("AGENTIC_RASHI_MULTI_PROFILE", "false"))
    enable_memory: bool = True
    strict_guardrails: bool = True


__all__ = ["LLMConfig", "RashiAgentConfig", "get_llm"]
