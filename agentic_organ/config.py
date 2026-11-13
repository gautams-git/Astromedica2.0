"""LLM and runtime configuration (Gemini)."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")
    model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    temperature: float = 0.0
    max_tokens: int = 1500
    api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
    # The Gemini client defaults to no timeout and 6 retries, so a blocked key looks like a hang.
    timeout: float = float(os.getenv("LLM_TIMEOUT", "20"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    confidence_threshold: float = 0.65
    max_replan_attempts: int = 2      # bound on critic-triggered re-runs
    enable_memory: bool = True
    strict_guardrails: bool = True


def get_llm(config: LLMConfig):
    """Lazy-init Gemini. Falls back to a stub if no key is set."""
    if config.provider == "gemini" and config.api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            transport="rest",
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    return FakeListChatModel(
        responses=['{"confidence": 0.8, "issues": [], "notes": "stub"}'] * 10
    )