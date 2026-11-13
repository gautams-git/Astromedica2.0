"""LLM and runtime configuration."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "gemini")  # gemini | ollama | stub
    model: str = os.getenv("LLM_MODEL", "gemini-2.0-flash")
    temperature: float = 0.0
    max_tokens: int = 1500
    api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    base_url: Optional[str] = None
    # The Gemini client defaults to no timeout and 6 retries, so a blocked key looks like a hang.
    timeout: float = float(os.getenv("LLM_TIMEOUT", "20"))
    max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "1"))


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    confidence_threshold: float = 0.7
    max_replan_attempts: int = 2
    enable_multi_profile: bool = True
    enable_memory: bool = True
    strict_guardrails: bool = True


def get_llm(config: LLMConfig):
    """Lazy-init a Gemini LLM client. Falls back to a stub if no API key."""
    if config.provider == "gemini" and config.api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.api_key,
            temperature=config.temperature,
            max_output_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )
    if config.provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=config.model or "gemini-2.0-flash", temperature=config.temperature)

    # Deterministic stub so the pipeline works without any API key.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    return FakeListChatModel(responses=['{"confidence": 0.9, "issues": [], "notes": "stub"}'] * 10)