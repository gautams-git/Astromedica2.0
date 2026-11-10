"""LLM and runtime configuration."""
import os
from dataclasses import dataclass, field
from typing import Optional


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))  # gemini | ollama | stub
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-2.0-flash"))
    temperature: float = 0.0
    max_tokens: int = 1500
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    base_url: Optional[str] = None
    # The Gemini client defaults to no timeout and 6 retries, so a blocked key looks like a hang.
    timeout: float = field(default_factory=lambda: float(os.getenv("LLM_TIMEOUT", "20")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_RETRIES", "1")))

    @property
    def is_live(self) -> bool:
        """True when a real model will be called. Otherwise the agent runs rule-based only."""
        return (self.provider == "gemini" and bool(self.api_key)) or self.provider == "ollama"


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    confidence_threshold: float = 0.7
    max_replan_attempts: int = 2
    # Off by default: only the "clinical" profile (today's weights) runs, so the result is
    # identical to calling the engine directly. AGENTIC_MULTI_PROFILE=true tries the others.
    enable_multi_profile: bool = field(default_factory=lambda: _flag("AGENTIC_MULTI_PROFILE", "false"))
    enable_memory: bool = True
    strict_guardrails: bool = True


def get_llm(config: LLMConfig):
    """Lazy-init the LLM client. Falls back to a stub when there is no live model."""
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

    # Never called (is_live is False), but keeps the agent's interface the same.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    return FakeListChatModel(responses=["stub"] * 10)
