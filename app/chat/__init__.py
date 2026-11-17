"""Chat and RL feature flags. Classic chat is the default."""

from __future__ import annotations

from app.core.config import load_data_settings
from app.domain.schema import CHAT_MODES, RL_HINT_MODES


def chat_mode() -> str:
    mode = load_data_settings().chat_mode
    if mode not in CHAT_MODES:
        return "classic"
    return mode


def rl_hint_mode() -> str:
    mode = load_data_settings().rl_hint_mode
    if mode not in RL_HINT_MODES:
        return "off"
    return mode


def model_assisted_enabled() -> bool:
    return chat_mode() == "model_assisted"


def rl_hints_may_change_prompt() -> bool:
    """Shadow mode records a hint and does not change the patient-facing prompt."""
    return rl_hint_mode() == "enabled"


def rl_failure_blocks_reply() -> bool:
    return False
