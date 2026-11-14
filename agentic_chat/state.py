"""Per-turn state."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TurnState:
    session_id: str
    patient_id: str
    user_message: str
    turn: int

    # Populated by nodes, in order
    rule_label: str = ""
    label: str = ""
    extracted: Dict[str, Any] = field(default_factory=dict)
    hint: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    reply: str = ""
    is_final: bool = False

    # Diagnostics, stored in the chat log's `trace`
    trace: Dict[str, Any] = field(default_factory=dict)