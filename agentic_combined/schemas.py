"""Typed schemas for tool I/O and agent state."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Tool inputs ─────────────────────────────────────────
class ProcessedInput(BaseModel):
    """The saved results of the five algorithms (the `processed` dict of jataka_logic.run_logic)."""
    processed: Dict[str, Any]


# ── Agent state ─────────────────────────────────────────
class AgentState(BaseModel):
    processed: Dict[str, Any]
    user_id: Optional[str] = None

    validation: Optional[Dict[str, Any]] = None
    engine_result: Optional[Dict[str, Any]] = None

    critique: Optional[Dict[str, Any]] = None
    confidence: float = 0.0

    summary_payload: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    llm_used: bool = False
    llm_error: Optional[str] = None
    guardrail_report: Optional[Dict[str, Any]] = None
    memory_snapshot: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None


class CriticOutput(BaseModel):
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""


class GuardrailReport(BaseModel):
    valid: bool
    hallucinated_terms: List[str] = Field(default_factory=list)
    gender_violations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
