"""Typed schemas for organ-truth tool I/O and agent state."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Tool input ───────────────────────────────────────────
class ChartInput(BaseModel):
    chart_data: Dict[str, Any]
    rule1_result: Dict[str, Any]
    gender: Optional[str] = None


# ── Agent state ──────────────────────────────────────────
class AgentState(BaseModel):
    chart_data: Dict[str, Any]
    rule1_result: Dict[str, Any]
    gender: Optional[str] = None
    user_id: Optional[str] = None

    validation: Optional[Dict[str, Any]] = None
    engine_result: Optional[Dict[str, Any]] = None

    critique: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    replan_count: int = 0

    summary: Optional[str] = None
    guardrail_report: Optional[Dict[str, Any]] = None
    memory_snapshot: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None


# ── LLM outputs ──────────────────────────────────────────
class CriticOutput(BaseModel):
    issues: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    notes: str = ""
    request_rerun: bool = False


class GuardrailReport(BaseModel):
    valid: bool
    hallucinated_organs: List[str] = Field(default_factory=list)
    hallucinated_diseases: List[str] = Field(default_factory=list)
    gender_violations: List[str] = Field(default_factory=list)
    disclaimer_added: bool = False
    errors: List[str] = Field(default_factory=list)