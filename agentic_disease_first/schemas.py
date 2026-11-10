"""Typed schemas for tool I/O and agent state."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Tool inputs ─────────────────────────────────────────
class ChartInput(BaseModel):
    chart_data: Dict[str, Any]
    rule1_result: Dict[str, Any]
    gender: Optional[str] = None


class PlanetTermsInput(BaseModel):
    planet_name: str


class NakshatraTermsInput(BaseModel):
    nakshatra_name: str
    pada: int = Field(ge=1, le=4)


class RashiTermsInput(BaseModel):
    rashi_name: str


class GenderCheckInput(BaseModel):
    term: str
    gender: Optional[str] = None


class DiseaseOrgansInput(BaseModel):
    disease: str


# ── Agent state ─────────────────────────────────────────
class AgentState(BaseModel):
    chart_data: Dict[str, Any]
    rule1_result: Dict[str, Any]
    gender: Optional[str] = None
    user_id: Optional[str] = None

    validation: Optional[Dict[str, Any]] = None
    profile_results: Dict[str, Dict[str, Any]] = {}
    chosen_profile: Optional[str] = None
    engine_result: Optional[Dict[str, Any]] = None

    critique: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    replan_count: int = 0

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
    suggested_profile: Optional[str] = None


class GuardrailReport(BaseModel):
    valid: bool
    hallucinated_terms: List[str] = Field(default_factory=list)
    gender_violations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
