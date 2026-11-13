"""Main agentic orchestrator using LangGraph.

Flow:
  validate → multi_score → compare → critique →
    (replan if low confidence) → summarize → guardrail → finalize
"""
import json
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agentic.config import AgentConfig, get_llm
from agentic.critic import critique
from agentic.guardrails import add_disclaimer, validate_output
from agentic.memory import MemoryStore
from agentic.profiles import PROFILES, profile_context, top_terms
from agentic.schemas import AgentState, ChartInput
from agentic.tools import build_tool_registry

SUMMARIZER_SYSTEM = """You are a medical-astrology EXPLAINER.
You receive a deterministic scoring result from an engine.
Write a clear, cautious 3-5 sentence summary for a user.

HARD RULES:
- ONLY mention diseases/organ terms that appear in the provided trails.
- Do NOT invent new terms.
- Do NOT contradict the ranking.
- Reference WHY the top system ranked first (cite 2-3 contributing terms).
- If the critic flagged low support or ambiguity, mention it honestly.
- End with an encouraging, non-alarmist note.
- NEVER give medical advice. NEVER prescribe.
"""


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text
    return text[start:end + 1]


class HitlistAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.cfg = config or AgentConfig()
        self.llm = get_llm(self.cfg.llm)
        self.tools = build_tool_registry()
        self.memory = MemoryStore() if self.cfg.enable_memory else None
        self.graph = self._build_graph()

    # ── Graph ────────────────────────────────────────────
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("validate", self._node_validate)
        g.add_node("score", self._node_score)
        g.add_node("critique", self._node_critique)
        g.add_node("summarize", self._node_summarize)
        g.add_node("guardrail", self._node_guardrail)
        g.add_node("finalize", self._node_finalize)

        g.set_entry_point("validate")
        g.add_edge("validate", "score")
        g.add_edge("score", "critique")
        g.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"rescore": "score", "summarize": "summarize"},
        )
        g.add_edge("summarize", "guardrail")
        g.add_edge("guardrail", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def _route_after_critique(self, state: AgentState) -> str:
        if (
            state.confidence < self.cfg.confidence_threshold
            and state.replan_count < self.cfg.max_replan_attempts
        ):
            return "rescore"
        return "summarize"

    # ── Nodes ────────────────────────────────────────────
    def _node_validate(self, state: AgentState) -> AgentState:
        issues = []
        if not state.chart_data.get("planets"):
            issues.append("chart_data has no planets")
        if not state.rule1_result:
            issues.append("rule1_result empty")
        gender = (state.gender or state.chart_data.get("gender") or "").lower()
        if gender not in ("male", "female", "m", "f", ""):   # jataka_logic passes the M/F code
            issues.append(f"unknown gender '{gender}'")
        state.validation = {"ok": not issues, "issues": issues}
        if issues:
            raise ValueError(f"Input validation failed: {issues}")
        return state

    def _node_score(self, state: AgentState) -> AgentState:
        # Re-entering after the critic means this is a re-score: count it, or the
        # max_replan_attempts limit never takes effect.
        if state.critique is not None:
            state.replan_count += 1
        profile_names = list(PROFILES.keys()) if self.cfg.enable_multi_profile else ["clinical"]
        results = {}

        for name in profile_names:
            with profile_context(name):
                res = self.tools.call(
                    "run_hitlist_analysis",
                    chart_data=state.chart_data,
                    rule1_result=state.rule1_result,
                    gender=state.gender,
                )
            results[name] = res

        state.profile_results = results

        # If re-scoring (replan), rotate to a different profile
        if state.replan_count > 0 and len(profile_names) > 1:
            alt = profile_names[state.replan_count % len(profile_names)]
            state.chosen_profile = alt
        else:
            state.chosen_profile = self._pick_stable_profile(results)

        state.engine_result = results[state.chosen_profile]
        return state

    def _pick_stable_profile(self, results: Dict[str, Dict]) -> str:
        """Prefer profile whose top-2 overlaps the most with the others."""
        if len(results) <= 1:
            return next(iter(results))
        tops = {name: set(top_terms(r, 2)) for name, r in results.items()}
        scores = {}
        for name, own in tops.items():
            overlap = sum(len(own & other) for n, other in tops.items() if n != name)
            scores[name] = overlap
        return max(scores, key=scores.get)

    def _node_critique(self, state: AgentState) -> AgentState:
        c = critique(self.llm, state.engine_result or {}, state.gender or "",
                     fallback_confidence=self.cfg.confidence_threshold)
        state.critique = c.model_dump()
        state.confidence = c.confidence

        # If critic suggests a profile switch and we have budget, replan
        if (
            c.suggested_profile
            and c.suggested_profile in state.profile_results
            and c.suggested_profile != state.chosen_profile
            and state.replan_count < self.cfg.max_replan_attempts
        ):
            state.chosen_profile = c.suggested_profile
            state.engine_result = state.profile_results[c.suggested_profile]
            state.replan_count += 1
            state.confidence = max(c.confidence, self.cfg.confidence_threshold)

        return state

    def _node_summarize(self, state: AgentState) -> AgentState:
        result = state.engine_result or {}
        payload = {
            "top4": [
                {
                    "label": g["label"],
                    "score": g["score"],
                    "hitlists": g["hitlists"],
                    "trail": g["trail"][:10],
                }
                for g in result.get("top4", [])
            ],
            "critic_notes": (state.critique or {}).get("notes", ""),
            "critic_issues": (state.critique or {}).get("issues", []),
            "gender": state.gender,
            "profile_used": state.chosen_profile,
        }

        try:
            resp = self.llm.invoke([
                SystemMessage(content=SUMMARIZER_SYSTEM),
                HumanMessage(content=json.dumps(payload, default=str)),
            ])
            text = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            text = f"(Summary unavailable: {e}. See trail data.)"

        state.summary = text
        return state

    def _node_guardrail(self, state: AgentState) -> AgentState:
        # Extract mentioned terms from summary via simple KB match
        summary = state.summary or ""
        mentioned = []
        try:
            from term_to_systems import TERM_TO_SYSTEMS
            mentioned = [t for t in TERM_TO_SYSTEMS if t.lower() in summary.lower()]
        except Exception:
            pass

        report = validate_output(
            summary=summary,
            mentioned_terms=mentioned,
            engine_result=state.engine_result or {},
            gender=state.gender or "",
            strict=self.cfg.strict_guardrails,
        )
        state.guardrail_report = report.model_dump()
        return state

    def _node_finalize(self, state: AgentState) -> AgentState:
        # Memory
        mem_snapshot = None
        if self.memory:
            key = MemoryStore.chart_key(state.chart_data, state.user_id)
            top4_scores = {
                g["label"]: g["score"]
                for g in (state.engine_result or {}).get("top4", [])
            }
            self.memory.remember(key, {
                "top4_scores": top4_scores,
                "profile": state.chosen_profile,
                "confidence": state.confidence,
            })
            mem_snapshot = {
                "key": key,
                "trend": self.memory.trend(key),
            }
        state.memory_snapshot = mem_snapshot

        summary = state.summary or ""
        if state.guardrail_report and state.guardrail_report.get("valid"):
            summary = add_disclaimer(summary)

        state.final = {
            "summary": summary,
            "profile_used": state.chosen_profile,
            "profile_scores": {
                name: [g["label"] for g in r.get("top4", [])]
                for name, r in state.profile_results.items()
            },
            "confidence": state.confidence,
            "critique": state.critique,
            "guardrail": state.guardrail_report,
            "memory": state.memory_snapshot,
            "engine_result": state.engine_result,
        }
        return state

    # ── Public API ───────────────────────────────────────
    def invoke(self, chart_data: Dict, rule1_result: Dict,
               gender: str | None = None,
               user_id: str | None = None) -> Dict[str, Any]:
        initial = AgentState(
            chart_data=chart_data,
            rule1_result=rule1_result,
            gender=gender or chart_data.get("gender"),
            user_id=user_id,
        )
        print("Invoking graph")
        out = self.graph.invoke(initial)
        return out.get("final", {})


# ── Convenience function ────────────────────────────────
def run_agentic_analysis(chart_data, rule1_result, gender=None,
                        user_id=None, config=None) -> Dict[str, Any]:
    agent = HitlistAgent(config)
    print("Config is",agent.cfg)
    return agent.invoke(chart_data, rule1_result, gender, user_id)