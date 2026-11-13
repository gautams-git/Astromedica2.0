"""Main organ-truth agentic orchestrator (LangGraph). Weights are fixed."""
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from agentic_organ.config import AgentConfig, get_llm
from agentic_organ.critic import critique
from agentic_organ.guardrails import add_disclaimer, validate_output
from agentic_organ.memory import MemoryStore
from agentic_organ.schemas import AgentState
from agentic_organ.summarizer import summarize
from agentic_organ.tools import build_tool_registry


class OrganTruthAgent:
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
            {"rerun": "score", "summarize": "summarize"},
        )
        g.add_edge("summarize", "guardrail")
        g.add_edge("guardrail", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    def _route_after_critique(self, state: AgentState) -> str:
        c = state.critique or {}
        wants_rerun = bool(c.get("request_rerun"))
        low_confidence = state.confidence < self.cfg.confidence_threshold
        if (
            wants_rerun
            and low_confidence
            and state.replan_count < self.cfg.max_replan_attempts
        ):
            return "rerun"
        return "summarize"

    # ── Nodes ────────────────────────────────────────────
    def _node_validate(self, state: AgentState) -> AgentState:
        issues = []
        if not state.chart_data.get("planets"):
            issues.append("chart_data has no planets")
        if not state.rule1_result:
            issues.append("rule1_result empty")
        gender = (state.gender or state.chart_data.get("gender") or "").lower()
        if gender not in ("male", "female", "m", "f", ""):
            issues.append(f"unknown gender '{gender}'")
        state.validation = {"ok": not issues, "issues": issues}
        if issues:
            raise ValueError(f"Input validation failed: {issues}")
        return state

    def _node_score(self, state: AgentState) -> AgentState:
        # Single deterministic call — weights are NOT modified.
        # Re-entering after the critic means this is a re-score: count it, or the
        # max_replan_attempts limit never takes effect.
        if state.critique is not None:
            state.replan_count += 1
        state.engine_result = self.tools.call(
            "run_organ_truth_analysis",
            chart_data=state.chart_data,
            rule1_result=state.rule1_result,
            gender=state.gender,
        )
        return state

    def _node_critique(self, state: AgentState) -> AgentState:
        c = critique(self.llm, state.engine_result or {}, state.gender or "",
                     fallback_confidence=self.cfg.confidence_threshold)
        state.critique = c.model_dump()
        state.confidence = c.confidence
        if c.request_rerun:
            state.replan_count += 1
        return state

    def _node_summarize(self, state: AgentState) -> AgentState:
        state.summary = summarize(
            self.llm,
            state.engine_result or {},
            state.critique or {},
            state.gender or "",
        )
        return state

    def _node_guardrail(self, state: AgentState) -> AgentState:
        summary = state.summary or ""
        mentioned = []
        try:
            allowed = set()
            for o in (state.engine_result or {}).get("all_organs", []):
                allowed.add(o["organ"])
                for d in o.get("related_diseases", []):
                    allowed.add(d["disease"])
            for d in (state.engine_result or {}).get("all_diseases", []):
                allowed.add(d["disease"])
            mentioned = [t for t in allowed if t.lower() in summary.lower()]
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
        mem_snapshot = None
        if self.memory:
            key = MemoryStore.chart_key(state.chart_data, state.user_id)
            top_organs = {
                o["organ"]: o["score"]
                for o in (state.engine_result or {}).get("top_organs", [])
            }
            self.memory.remember(key, {
                "top_organs": top_organs,
                "confidence": state.confidence,
            })
            mem_snapshot = {"key": key, "trend": self.memory.trend(key)}
        state.memory_snapshot = mem_snapshot

        summary = state.summary or ""
        if state.guardrail_report and state.guardrail_report.get("valid"):
            summary = add_disclaimer(summary)
            state.guardrail_report["disclaimer_added"] = True

        state.final = {
            "summary": summary,
            "confidence": state.confidence,
            "critique": state.critique,
            "guardrail": state.guardrail_report,
            "memory": state.memory_snapshot,
            "reruns": state.replan_count,
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
        out = self.graph.invoke(initial)
        return out.get("final", {})


# ── Convenience function ────────────────────────────────
def run_agentic_organ_analysis(chart_data, rule1_result, gender=None,
                               user_id=None, config=None) -> Dict[str, Any]:
    if rule1_result is None or rule1_result == {}:
        raise RuntimeError(
            "run_agentic_organ_analysis called with empty rule1_result. "
            "Call jataka_logic.run_logic(chart, gender) first."
        )
    agent = OrganTruthAgent(config)
    print("Organ Truth Agent")
    return agent.invoke(chart_data, rule1_result, gender, user_id)