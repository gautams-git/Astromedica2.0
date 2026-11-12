"""Agentic Combined Algo (LangGraph).

Flow:
  validate -> score -> critique -> summarize -> guardrail -> finalize

The scoring is still the deterministic algorithm_comparison.build_combined_algo_summary, so the
result is identical to calling it directly. The Combined Algo is a fixed formula (rank score +
median of five algorithms): there are no weights to try, so unlike the other agents there is no
profile switch and no re-score loop - the graph has no cycle at all.

With no live LLM (no GEMINI_API_KEY) the critic is rule-based and the summary is built in
code, so the agent still works and never calls out to a model.
"""
import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agentic_combined.config import AgentConfig, get_llm
from agentic_combined.critic import critique
from agentic_combined.guardrails import add_disclaimer, validate_output
from agentic_combined.memory import MemoryStore
from agentic_combined.schemas import AgentState
from agentic_combined.tools import algorithms_with_data, build_tool_registry
from algorithm_comparison import build_combined_algo_summary

TITLE = "Combined Algo"
ALGORITHM = "combined_algo"
TOOL = "build_combined_algo_summary"
GRAPH_STEP_LIMIT = 40

SUMMARIZER_SYSTEM = """You are a medical-astrology EXPLAINER.
You receive the deterministic "Combined Algo" result: the top organs by the MEDIAN consensus
of five algorithms, each with the algorithms that named it and its top diseases.
Write a clear, cautious 3-5 sentence summary for a user.

HARD RULES:
- ONLY mention organs and diseases that appear in the provided data.
- Do NOT invent new terms.
- Do NOT change the ranking.
- Say WHY the top organ ranked first (how many algorithms agree, its consensus score).
- If the critic flagged weak agreement or ambiguity, mention it honestly.
- End with an encouraging, non-alarmist note.
- NEVER give medical advice. NEVER prescribe.
"""


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def deterministic_summary(top: List[Dict[str, Any]]) -> str:
    """Summary built in code from the engine data only. Used with no live LLM, or when the LLM
    output fails the guardrail."""
    if not top:
        return f"{TITLE} found no organ with enough support across the algorithms."
    lead = top[0]
    text = (f"{TITLE} ranks {lead['organ']} first (consensus score {lead['consensus_score']}), "
            f"named by {lead['support_count']} of 5 algorithms")
    if lead.get("algorithms"):
        text += f": {', '.join(lead['algorithms'])}"
    text += "."
    if lead.get("diseases"):
        text += f" Its leading diseases: {', '.join(lead['diseases'])}."
    rest = [o["organ"] for o in top[1:3]]
    if rest:
        text += f" Next in the ranking: {', '.join(rest)}."
    return text


class CombinedAlgoAgent:
    def __init__(self, config: AgentConfig | None = None):
        self.cfg = config or AgentConfig()
        self.live = self.cfg.llm.is_live
        self.llm = get_llm(self.cfg.llm)
        self.tools = build_tool_registry()
        self.memory = MemoryStore.shared() if self.cfg.enable_memory else None
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
        g.add_edge("critique", "summarize")
        g.add_edge("summarize", "guardrail")
        g.add_edge("guardrail", "finalize")
        g.add_edge("finalize", END)
        return g.compile()

    # ── Nodes ────────────────────────────────────────────
    def _node_validate(self, state: AgentState) -> AgentState:
        issues = []
        if not state.processed:
            issues.append("processed results are empty")
        present = algorithms_with_data(state.processed)
        # A missing algorithm is not fatal (the plain function copes with it): the critic reports it.
        state.validation = {"ok": not issues, "issues": issues, "algorithms_with_data": present}
        if issues:
            raise ValueError(f"Input validation failed: {issues}")
        return state

    def _node_score(self, state: AgentState) -> AgentState:
        state.engine_result = self.tools.call(TOOL, processed=state.processed)
        return state

    def _node_critique(self, state: AgentState) -> AgentState:
        present = (state.validation or {}).get("algorithms_with_data", {})
        c = critique(self.llm, state.engine_result or {}, present, use_llm=self.live)
        state.critique = c.model_dump()
        state.confidence = c.confidence
        return state

    def _payload(self, state: AgentState) -> Dict[str, Any]:
        result = state.engine_result or {}
        top = [
            {
                "organ": o["organ"],
                "consensus_score": o.get("consensus_score"),
                "support_count": o.get("algorithm_support_count"),
                "algorithms": o.get("supporting_algorithms") or [],
                "diseases": [d["disease"] for d in (o.get("top_diseases") or [])],
                "related_organs": [r["organ"] for r in (o.get("related_organs") or [])],
            }
            for o in (result.get("top_organs") or [])
        ]
        return {
            "top": top,
            "critic_notes": (state.critique or {}).get("notes", ""),
            "critic_issues": (state.critique or {}).get("issues", []),
        }

    def _node_summarize(self, state: AgentState) -> AgentState:
        payload = self._payload(state)
        state.summary_payload = payload
        fallback = deterministic_summary(payload["top"])

        if not self.live:
            state.summary = fallback
            return state

        try:
            resp = self.llm.invoke([
                SystemMessage(content=SUMMARIZER_SYSTEM),
                HumanMessage(content=json.dumps(payload, default=str)),
            ])
            state.summary = _text(resp.content if hasattr(resp, "content") else resp).strip() or fallback
            state.llm_used = True
        except Exception as e:
            state.summary = fallback
            state.llm_error = f"{type(e).__name__}: {e}"
        return state

    def _node_guardrail(self, state: AgentState) -> AgentState:
        top = (state.summary_payload or {}).get("top", [])
        report = validate_output(
            summary=state.summary or "",
            allowed_diseases=[d for o in top for d in o["diseases"]],
            allowed_organs=[o["organ"] for o in top] + [r for o in top for r in o["related_organs"]],
            strict=self.cfg.strict_guardrails,
        )
        state.guardrail_report = report.model_dump()
        if not report.valid:
            # The model wrote something that is not in the data: keep the facts, drop the prose.
            state.summary = deterministic_summary(top)
            state.llm_used = False
        return state

    def _node_finalize(self, state: AgentState) -> AgentState:
        mem_snapshot = None
        if self.memory:
            key = MemoryStore.chart_key(state.processed, state.user_id, ALGORITHM)
            top_scores = {
                o["organ"]: o["consensus_score"]
                for o in (state.engine_result or {}).get("top_organs", [])
            }
            self.memory.remember(key, {"top_scores": top_scores, "confidence": state.confidence})
            mem_snapshot = {"key": key, "trend": self.memory.trend(key)}
        state.memory_snapshot = mem_snapshot

        state.final = {
            "summary": add_disclaimer(state.summary or ""),
            "confidence": state.confidence,
            "critique": state.critique,
            "guardrail": state.guardrail_report,
            "memory": state.memory_snapshot,
            "llm_used": state.llm_used,
            "llm_error": state.llm_error,
            "algorithms_with_data": (state.validation or {}).get("algorithms_with_data"),
            "engine_result": state.engine_result,
        }
        return state

    # ── Public API ───────────────────────────────────────
    def invoke(self, processed: Dict, user_id: str | None = None) -> Dict[str, Any]:
        initial = AgentState(processed=processed, user_id=user_id)
        out = self.graph.invoke(initial, config={"recursion_limit": GRAPH_STEP_LIMIT})
        return out.get("final", {})


# ── Convenience function ────────────────────────────────
def run_agentic_combined(processed, user_id=None, config=None) -> Dict[str, Any]:
    """Agentic Combined Algo. If the agent itself fails for any reason, the plain function's
    result is returned so a chart is never lost because of the agent."""
    try:
        agent = CombinedAlgoAgent(config)
        return agent.invoke(processed, user_id)
    except Exception as exc:
        print(f"[Agentic] {TITLE} agent failed ({type(exc).__name__}: {exc}) - using the plain function")
        plain = build_combined_algo_summary(processed)
        return {"engine_result": plain, "agentic_error": f"{type(exc).__name__}: {exc}"}
