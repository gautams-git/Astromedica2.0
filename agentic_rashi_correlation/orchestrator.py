"""Agentic Rashi Correlation (LangGraph).

Flow:
  validate -> score -> critique -> (re-score if low confidence and there is another profile)
           -> summarize -> guardrail -> finalize

The scoring is still the deterministic rashi_correlation_engine. By default only the
"clinical" profile (today's weights) runs, so the result is identical to calling the engine
directly. Set AGENTIC_RASHI_MULTI_PROFILE=true to also try the alternative weight profiles.

With no live LLM (no GEMINI_API_KEY) the critic is rule-based and the summary is built in
code, so the agent still works and never calls out to a model.

The engine output (top3, all_groups, anchors, ...) is returned untouched under "engine_result".
"""
import json
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from agentic_rashi_correlation.config import RashiAgentConfig, get_llm
from agentic_rashi_correlation.critic import TOP_N, anchor_signs, critique, top_groups_payload
from agentic_rashi_correlation.guardrails import add_disclaimer, validate_output
from agentic_rashi_correlation.memory import ALGORITHM, RashiMemoryStore
from agentic_rashi_correlation.profiles import PROFILES, profile_context, top_terms
from agentic_rashi_correlation.schemas import AgentState
from agentic_rashi_correlation.tools import build_tool_registry
from rashi_correlation_engine import run_rashi_correlation_analysis

TITLE = "Rashi Correlation"
TOOL = "run_rashi_correlation_analysis"
GRAPH_STEP_LIMIT = 40

SUMMARIZER_SYSTEM = """You are a medical-astrology EXPLAINER.
You receive a deterministic result from the "Rashi Correlation" engine: Rule 1 source evidence
matched against the organs of the 6th house rashi (P6) and the ascendant rashi (P7), merged
into organ-system groups.
Write a clear, cautious 3-5 sentence summary for a user.

HARD RULES:
- ONLY mention groups, diseases and organs that appear in the provided data.
- Do NOT invent new terms.
- Do NOT contradict the ranking.
- Say WHY the top group ranked first (cite its anchors, priorities and organs).
- If the critic flagged low support or ambiguity, mention it honestly.
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
        return f"{TITLE} found no organ groups to rank for this chart."
    lead = top[0]
    text = f"{TITLE} ranks {lead['label']} first (score {lead['score']})"
    if lead.get("anchors"):
        text += f", matched against {', '.join(lead['anchors'])}"
    if lead.get("priorities"):
        text += f" using {', '.join(lead['priorities'])} evidence"
    if lead.get("organs"):
        text += f", linked to {', '.join(lead['organs'])}"
    text += "."
    rest = [g["label"] for g in top[1:4]]
    if rest:
        text += f" Next in the ranking: {', '.join(rest)}."
    return text


class RashiCorrelationAgent:
    def __init__(self, config: RashiAgentConfig | None = None):
        self.cfg = config or RashiAgentConfig()
        self.live = self.cfg.llm.is_live
        self.llm = get_llm(self.cfg.llm)
        self.tools = build_tool_registry()
        self.memory = RashiMemoryStore.shared() if self.cfg.enable_memory else None
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
        # A re-score only helps when there is another profile to try. _node_score counts
        # every re-score, so max_replan_attempts really is the limit.
        if (
            len(state.profile_results) > 1
            and state.confidence < self.cfg.confidence_threshold
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
        state.validation = {"ok": not issues, "issues": issues}
        if issues:
            raise ValueError(f"Input validation failed: {issues}")
        return state

    def _node_score(self, state: AgentState) -> AgentState:
        # Re-entering after the critic means this is a re-score: count it.
        if state.critique is not None:
            state.replan_count += 1

        profile_names = list(PROFILES.keys()) if self.cfg.enable_multi_profile else ["clinical"]

        # The engine is deterministic, so a re-score reuses the results already computed
        # and only rotates the chosen profile.
        if not state.profile_results:
            results = {}
            for name in profile_names:
                with profile_context(name):
                    results[name] = self.tools.call(
                        TOOL,
                        chart_data=state.chart_data,
                        rule1_result=state.rule1_result,
                        gender=state.gender,
                    )
            state.profile_results = results

        results = state.profile_results
        if state.replan_count > 0 and len(profile_names) > 1:
            state.chosen_profile = profile_names[state.replan_count % len(profile_names)]
        else:
            state.chosen_profile = self._pick_stable_profile(results)

        state.engine_result = results[state.chosen_profile]
        return state

    def _pick_stable_profile(self, results: Dict[str, Dict]) -> str:
        """Prefer the profile whose top-3 groups overlap the most with the others.
        Ties keep "clinical" (the first one)."""
        if len(results) <= 1:
            return next(iter(results))
        tops = {name: set(top_terms(r, 3)) for name, r in results.items()}
        scores = {}
        for name, own in tops.items():
            scores[name] = sum(len(own & other) for n, other in tops.items() if n != name)
        return max(scores, key=scores.get)

    def _node_critique(self, state: AgentState) -> AgentState:
        c = critique(self.llm, state.engine_result or {}, state.gender or "", use_llm=self.live)
        state.critique = c.model_dump()
        state.confidence = c.confidence

        # If the critic suggests a profile switch and we have budget, switch
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

    def _payload(self, state: AgentState) -> Dict[str, Any]:
        result = state.engine_result or {}
        return {
            "top": top_groups_payload(result, TOP_N),
            "anchor_signs": anchor_signs(result),
            "fallback_note": result.get("fallback_note") or "",
            "critic_notes": (state.critique or {}).get("notes", ""),
            "critic_issues": (state.critique or {}).get("issues", []),
            "gender": state.gender,
            "profile_used": state.chosen_profile,
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
        payload = state.summary_payload or {}
        top = payload.get("top", [])
        report = validate_output(
            summary=state.summary or "",
            allowed_terms=[
                *(g["label"] for g in top),
                *(o for g in top for o in g["organs"]),
                *(d for g in top for d in g["diseases"]),
            ],
            gender=state.gender or "",
            strict=self.cfg.strict_guardrails,
        )
        state.guardrail_report = report.model_dump()
        if not report.valid:
            # The model wrote something that is not in the data: keep the facts, drop the prose.
            state.summary = deterministic_summary(top)
            state.llm_used = False
        return state

    def _node_finalize(self, state: AgentState) -> AgentState:
        engine_result = state.engine_result or {}
        mem_snapshot = None
        if self.memory:
            key = RashiMemoryStore.chart_key(state.chart_data, state.user_id, ALGORITHM)
            top_scores = {g["label"]: g["score"] for g in (engine_result.get("top3") or [])[:TOP_N]}
            self.memory.remember(key, {
                "top_scores": top_scores,
                "profile": state.chosen_profile,
                "confidence": state.confidence,
            })
            mem_snapshot = {"key": key, "trend": self.memory.trend(key)}
        state.memory_snapshot = mem_snapshot

        state.final = {
            "summary": add_disclaimer(state.summary or ""),
            "profile_used": state.chosen_profile,
            "profile_scores": {
                name: top_terms(r, 3) for name, r in state.profile_results.items()
            },
            "confidence": state.confidence,
            "critique": state.critique,
            "guardrail": state.guardrail_report,
            "memory": state.memory_snapshot,
            "llm_used": state.llm_used,
            "llm_error": state.llm_error,
            "engine_result": engine_result,
            # Clinical-profile ranking, exposed for the existing regression test.
            # It does not alter scores.
            "baseline_top3": (state.profile_results.get("clinical") or engine_result).get("top3") or [],
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
        out = self.graph.invoke(initial, config={"recursion_limit": GRAPH_STEP_LIMIT})
        final = out.get("final") if isinstance(out, dict) else getattr(out, "final", None)
        if not final:
            raise RuntimeError("Rashi agent graph finished without a result")
        return final


# ── Convenience function ────────────────────────────────
def run_agentic_rashi_correlation(chart_data, rule1_result, gender=None,
                                  user_id=None, config=None) -> Dict[str, Any]:
    """Agentic Rashi Correlation. If the agent itself fails for any reason, the plain engine
    result is returned so a chart is never lost because of the agent."""
    try:
        agent = RashiCorrelationAgent(config)
        return agent.invoke(chart_data, rule1_result, gender, user_id)
    except Exception as exc:
        print(f"[Agentic] {TITLE} agent failed ({type(exc).__name__}: {exc}) - using the plain engine")
        with profile_context("clinical"):       # takes the lock, so no thread is mid-swap
            plain = run_rashi_correlation_analysis(chart_data, rule1_result, gender=gender)
        return {"engine_result": plain, "agentic_error": f"{type(exc).__name__}: {exc}"}