"""Tool registry — thin, typed wrappers around disease_compare_logic functions."""
from typing import Any, Callable, Dict

from agentic_disease_compare.schemas import (
    ChartInput, DiseaseOrgansInput, DiseasePairInput, GenderCheckInput, RashiTermsInput,
)

# Existing functions — NOT modified
from disease_compare_logic import (
    run_disease_compare_logic,
    _disease_organs,
    _is_very_close_disease,
)
from disease_first_logic import _rashi_terms, _term_allowed


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable, schema, description: str):
        self._tools[name] = {
            "fn": fn,
            "schema": schema,
            "description": description,
        }

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        schema = self._tools[name]["schema"]
        validated = schema(**kwargs)
        return self._tools[name]["fn"](**validated.model_dump())

    def describe(self) -> str:
        return "\n".join(
            f"- {n}: {t['description']}" for n, t in self._tools.items()
        )

    def names(self):
        return list(self._tools.keys())


def _run_disease_compare_tool(chart_data, rule1_result, gender=None):
    """Deterministic scoring tool."""
    return run_disease_compare_logic(chart_data, rule1_result, gender=gender)


def _compare_diseases_tool(source_disease, candidate_disease):
    is_close, reason, similarity = _is_very_close_disease(source_disease, candidate_disease)
    return {"is_close": is_close, "reason": reason, "similarity": similarity}


def _disease_organs_tool(disease, gender=None):
    return _disease_organs(disease, gender)


def _rashi_terms_tool(rashi_name):
    return _rashi_terms(rashi_name)


def _gender_filter_tool(term, gender=None):
    return _term_allowed(term, gender)


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        "run_disease_compare_logic",
        _run_disease_compare_tool,
        ChartInput,
        "Compare the Rule 1 source planets' nakshatra diseases with every known disease; only "
        "very-close wordings with shared organs are scored. Returns top_diseases + all_diseases "
        "with the source-disease matches. Deterministic.",
    )
    reg.register(
        "compare_diseases",
        _compare_diseases_tool,
        DiseasePairInput,
        "Tell whether two disease names are a very close match, with the reason and similarity.",
    )
    reg.register(
        "disease_organs",
        _disease_organs_tool,
        DiseaseOrgansInput,
        "Return the organs a nakshatra disease maps to (gender-filtered).",
    )
    reg.register(
        "get_rashi_terms",
        _rashi_terms_tool,
        RashiTermsInput,
        "Return organ terms for a rashi.",
    )
    reg.register(
        "is_term_allowed",
        _gender_filter_tool,
        GenderCheckInput,
        "Check if a term passes gender filtering.",
    )
    return reg
