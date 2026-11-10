"""Tool registry — thin, typed wrappers around disease_first_logic functions."""
from typing import Any, Callable, Dict

from agentic_disease_first.schemas import (
    ChartInput, DiseaseOrgansInput, GenderCheckInput,
    NakshatraTermsInput, PlanetTermsInput, RashiTermsInput,
)

# Existing functions — NOT modified
from disease_first_logic import (
    run_disease_first_logic,
    _nakshatra_terms,
    _planet_terms,
    _rashi_terms,
    _term_allowed,
)
from rashi_truth_review import NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS


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


def _run_disease_first_tool(chart_data, rule1_result, gender=None):
    """Deterministic scoring tool."""
    return run_disease_first_logic(chart_data, rule1_result, gender=gender)


def _nakshatra_terms_tool(nakshatra_name, pada):
    return _nakshatra_terms(nakshatra_name, pada)


def _planet_terms_tool(planet_name):
    return _planet_terms(planet_name)


def _rashi_terms_tool(rashi_name):
    return _rashi_terms(rashi_name)


def _gender_filter_tool(term, gender=None):
    return _term_allowed(term, gender)


def _disease_organs_tool(disease):
    return list(NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS.get(disease, []))


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        "run_disease_first_logic",
        _run_disease_first_tool,
        ChartInput,
        "Score diseases from the Rule 1 source planets' nakshatra diseases, backed by "
        "planet-organ and rashi-organ support. Returns top_diseases + all_diseases. Deterministic.",
    )
    reg.register(
        "get_nakshatra_terms",
        _nakshatra_terms_tool,
        NakshatraTermsInput,
        "Return disease terms for a nakshatra + pada.",
    )
    reg.register(
        "get_planet_terms",
        _planet_terms_tool,
        PlanetTermsInput,
        "Return raw disease/organ terms for a planet.",
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
    reg.register(
        "disease_organs",
        _disease_organs_tool,
        DiseaseOrgansInput,
        "Return the organs a nakshatra disease maps to.",
    )
    return reg
