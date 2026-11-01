"""Tool registry — thin, typed wrappers around hitlist_engine functions."""
from typing import Any, Callable, Dict

from agentic.schemas import (
    ChartInput, GenderCheckInput, NakshatraTermsInput,
    PlanetTermsInput, RashiTermsInput,
)

# Import your existing functions — NOT modified
from hitlist_engine import (
    run_hitlist_analysis,
    get_planet_terms,
    get_nakshatra_terms,
    get_rashi_terms,
    is_term_allowed,
)
from term_to_systems import TERM_TO_SYSTEMS


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


def _run_hitlist_analysis_tool(chart_data, rule1_result, gender=None):
    """Deterministic scoring tool."""
    return run_hitlist_analysis(chart_data, rule1_result, gender=gender)


def _get_planet_terms_tool(planet_name):
    return get_planet_terms(planet_name)


def _get_nakshatra_terms_tool(nakshatra_name, pada):
    return get_nakshatra_terms(nakshatra_name, pada)


def _get_rashi_terms_tool(rashi_name):
    return get_rashi_terms(rashi_name)


def _gender_filter_tool(term, gender=None):
    return is_term_allowed(term, gender)


def _lookup_systems_tool(term):
    return TERM_TO_SYSTEMS.get(term, [])


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        "run_hitlist_analysis",
        _run_hitlist_analysis_tool,
        ChartInput,
        "Score disease systems via 6th-house priority planets / nakshatras / rashis. "
        "Returns top4 + all_groups + trails. Deterministic.",
    )
    reg.register(
        "get_planet_terms",
        _get_planet_terms_tool,
        PlanetTermsInput,
        "Return raw disease terms for a planet.",
    )
    reg.register(
        "get_nakshatra_terms",
        _get_nakshatra_terms_tool,
        NakshatraTermsInput,
        "Return disease terms for a nakshatra + pada.",
    )
    reg.register(
        "get_rashi_terms",
        _get_rashi_terms_tool,
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
        "lookup_systems",
        _lookup_systems_tool,
        PlanetTermsInput,  # reuse single-string schema
        "Map a single term to systems in TERM_TO_SYSTEMS.",
    )
    return reg