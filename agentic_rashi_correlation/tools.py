"""Tool registry for the Rashi Correlation agent: thin, typed wrappers over the engine functions.

The registry class and the input schemas are the disease agent's; only the tools are new.
"""
from agentic_disease_first.tools import ToolRegistry

from agentic_rashi_correlation.schemas import (
    ChartInput,
    GenderCheckInput,
    NakshatraTermsInput,
    PlanetTermsInput,
    RashiTermsInput,
)

# Existing engine functions: NOT modified
from rashi_correlation_engine import (
    run_rashi_correlation_analysis,
    _planet_terms,
    _nakshatra_terms,
    _rashi_terms,
    _term_allowed,
)


def _run_rashi_tool(chart_data, rule1_result, gender=None):
    """Deterministic scoring tool."""
    return run_rashi_correlation_analysis(chart_data, rule1_result, gender=gender)


def _nakshatra_terms_tool(nakshatra_name, pada):
    return _nakshatra_terms(nakshatra_name, pada)


def _planet_terms_tool(planet_name):
    return _planet_terms(planet_name)


def _rashi_terms_tool(rashi_name):
    return _rashi_terms(rashi_name)


def _gender_filter_tool(term, gender=None):
    return _term_allowed(term, gender)


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        "run_rashi_correlation_analysis",
        _run_rashi_tool,
        ChartInput,
        "Match Rule 1 source evidence against the 6th-house and ascendant rashi organs. "
        "Returns top3 + all_groups. Deterministic.",
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
    return reg
