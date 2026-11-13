"""Repository map used by the benchmark. Gaps are recorded, not invented."""

from __future__ import annotations

# Track C (LLM-selected weights) is implemented only for Rashi Correlation.
# The other four agents can select a named profile, but they do not expose an
# LLM weight-selection mechanism that this benchmark is allowed to extend.

ALGORITHM_MAP = (
    {
        "algorithm": "HitList",
        "original": "hitlist_engine.run_hitlist_analysis",
        "agentic": "agentic.orchestrator.run_agentic_analysis",
        "weight_source": "HitList engine constants. The agent can A/B named profiles when multi-profile is on. Track B forces the default profile and disables the LLM.",
        "existing_tests": "tests/test_agents_regression_and_demo.py (shared agent regression). No dedicated HitList oracle in this repository.",
        "track_a": True,
        "track_b": True,
        "track_c": False,
        "track_c_reason": "No LLM weight-selection contract is implemented for HitList. Track C is left unavailable rather than invented.",
    },
    {
        "algorithm": "Rashi Correlation",
        "original": "rashi_correlation_engine.run_rashi_correlation_analysis",
        "agentic": "agentic_rashi_correlation.orchestrator.run_agentic_rashi_correlation",
        "weight_source": "rashi_correlation_engine.PRIORITY_WEIGHTS, LAYER_WEIGHTS, ANCHOR_WEIGHTS. The agent swaps those tables only inside agentic_rashi_correlation.profiles.weights_context. Named profiles: clinical, nakshatra_heavy, anchor_heavy. The critic may suggest one of those names.",
        "existing_tests": "tests/test_agentic_rashi_correlation.py and agentic_rashi_correlation/test_agentic_rashi_correlation.py",
        "track_a": True,
        "track_b": True,
        "track_c": True,
        "track_c_reason": "Benchmark-only constrained weight proposal. The engine formulas are not modified. Proposed weights are validated, then applied through the existing weights_context.",
    },
    {
        "algorithm": "Organ Truth",
        "original": "organ_truth_correlation_engine.run_organ_truth_correlation_analysis",
        "agentic": "agentic_organ.orchestrator.run_agentic_organ_analysis",
        "weight_source": "Organ Truth engine anchor and layer constants. The agent does not expose a weight-swap profile table.",
        "existing_tests": "Covered only indirectly by agent regression tests. No dedicated Organ Truth mathematical oracle was already in the repository.",
        "track_a": True,
        "track_b": True,
        "track_c": False,
        "track_c_reason": "No LLM weight-selection mechanism is exposed. Track C is not fabricated.",
    },
    {
        "algorithm": "Disease First",
        "original": "disease_first_logic.run_disease_first_logic",
        "agentic": "agentic_disease_first.orchestrator.run_agentic_disease_first",
        "weight_source": "disease_first_logic weight tables. agentic_disease_first.profiles can swap named profiles (clinical, planet_heavy, rashi_heavy) when AGENTIC_MULTI_PROFILE is on. Track B keeps the clinical profile.",
        "existing_tests": "tests/test_agents_regression_and_demo.py",
        "track_a": True,
        "track_b": True,
        "track_c": False,
        "track_c_reason": "The existing LLM critic audits output. It is not a validated free weight-selection mechanism. Track C is not added for this algorithm.",
    },
    {
        "algorithm": "Disease Compare",
        "original": "disease_compare_logic.run_disease_compare_logic",
        "agentic": "agentic_disease_compare.orchestrator.run_agentic_disease_compare",
        "weight_source": "disease_compare_logic weight tables. Named profiles follow the same pattern as Disease First. Track B keeps the default profile.",
        "existing_tests": "tests/test_agents_regression_and_demo.py",
        "track_a": True,
        "track_b": True,
        "track_c": False,
        "track_c_reason": "No separate LLM weight-selection path is added. The gap is reported.",
    },
)


def map_by_name() -> dict:
    return {row["algorithm"]: row for row in ALGORITHM_MAP}
