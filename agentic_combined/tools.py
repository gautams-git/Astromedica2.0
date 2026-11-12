"""Tool registry - a thin, typed wrapper around the Combined Algo function."""
from typing import Any, Callable, Dict

from agentic_combined.schemas import ProcessedInput

# Existing function - NOT modified
from algorithm_comparison import build_combined_algo_summary

# The five algorithms the Combined Algo takes its votes from: (key in `processed`, list that holds its ranking)
ALGORITHM_INPUTS = {
    "System Priority (HitList)": ("hitlist", "top4"),
    "Rashi Priority Correlation": ("rashi_correlation", "top3"),
    "Organ Truth Correlation": ("organ_truth_correlation", "top_organs"),
    "Disease First Logic": ("disease_first_logic", "top_diseases"),
    "Disease Compare Logic": ("disease_compare_logic", "top_diseases"),
}


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, fn: Callable, schema, description: str):
        self._tools[name] = {"fn": fn, "schema": schema, "description": description}

    def call(self, name: str, **kwargs) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        validated = self._tools[name]["schema"](**kwargs)
        return self._tools[name]["fn"](**validated.model_dump())

    def describe(self) -> str:
        return "\n".join(f"- {n}: {t['description']}" for n, t in self._tools.items())

    def names(self):
        return list(self._tools.keys())


def _combined_tool(processed):
    """Deterministic scoring tool."""
    return build_combined_algo_summary(processed)


def algorithms_with_data(processed: Dict[str, Any]) -> Dict[str, bool]:
    """For every algorithm: does `processed` hold a non-empty ranking that the Combined Algo can vote from?"""
    out = {}
    for label, (key, field) in ALGORITHM_INPUTS.items():
        block = processed.get(key) or {}
        out[label] = bool(isinstance(block, dict) and block.get(field))
    return out


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "build_combined_algo_summary",
        _combined_tool,
        ProcessedInput,
        "Median consensus of the five algorithms: every organ gets a rank score from each "
        "algorithm, the median is its consensus score. Returns the top 3 organs with 3 "
        "diseases each. Deterministic.",
    )
    return reg
