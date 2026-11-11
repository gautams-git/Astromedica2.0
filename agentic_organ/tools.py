"""Typed wrapper around organ_truth_correlation_engine."""
from typing import Any, Callable, Dict

from agentic_organ.schemas import ChartInput

from organ_truth_correlation_engine import (
    run_organ_truth_correlation_analysis,
)


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name, fn, schema, description):
        self._tools[name] = {"fn": fn, "schema": schema, "description": description}

    def call(self, name, **kwargs):
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        validated = self._tools[name]["schema"](**kwargs)
        return self._tools[name]["fn"](**validated.model_dump())

    def names(self):
        return list(self._tools.keys())


def _run_engine_tool(chart_data, rule1_result, gender=None):
    return run_organ_truth_correlation_analysis(
        chart_data, rule1_result, gender=gender
    )


def build_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "run_organ_truth_analysis",
        _run_engine_tool,
        ChartInput,
        "Score organ ↔ disease correlations from 6th-house anchors, "
        "nakshatra diseases, and 11th-lord confirmation. Deterministic. "
        "Uses fixed engine weights.",
    )
    return reg