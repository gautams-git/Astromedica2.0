"""Rule-based critic checks. An LLM agreeing with another LLM is not treated as proof."""

from __future__ import annotations

from agentic_rashi_correlation.critic import rule_based_audit


# Independently labelled critic scenarios. expected_issue is a substring, or None
# when the critic should stay quiet about that fault.
SCENARIOS = (
    {
        "name": "empty",
        "result": {"top3": [], "fallback_note": ""},
        "expect_issue": "no groups",
        "expect_flag": True,
    },
    {
        "name": "single_priority",
        "result": {"top3": [{"label": "Head", "score": 10, "supporting_priorities": ["P1"], "matched_organs": ["Head"]}], "fallback_note": ""},
        "expect_issue": "single priority",
        "expect_flag": True,
    },
    {
        "name": "clear_margin",
        "result": {
            "top3": [
                {"label": "Head", "score": 20, "supporting_priorities": ["P1", "P2"], "matched_organs": ["Head"]},
                {"label": "Stomach", "score": 10, "supporting_priorities": ["P1", "P4"], "matched_organs": ["Stomach"]},
            ],
            "fallback_note": "",
        },
        "expect_issue": None,
        "expect_flag": False,
    },
    {
        "name": "close_top_two",
        "result": {
            "top3": [
                {"label": "Head", "score": 10.0, "supporting_priorities": ["P1", "P2"], "matched_organs": ["Head"]},
                {"label": "Stomach", "score": 9.5, "supporting_priorities": ["P1", "P4"], "matched_organs": ["Stomach"]},
            ],
            "fallback_note": "",
        },
        "expect_issue": "ambiguous",
        "expect_flag": True,
    },
    {
        "name": "fallback_note",
        "result": {
            "top3": [{"label": "Head", "score": 8, "supporting_priorities": ["P1", "P2"], "matched_organs": ["Head"]}],
            "fallback_note": "11th house occupant used",
        },
        "expect_issue": "11th house",
        "expect_flag": True,
    },
)


def evaluate_scenarios() -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    rows = []
    for scenario in SCENARIOS:
        issues, confidence = rule_based_audit(scenario["result"])
        text = " ".join(issues).lower()
        flagged = bool(scenario["expect_issue"]) and scenario["expect_issue"] in text
        if scenario["expect_issue"] is None:
            flagged = any("ambiguous" in item.lower() or "single priority" in item.lower() or "no groups" in item.lower() for item in issues)
        expected = scenario["expect_flag"]
        if expected and flagged:
            kind = "tp"
        elif expected and not flagged:
            kind = "fn"
        elif not expected and flagged:
            kind = "fp"
        else:
            kind = "tn"
        counts[kind] += 1
        rows.append({
            "name": scenario["name"],
            "kind": kind,
            "issues": issues,
            "confidence": confidence,
            "expected_flag": expected,
        })
    return {"counts": counts, "rows": rows, "authority": "rule_based_audit only; no LLM critic vote is used"}
