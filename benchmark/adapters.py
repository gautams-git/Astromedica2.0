"""Call the existing engines and agents. This module does not edit their formulas."""

from __future__ import annotations

import dataclasses
import time
import traceback
from copy import deepcopy
from typing import Any, Callable

from benchmark.architecture import map_by_name
from benchmark.normalize import confidence_from_agent, fallback_note, rankings_from_engine, unwrap_agentic
from benchmark.oracles import verify_rashi
from benchmark.rashi_weights import propose_weights

TRACK_C_GAP = (
    "Track C is not implemented for this algorithm. "
    "LLM weight selection is restricted to Rashi Correlation."
)


def _config_for(module_name: str, *, multi_profile: bool):
    import importlib
    module = importlib.import_module(module_name)
    config_cls = None
    for name in ("RashiAgentConfig", "AgentConfig"):
        if hasattr(module, name):
            config_cls = getattr(module, name)
            break
    if config_cls is None:
        return None
    # Build the LLM config from the same module when it defines LLMConfig,
    # otherwise from the field's default factory.
    fields = {item.name: item for item in dataclasses.fields(config_cls)}
    kwargs = {}
    if "llm" in fields:
        factory = fields["llm"].default_factory
        sample = factory() if factory is not dataclasses.MISSING else None
        llm_cls = type(sample)
        llm_fields = {item.name for item in dataclasses.fields(llm_cls)}
        llm_kwargs = {}
        if "provider" in llm_fields:
            llm_kwargs["provider"] = "stub"
        if "api_key" in llm_fields:
            llm_kwargs["api_key"] = None
        if "temperature" in llm_fields:
            llm_kwargs["temperature"] = 0.0
        kwargs["llm"] = llm_cls(**llm_kwargs)
    if "enable_memory" in fields:
        kwargs["enable_memory"] = False
    if "enable_multi_profile" in fields:
        kwargs["enable_multi_profile"] = multi_profile
    return config_cls(**kwargs)


def _call(fn: Callable, chart: dict, rule1: dict, gender: str | None, config=None) -> dict:
    if config is None:
        return fn(deepcopy(chart), deepcopy(rule1), gender=gender)
    return fn(deepcopy(chart), deepcopy(rule1), gender=gender, config=config)


def _pack(ok: bool, raw: dict | None, error: str | None, started: float, *, weights=None, weight_audit=None, verification=None) -> dict[str, Any]:
    engine, agent = unwrap_agentic(raw or {})
    return {
        "ok": ok,
        "error": error,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "rankings": rankings_from_engine(engine),
        "fallback_note": fallback_note(engine),
        "confidence": confidence_from_agent(agent),
        "agent": _public_agent(agent),
        "weights": weights,
        "weight_audit": weight_audit,
        "verification": verification,
        "raw": raw,
    }


def _public_agent(agent: dict | None) -> dict | None:
    if not agent:
        return None
    keep = (
        "summary", "profile_used", "profile_scores", "confidence", "critique",
        "guardrail", "llm_used", "llm_error", "agentic_error", "reruns",
    )
    public = {key: agent.get(key) for key in keep if key in agent}
    critique = public.get("critique") or {}
    if isinstance(critique, dict):
        public["stages"] = {
            "profile": agent.get("profile_used"),
            "critic_issues": critique.get("issues"),
            "critic_notes": critique.get("notes"),
            "suggested_profile": critique.get("suggested_profile"),
            "guardrail_valid": (agent.get("guardrail") or {}).get("valid") if isinstance(agent.get("guardrail"), dict) else None,
            "llm_used": agent.get("llm_used"),
        }
    return public


def _rashi_weights_snapshot() -> dict:
    import rashi_correlation_engine as rce
    return {
        "PRIORITY_WEIGHTS": dict(rce.PRIORITY_WEIGHTS),
        "LAYER_WEIGHTS": dict(rce.LAYER_WEIGHTS),
        "ANCHOR_WEIGHTS": dict(rce.ANCHOR_WEIGHTS),
    }


def run_track(algorithm: str, track: str, chart: dict, rule1: dict, gender: str | None) -> dict[str, Any]:
    spec = map_by_name()[algorithm]
    started = time.perf_counter()
    if track == "C" and not spec["track_c"]:
        return _pack(False, None, TRACK_C_GAP, started, weight_audit={"available": False, "reason": spec["track_c_reason"]})

    try:
        if algorithm == "HitList":
            return _hitlist(track, chart, rule1, gender, started)
        if algorithm == "Rashi Correlation":
            return _rashi(track, chart, rule1, gender, started)
        if algorithm == "Organ Truth":
            return _organ(track, chart, rule1, gender, started)
        if algorithm == "Disease First":
            return _disease_first(track, chart, rule1, gender, started)
        if algorithm == "Disease Compare":
            return _disease_compare(track, chart, rule1, gender, started)
        return _pack(False, None, f"unknown algorithm {algorithm}", started)
    except Exception as exc:
        return _pack(False, None, f"{type(exc).__name__}: {exc}", started, verification={"status": "not_run", "reason": traceback.format_exc()[-500:]})


def _hitlist(track: str, chart, rule1, gender, started):
    if track == "A":
        from hitlist_engine import run_hitlist_analysis
        raw = _call(run_hitlist_analysis, chart, rule1, gender)
        return _pack(True, raw, None, started, verification=_limited_trail(raw))
    from agentic.orchestrator import run_agentic_analysis
    config = _config_for("agentic.config", multi_profile=False)
    raw = _call(run_agentic_analysis, chart, rule1, gender, config=config)
    engine, _agent = unwrap_agentic(raw)
    return _pack(True, raw, None, started, weights="engine default / clinical profile", verification=_limited_trail(engine))


def _organ(track, chart, rule1, gender, started):
    if track == "A":
        from organ_truth_correlation_engine import run_organ_truth_correlation_analysis
        raw = _call(run_organ_truth_correlation_analysis, chart, rule1, gender)
        return _pack(True, raw, None, started, verification=_limited_trail(raw))
    from agentic_organ.orchestrator import run_agentic_organ_analysis
    config = _config_for("agentic_organ.config", multi_profile=False)
    raw = _call(run_agentic_organ_analysis, chart, rule1, gender, config=config)
    engine, _agent = unwrap_agentic(raw)
    return _pack(True, raw, None, started, weights="engine default", verification=_limited_trail(engine))


def _disease_first(track, chart, rule1, gender, started):
    if track == "A":
        from disease_first_logic import run_disease_first_logic
        raw = _call(run_disease_first_logic, chart, rule1, gender)
        return _pack(True, raw, None, started, verification=_limited_trail(raw))
    from agentic_disease_first.orchestrator import run_agentic_disease_first
    config = _config_for("agentic_disease_first.config", multi_profile=False)
    raw = _call(run_agentic_disease_first, chart, rule1, gender, config=config)
    engine, _agent = unwrap_agentic(raw)
    return _pack(True, raw, None, started, weights="clinical profile", verification=_limited_trail(engine))


def _disease_compare(track, chart, rule1, gender, started):
    if track == "A":
        from disease_compare_logic import run_disease_compare_logic
        raw = _call(run_disease_compare_logic, chart, rule1, gender)
        return _pack(True, raw, None, started, verification=_limited_trail(raw))
    from agentic_disease_compare.orchestrator import run_agentic_disease_compare
    config = _config_for("agentic_disease_compare.config", multi_profile=False)
    raw = _call(run_agentic_disease_compare, chart, rule1, gender, config=config)
    engine, _agent = unwrap_agentic(raw)
    return _pack(True, raw, None, started, weights="clinical profile", verification=_limited_trail(engine))


def _rashi(track: str, chart, rule1, gender, started):
    from rashi_correlation_engine import run_rashi_correlation_analysis
    from agentic_rashi_correlation.orchestrator import run_agentic_rashi_correlation
    from agentic_rashi_correlation.profiles import weights_context

    if track == "A":
        raw = _call(run_rashi_correlation_analysis, chart, rule1, gender)
        return _pack(
            True, raw, None, started,
            weights=_rashi_weights_snapshot(),
            verification=verify_rashi(chart, rule1, gender, raw),
        )

    if track == "B":
        config = _config_for("agentic_rashi_correlation.config", multi_profile=False)
        raw = _call(run_agentic_rashi_correlation, chart, rule1, gender, config=config)
        engine, _agent = unwrap_agentic(raw)
        return _pack(
            True, raw, None, started,
            weights=_rashi_weights_snapshot(),
            weight_audit={"llm_selected": False, "profile": "clinical"},
            verification=verify_rashi(chart, rule1, gender, engine),
        )

    proposal = propose_weights(_chart_summary(chart, rule1, gender))
    validation = proposal.get("validation") or {}
    if not validation.get("valid") or not validation.get("weights"):
        return _pack(
            False, None,
            "LLM weight proposal was not applied: " + "; ".join(validation.get("errors") or ["no valid weights"]),
            started,
            weights=_rashi_weights_snapshot(),
            weight_audit=proposal,
            verification={"status": "not_run", "reason": "Invalid or unavailable weight proposal was not executed."},
        )

    with weights_context(validation["weights"]):
        engine = run_rashi_correlation_analysis(deepcopy(chart), deepcopy(rule1), gender=gender)
        config = _config_for("agentic_rashi_correlation.config", multi_profile=False)
        agent_raw = run_agentic_rashi_correlation(deepcopy(chart), deepcopy(rule1), gender=gender, config=config)
        verification = verify_rashi(chart, rule1, gender, engine)
        applied_weights = _rashi_weights_snapshot()
    agent_engine, agent_meta = unwrap_agentic(agent_raw)
    # The score of record is the engine executed under the validated weights.
    # The agent run is kept to show that orchestration still returns that engine result.
    raw = {"engine_result": engine, "agent_under_proposed_weights": agent_meta, "agent_engine_top": [
        item.get("label") for item in (agent_engine.get("top3") or [])
    ]}
    audit = dict(proposal)
    audit["ablation"] = _ablate(chart, rule1, gender, validation.get("changed") or [])
    return _pack(
        True, raw, None, started,
        weights=applied_weights,
        weight_audit=audit,
        verification=verification,
    )


def _chart_summary(chart: dict, rule1: dict, gender: str | None) -> dict:
    planets = []
    for planet in chart.get("planets") or []:
        if not isinstance(planet, dict):
            continue
        sign = planet.get("sign") if isinstance(planet.get("sign"), dict) else {}
        nakshatra = planet.get("nakshatra") if isinstance(planet.get("nakshatra"), dict) else {}
        planets.append({
            "name": planet.get("name"),
            "house": planet.get("house"),
            "sign": sign.get("name"),
            "nakshatra": nakshatra.get("name"),
        })
    return {
        "gender": gender,
        "ascendant": (chart.get("ascendant") or {}).get("name") if isinstance(chart.get("ascendant"), dict) else None,
        "sixth_house_sign": (rule1 or {}).get("sixth_house_sign"),
        "planets": planets,
    }


def _ablate(chart, rule1, gender, changes: list[dict]) -> list[dict]:
    """One-at-a-time weight changes, so a score move can be traced to a key."""
    if not changes:
        return []
    from rashi_correlation_engine import run_rashi_correlation_analysis
    from agentic_rashi_correlation.profiles import weights_context
    from benchmark.rashi_weights import baseline_tables

    baseline = baseline_tables()
    base_engine = run_rashi_correlation_analysis(deepcopy(chart), deepcopy(rule1), gender=gender)
    base_top = rankings_from_engine(base_engine)[:3]
    rows = []
    for change in changes:
        trial = {name: dict(table) for name, table in baseline.items()}
        trial[change["table"]][change["key"]] = change["proposed"]
        with weights_context(trial):
            engine = run_rashi_correlation_analysis(deepcopy(chart), deepcopy(rule1), gender=gender)
        top = rankings_from_engine(engine)[:3]
        rows.append({
            "table": change["table"],
            "key": change["key"],
            "from": change["baseline"],
            "to": change["proposed"],
            "baseline_top": base_top,
            "changed_top": top,
            "top_label_changed": [item["label"] for item in base_top] != [item["label"] for item in top],
            "top_score_delta": None if not base_top or not top or base_top[0]["score"] is None or top[0]["score"] is None else round(top[0]["score"] - base_top[0]["score"], 4),
        })
    return rows


def _limited_trail(engine: dict | None) -> dict:
    if not isinstance(engine, dict):
        return {"status": "limited", "reason": "No engine object was returned."}
    rows = engine.get("all_groups") or engine.get("all_organs") or engine.get("all_diseases") or []
    if not isinstance(rows, list) or not rows:
        return {"status": "limited", "reason": "No ranked rows with trails were exposed. This is not scored as a pass."}
    issues = []
    checked = 0
    for item in rows:
        trail = item.get("trail")
        if not isinstance(trail, list) or not trail:
            continue
        weights = [step.get("weight") for step in trail if isinstance(step, dict) and step.get("weight") is not None]
        if not weights or item.get("score") is None:
            continue
        checked += 1
        total = round(sum(float(value) for value in weights), 2)
        if abs(total - float(item["score"])) > 0.05:
            issues.append({"label": item.get("label") or item.get("organ") or item.get("disease"), "score": item.get("score"), "trail_sum": total})
    if checked == 0:
        return {"status": "limited", "reason": "Trails do not expose numeric weights, so the score was not reconstructed."}
    if issues:
        return {"status": "mismatch", "reason": "Exposed trail weights do not add up to the reported score.", "issues": issues}
    return {"status": "passed", "reason": "Exposed trail weights add up to the reported score.", "rows_checked": checked}
