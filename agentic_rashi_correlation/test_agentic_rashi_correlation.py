"""Tests for agentic_rashi_correlation (run from the project root: pytest tests/).

The engine's knowledge tables are replaced by a small fixture for the duration of each test, so
the assertions do not depend on the contents of knowledge_base / term_to_systems.
"""
import copy
import inspect
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import rashi_correlation_engine as rce

# ── Fixture data ────────────────────────────────────────
TERM_TO_SYSTEMS = {
    "Liver disorders": ["digestive"], "Stomach": ["digestive"], "Jaundice": ["digestive"],
    "Joint pain": ["musculoskeletal"], "Knees": ["musculoskeletal"], "Bones": ["musculoskeletal"],
    "Kidney trouble": ["urinary"], "Kidneys": ["urinary"],
    "Skin disease": ["dermatological"], "Skin": ["dermatological"],
    "Fever": ["systemic"],
    "Nervous disorders": ["nervous"], "Head": ["nervous"],
}
MERGE_GROUPS = {
    "G_DIG": {"label": "Digestive", "systems": ["digestive"]},
    "G_MSK": {"label": "Musculoskeletal", "systems": ["musculoskeletal"]},
    "G_URI": {"label": "Urinary", "systems": ["urinary"]},
    "G_NER": {"label": "Nervous", "systems": ["nervous"]},
    "G_DER": {"label": "Dermatological", "systems": ["dermatological"]},
    "G_SYS": {"label": "Systemic", "systems": ["systemic"]},
}
PLANET_DISEASES = {
    "Mars": "Liver disorders, Fever",
    "Saturn": "Joint pain, Skin disease",
    "Mercury": "Nervous disorders",
    "Venus": "Kidney trouble",
}
NAKSHATRA_DISEASES = {
    "Ashwini": {"1, 2": ["Liver disorders", "Jaundice"], "3, 4": ["Nervous disorders"]},
    "Pushya": {"1, 2, 3": ["Joint pain"], "4": ["Kidney trouble"]},
    "Ashlesha": {"1, 2, 3, 4": ["Nervous disorders"]},
}
RASHI_ORGANS = {"Karka": "Stomach, Kidneys", "Mesha": "Head, Knees"}

CHART = {
    "gender": "M",
    "ascendant": {"name": "Mesha"},
    "planets": [
        {"name": "Mars", "house": 6, "sign": {"name": "Mesha"}, "nakshatra": {"name": "Ashwini", "pada": 1}},
        {"name": "Saturn", "house": 8, "sign": {"name": "Karka"}, "nakshatra": {"name": "Pushya", "pada": 2}},
        {"name": "Mercury", "house": 3, "sign": {"name": "Mesha"}, "nakshatra": {"name": "Ashlesha", "pada": 1}},
        {"name": "Venus", "house": 2, "sign": {"name": "Karka"}, "nakshatra": {"name": "Pushya", "pada": 4}},
    ],
}
RULE1 = {
    "rogkaraka_1": [{"name": "Mars"}],
    "rogkaraka_3": {"name": "Saturn"},
    "drishti_on_6th": [{"planet": "Mercury"}],
    "sixth_house_lord": "Venus",
    "ascendant_lord": "Mars",
    "sixth_house_sign": "Karka",
}
GENDER = "M"


@pytest.fixture(autouse=True)
def engine_fixture(monkeypatch):
    monkeypatch.setattr(rce, "TERM_TO_SYSTEMS", TERM_TO_SYSTEMS)
    monkeypatch.setattr(rce, "MERGE_GROUPS", MERGE_GROUPS)
    monkeypatch.setattr(rce, "GROUP_LOOKUP", rce._group_lookup())
    monkeypatch.setattr(rce, "PLANET_DISEASES", PLANET_DISEASES)
    monkeypatch.setattr(rce, "NAKSHATRA_DISEASES", NAKSHATRA_DISEASES)
    monkeypatch.setattr(rce, "RASHI_ORGANS", RASHI_ORGANS)
    stub = types.ModuleType("eleventh_lord_confirmation")
    stub.get_eleventh_lord_confirmation_source = lambda chart, rule1, gender: {}
    stub.find_disease_confirmation = lambda disease, source: None
    stub.confirmation_boost = lambda score: 0.0
    monkeypatch.setitem(sys.modules, "eleventh_lord_confirmation", stub)
    # Never call a real model, never write a memory file.
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_PATH", raising=False)
    monkeypatch.delenv("AGENT_MEMORY_PATH_RASHI", raising=False)
    monkeypatch.delenv("AGENTIC_RASHI_MULTI_PROFILE", raising=False)


def plain():
    return rce.run_rashi_correlation_analysis(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER)


def tables():
    return {n: dict(getattr(rce, n)) for n in ("PRIORITY_WEIGHTS", "LAYER_WEIGHTS", "ANCHOR_WEIGHTS")}


# ── A. import ───────────────────────────────────────────
def test_a_import():
    import agentic_rashi_correlation
    assert callable(agentic_rashi_correlation.run_agentic_rashi_correlation)
    assert agentic_rashi_correlation.RashiCorrelationAgent


# ── B. plain engine ─────────────────────────────────────
def test_b_plain_engine_works_and_is_deterministic():
    a, b = plain(), plain()
    assert a == b
    assert a["top3"] and a["all_groups"]
    assert a["top3"] == a["all_groups"][:3]


# ── C. agentic run ──────────────────────────────────────
def test_c_agentic_run():
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    out = run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER)
    assert "agentic_error" not in out
    assert out["profile_used"] == "clinical"
    assert out["llm_used"] is False
    assert "not a medical diagnosis" in out["summary"].lower()
    assert out["guardrail"]["valid"] is True
    assert 0.3 <= out["confidence"] <= 0.95
    assert out["memory"]["key"]
    json.dumps(out)                       # the pipeline json.dumps the result


# ── D. integration: the pipeline calls the wrapper ──────
def test_d_run_logic_uses_wrapper(monkeypatch, tmp_path):
    import jataka_logic as jl
    src = inspect.getsource(jl.run_logic)
    assert "_run_rashi_correlation(chart, rule1_result, gender_code)" in src
    assert "run_rashi_correlation_analysis(" not in src

    for name in ("agentic", "organ_truth_correlation_engine", "disease_first_logic", "disease_compare_logic",
                 "common_findings_engine", "related_house_severity_engine", "dasha_chat_priority",
                 "algorithm_comparison", "chat_ai_context_builder"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["agentic"].run_agentic_analysis = lambda *a, **k: {"top4": []}
    sys.modules["organ_truth_correlation_engine"].run_organ_truth_correlation_analysis = lambda *a, **k: {}
    sys.modules["disease_first_logic"].build_current_dasha_priority_sources = lambda *a, **k: {}
    sys.modules["common_findings_engine"].build_common_findings = lambda *a, **k: {}
    sys.modules["related_house_severity_engine"].build_related_house_severity = lambda *a, **k: {}
    sys.modules["dasha_chat_priority"].build_dasha_chat_priority = lambda *a, **k: {}
    sys.modules["algorithm_comparison"].write_algorithm_comparison = lambda *a, **k: {}
    sys.modules["chat_ai_context_builder"].build_chat_ai_context = lambda *a, **k: None
    monkeypatch.setattr(jl, "_run_disease_first", lambda *a, **k: {})
    monkeypatch.setattr(jl, "_run_disease_compare", lambda *a, **k: {})
    monkeypatch.setattr(jl, "apply_rule_1", lambda chart: copy.deepcopy(RULE1))
    monkeypatch.setattr(jl, "analyse_dasha", lambda *a, **k: {})
    monkeypatch.setattr(jl, "complete_risk_analysis", lambda **k: {})
    monkeypatch.setattr(jl, "generate_1_year_health_forecast", lambda *a, **k: [])
    monkeypatch.setattr(jl, "apply_bhadravathi_overrides", lambda *a, **k: None)
    monkeypatch.setattr(jl, "apply_person2_overrides", lambda *a, **k: None)

    calls = []
    sentinel = {"top3": [], "all_groups": [], "sentinel": True}
    monkeypatch.setattr(jl, "_run_rashi_correlation", lambda *a: calls.append(a) or sentinel)

    folder = tmp_path / "patients" / "T1"
    folder.mkdir(parents=True)
    (folder / "raw_chart.json").write_text(json.dumps({
        "gender": "Male",
        "planet_position": {"data": {"planet_position": [
            {"name": "Ascendant", "rasi": {"name": "Mesha", "id": 0}, "degree": 5.0, "position": 1},
            {"name": "Sun", "rasi": {"name": "Simha"}, "degree": 10.0, "position": 5},
        ]}},
    }))
    (folder / "raw_dasha.json").write_text(json.dumps({"dasha_data": {"data": {}}}))
    monkeypatch.chdir(tmp_path)
    processed = jl.run_logic("T1")

    assert len(calls) == 1
    chart_arg, rule1_arg, gender_arg = calls[0]
    assert chart_arg["planets"] and rule1_arg == RULE1
    assert processed["rashi_correlation"] is sentinel


# ── E. fallback ─────────────────────────────────────────
def test_e_fallback_when_agent_import_fails(monkeypatch, capsys):
    import jataka_logic as jl
    monkeypatch.setitem(sys.modules, "agentic_rashi_correlation", None)   # makes the import raise
    monkeypatch.setenv("AGENTIC_DISEASE_LOGIC", "true")
    out = jl._run_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER)
    assert out == plain()
    assert "agentic" not in out
    assert "Rashi agent unavailable" in capsys.readouterr().out


def test_e_fallback_when_agent_crashes(monkeypatch):
    import jataka_logic as jl
    import agentic_rashi_correlation.orchestrator as orch

    def boom(self, *a, **k):
        raise RuntimeError("graph exploded")
    monkeypatch.setattr(orch.RashiCorrelationAgent, "invoke", boom)
    monkeypatch.setenv("AGENTIC_DISEASE_LOGIC", "true")
    out = jl._run_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER)
    # engine output stays on top, the failure is recorded under "agentic"
    assert {k: v for k, v in out.items() if k != "agentic"} == plain()
    assert "graph exploded" in out["agentic"]["agentic_error"]


def test_e_bad_input_falls_back_to_plain_engine():
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    out = run_agentic_rashi_correlation({"planets": []}, copy.deepcopy(RULE1), gender=GENDER)
    assert "Input validation failed" in out["agentic_error"]
    assert out["engine_result"] is not None


# ── F. feature flag (AGENTIC_DISEASE_LOGIC, same as Disease First / Compare) ──
@pytest.mark.parametrize("value,expected", [
    (None, True), ("true", True), ("1", True), ("yes", True),
    ("false", False), ("0", False), ("no", False), ("off", False),
])
def test_f_feature_flag(monkeypatch, value, expected):
    import jataka_logic as jl
    monkeypatch.delenv("AGENTIC_DISEASE_LOGIC", raising=False) if value is None \
        else monkeypatch.setenv("AGENTIC_DISEASE_LOGIC", value)
    assert jl._agentic_disease_logic_on() is expected
    assert not hasattr(jl, "_agentic_rashi_on")        # no separate Rashi flag

    result = jl._run_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER)
    assert ("agentic" in result) is expected
    assert result["top3"] and result["all_groups"]
    if not expected:
        assert result == plain()


# ── G. profile isolation ────────────────────────────────
def test_g_profiles_are_immutable_and_restore_tables():
    from agentic_rashi_correlation import profiles
    base = tables()
    assert {n: dict(t) for n, t in profiles.PROFILES["clinical"].items()} == base
    assert profiles.PROFILES["nakshatra_heavy"]["LAYER_WEIGHTS"]["nakshatra"] == 1.875
    assert profiles.PROFILES["nakshatra_heavy"]["LAYER_WEIGHTS"]["planet"] == base["LAYER_WEIGHTS"]["planet"]
    assert profiles.PROFILES["anchor_heavy"]["ANCHOR_WEIGHTS"] == {"P6": 4.5, "P7": 2.25}
    assert profiles.PROFILES["anchor_heavy"]["LAYER_WEIGHTS"] == base["LAYER_WEIGHTS"]
    with pytest.raises(TypeError):
        profiles.PROFILES["clinical"]["LAYER_WEIGHTS"]["nakshatra"] = 99
    with pytest.raises(TypeError):
        profiles.PROFILES["clinical"] = {}

    for name in profiles.PROFILES:
        with profiles.profile_context(name):
            if name == "nakshatra_heavy":
                assert rce.LAYER_WEIGHTS["nakshatra"] == 1.875
            if name == "anchor_heavy":
                assert rce.ANCHOR_WEIGHTS["P6"] == 4.5
        assert tables() == base
    assert {n: dict(t) for n, t in profiles.PROFILES["clinical"].items()} == base


def test_g_tables_restored_after_engine_error():
    from agentic_rashi_correlation import profiles
    base = tables()
    with pytest.raises(RuntimeError):
        with profiles.profile_context("nakshatra_heavy"):
            assert rce.LAYER_WEIGHTS["nakshatra"] == 1.875
            raise RuntimeError("engine failed")
    assert tables() == base


def test_g_profiles_change_scores_but_clinical_matches_plain():
    from agentic_rashi_correlation import profiles
    results = {}
    for name in profiles.PROFILES:
        with profiles.profile_context(name):
            results[name] = plain()
    assert results["clinical"] == plain()
    assert results["nakshatra_heavy"]["top3"][0]["score"] != results["clinical"]["top3"][0]["score"]
    assert results["anchor_heavy"]["top3"][0]["score"] != results["clinical"]["top3"][0]["score"]


def test_g_concurrent_requests_do_not_leak_weights():
    from agentic_rashi_correlation import profiles
    base = tables()
    names = list(profiles.PROFILES)
    expected = {}
    for n in names:
        with profiles.profile_context(n):
            expected[n] = plain()

    barrier = threading.Barrier(12)

    def work(i):
        name = names[i % len(names)]
        barrier.wait()
        with profiles.profile_context(name):
            time.sleep(0.005)              # widen the window in which another thread could interfere
            return name, plain()

    with ThreadPoolExecutor(12) as pool:
        for name, result in pool.map(work, range(12)):
            assert result == expected[name], f"{name} saw foreign weights"
    assert tables() == base


def test_g_concurrent_agents_multi_profile(monkeypatch):
    monkeypatch.setenv("AGENTIC_RASHI_MULTI_PROFILE", "true")
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    base = tables()
    serial = run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER, user_id="serial")
    assert set(serial["profile_scores"]) == {"clinical", "nakshatra_heavy", "anchor_heavy"}

    with ThreadPoolExecutor(8) as pool:
        outs = list(pool.map(
            lambda i: run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1),
                                                    gender=GENDER, user_id=f"u{i}"),
            range(8)))
    for out in outs:
        assert out["engine_result"] == serial["engine_result"]
        assert out["profile_scores"] == serial["profile_scores"]
    assert tables() == base
    # "baseline_top3" is always what the plain engine returns
    assert [g["label"] for g in serial["baseline_top3"]] == [g["label"] for g in plain()["top3"]]


# ── H. disease agent regression ─────────────────────────
def test_h_disease_agent_unaffected():
    import disease_first_logic as dfl
    from agentic_disease_first import profiles as dprofiles
    before = {n: dict(getattr(dfl, n)) for n in dprofiles.WEIGHT_TABLES}
    from agentic_rashi_correlation import profiles
    for name in profiles.PROFILES:
        with profiles.profile_context(name):
            plain()
    assert {n: dict(getattr(dfl, n)) for n in dprofiles.WEIGHT_TABLES} == before
    assert set(dprofiles.PROFILES) == {"clinical", "planet_heavy", "rashi_heavy"}
    assert profiles.PROFILE_LOCK is not dprofiles.PROFILE_LOCK      # independent locks, independent tables
    from agentic_disease_first.memory import MemoryStore
    from agentic_rashi_correlation.memory import RashiMemoryStore
    assert RashiMemoryStore.shared() is not MemoryStore.shared()
    assert RashiMemoryStore.shared() is RashiMemoryStore.shared()


# ── I. output contract ──────────────────────────────────
def test_i_output_contract_matches_plain_engine():
    import jataka_logic as jl
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    expected = plain()
    raw = run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER)
    assert raw["engine_result"] == expected                    # untouched

    unwrapped = jl._unwrap_agentic(raw)
    assert unwrapped["top3"] == expected["top3"]
    assert unwrapped["all_groups"] == expected["all_groups"]
    assert {k: v for k, v in unwrapped.items() if k != "agentic"} == expected
    assert set(unwrapped["agentic"]) >= {"summary", "critique", "guardrail", "memory", "profile_used", "confidence"}
    assert "engine_result" not in unwrapped["agentic"]


# ── Critic / guardrail / live-LLM paths ─────────────────
def test_critic_rule_based_audit():
    from agentic_rashi_correlation.critic import rule_based_audit
    assert rule_based_audit({"top3": []}) == (["no groups were scored"], 0.3)
    result = {
        "top3": [
            {"label": "A", "score": 10.0, "supporting_priorities": ["P1"], "matched_organs": []},
            {"label": "B", "score": 9.5, "supporting_priorities": ["P1", "P2"], "matched_organs": ["x"]},
        ],
        "fallback_note": "11th house used",
    }
    issues, conf = rule_based_audit(result)
    assert "'A' is backed by a single priority" in issues
    assert any("ambiguous" in i for i in issues)
    assert "11th house used" in issues
    assert "'A' has no matched organs" in issues
    assert conf == pytest.approx(0.55)      # 0.9 - 0.05 - 0.1 - 0.1 - 0.1, floor 0.3


def test_critic_payload_field_names():
    from agentic_rashi_correlation.critic import top_groups_payload
    top = top_groups_payload(plain())
    assert top and set(top[0]) == {"label", "score", "anchors", "priorities", "organs", "diseases"}


def test_guardrail_blocks_invented_terms_and_gender(monkeypatch):
    from agentic_rashi_correlation import guardrails
    monkeypatch.setattr(guardrails, "_VOCABULARY", ["Kidney trouble", "Stomach", "Bones", "Uterus"])
    ok = guardrails.validate_output("Digestive ranks first via the stomach.", ["Digestive", "Stomach"], "M")
    assert ok.valid and not ok.hallucinated_terms
    bad = guardrails.validate_output("Bones are also involved.", ["Digestive", "Stomach"], "M")
    assert not bad.valid and bad.hallucinated_terms == ["Bones"]
    lax = guardrails.validate_output("Bones are also involved.", ["Digestive"], "M", strict=False)
    assert lax.valid and lax.hallucinated_terms == ["Bones"]
    assert "not a medical diagnosis" in guardrails.add_disclaimer("x").lower()


def _live_agent(responses):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from agentic_rashi_correlation.orchestrator import RashiCorrelationAgent
    agent = RashiCorrelationAgent()
    agent.llm, agent.live = FakeListChatModel(responses=responses), True
    return agent


CRITIC_JSON = json.dumps({"issues": ["llm says hello"], "confidence": 0.8, "notes": "fine", "suggested_profile": None})


def test_live_llm_summary_used_when_it_passes_guardrail():
    agent = _live_agent([CRITIC_JSON, "Digestive ranks first, linked to the Stomach. Stay well."])
    out = agent.invoke(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER, "live1")
    assert out["llm_used"] is True and out["guardrail"]["valid"] is True
    assert out["summary"].startswith("Digestive ranks first")
    assert "llm says hello" in out["critique"]["issues"]
    assert out["engine_result"] == plain()


def test_live_llm_invented_term_falls_back_to_deterministic_summary():
    agent = _live_agent([CRITIC_JSON, "Digestive ranks first, but mostly the Liver disorders and Fever."])
    # make "Fever" part of the vocabulary but absent from the top groups' data
    from agentic_rashi_correlation import guardrails
    guardrails._VOCABULARY.append("Fever")
    try:
        out = agent.invoke(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER, "live2")
    finally:
        guardrails._VOCABULARY.remove("Fever")
    top_labels = [g["label"] for g in plain()["top3"]]
    assert out["llm_used"] is False
    assert out["guardrail"]["valid"] is False and "Fever" in out["guardrail"]["hallucinated_terms"]
    assert out["summary"].startswith("Rashi Correlation ranks " + top_labels[0])


def test_live_llm_garbage_critic_falls_back_to_rule_based():
    agent = _live_agent(["not json at all", "Digestive ranks first. Stay well."])
    out = agent.invoke(copy.deepcopy(CHART), copy.deepcopy(RULE1), GENDER, "live3")
    assert any(i.startswith("llm_critic_failed") for i in out["critique"]["issues"])
    assert out["engine_result"] == plain()


def test_memory_trend_per_chart_and_algorithm():
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    a = run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER, user_id="trend")
    b = run_agentic_rashi_correlation(copy.deepcopy(CHART), copy.deepcopy(RULE1), gender=GENDER, user_id="trend")
    assert a["memory"]["key"] == b["memory"]["key"]
    assert b["memory"]["trend"]["previous_runs"] >= 2
    assert all(v == 0 for v in b["memory"]["trend"]["deltas"].values())


def test_memory_files_never_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MEMORY_PATH", str(tmp_path / "mem.json"))
    from agentic_rashi_correlation.memory import RashiMemoryStore
    assert RashiMemoryStore().path == str(tmp_path / "mem_rashi.json")
    monkeypatch.setenv("AGENT_MEMORY_PATH_RASHI", str(tmp_path / "r.json"))
    assert RashiMemoryStore().path == str(tmp_path / "r.json")
