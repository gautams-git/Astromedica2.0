"""Disease agent regression (against the REAL disease_first_logic) + demo script smoke tests.

Run from the project root: pytest tests/. The sample tables from demo/show_engines_vs_agents.py are
installed through monkeypatch, so every test leaves the real tables exactly as it found them.
"""
import copy
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "demo"))
import show_engines_vs_agents as demo  # noqa: E402


@pytest.fixture(autouse=True)
def sample_tables(monkeypatch):
    demo.install_fixture(monkeypatch.setattr, monkeypatch.setitem)
    monkeypatch.setenv("LLM_PROVIDER", "stub")
    for name in ("GEMINI_API_KEY", "AGENT_MEMORY_PATH", "AGENT_MEMORY_PATH_RASHI",
                 "AGENTIC_MULTI_PROFILE", "AGENTIC_RASHI_MULTI_PROFILE"):
        monkeypatch.delenv(name, raising=False)


def _args():
    return copy.deepcopy(demo.CHART), copy.deepcopy(demo.RULE1)


def _disease_tables():
    import disease_first_logic as dfl
    return {n: dict(getattr(dfl, n)) for n in ("SOURCE_WEIGHTS", "PLANET_LAYER_WEIGHTS", "RASHI_LAYER_WEIGHTS")}


def test_disease_agent_engine_result_equals_real_plain_engine():
    from disease_first_logic import run_disease_first_logic
    from agentic_disease_first import run_agentic_disease_first
    chart, rule1 = _args()
    out = run_agentic_disease_first(chart, rule1, gender=demo.GENDER)
    assert "agentic_error" not in out and out["profile_used"] == "clinical"
    assert out["engine_result"] == run_disease_first_logic(*_args(), gender=demo.GENDER)
    assert out["engine_result"]["top_diseases"]


def test_both_agents_unwrap_to_the_shape_the_app_reads():
    import jataka_logic as jl
    from disease_first_logic import run_disease_first_logic
    from rashi_correlation_engine import run_rashi_correlation_analysis
    from agentic_disease_first import run_agentic_disease_first
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    d = jl._unwrap_agentic(run_agentic_disease_first(*_args(), gender=demo.GENDER))
    r = jl._unwrap_agentic(run_agentic_rashi_correlation(*_args(), gender=demo.GENDER))
    assert d["top_diseases"] and r["top3"] and r["all_groups"]
    assert {k: v for k, v in d.items() if k != "agentic"} == run_disease_first_logic(*_args(), gender=demo.GENDER)
    assert {k: v for k, v in r.items() if k != "agentic"} == run_rashi_correlation_analysis(*_args(), gender=demo.GENDER)
    assert set(d["agentic"]) == set(r["agentic"])         # same output contract for both agents


def test_agents_use_independent_tables_and_locks_with_multi_profile_on(monkeypatch):
    monkeypatch.setenv("AGENTIC_MULTI_PROFILE", "true")
    monkeypatch.setenv("AGENTIC_RASHI_MULTI_PROFILE", "true")
    import rashi_correlation_engine as rce
    from agentic_disease_first import run_agentic_disease_first
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    before_d = _disease_tables()
    before_r = {n: dict(getattr(rce, n)) for n in ("PRIORITY_WEIGHTS", "LAYER_WEIGHTS", "ANCHOR_WEIGHTS")}
    d = run_agentic_disease_first(*_args(), gender=demo.GENDER)
    r = run_agentic_rashi_correlation(*_args(), gender=demo.GENDER)
    assert set(d["profile_scores"]) == {"clinical", "planet_heavy", "rashi_heavy"}
    assert set(r["profile_scores"]) == {"clinical", "nakshatra_heavy", "anchor_heavy"}
    assert _disease_tables() == before_d
    assert {n: dict(getattr(rce, n)) for n in before_r} == before_r


def test_threads_mixing_both_agents_never_see_foreign_weights(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    monkeypatch.setenv("AGENTIC_MULTI_PROFILE", "true")
    monkeypatch.setenv("AGENTIC_RASHI_MULTI_PROFILE", "true")
    from agentic_disease_first import run_agentic_disease_first
    from agentic_rashi_correlation import run_agentic_rashi_correlation
    exp_d = run_agentic_disease_first(*_args(), gender=demo.GENDER, user_id="s")
    exp_r = run_agentic_rashi_correlation(*_args(), gender=demo.GENDER, user_id="s")

    def job(i):
        fn = run_agentic_disease_first if i % 2 else run_agentic_rashi_correlation
        return i % 2, fn(*_args(), gender=demo.GENDER, user_id=f"t{i}")

    with ThreadPoolExecutor(10) as pool:
        for is_disease, out in pool.map(job, range(20)):
            exp = exp_d if is_disease else exp_r
            assert out["engine_result"] == exp["engine_result"]
            assert out["profile_scores"] == exp["profile_scores"]


def _run_demo(*extra):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([ROOT, os.environ.get("PYTHONPATH", "")])}
    env.pop("GEMINI_API_KEY", None)
    return subprocess.run([sys.executable, os.path.join(ROOT, "demo", "show_engines_vs_agents.py"), "--fixture", *extra],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("extra", [(), ("--multi-profile",), ("--only", "rashi")])
def test_demo_script_runs_and_reports_identical_engine_output(extra):
    proc = _run_demo(*extra)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    n_algos = 1 if "--only" in extra else 2
    assert out.count("engine_result identical to the plain engine output: True") == n_algos
    for node in ("validate", "score", "critique", "summarize", "guardrail", "finalize"):
        assert f"[{node}" in out
    assert "RASHI CORRELATION" in out and ("DISEASE FIRST LOGIC" in out) == (n_algos == 2)
