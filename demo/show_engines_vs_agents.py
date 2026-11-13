#!/usr/bin/env python3
"""Show how the Rashi and Disease First agents process a chart, and compare each agent with its plain engine.

    python demo/show_engines_vs_agents.py --patient 001          # your real patient (patients/001/raw_*.json)
    python demo/show_engines_vs_agents.py --fixture              # built-in sample chart + sample tables
    ... --multi-profile   also try the alternative weight profiles (sets the two MULTI_PROFILE env vars)
    ... --json out.json   also write everything that was printed to a file

For each algorithm it prints: (1) the graph, node by node, as it runs; (2) the plain engine result;
(3) the agentic result; (4) a score-by-score comparison. Nothing here changes any project file or
any engine; with --patient nothing is written unless you pass --json.
--fixture REPLACES the knowledge tables with a tiny sample (in memory only) so it runs anywhere.
"""
import argparse, copy, json, os, sys, textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Sample data (used by --fixture and by the regression tests) ─────────────────────────
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
    "rogkaraka_1": [{"name": "Mars"}], "rogkaraka_3": {"name": "Saturn"},
    "drishti_on_6th": [{"planet": "Mercury"}], "sixth_house_lord": "Venus",
    "ascendant_lord": "Mars", "sixth_house_sign": "Karka",
}
GENDER = "M"
TERM_TO_SYSTEMS = {
    "Liver disorders": ["digestive"], "Stomach": ["digestive"], "Jaundice": ["digestive"], "Liver": ["digestive"],
    "Joint pain": ["musculoskeletal"], "Knees": ["musculoskeletal"], "Bones": ["musculoskeletal"],
    "Kidney trouble": ["urinary"], "Kidneys": ["urinary"],
    "Skin disease": ["dermatological"], "Skin": ["dermatological"], "Fever": ["systemic"],
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
    "Mars": "Liver disorders, Fever, Liver", "Saturn": "Joint pain, Skin disease, Bones",
    "Mercury": "Nervous disorders, Head", "Venus": "Kidney trouble, Kidneys",
}
NAKSHATRA_DISEASES = {
    "Ashwini": {"1, 2": ["Liver disorders", "Jaundice"], "3, 4": ["Nervous disorders"]},
    "Pushya": {"1, 2, 3": ["Joint pain"], "4": ["Kidney trouble"]},
    "Ashlesha": {"1, 2, 3, 4": ["Nervous disorders"]},
}
RASHI_ORGANS = {"Karka": "Stomach, Kidneys", "Mesha": "Head, Knees"}
ALL_ORGANS = ["Stomach", "Kidneys", "Head", "Knees", "Liver", "Bones"]
DISEASE_TO_ORGANS = {
    "Liver disorders": ["Liver", "Stomach"], "Jaundice": ["Liver", "Stomach"], "Joint pain": ["Knees", "Bones"],
    "Kidney trouble": ["Kidneys"], "Nervous disorders": ["Head"],
}


def install_fixture(set_attr=setattr, set_item=lambda d, k, v: d.__setitem__(k, v)):
    """Replace the knowledge tables of BOTH engines (and the guardrail vocabularies) with the sample.
    `set_attr` / `set_item` let tests pass pytest's monkeypatch so everything is undone afterwards."""
    import types
    import rashi_correlation_engine as rce
    import disease_first_logic as dfl
    from agentic_disease_first import guardrails as dg
    from agentic_rashi_correlation import guardrails as rg
    for name, val in (("TERM_TO_SYSTEMS", TERM_TO_SYSTEMS), ("MERGE_GROUPS", MERGE_GROUPS),
                      ("PLANET_DISEASES", PLANET_DISEASES), ("NAKSHATRA_DISEASES", NAKSHATRA_DISEASES),
                      ("RASHI_ORGANS", RASHI_ORGANS)):
        set_attr(rce, name, val)
    set_attr(rce, "GROUP_LOOKUP", rce._group_lookup())
    for name, val in (("PLANET_DISEASES", PLANET_DISEASES), ("NAKSHATRA_DISEASES", NAKSHATRA_DISEASES),
                      ("RASHI_ORGANS", RASHI_ORGANS), ("NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS", DISEASE_TO_ORGANS),
                      ("ALL_ORGANS_LOOKUP", {o.lower(): o for o in ALL_ORGANS})):
        set_attr(dfl, name, val)
    set_attr(rg, "_VOCABULARY", sorted(TERM_TO_SYSTEMS, key=len, reverse=True))
    set_attr(dg, "_DISEASES", sorted(DISEASE_TO_ORGANS, key=len, reverse=True))
    set_attr(dg, "_ORGANS", sorted(ALL_ORGANS, key=len, reverse=True))
    stub = types.ModuleType("eleventh_lord_confirmation")
    stub.get_eleventh_lord_confirmation_source = lambda chart, rule1, gender: {}
    stub.find_disease_confirmation = lambda disease, source: None
    stub.confirmation_boost = lambda score: 0.0
    set_item(sys.modules, "eleventh_lord_confirmation", stub)


# ── Algorithm adapters ──────────────────────────────────────────────────────────────────
def _algorithms():
    from rashi_correlation_engine import run_rashi_correlation_analysis
    from disease_first_logic import run_disease_first_logic
    from agentic_rashi_correlation import RashiCorrelationAgent, run_agentic_rashi_correlation
    from agentic_disease_first import DiseaseFirstAgent, run_agentic_disease_first
    return {
        "rashi": dict(
            title="RASHI CORRELATION", plain=run_rashi_correlation_analysis, agent=RashiCorrelationAgent,
            agentic=run_agentic_rashi_correlation, name_key="label", list_key="all_groups", top_key="top3",
            detail=lambda g: f"anchors={g.get('matched_anchors')} priorities={g.get('supporting_priorities')}"),
        "disease": dict(
            title="DISEASE FIRST LOGIC", plain=run_disease_first_logic, agent=DiseaseFirstAgent,
            agentic=run_agentic_disease_first, name_key="disease", list_key="all_diseases", top_key="top_diseases",
            detail=lambda d: f"priorities={d.get('source_priorities')} labels={d.get('source_labels')}"),
    }


def _short(text, n=110):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _describe(node, st, prev, A):
    """One line of plain English for what a graph node just did (read from the state it returned)."""
    nk, tk = A["name_key"], A["top_key"]
    if node == "validate":
        return f"input ok={st['validation']['ok']} issues={st['validation']['issues']}"
    if node == "score":
        profs = {p: [x[nk] for x in (r.get(tk) or [])[:3]] for p, r in st["profile_results"].items()}
        return f"engine ran for profiles {list(profs)}; chose '{st['chosen_profile']}'; top-3 per profile: {profs}"
    if node == "critique":
        c = st["critique"]
        return (f"confidence={st['confidence']} replans={st['replan_count']} issues={len(c['issues'])} "
                f"suggested_profile={c.get('suggested_profile')} :: {_short(c['notes'], 60)}")
    if node == "summarize":
        return f"llm_used={st['llm_used']} llm_error={st['llm_error']} :: {_short(st['summary'])}"
    if node == "guardrail":
        g = st["guardrail_report"]
        return f"valid={g['valid']} hallucinated={g['hallucinated_terms']} gender_violations={g['gender_violations']}"
    if node == "finalize":
        m = st.get("memory_snapshot") or {}
        return f"memory key={m.get('key')} previous_runs={(m.get('trend') or {}).get('previous_runs')}"
    return ""


def trace_agent(A, chart, rule1, gender, say):
    """Run the agent's own LangGraph step by step and narrate each node. Returns its final dict."""
    agent = A["agent"]()
    from agentic_disease_first.schemas import AgentState
    state = AgentState(chart_data=copy.deepcopy(chart), rule1_result=copy.deepcopy(rule1),
                       gender=gender or chart.get("gender"), user_id=None)
    say(f"  live LLM: {agent.live}   multi-profile: {agent.cfg.enable_multi_profile}   "
        f"threshold: {agent.cfg.confidence_threshold}   max re-scores: {agent.cfg.max_replan_attempts}")
    current, final = state.model_dump(), None
    for chunk in agent.graph.stream(state, config={"recursion_limit": 40}, stream_mode="updates"):
        for node, update in chunk.items():
            update = update if isinstance(update, dict) else update.model_dump()
            current = {**current, **update}          # an update holds only the fields the node changed
            say(f"  [{node:<9}] {_describe(node, current, None, A)}")
            final = current.get("final") or final
    return final


def table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = lambda r: "  " + "  ".join(str(c).ljust(w) for c, w in zip(r, widths))
    return "\n".join([line(headers), "  " + "  ".join("-" * w for w in widths)] + [line(r) for r in rows])


def run_one(key, A, chart, rule1, gender, say):
    say(f"\n{'=' * 100}\n{A['title']}\n{'=' * 100}")
    nk, lk = A["name_key"], A["list_key"]

    say("\n1) HOW THE AGENT PROCESSES THE CHART (its graph, node by node)")
    traced = trace_agent(A, chart, rule1, gender, say)

    plain = A["plain"](copy.deepcopy(chart), copy.deepcopy(rule1), gender=gender)
    say("\n2) PLAIN ENGINE RESULT (the deterministic engine called directly)")
    say(table([(i + 1, x[nk], x["score"], _short(A["detail"](x), 80)) for i, x in enumerate(plain[lk][:8])],
              ["#", nk, "score", "evidence"]))
    say(f"   total {nk}s scored: {len(plain[lk])}")

    prod = A["agentic"](copy.deepcopy(chart), copy.deepcopy(rule1), gender=gender)   # the exact call jataka_logic makes
    engine = prod.get("engine_result") or {}
    say("\n3) AGENTIC RESULT (what run_agentic_* returns; the engine output is under 'engine_result')")
    say(table([(i + 1, x[nk], x["score"], _short(A["detail"](x), 80)) for i, x in enumerate(engine.get(lk, [])[:8])],
              ["#", nk, "score", "evidence"]))
    if prod.get("agentic_error"):
        say(f"   !! agent fell back to the plain engine: {prod['agentic_error']}")
    say(f"   profile_used={prod.get('profile_used')}  profile_scores={prod.get('profile_scores')}")
    say(f"   confidence={prod.get('confidence')}  llm_used={prod.get('llm_used')}  llm_error={prod.get('llm_error')}")
    say(f"   critic issues: {(prod.get('critique') or {}).get('issues')}")
    say(f"   guardrail: {prod.get('guardrail')}")
    say(f"   memory: {prod.get('memory')}")
    say("   summary: " + textwrap.fill(prod.get("summary", ""), 96, subsequent_indent="            "))

    say("\n4) COMPARISON: plain engine vs agentic (same labels, score by score)")
    agentic_scores = {x[nk]: x["score"] for x in engine.get(lk, [])}
    rows, all_same = [], True
    for x in plain[lk][:8]:
        a = agentic_scores.get(x[nk])
        same = a == x["score"]
        all_same &= same
        rows.append((x[nk], x["score"], a, "-" if a is None else round(a - x["score"], 4), "same" if same else "DIFFERENT"))
    say(table(rows, [nk, "plain", "agentic", "diff", "verdict"]))
    identical = engine == plain
    say(f"   engine_result identical to the plain engine output: {identical}")
    say("   " + ("-> the agent adds explanation/audit around the engine; it did not change any score."
                 if identical else
                 f"-> scores differ because the agent chose profile '{prod.get('profile_used')}' (non-default weights)."))
    return {"plain": plain, "agentic": prod, "traced_final_matches_prod_engine": (traced or {}).get("engine_result") == engine,
            "identical": identical}


def load_patient(pid):
    import jataka_logic as jl
    from rules import apply_rule_1
    from knowledge_base import normalize_gender
    folder = os.path.join("patients", pid)
    raw_chart = json.load(open(os.path.join(folder, "raw_chart.json")))
    raw_dasha = json.load(open(os.path.join(folder, "raw_dasha.json")))
    chart = jl.build_chart_dict(raw_chart, raw_dasha)
    return chart, apply_rule_1(chart), normalize_gender(raw_chart.get("gender", "Unknown"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--patient"); src.add_argument("--fixture", action="store_true")
    ap.add_argument("--multi-profile", action="store_true"); ap.add_argument("--json")
    ap.add_argument("--only", choices=["rashi", "disease"])
    args = ap.parse_args()
    if args.multi_profile:
        os.environ["AGENTIC_RASHI_MULTI_PROFILE"] = os.environ["AGENTIC_MULTI_PROFILE"] = "true"

    if args.fixture:
        install_fixture()
        chart, rule1, gender = CHART, RULE1, GENDER
        print("SOURCE: built-in sample chart and sample knowledge tables (NOT real patient data)")
    else:
        chart, rule1, gender = load_patient(args.patient)
        print(f"SOURCE: patient {args.patient} (real raw_chart.json / raw_dasha.json)")
    print(f"INPUT: gender={gender} ascendant={(chart.get('ascendant') or {}).get('name')} planets={len(chart['planets'])}")
    print("       rule1 sources: P1=" + str([x.get("name") for x in rule1.get("rogkaraka_1", [])]) +
          f" P2={(rule1.get('rogkaraka_3') or {}).get('name')} P3={[x.get('planet') for x in rule1.get('drishti_on_6th', [])]}"
          f" P4={rule1.get('sixth_house_lord')} P5={rule1.get('ascendant_lord')}")
    print("       LLM:", "LIVE (GEMINI_API_KEY set)" if os.getenv("GEMINI_API_KEY") and os.getenv("LLM_PROVIDER", "gemini") == "gemini"
          else "none (rule-based critic + code-built summary)")

    out = {}
    for key, A in _algorithms().items():
        if args.only and key != args.only:
            continue
        out[key] = run_one(key, A, chart, rule1, gender, print)
    if args.json:
        json.dump(out, open(args.json, "w"), indent=2, default=str)
        print(f"\n[OK] full results written to {args.json}")


if __name__ == "__main__":
    main()
