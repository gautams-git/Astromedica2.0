from __future__ import annotations

"""
Combined Algo Summary v2 -- runs all 5 v2 engines and merges them with
MAGNITUDE-weighted consensus instead of the old ordinal rank_score
(100/85/70/55 by position). Each algorithm's contribution is normalized
against its own max score (score / max_score * 100), so a result that
crushed it in one algorithm isn't flattened to the same level as a result
that barely scraped into that algorithm's top list.
Fully isolated from algorithm_comparison.py; no chart-specific overrides
are applied anywhere in this module.
"""

from hitlist_engine_v2 import run_hitlist_analysis_v2
from disease_first_logic_v2 import run_disease_first_logic_v2
from disease_compare_logic_v2 import run_disease_compare_logic_v2
from rashi_correlation_engine_v2 import run_rashi_correlation_analysis_v2
from organ_truth_correlation_engine_v2 import run_organ_truth_correlation_analysis_v2

ALGORITHM_NAMES = {
    "hitlist": "System Priority (HitList) v2",
    "rashi": "Rashi Priority Correlation v2",
    "organ_truth": "Organ Truth Correlation v2",
    "disease_first": "Disease First Logic v2",
    "disease_compare": "Disease Compare Logic v2",
}

ORGAN_ALIASES = {
    "heart": "Heart / Blood", "blood": "Heart / Blood", "heart / blood": "Heart / Blood",
    "brain": "Brain / Nervous System", "head": "Brain / Nervous System", "mind": "Brain / Nervous System",
    "nervous system": "Brain / Nervous System", "brain / nervous system": "Brain / Nervous System",
    "flesh": "Skin / Flesh / Immunity", "skin": "Skin / Flesh / Immunity", "immunity": "Skin / Flesh / Immunity",
    "sensory": "Skin / Flesh / Immunity", "skin / immunity / sensory": "Skin / Flesh / Immunity",
    "skin / flesh / immunity": "Skin / Flesh / Immunity",
    "liver": "Digestive / Liver", "digestive": "Digestive / Liver", "abdomen": "Digestive / Liver",
    "stomach": "Digestive / Liver", "upper stomach": "Digestive / Liver", "digestive / liver": "Digestive / Liver",
    "lungs": "Respiratory / Lungs", "respiratory": "Respiratory / Lungs", "respiratory / lungs": "Respiratory / Lungs",
}

GROUP_MAIN_ORGAN = {
    "Heart / Blood": "Heart", "Brain / Nervous System": "Brain", "Skin / Flesh / Immunity": "Skin",
    "Digestive / Liver": "Liver", "Respiratory / Lungs": "Lungs",
}

BODY_REGION_TERMS = {
    "upper back", "back", "abdomen", "upper stomach", "stomach", "forehead",
    "face", "feet", "toes", "groins", "loins", "joints",
}

PREFERRED_MAIN_ORGANS = {
    "heart", "brain", "skin", "liver", "lungs", "kidney", "bones", "blood",
    "spleens", "lymphatic system", "flesh", "head", "mind", "spine",
}


def _normalize(value) -> str:
    return " ".join(str(value or "").strip().lower().replace("/", " / ").split())


def _canonical_organ(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    key = _normalize(text)
    if key in ORGAN_ALIASES:
        return ORGAN_ALIASES[key]
    for part in key.split(" / "):
        if part in ORGAN_ALIASES:
            return ORGAN_ALIASES[part]
    return text


def _display_organ(value) -> str:
    canonical = _canonical_organ(value)
    return GROUP_MAIN_ORGAN.get(canonical, canonical)


def _can_be_main_organ(value) -> bool:
    key = _normalize(value)
    if not key or key in BODY_REGION_TERMS:
        return False
    return key in PREFERRED_MAIN_ORGANS or value in GROUP_MAIN_ORGAN or " " not in key


def _split_label(label) -> list[str]:
    return [part.strip() for part in str(label or "").split("/") if part.strip()]


def _normalized_score(raw_score: float, max_score: float) -> float:
    if not max_score:
        return 0.0
    return round(min(100.0, (max(0.0, float(raw_score or 0)) / max_score) * 100.0), 2)


def build_combined_algo_summary_v2(chart_data: dict, rule1_result: dict, gender: str | None = None) -> dict:
    hitlist = run_hitlist_analysis_v2(chart_data, rule1_result, gender)
    rashi = run_rashi_correlation_analysis_v2(chart_data, rule1_result, gender)
    organ_truth = run_organ_truth_correlation_analysis_v2(chart_data, rule1_result, gender)
    disease_first = run_disease_first_logic_v2(chart_data, rule1_result, gender)
    disease_compare = run_disease_compare_logic_v2(chart_data, rule1_result, gender)

    algorithm_keys = list(ALGORITHM_NAMES)
    organs: dict[str, dict] = {}

    def ensure_organ(organ):
        name = _canonical_organ(organ)
        if not name:
            return None
        return organs.setdefault(name, {
            "organ": name,
            "algorithm_scores": {key: 0.0 for key in algorithm_keys},
            "related_organs": {},
            "diseases": {},
        })

    def add_evidence(organ, algorithm, normalized_score):
        item = ensure_organ(organ)
        if not item:
            return
        item["algorithm_scores"][algorithm] = max(item["algorithm_scores"][algorithm], normalized_score)

    def add_related(organ_list, score):
        clean = [str(o or "").strip() for o in organ_list if str(o or "").strip()]
        for organ in clean:
            main = ensure_organ(organ)
            if not main:
                continue
            for related in clean:
                related_name = _display_organ(related)
                if not related_name or _normalize(related_name) == _normalize(_display_organ(main["organ"])):
                    continue
                entry = main["related_organs"].setdefault(related_name, {"organ": related_name, "score": 0.0})
                entry["score"] += float(score or 1)

    def add_diseases(organ_list, disease_list, source, score):
        clean_organs = [str(o or "").strip() for o in organ_list if str(o or "").strip()]
        clean_diseases = [str(d or "").strip() for d in disease_list if str(d or "").strip()]
        for organ in clean_organs:
            item = ensure_organ(organ)
            if not item:
                continue
            for disease in clean_diseases:
                entry = item["diseases"].setdefault(disease, {"disease": disease, "score": 0.0, "sources": set()})
                entry["score"] += float(score or 0)
                entry["sources"].add(source)

    # -- HitList v2: groups carry a "/"-joined label of organs/systems --
    hl_max = max((g["score"] for g in hitlist["all_groups"]), default=0)
    for item in hitlist["top4"]:
        organs_in_label = _split_label(item.get("label"))
        norm = _normalized_score(item.get("score", 0), hl_max)
        for organ in organs_in_label:
            add_evidence(organ, "hitlist", norm)
        add_related(organs_in_label, item.get("score", 0))
        hl_diseases = [
            step.get("term") for step in item.get("trail", []) or []
            if "HL2" in str(step.get("source", ""))
        ]
        add_diseases(organs_in_label, hl_diseases, ALGORITHM_NAMES["hitlist"], item.get("score", 0))

    # -- Rashi Correlation v2 --
    rc_max = max((g["score"] for g in rashi["all_groups"]), default=0)
    for item in rashi["top3"]:
        organs_in_label = _split_label(item.get("label"))
        norm = _normalized_score(item.get("score", 0), rc_max)
        for organ in organs_in_label:
            add_evidence(organ, "rashi", norm)
        add_related(organs_in_label, item.get("score", 0))
        rc_diseases = [
            step.get("evidence_term") for step in item.get("trail", []) or []
            if step.get("layer") == "nakshatra"
        ]
        add_diseases(organs_in_label, rc_diseases, ALGORITHM_NAMES["rashi"], item.get("score", 0))

    # -- Organ Truth v2 --
    ot_max = max((o["score"] for o in organ_truth["all_organs"]), default=0)
    for item in organ_truth["top_organs"]:
        norm = _normalized_score(item.get("score", 0), ot_max)
        add_evidence(item.get("organ", ""), "organ_truth", norm)
        ot_diseases = [d.get("disease") for d in item.get("related_diseases", []) or []]
        add_diseases([item.get("organ", "")], ot_diseases, ALGORITHM_NAMES["organ_truth"], item.get("score", 0))

    # -- Disease First v2 --
    df_max = max((d["score"] for d in disease_first["all_diseases"]), default=0)
    for item in disease_first["top_diseases"]:
        item_organs = [o.get("organ") for o in item.get("supporting_organs", [])]
        norm = _normalized_score(item.get("score", 0), df_max)
        for organ in item_organs:
            add_evidence(organ, "disease_first", norm)
        add_related(item_organs, item.get("score", 0))
        add_diseases(item_organs, [item.get("disease")], ALGORITHM_NAMES["disease_first"], item.get("score", 0))

    # -- Disease Compare v2 --
    dc_scores = [d["score"] for d in disease_compare["all_diseases"]] if disease_compare.get("all_diseases") else [d["score"] for d in disease_compare["top_diseases"]]
    dc_max = max(dc_scores, default=0)
    for item in disease_compare["top_diseases"]:
        item_organs = sorted({
            organ
            for match in item.get("source_disease_matches", [])
            for organ in match.get("shared_organs", [])
        })
        norm = _normalized_score(item.get("score", 0), dc_max)
        for organ in item_organs:
            add_evidence(organ, "disease_compare", norm)
        add_related(item_organs, item.get("score", 0))
        add_diseases(item_organs, [item.get("disease")], ALGORITHM_NAMES["disease_compare"], item.get("score", 0))

    ranked = []
    for item in organs.values():
        scores = list(item["algorithm_scores"].values())
        sorted_scores = sorted(scores)
        median_score = sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0
        evidence_count = sum(score > 0 for score in scores)
        main_organ = _display_organ(item["organ"])
        if not evidence_count or not _can_be_main_organ(main_organ):
            continue
        ranked.append({
            "organ": main_organ,
            "canonical_group": item["organ"],
            "consensus_score": round(median_score, 2),
            "algorithm_support_count": evidence_count,
            "algorithm_scores": {ALGORITHM_NAMES[k]: round(v, 2) for k, v in item["algorithm_scores"].items()},
            "supporting_algorithms": [ALGORITHM_NAMES[k] for k, v in item["algorithm_scores"].items() if v > 0],
            "related_organs": [
                {"organ": r["organ"], "score": round(r["score"], 2)}
                for r in sorted(item["related_organs"].values(), key=lambda v: v["score"], reverse=True)[:2]
            ],
            "top_diseases": [
                {"disease": d["disease"], "score": round(d["score"], 2), "sources": sorted(d["sources"])}
                for d in sorted(item["diseases"].values(), key=lambda v: (len(v["sources"]), v["score"]), reverse=True)[:3]
            ],
        })

    ranked.sort(
        key=lambda item: (item["consensus_score"], item["algorithm_support_count"], sum(item["algorithm_scores"].values())),
        reverse=True,
    )

    return {
        "method": "Magnitude-weighted consensus (score normalized against each algorithm's own max, not ordinal rank)",
        "top_organs": ranked[:3],
        "all_organs": ranked,
        "raw_outputs": {
            "hitlist": hitlist,
            "rashi_correlation": rashi,
            "organ_truth_correlation": organ_truth,
            "disease_first_logic": disease_first,
            "disease_compare_logic": disease_compare,
        },
    }
