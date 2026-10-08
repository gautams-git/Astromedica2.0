import json
import math
import os
from datetime import datetime


ALGORITHM_NAMES = {
    "hitlist": "System Priority (HitList)",
    "rashi": "Rashi Priority Correlation",
    "organ_truth": "Organ Truth Correlation",
    "disease_first": "Disease First Logic",
    "disease_compare": "Disease Compare Logic",
}

ORGAN_ALIASES = {
    "heart": "Heart / Blood",
    "blood": "Heart / Blood",
    "heart / blood": "Heart / Blood",
    "brain": "Brain / Nervous System",
    "head": "Brain / Nervous System",
    "mind": "Brain / Nervous System",
    "nervous system": "Brain / Nervous System",
    "brain / nervous system": "Brain / Nervous System",
    "flesh": "Skin / Flesh / Immunity",
    "skin": "Skin / Flesh / Immunity",
    "immunity": "Skin / Flesh / Immunity",
    "sensory": "Skin / Flesh / Immunity",
    "skin / immunity / sensory": "Skin / Flesh / Immunity",
    "skin / flesh / immunity": "Skin / Flesh / Immunity",
    "liver": "Digestive / Liver",
    "digestive": "Digestive / Liver",
    "abdomen": "Digestive / Liver",
    "stomach": "Digestive / Liver",
    "upper stomach": "Digestive / Liver",
    "digestive / liver": "Digestive / Liver",
    "lungs": "Respiratory / Lungs",
    "respiratory": "Respiratory / Lungs",
    "respiratory / lungs": "Respiratory / Lungs",
}

GROUP_MAIN_ORGAN = {
    "Heart / Blood": "Heart",
    "Brain / Nervous System": "Brain",
    "Skin / Flesh / Immunity": "Skin",
    "Digestive / Liver": "Liver",
    "Respiratory / Lungs": "Lungs",
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


def _rank_score(raw_score, rank) -> float:
    normalized_rank = max(1, int(rank or 1))
    rank_score = max(35, 115 - (normalized_rank * 15))
    score_boost = min(10, math.log10(abs(float(raw_score or 0)) + 1) * 2)
    return min(100, rank_score + score_boost)


def _algorithm_sections(processed: dict) -> dict:
    hitlist = processed.get("hitlist", {}) or {}
    rashi = processed.get("rashi_correlation", {}) or {}
    organ_truth = processed.get("organ_truth_correlation", {}) or {}
    disease_first = processed.get("disease_first_logic", {}) or {}
    disease_compare = processed.get("disease_compare_logic", {}) or {}

    return {
        ALGORITHM_NAMES["hitlist"]: {
            "top_organs_and_systems": hitlist.get("top4", []),
            "top_diseases": [
                disease
                for item in hitlist.get("top4", [])
                for disease in item.get("diseases", item.get("matched_diseases", []))
            ],
            "raw_output": hitlist,
        },
        ALGORITHM_NAMES["rashi"]: {
            "top_organs_and_systems": rashi.get("top3", []),
            "top_diseases": [
                disease
                for item in rashi.get("top3", [])
                for disease in item.get("matched_diseases", [])
            ],
            "raw_output": rashi,
        },
        ALGORITHM_NAMES["organ_truth"]: {
            "top_organs": organ_truth.get("top_organs", []),
            "top_diseases": organ_truth.get("top_diseases", []),
            "raw_output": organ_truth,
        },
        ALGORITHM_NAMES["disease_first"]: {
            "top_organs": [
                organ
                for item in disease_first.get("top_diseases", [])
                for organ in item.get("supporting_organs", [])
            ],
            "top_diseases": disease_first.get("top_diseases", []),
            "raw_output": disease_first,
        },
        ALGORITHM_NAMES["disease_compare"]: {
            "top_organs": [
                organ
                for item in disease_compare.get("top_diseases", [])
                for organ in item.get("supporting_organs", [])
            ],
            "top_diseases": disease_compare.get("top_diseases", []),
            "raw_output": disease_compare,
        },
    }


def build_combined_algo_summary(processed: dict) -> dict:
    algorithm_keys = list(ALGORITHM_NAMES)
    organs = {}

    def ensure_organ(organ):
        name = _canonical_organ(organ)
        if not name:
            return None
        return organs.setdefault(name, {
            "organ": name,
            "algorithm_scores": {key: 0 for key in algorithm_keys},
            "related_organs": {},
            "diseases": {},
        })

    def add_evidence(organ, algorithm, raw_score, rank):
        item = ensure_organ(organ)
        if not item:
            return
        item["algorithm_scores"][algorithm] = max(
            item["algorithm_scores"][algorithm],
            _rank_score(raw_score, rank),
        )

    def add_related(organ_list, source, score):
        clean = [str(item or "").strip() for item in organ_list if str(item or "").strip()]
        for organ in clean:
            main = ensure_organ(organ)
            if not main:
                continue
            for related in clean:
                related_name = _display_organ(related)
                if not related_name or _normalize(related_name) == _normalize(_display_organ(main["organ"])):
                    continue
                entry = main["related_organs"].setdefault(
                    related_name, {"organ": related_name, "score": 0, "sources": set()}
                )
                entry["score"] += float(score or 1)
                entry["sources"].add(source)

    def add_diseases(organ_list, disease_list, source, score):
        for organ in organ_list:
            item = ensure_organ(organ)
            if not item:
                continue
            for disease in disease_list:
                disease_name = str(disease or "").strip()
                if not disease_name:
                    continue
                entry = item["diseases"].setdefault(
                    disease_name, {"disease": disease_name, "score": 0, "sources": set()}
                )
                entry["score"] += float(score or 0)
                entry["sources"].add(source)

    for rank, item in enumerate((processed.get("hitlist", {}) or {}).get("top4", []), 1):
        trail = item.get("trail", []) or []
        organ_trail = [
            step.get("term") for step in trail
            if "nakshatra" not in str(step.get("source", "")).lower()
        ]
        disease_trail = [
            step.get("term") for step in trail
            if "nakshatra" in str(step.get("source", "")).lower()
        ]
        item_organs = (
            _split_label(item.get("label"))
            + item.get("organs", [])
            + item.get("matched_organs", [])
            + organ_trail
        )
        for organ in item_organs:
            add_evidence(organ, "hitlist", item.get("score", 0), rank)
        add_related(item_organs, ALGORITHM_NAMES["hitlist"], item.get("score", 0))
        add_diseases(
            item_organs,
            item.get("diseases", item.get("matched_diseases", item.get("related_diseases", disease_trail))),
            ALGORITHM_NAMES["hitlist"],
            item.get("score", 0),
        )

    for rank, item in enumerate((processed.get("rashi_correlation", {}) or {}).get("top3", []), 1):
        item_organs = _split_label(item.get("label")) + item.get("matched_organs", [])
        for organ in item_organs:
            add_evidence(organ, "rashi", item.get("score", 0), rank)
        add_related(item_organs, ALGORITHM_NAMES["rashi"], item.get("score", 0))
        add_diseases(
            item_organs,
            item.get("matched_diseases", item.get("related_diseases", [])),
            ALGORITHM_NAMES["rashi"],
            item.get("score", 0),
        )

    for rank, item in enumerate((processed.get("organ_truth_correlation", {}) or {}).get("top_organs", []), 1):
        organ = item.get("organ", "")
        add_evidence(organ, "organ_truth", item.get("score", 0), rank)
        for disease in item.get("related_diseases", []):
            add_diseases(
                [organ],
                [disease.get("disease") if isinstance(disease, dict) else disease],
                ALGORITHM_NAMES["organ_truth"],
                disease.get("score", item.get("score", 0)) if isinstance(disease, dict) else item.get("score", 0),
            )

    for source_key, processed_key in (
        ("disease_first", "disease_first_logic"),
        ("disease_compare", "disease_compare_logic"),
    ):
        for rank, item in enumerate((processed.get(processed_key, {}) or {}).get("top_diseases", []), 1):
            item_organs = [
                organ.get("organ") for organ in item.get("supporting_organs", [])
                if isinstance(organ, dict)
            ]
            for organ in item_organs:
                add_evidence(organ, source_key, item.get("score", 0), rank)
            add_related(item_organs, ALGORITHM_NAMES[source_key], item.get("score", 0))
            add_diseases(
                item_organs,
                [item.get("disease")],
                ALGORITHM_NAMES[source_key],
                item.get("score", 0),
            )

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
            "algorithm_scores": {
                ALGORITHM_NAMES[key]: round(value, 2)
                for key, value in item["algorithm_scores"].items()
            },
            "supporting_algorithms": [
                ALGORITHM_NAMES[key]
                for key, value in item["algorithm_scores"].items()
                if value > 0
            ],
            "related_organs": [
                {
                    "organ": related["organ"],
                    "score": round(related["score"], 2),
                    "sources": sorted(related["sources"]),
                }
                for related in sorted(
                    item["related_organs"].values(),
                    key=lambda value: (len(value["sources"]), value["score"]),
                    reverse=True,
                )[:2]
            ],
            "top_diseases": [
                {
                    "disease": disease["disease"],
                    "score": round(disease["score"], 2),
                    "sources": sorted(disease["sources"]),
                }
                for disease in sorted(
                    item["diseases"].values(),
                    key=lambda value: (len(value["sources"]), value["score"]),
                    reverse=True,
                )[:3]
            ],
        })

    ranked.sort(
        key=lambda item: (
            item["consensus_score"],
            item["algorithm_support_count"],
            sum(item["algorithm_scores"].values()),
        ),
        reverse=True,
    )
    return {
        "method": "Median consensus across all five algorithms",
        "top_organs": ranked[:3],
    }


def write_algorithm_comparison(folder: str, patient_id: str, processed: dict) -> dict:
    output = {
        "patient_id": patient_id,
        "generated_at": datetime.now().isoformat(),
        "algorithms": _algorithm_sections(processed),
        "Combined Algo Summary": build_combined_algo_summary(processed),
    }
    output_file = os.path.join(folder, "algorithm_comparison.json")
    with open(output_file, "w") as file:
        json.dump(output, file, indent=2)
    return output
