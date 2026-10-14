#!/usr/bin/env python3
"""
Module C — Output Aggregation
Aggregates diagnosis.json, chat_log.json, and processed_logic.json
to produce a final actionable health report.
"""

import os
import re
import json
from datetime import datetime, timedelta
from symptom_mapper import extract_cluster_terms
from combined_algo_clusters import (
    CHAT_PLAN_SOURCE,
    combined_algo_as_hitlist,
    compute_combined_algo_summary,
)

# ── Risk Grade Constants ───────────────────────────────────────
RISK_GRADES = {
    "VERY_HIGH": {
        "min_score":  15,
        "label":      "Very High Risk",
        "color":      "#DC2626",
        "action":     "Immediate medical attention strongly advised"
    },
    "HIGH": {
        "min_score":  8,
        "label":      "Serious Risk",
        "color":      "#EF4444",
        "action":     "Consult a doctor within 2-4 weeks"
    },
    "MODERATE": {
        "min_score":  6,
        "label":      "Moderate Risk",
        "color":      "#F59E0B",
        "action":     "Monitor closely, Ayurvedic care advised"
    },
    "LOW": {
        "min_score":  3,
        "label":      "Low Risk",
        "color":      "#22C55E",
        "action":     "Preventive care and lifestyle awareness"
    },
    "VERY_LOW": {
        "min_score":  0,
        "label":      "Very Low Risk",
        "color":      "#4ADE80",
        "action":     "General wellness maintenance"
    }
}

# ── Recommendation Tables ──────────────────────────────────────
FOOD_TABLE = {
    "Vatha": {
        "avoid": [
            "raw vegetables", "cold foods", "carbonated drinks", "beans",
            "dry snacks", "caffeine"
        ],
        "vegetarian": {
            "eat": [
                "warm cooked foods", "ghee and sesame oil",
                "sweet fruits like mango banana", "warm milk with turmeric",
                "rice and wheat", "root vegetables", "nuts and seeds"
            ]
        },
        "non_vegetarian": {
            "eat": [
                "chicken soup", "warm fish curry", "eggs", "bone broth"
            ]
        }
    },
    "Pittha": {
        "avoid": [
            "spicy food", "sour food", "fried food", "alcohol",
            "red meat", "excess salt", "fermented food"
        ],
        "vegetarian": {
            "eat": [
                "cooling foods like cucumber", "coconut water",
                "sweet fruits like melons", "leafy greens",
                "milk and butter", "basmati rice", "coriander and fennel"
            ]
        },
        "non_vegetarian": {
            "eat": [
                "white fish", "chicken (not spicy)", "egg whites"
            ]
        }
    },
    "Kapha": {
        "avoid": [
            "heavy oily foods", "dairy excess", "sweets", "cold drinks",
            "wheat excess", "fried foods", "red meat"
        ],
        "vegetarian": {
            "eat": [
                "light spicy warm foods", "honey", "ginger tea",
                "lentils and legumes", "leafy vegetables",
                "barley and millet", "bitter gourd"
            ]
        },
        "non_vegetarian": {
            "eat": [
                "lean chicken grilled", "fish (light preparation)", "eggs (boiled not fried)"
            ]
        }
    }
}

LIFESTYLE_TABLE = {
    "Vatha": [
        "Sleep before 10 PM, wake by 6 AM",
        "Warm oil massage (sesame) daily",
        "Avoid cold and windy environments",
        "Gentle yoga and walking — no intense exercise",
        "Meditation 10 minutes daily",
        "Keep warm — especially feet and lower back",
        "Eat at fixed times — no skipping meals"
    ],
    "Pittha": [
        "Avoid direct sun between 10 AM and 2 PM",
        "Cool water bath in morning",
        "Avoid anger and stressful situations",
        "Moderate exercise — swimming or walking",
        "Eat on time — never skip meals",
        "Coconut oil massage weekly",
        "Spend time in nature — near water"
    ],
    "Kapha": [
        "Wake before 6 AM — avoid oversleeping",
        "Vigorous exercise daily — running or cycling",
        "Dry massage (garshana) with silk gloves",
        "Avoid daytime sleeping",
        "Keep environment warm and dry",
        "Intermittent fasting beneficial",
        "Stay socially active — avoid isolation"
    ]
}

WELLNESS_TABLE = {
    "Vatha": [
        "Ashwagandha supplement daily",
        "Triphala at bedtime",
        "Warm ginger and honey tea morning",
        "Pranayama — Nadi Shodhana breathing"
    ],
    "Pittha": [
        "Shatavari supplement",
        "Amla juice daily",
        "Brahmi for mental cooling",
        "Chandra Namaskar (moon salutation)"
    ],
    "Kapha": [
        "Trikatu (ginger, pepper, pippali)",
        "Honey with warm water morning",
        "Surya Namaskar daily",
        "Dry ginger and turmeric milk"
    ]
}


# ── Internal Logic Functions ──────────────────────────────────

def get_risk_grade(score: float) -> dict:
    if score >= 15:
        return RISK_GRADES["VERY_HIGH"]
    elif score >= 10:
        return RISK_GRADES["HIGH"]
    elif score >= 6:
        return RISK_GRADES["MODERATE"]
    elif score >= 3:
        return RISK_GRADES["LOW"]
    else:
        return RISK_GRADES["VERY_LOW"]


def calculate_disease_probability(diagnosis: dict, chat_log: dict) -> list:
    from knowledge_base import PLANET_DISEASES

    # Weighted Base Score
    score_ctx = diagnosis.get("risk_score", {})
    wbs = (
        score_ctx.get("step_a", 0) * 0.4 +
        score_ctx.get("step_b", 0) * 0.3 +
        score_ctx.get("step_c", 0) * 0.3
    )

    # Keywords for Chat Score
    body_systems = [s.strip().lower() for s in diagnosis.get("body_systems", [])]
    dosha_symptoms_text = " ".join(diagnosis.get("dosha_symptoms", {}).values()).lower()
    
    keywords = []
    for s in body_systems:
        keywords += [k.strip() for k in s.split(",")]
    keywords += [k.strip() for k in dosha_symptoms_text.split(",")]
    keywords = list(set([k for k in keywords if k]))

    # Scan chat for matches
    chat_score_total = 0
    matched_keywords = []
    user_msgs = [t["message"].lower() for t in chat_log.get("turns", []) if t["role"] == "user"]
    
    for kw in keywords:
        for msg in user_msgs:
            if kw in msg:
                chat_score_total += 1
                matched_keywords.append(kw)
                break # count each keyword once

    chat_score_total = min(chat_score_total, 5)

    # Calculate for each condition in high_risk_conditions
    active_dasha_planets = [a["planet"] for a in diagnosis.get("active_dasha_alerts", [])]
    
    results = []
    for cond in diagnosis.get("high_risk_conditions", []):
        # Dasha Multiplier check
        multiplier = 1.0
        for planet in active_dasha_planets:
            planet_sys = PLANET_DISEASES.get(planet, "").lower()
            if cond.lower() in planet_sys:
                multiplier = 1.5
                break

        final_score = (wbs + chat_score_total) * multiplier
        grade_info = get_risk_grade(final_score)

        results.append({
            "condition":         cond,
            "base_score":        round(wbs, 1),
            "chat_score":        chat_score_total,
            "dasha_multiplier":  multiplier,
            "final_score":       round(final_score, 1),
            "grade":             [g for g in RISK_GRADES if RISK_GRADES[g]["label"] == grade_info["label"]][0],
            "grade_label":       grade_info["label"],
            "grade_color":       grade_info["color"],
            "action":            grade_info["action"],
            "matched_keywords":  list(set(matched_keywords))
        })

    return results


def calculate_care_period(diagnosis: dict, chat_log: dict, logic: dict) -> dict:
    # 1. Dasha End Dates
    alerts = diagnosis.get("active_dasha_alerts", [])
    end_dates = []
    for a in alerts:
        try:
            end_dates.append(datetime.strptime(a["end_date"], "%Y-%m-%d"))
        except: pass
    
    if not end_dates:
        primary_until = datetime.now() + timedelta(days=365)
    else:
        primary_until = max(end_dates)

    # 2. Chronic Indicators
    chronic_keywords = [
        "months", "years", "since", "long", "chronic", 
        "recurring", "always", "never goes away"
    ]
    is_chronic = False
    for t in chat_log.get("turns", []):
        if t["role"] == "user":
            msg = t["message"].lower()
            if any(kw in msg for kw in chronic_keywords):
                is_chronic = True
                primary_until += timedelta(days=180) # +6 months
                break

    # 3. Pratyantardasha (from logic)
    p_dasha = logic.get("dasha", {}).get("pratyantardasha", {})
    review_date_str = p_dasha.get("end_date", "")
    if review_date_str:
        review_date = datetime.strptime(review_date_str, "%Y-%m-%d")
    else:
        review_date = datetime.now() + timedelta(days=90)

    duration = (primary_until - datetime.now()).days // 30

    return {
        "primary_care_until":  primary_until.strftime("%Y-%m-%d"),
        "review_date":         review_date.strftime("%Y-%m-%d"),
        "duration_months":     max(duration, 0),
        "basis":               "Active Dasha and symptom persistence",
        "chronic_indicator":   is_chronic,
        "recommendation":      f"Monitor health closely until {primary_until.strftime('%b %Y')}. Professional review suggested by {review_date.strftime('%b %Y')}."
    }


def get_food_recommendations(diagnosis: dict) -> dict:
    doshas = diagnosis.get("dosha_imbalance", [])
    avoid = []
    veg_eat = []
    non_veg_eat = []

    for d in doshas:
        entry = FOOD_TABLE.get(d, {})
        if entry:
            avoid += entry.get("avoid", [])
            veg_eat += entry.get("vegetarian", {}).get("eat", [])
            non_veg_eat += entry.get("non_vegetarian", {}).get("eat", [])

    return {
        "doshas":  doshas,
        "avoid":   list(set(avoid)),
        "vegetarian": {"eat": list(set(veg_eat))},
        "non_vegetarian": {"eat": list(set(non_veg_eat))},
        "note": f"Personalized diet for {', '.join(doshas)} constitution."
    }


def get_doctor_recommendation(disease_prob: list, diagnosis: dict, chat_log: dict) -> dict:
    # Logic based on requirements
    top_grade = "VERY_LOW"
    grades_priority = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    
    high_count = 0
    for d in disease_prob:
        grade = d["grade"]
        if grades_priority.index(grade) > grades_priority.index(top_grade):
            top_grade = grade
        if grades_priority.index(grade) >= grades_priority.index("HIGH"):
            high_count += 1

    has_active_dasha = len(diagnosis.get("active_dasha_alerts", [])) > 0
    
    chronic_keywords = ["years", "chronic", "always", "recurring", "never goes away"]
    is_chronic = False
    for t in chat_log.get("turns", []):
        if t["role"] == "user" and any(kw in t["message"].lower() for kw in chronic_keywords):
            is_chronic = True
            break

    # Decision logic
    see_doctor = False
    urgency = "Not immediate"
    reasoning = "General wellness guidelines are sufficient."
    
    priority_level = grades_priority.index(top_grade)
    if has_active_dasha: priority_level += 1
    if is_chronic: priority_level += 1
    if high_count >= 2: see_doctor = True

    if priority_level >= 4: # Equivalent to HIGH/VERY_HIGH
        see_doctor = True
        urgency = "Within 2 weeks"
        reasoning = "Presence of high-risk indicators combined with active dasha and/or chronic history."
    elif priority_level >= 3: # MODERATE
        see_doctor = True
        urgency = "Within 4 weeks"
        reasoning = "Moderate risk alignment detected. Proactive screening recommended."
    
    return {
        "see_doctor": see_doctor,
        "urgency":    urgency,
        "reasoning":  reasoning,
        "specialist": "General Physician and Ayurvedic Practitioner",
        "score_basis": {
            "risk_grade": top_grade,
            "dasha_active": has_active_dasha,
            "chronic_found": is_chronic,
            "high_conditions": high_count
        }
    }


def get_lifestyle_recommendations(diagnosis: dict) -> list:
    doshas = diagnosis.get("dosha_imbalance", [])
    recs = []
    for d in doshas:
        recs += LIFESTYLE_TABLE.get(d, [])
    return list(set(recs))


def get_wellness_suggestions(diagnosis: dict) -> list:
    doshas = diagnosis.get("dosha_imbalance", [])
    recs = []
    for d in doshas:
        recs += WELLNESS_TABLE.get(d, [])
    return list(set(recs))


def build_hitlist_confirmation_summary(logic: dict, chat_log: dict) -> list:
    scores = chat_log.get("hitlist_cluster_scores", {}) or {}
    top4 = logic.get("hitlist", {}).get("top4", [])
    summary = []

    for index, cluster in enumerate(top4[:4], start=1):
        score_data = scores.get(str(index), {})
        support_score = score_data.get("score", 0)
        yes_strong = score_data.get("yes_strong", 0)
        yes_but = score_data.get("yes_but", 0)
        no_but = score_data.get("no_but", 0)
        no_strong = score_data.get("no_strong", 0)
        gate_status = score_data.get("status", "")

        if gate_status == "confirmed" or yes_strong >= 3:
            status = "Confirmed"
            confidence = "High"
        elif gate_status in ["drilldown", "suspected"] or yes_strong >= 1 or yes_but >= 1:
            status = "Possible"
            confidence = "Medium"
        elif gate_status == "rejected" or no_strong >= 2:
            status = "Rejected"
            confidence = "Low"
        else:
            status = "Not strongly supported"
            confidence = "Low"

        terms = extract_cluster_terms(cluster)
        answers = score_data.get("answers", [])
        confirmed = [
            a.get("answer", "")
            for a in answers
            if a.get("result") in ["yes_strong", "yes_but"]
        ]

        summary.append({
            "rank": index,
            "original_rank": index,
            "system": cluster.get("label", f"Cluster {index}"),
            "hitlist_score": cluster.get("score", 0),
            "convergence": cluster.get("convergence", 0),
            "status": status,
            "confidence": confidence,
            "gate_status": gate_status,
            "support_score": support_score,
            "yes_strong": yes_strong,
            "yes_but": yes_but,
            "no_but": no_but,
            "no_strong": no_strong,
            "confirmed_patient_signals": confirmed[:4],
            "internal_organs": terms.get("organs", []),
            "internal_diseases": terms.get("diseases", []),
            "confirmed_organs": score_data.get("confirmed_organs", []),
            "confirmed_diseases": score_data.get("confirmed_diseases", []),
            "possible_organs": score_data.get("possible_organs", []),
            "possible_diseases": score_data.get("possible_diseases", []),
            "branches": list((score_data.get("branches", {}) or {}).values()),
        })

    # Re-rank the confirmation view using patient-supported evidence first.
    # Raw astrology score remains as a tie-breaker and is preserved as original_rank.
    summary.sort(
        key=lambda item: (
            item.get("support_score", 0),
            item.get("yes_strong", 0),
            item.get("yes_but", 0),
            -item.get("no_strong", 0),
            -item.get("no_but", 0),
            item.get("hitlist_score", 0),
            item.get("convergence", 0),
            -item.get("original_rank", 99),
        ),
        reverse=True
    )

    for new_rank, item in enumerate(summary, start=1):
        item["rank"] = new_rank

    return summary


# ── RL handoff — "For RL Report.json" for Abhishek's RL system ──
# Written in exactly the input format of his Medical Report Enhancer, so the
# file can be uploaded to it as-is:
#   {"patient_id": "...",
#    "conditions": [{"Organ": ["disease", ...]}, ...],   # Combined Algo top 3
#    "chat_log":   [ <current chat session> ]}            # [] until a chat starts
# Created at chart generation, then its chat_log is refreshed after every turn.
# diseases stay a JSON list, not a comma-joined string, because some disease
# names contain commas ("Diabetes, Insulin deficiency").

def _norm_term(text: str) -> str:
    return " ".join(str(text).lower().replace("&", " and ").split())


def _is_body_part_term(disease: str) -> bool:
    # Planet/rashi indications like "Kidney", "Blood", "Water in the body"
    # get stored as diseases upstream — they're body parts, not diseases.
    # ALL_ORGANS also lists a few real diseases (e.g. "Leprosy"), so a term in
    # the known-disease table is always kept.
    from rashi_truth_review import ALL_ORGANS, NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS
    term = _norm_term(disease)
    if term in {_norm_term(d) for d in NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS}:
        return False
    return term in {_norm_term(o) for o in ALL_ORGANS}


def _group_by_organ(pairs: list) -> list:
    grouped = {}
    for organ, disease in pairs:
        if not organ or not disease or _is_body_part_term(disease):
            continue
        diseases = grouped.setdefault(organ, [])
        if _norm_term(disease) not in {_norm_term(d) for d in diseases}:
            diseases.append(disease)
    return [{"organ": o, "diseases": d} for o, d in grouped.items() if d]


def build_rl_conditions(summary: dict) -> list:
    pairs = [
        (entry.get("organ", ""), d.get("disease", ""))
        for entry in (summary.get("top_organs") or [])[:3]
        for d in (entry.get("top_diseases") or [])
    ]
    return [{g["organ"]: g["diseases"]} for g in _group_by_organ(pairs)]


def _current_session_log(chat_log: dict) -> dict:
    # chat_log.json appends every chat the patient has had; RL gets only the
    # current one (from the last assistant turn numbered 1).
    turns = chat_log.get("turns") or []
    start = 0
    for i, t in enumerate(turns):
        if t.get("role") == "assistant" and t.get("turn") == 1:
            start = i
    session_turns = turns[start:]
    started_at = session_turns[0].get("timestamp", "") if session_turns else ""

    # The top-level completed / hitlist_cluster_scores fields belong to the last
    # chat that finished. If that isn't this one (still in progress), use the
    # live cluster scores saved with this chat's latest assistant turn instead.
    finished_here = bool(chat_log.get("completed")) and str(chat_log.get("completed_at", "")) >= str(started_at)
    if finished_here:
        scores = chat_log.get("hitlist_cluster_scores") or {}
    else:
        scores = next(
            ((t.get("metadata") or {}).get("cluster_scores") for t in reversed(session_turns)
             if t.get("role") == "assistant" and (t.get("metadata") or {}).get("cluster_scores")),
            {},
        )

    # Saved with each turn since the RL chat-log update (older chats lack it);
    # RL uses it to find and delete a stored chat.
    session_id = next(
        ((t.get("metadata") or {}).get("session_id") for t in session_turns
         if (t.get("metadata") or {}).get("session_id")),
        None,
    )
    session = {
        "patient_id": chat_log.get("patient_id"),
        **({"session_id": session_id} if session_id else {}),
        "started_at": started_at,
        "turns": session_turns,
        "completed": finished_here,
        "hitlist_cluster_scores": scores,
    }
    if finished_here:
        session["completed_at"] = chat_log.get("completed_at")
        session["final_assessment"] = chat_log.get("final_assessment")
        session["chat_plan_source"] = chat_log.get("chat_plan_source")
    return session


# ── Patient-facing confirmed findings (results page) ──
# Only the one confirmed cluster: its organ, the diseases the chat confirmed,
# and for each disease the questions the patient answered yes / no to.

YES_RESULTS = {"yes_strong", "yes_but"}
NO_RESULTS = {"no_strong", "no_but"}


def _latest_session_turns(chat_log: dict) -> list:
    # chat_log.json appends every chat the patient has had; a new chat starts
    # at the last assistant turn numbered 1. hitlist_cluster_scores belongs to
    # the chat that finished at completed_at, so ignore anything after it
    # (a newer chat that was started but never finished).
    turns = chat_log.get("turns") or []
    completed_at = chat_log.get("completed_at")
    if completed_at:
        turns = [t for t in turns if str(t.get("timestamp", "")) <= str(completed_at)]
    start = 0
    for i, t in enumerate(turns):
        if t.get("role") == "assistant" and t.get("turn") == 1:
            start = i
    return turns[start:]


def _chat_message_text(message) -> str:
    text = str(message or "").strip()
    try:
        obj = json.loads(message)
        if isinstance(obj, dict) and obj.get("chat_message"):
            text = str(obj["chat_message"]).strip()
    except (TypeError, ValueError):
        pass
    # Replies are "acknowledgement sentence + one question"; keep the question.
    questions = re.findall(r"[^.!?]*\?", text)
    return questions[-1].strip() if questions else text


def _answered_questions(chat_log: dict) -> list:
    answered = []
    question = ""
    for t in _latest_session_turns(chat_log):
        if t.get("role") == "assistant":
            question = _chat_message_text(t.get("message"))
        elif t.get("role") == "user":
            md = t.get("metadata") or {}
            answered.append({
                "question": question,
                "answer": str(t.get("message", "")).strip(),
                "result": md.get("answer_result", ""),
                "cluster_rank": md.get("answered_cluster_rank"),
                "branch_id": md.get("answered_branch_id", ""),
            })
    return answered


def _split_yes_no(answers: list) -> tuple[list, list]:
    def pick(results):
        out, seen = [], set()
        for a in answers:
            key = (_norm_term(a["question"]), _norm_term(a["answer"]))
            if a["result"] in results and key not in seen:
                seen.add(key)
                out.append({"question": a["question"], "answer": a["answer"]})
        return out
    return pick(YES_RESULTS), pick(NO_RESULTS)


ANSWER_POINTS = {"yes_strong": 1.0, "yes_but": 0.5, "no_but": 0.0, "no_strong": 0.0}


def _disease_branch_id(disease: str) -> str:
    # Same keyword rule symptom_mapper uses to put a disease into a question
    # branch when it builds the chat plan.
    from symptom_mapper import map_term, _rule_id
    rule = map_term(disease)
    return _rule_id(rule) if rule else ""


def _score_disease(name: str, cluster_answers: list) -> dict:
    # Gemini-written plans have one branch per disease (question_generator);
    # built-in plans group diseases by keyword rule.
    from question_generator import generated_branch_id
    branch_ids = {generated_branch_id(name), _disease_branch_id(name)} - {""}
    scored, seen = [], set()
    for a in cluster_answers:
        key = (_norm_term(a["question"]), _norm_term(a["answer"]), a["result"])
        if a["branch_id"] in branch_ids and a["result"] in ANSWER_POINTS and key not in seen:
            seen.add(key)
            scored.append(a)
    said_yes, said_no = _split_yes_no(scored)
    if not scored:
        return {"name": name, "asked": False, "score": None, "yes_count": 0,
                "question_count": 0, "said_yes": [], "said_no": []}
    # 0-10: share of answers that were yes (a "yes, but" counts half).
    score = round(sum(ANSWER_POINTS[a["result"]] for a in scored) / len(scored) * 10, 1)
    return {
        "name": name,
        "asked": True,
        "score": score,
        "yes_count": sum(1 for a in scored if a["result"] in YES_RESULTS),
        "question_count": len(scored),
        "said_yes": said_yes,
        "said_no": said_no,
    }


def build_confirmed_findings(chat_log: dict, combined_summary: dict | None = None) -> dict:
    clusters = (chat_log.get("hitlist_cluster_scores") or {}).values()
    confirmed = [c for c in clusters if c.get("status") == "confirmed"]
    if not confirmed:
        return {"confirmed": False}

    best = max(confirmed, key=lambda c: (c.get("yes_strong", 0), c.get("score", 0)))
    organ = best.get("system", "")
    cluster_answers = [
        a for a in _answered_questions(chat_log)
        if str(a["cluster_rank"]) == str(best.get("rank"))
    ]

    # All of the Combined Algo's diseases for the confirmed organ, including
    # ones the chat never asked about. Chats that predate the Combined Algo
    # plan (clusters named like "Kidney / Urinary System") fall back to the
    # diseases the chat itself confirmed.
    algo_entry = next(
        (e for e in (combined_summary or {}).get("top_organs") or []
         if _norm_term(e.get("organ", "")) == _norm_term(organ)),
        None,
    )
    if algo_entry:
        candidates = [d.get("disease", "") for d in algo_entry.get("top_diseases") or []]
    else:
        candidates = [
            d for b in (best.get("branches") or {}).values()
            if b.get("status") == "confirmed"
            for d in b.get("confirmed_diseases", [])
        ]

    diseases, seen = [], set()
    for name in candidates:
        if not name or _is_body_part_term(name) or _norm_term(name) in seen:
            continue
        seen.add(_norm_term(name))
        diseases.append(_score_disease(name, cluster_answers))
    diseases.sort(key=lambda d: (d["asked"], d["score"] or 0), reverse=True)

    said_yes, said_no = _split_yes_no(cluster_answers)
    return {
        "confirmed": True,
        "organ": organ,
        "diseases": diseases,
        "said_yes": said_yes,
        "said_no": said_no,
    }


RL_REPORT_FILE = "For RL Report.json"


def _save_rl_report(folder: str, rl_report: dict) -> dict:
    with open(os.path.join(folder, RL_REPORT_FILE), "w", encoding="utf-8") as f:
        json.dump(rl_report, f, indent=2, ensure_ascii=False)
    session = rl_report["chat_log"][0] if rl_report["chat_log"] else {}
    # Logged at chart generation and chat completion only — not every turn.
    if not session or session.get("completed"):
        print(f"[RL] {RL_REPORT_FILE} written -> {len(rl_report['conditions'])} organs, "
              f"{len(session.get('turns', []))} chat turns")
    return rl_report


def write_rl_report(folder: str, combined_summary: dict | None = None, chat_log: dict | None = None) -> dict:
    # At chart generation chat_log is None -> "chat_log": [].
    if combined_summary is None:
        combined_summary = compute_combined_algo_summary(folder)
    return _save_rl_report(folder, {
        "patient_id": os.path.basename(os.path.normpath(folder)),
        "conditions": build_rl_conditions(combined_summary),
        "chat_log": [_current_session_log(chat_log)] if chat_log else [],
    })


def ensure_rl_report(folder: str) -> None:
    # For patients opened from the list rather than re-generated: create the
    # file if missing, or rewrite one still in the old two-section layout
    # (which held no chat_log, so nothing is lost). A current file is kept.
    path = os.path.join(folder, RL_REPORT_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if "conditions" in json.load(f):
                return
    write_rl_report(folder)


def refresh_rl_report_chat(folder: str, chat_log: dict) -> dict:
    # After every chat turn: keep the conditions written at chart generation,
    # replace the chat_log with the chat as it stands now.
    path = os.path.join(folder, RL_REPORT_FILE)
    if not os.path.exists(path):
        return write_rl_report(folder, chat_log=chat_log)
    with open(path, encoding="utf-8") as f:
        rl_report = json.load(f)
    if "conditions" not in rl_report:  # file from the older two-section format
        return write_rl_report(folder, chat_log=chat_log)
    rl_report["chat_log"] = [_current_session_log(chat_log)]
    return _save_rl_report(folder, rl_report)


# ── Main Engine Call ───────────────────────────────────────────

def run_combine(patient_id: str) -> dict:
    folder = f"patients/{patient_id}"

    # Verify files
    try:
        with open(f"{folder}/diagnosis.json") as f:
            diagnosis = json.load(f)
        with open(f"{folder}/chat_log.json") as f:
            chat_log = json.load(f)
        with open(f"{folder}/processed_logic.json") as f:
            logic = json.load(f)
    except FileNotFoundError as e:
        print(f"[FAIL] Aborting aggregation: {e}")
        return {}

    disease_prob = calculate_disease_probability(diagnosis, chat_log)
    care_period  = calculate_care_period(diagnosis, chat_log, logic)
    food         = get_food_recommendations(diagnosis)
    doctor       = get_doctor_recommendation(disease_prob, diagnosis, chat_log)
    lifestyle    = get_lifestyle_recommendations(diagnosis)
    wellness     = get_wellness_suggestions(diagnosis)
    combined_summary = compute_combined_algo_summary(folder)
    if chat_log.get("chat_plan_source") == CHAT_PLAN_SOURCE:
        # The chat's cluster scores are keyed by the Combined Algo's clusters,
        # so label the confirmation summary with those, not hitlist v1's.
        logic = {**logic, "hitlist": combined_algo_as_hitlist(combined_summary)}
    hitlist_summary = build_hitlist_confirmation_summary(logic, chat_log)

    # Top score calculation
    top_score = max([d["final_score"] for d in disease_prob]) if disease_prob else diagnosis.get("risk_score",{}).get("total",0)
    overall = get_risk_grade(top_score)

    final_output = {
        "patient_id":   patient_id,
        "generated_at": datetime.now().isoformat(),
        "disease_probability": disease_prob,
        "confirmed_findings": build_confirmed_findings(chat_log, combined_summary),
        "hitlist_confirmation": hitlist_summary,
        "overall_grade": {
            "score": round(top_score, 1),
            "grade": [g for g in RISK_GRADES if RISK_GRADES[g]["label"] == overall["label"]][0],
            **overall
        },
        "care_period": care_period,
        "food": food,
        "doctor": doctor,
        "lifestyle": lifestyle,
        "wellness": wellness,
        "raw_scores": {
            "step_a": diagnosis.get("risk_score", {}).get("step_a", 0),
            "step_b": diagnosis.get("risk_score", {}).get("step_b", 0),
            "step_c": diagnosis.get("risk_score", {}).get("step_c", 0),
            "total":  diagnosis.get("risk_score", {}).get("total", 0),
            "chat_turns": len(chat_log.get("turns", []))
        }
    }

    with open(f"{folder}/final_output.json", "w") as f:
        json.dump(final_output, f, indent=2)
    from chat_ai_context_builder import build_chat_ai_context
    build_chat_ai_context(patient_id, folder)

    write_rl_report(folder, combined_summary, chat_log)

    print(f"[OK] Aggregation complete for patient {patient_id}")
    return final_output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", required=True)
    args = parser.parse_args()
    
    import sys
    # Ensure current directory is in path for knowledge_base
    sys.path.append(os.getcwd())
    
    result = run_combine(args.patient)
    if result:
        print("\n--- AGGREGATION RESULT (SNEAK PEEK) ---")
        print(f"Overall Grade: {result['overall_grade']['label']}")
        print(f"Top Score:     {result['overall_grade']['score']}")
        print(f"See Doctor:    {result['doctor']['see_doctor']} ({result['doctor']['urgency']})")
        print(f"Care Until:    {result['care_period']['primary_care_until']}")
        print(f"----------------------------------------")
