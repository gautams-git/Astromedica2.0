#!/usr/bin/env python3
"""
Module B — Chat Module
Handles health consultation chat sessions with Gemma AI.
Reads patient data from diagnosis.json / processed_logic.json.
Saves every Q&A turn to patients/00X/chat_log.json.
"""

import os
import json
import asyncio
from datetime import datetime
from symptom_mapper import (
    build_hitlist_chat_plan,
    classify_answer,
    score_answer,
)


# ── Function 1: Load patient context from files ──────────────────
def load_patient_context(patient_id: str) -> dict:
    """
    Reads diagnosis.json and processed_logic.json for this patient.
    Returns combined context dict.
    """
    folder = f"patients/{patient_id}"

    with open(f"{folder}/processed_logic.json") as f:
        logic = json.load(f)

    with open(f"{folder}/diagnosis.json") as f:
        diagnosis = json.load(f)

    # The chat asks about the Combined Algo's top 3 organs, not hitlist v1.
    # Falls back to hitlist v1 only if the Combined Algo produced nothing.
    from combined_algo_clusters import load_combined_algo_hitlist, CHAT_PLAN_SOURCE
    hitlist = load_combined_algo_hitlist(folder)
    chat_plan_source = CHAT_PLAN_SOURCE
    if not hitlist.get("top4"):
        hitlist = logic.get("hitlist", {})
        chat_plan_source = "hitlist_v1"
    print(f"[Chat] chat plan source: {chat_plan_source} -> "
          f"{[c.get('label') for c in hitlist.get('top4', [])]}")

    return {
        "patient_id":        patient_id,
        "rule1":             logic.get("rule1", {}),
        "planets":           logic.get("planets", []),
        "complete_analysis": logic.get("complete_analysis", {}),
        "dasha":             logic.get("dasha", {}),
        "disease_filter":    logic.get("disease_filter", {}),
        "most_probable":     logic.get("most_probable", []),
        "hitlist":           hitlist,
        "chat_plan_source":  chat_plan_source,
        "dasha_chat_priority": logic.get("dasha_chat_priority", {}),
        "diagnosis":         diagnosis,
        "gender":            logic.get("gender", "Unknown")
    }


# ── Function 2: Save one chat turn to chat_log.json ──────────────
def save_chat_turn(
    patient_id: str,
    turn: int,
    role: str,
    message: str,
    metadata: dict = None
) -> None:
    """
    Appends one Q&A turn to patients/00X/chat_log.json
    """
    folder   = f"patients/{patient_id}"
    log_file = f"{folder}/chat_log.json"

    if os.path.exists(log_file):
        with open(log_file) as f:
            log = json.load(f)
    else:
        log = {
            "patient_id": patient_id,
            "started_at": datetime.now().isoformat(),
            "turns": []
        }

    item = {
        "turn":      turn,
        "role":      role,
        "message":   message,
        "timestamp": datetime.now().isoformat()
    }
    if metadata:
        item["metadata"] = metadata
    log["turns"].append(item)

    with open(log_file, "w") as f:
        json.dump(log, f, indent=2)
    sync_rl_report(patient_id, log)


def sync_rl_report(patient_id: str, log: dict) -> None:
    # Keeps For RL Report.json's chat_log current after every turn. It's an extra
    # output for the RL system, so a failure here must never break the chat.
    try:
        from combine_output import refresh_rl_report_chat
        refresh_rl_report_chat(f"patients/{patient_id}", log)
    except Exception as e:
        print(f"[RL] could not update For RL Report.json: {e}")


def queue_rl(patient_id: str) -> None:
    # Runs Abhishek's RL enhancer on the finished chat (in its background
    # worker). Like sync_rl_report, a failure here must never break the chat.
    try:
        from rl_integration import queue_patient_rl
        queue_patient_rl(patient_id)
    except Exception as e:
        print(f"[RL] could not queue RL for patient {patient_id}: {e}")


def build_debug_context(context: dict, system_prompt: str, chat_plan: dict = None) -> dict:
    """
    Returns a structured payload for inspecting what the chat model receives.
    """
    hitlist_result = context.get("hitlist", {}) or {}
    dasha_result = context.get("dasha", {}) or {}
    dasha_chat_priority = context.get("dasha_chat_priority", {}) or {}

    def dedupe_keep_order(items):
        seen = set()
        output = []
        for item in items:
            clean = (item or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            output.append(clean)
        return output

    def summarize_cluster(cluster: dict, rank: int) -> dict:
        planet_terms = []
        nakshatra_terms = []
        rashi_terms = []

        for entry in cluster.get("trail", []):
            source = entry.get("source", "")
            term = entry.get("term", "")
            if "HL1" in source or "Planet" in source:
                planet_terms.append(term)
            elif "HL2" in source or "Nakshatra" in source:
                nakshatra_terms.append(term)
            elif "HL3" in source or "Rashi" in source:
                rashi_terms.append(term)

        planet_terms = dedupe_keep_order(planet_terms)
        nakshatra_terms = dedupe_keep_order(nakshatra_terms)
        rashi_terms = dedupe_keep_order(rashi_terms)

        return {
            "rank": rank,
            "label": cluster.get("label", f"Cluster {rank}"),
            "planet": planet_terms,
            "nakshatra": nakshatra_terms,
            "rashi": rashi_terms,
            "organs": rashi_terms,
            "probable_diseases": dedupe_keep_order(nakshatra_terms + planet_terms),
        }

    top2_clusters = [
        summarize_cluster(cluster, index)
        for index, cluster in enumerate(hitlist_result.get("top4", [])[:2], start=1)
    ]

    current_dasha = {
        "mahadasha": (dasha_result.get("mahadasha") or {}).get("planet", "Unknown"),
        "antardasha": (dasha_result.get("antardasha") or {}).get("planet", "Unknown"),
        "pratyantardasha": (dasha_result.get("pratyantardasha") or {}).get("planet", "Unknown"),
    }

    return {
        "patient_id": context.get("patient_id"),
        "gender": context.get("gender", "Unknown"),
        "system_prompt": system_prompt,
        "gemma_input": {
            "top2_clusters": top2_clusters,
            "current_dasha": current_dasha,
            "dasha_chat_priority": dasha_chat_priority,
        }
    }


def build_hitlist_system_prompt(chat_plan: dict, gender: str = "Unknown", dasha_chat_priority: dict = None) -> str:
    plan_json = json.dumps(chat_plan, indent=2)
    dasha_chat_priority = dasha_chat_priority or {}
    priority_json = json.dumps(dasha_chat_priority, indent=2)
    return f"""You are AstroMedica, a warm health interviewer.

You are running a 15-question symptom interview based ONLY on the chat plan below.
The chat plan's clusters are the top 3 organs from the Combined Algo (consensus of five algorithms), in priority order.
Do not use any other diagnosis, disease_risks, most_probable, dasha, or organ map logic.

Patient gender: {gender}

CHAT PLAN
{plan_json}

DASHA CHAT PRIORITY
{priority_json}

CORE RULES
1. Never mention disease names, internal organs (kidney, liver, lungs, thyroid, bladder...), planets, nakshatra, dasha, rogkaraka, or astrology to the patient. Everyday body parts people can see or feel (nose, ears, eyes, throat, chest, stomach, skin, back, legs) are fine.
2. Ask from symptom_targets and prepared question lists only. You may rewrite them warmly, but do not add unrelated medical areas. Each branch is one possible condition: set branch_id to the branch whose question you ask.
2b. Language for everyone - young or old, little schooling, English not their first language:
   - ONE symptom per question. Never join two symptoms with "or" / "and".
   - Simple everyday words, no medical terms. The question itself at most 18 words.
   - Answerable with yes / no / sometimes. Calm and neutral, never suggest the answer.
   - Never repeat a question already asked in this chat, even in other words.
3. Ask exactly one question per turn.
4. Turns 1-2: screen Cluster 1 first using two different symptom angles.
5. Turn 3: continue with Cluster 1 unless the patient has been clearly negative so far; only then check Cluster 2.
6. Turn 4: continue Cluster 1 by default, and use Cluster 2 only if Cluster 1 remains weak or negative.
6b. Move to Cluster 3 only when both Cluster 1 and Cluster 2 are clearly negative or rejected.
7. Turns 5-8: drill deeper into the most supported cluster, but prefer Cluster 1 when support is close or unclear.
8. Turns 9-11: ask duration, frequency, triggers, and severity questions for the most supported clusters.
9. Turns 12-13: ask red-flag checks from the most supported clusters.
10. Turn 14: ask one lifestyle/context question related to the most supported cluster.
11. Turn 15: give the final summary. Do not ask another question.
12. Before turn 15, do not give a final assessment.
13. Keep replies short: one acknowledgement sentence plus one question.
14. If the patient gives repeated strong yes answers in the same cluster, treat that cluster as confirmed and stop probing deeply. Move to simple lifestyle/context questions.
14b. Before those lifestyle/context questions, ask one screening question for every branch of the confirmed cluster that has not been asked yet (asked_count 0 in the runtime state), so each possible condition in it is checked.
15. If the patient gives mixed answers like yes but / no but, stay in drilldown mode and ask more specific symptom questions from the same cluster.
16. If the patient gives repeated hard no answers with no real support, treat that cluster as rejected and move to the next cluster.
17. Respect dasha_chat_priority.chat_mode for the opening and early turns:
   - direct_disease_first: ask the strongest relevant symptom question early and directly.
   - focused_symptom_first: ask symptom-led questions early, but keep the tone measured.
   - mild_symptom_first: begin gently, avoid the strongest alarm-style wording first.
   - general_lifestyle_first: begin with broad energy, sleep, digestion, routine, and lifestyle questions before disease-focused probing.
18. If dasha_chat_priority.risk_level is very_low, do not start with direct disease-style questioning.

OUTPUT FORMAT
Return only valid JSON, no markdown:
{{
  "reasoning": "Cluster rank, system, question type, and why this was selected.",
  "cluster_rank": 1,
  "branch_id": "id of the chat plan branch this question targets",
  "question_type": "screening|drilldown|severity|red_flag|context|final",
  "chat_message": "Patient-facing reply. No organ names, disease names, or astrology."
}}

FINAL TURN FORMAT
On turn 15, return:
{{
  "reasoning": "Final summary based on the chat plan clusters and patient answers.",
  "cluster_rank": 0,
  "question_type": "final",
  "chat_message": "Based on our conversation today, ... Please consult a qualified physician and Ayurvedic practitioner for proper guidance. This is wellness awareness only and not a medical diagnosis."
}}
"""


def ask_gemini(call_gemini_fn, prompt: str, messages: list, turn: int) -> str:
    # If Gemini still fails after call_gemini's retries (overloaded, network),
    # return "" so the turn falls back to the chat's own next question instead
    # of erroring. The patient's answer is already scored and saved by then,
    # and an error would make them resend it and have it counted twice.
    try:
        return asyncio.run(call_gemini_fn(prompt, messages))
    except Exception as e:
        print(f"[Chat] Turn {turn}: Gemini call failed ({str(e)[:200]}), using fallback question")
        return ""


def log_unreadable_reply(turn: int, raw) -> None:
    # The raw reply is otherwise lost once the fallback question replaces it.
    text = str(raw or "")
    if not text:
        return  # failed call, already logged by ask_gemini
    print(f"[Chat] Turn {turn}: Gemini reply was not valid JSON ({len(text)} chars), "
          f"using fallback question. Raw reply: {text[:500]!r}")


def parse_model_json(raw: str) -> dict:
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = text.replace("```json", "", 1).replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        return {}


def initial_cluster_scores(chat_plan: dict) -> dict:
    scores = {}
    for cluster in chat_plan.get("clusters", []):
        rank = cluster.get("rank")
        branch_scores = {}
        for branch in cluster.get("branches", []):
            branch_scores[branch.get("id")] = {
                "id": branch.get("id"),
                "label": branch.get("label", ""),
                "yes_strong": 0,
                "yes_but": 0,
                "no_but": 0,
                "no_strong": 0,
                "status": "screening",
                "asked_count": 0,
                "confirmed_organs": [],
                "confirmed_diseases": [],
                "possible_organs": [],
                "possible_diseases": [],
            }
        scores[str(rank)] = {
            "rank": rank,
            "system": cluster.get("system", ""),
            "score": 0,
            "yes_strong": 0,
            "yes_but": 0,
            "no_but": 0,
            "no_strong": 0,
            "status": "screening",
            "strong_yes_streak": 0,
            "strong_no_streak": 0,
            "confirmed_organs": [],
            "confirmed_diseases": [],
            "possible_organs": [],
            "possible_diseases": [],
            "dominant_branch": "",
            "branches": branch_scores,
            "answers": [],
        }
    return scores


def _dedupe_preserve(items):
    seen = set()
    output = []
    for item in items:
        clean = str(item or "").strip()
        key = clean.lower()
        if not clean or key in seen:
            continue
        seen.add(key)
        output.append(clean)
    return output


def branch_by_id(cluster: dict, branch_id: str) -> dict:
    for branch in cluster.get("branches", []):
        if branch.get("id") == branch_id:
            return branch
    return {}


def choose_branch_for_cluster(session: dict, cluster_rank: int, question_type: str) -> str:
    cluster = cluster_by_rank(session, cluster_rank)
    if not cluster:
        return ""
    cluster_score = (session.get("cluster_scores", {}) or {}).get(str(cluster_rank), {}) or {}
    branch_states = cluster_score.get("branches", {}) or {}
    branches = cluster.get("branches", []) or []
    if not branches:
        return ""

    def sort_key(branch: dict):
        state = branch_states.get(branch.get("id"), {}) or {}
        return (
            1 if state.get("status") == "confirmed" else 0,
            1 if state.get("status") in ["drilldown", "suspected"] else 0,
            state.get("yes_strong", 0),
            state.get("yes_but", 0),
            -state.get("no_strong", 0),
            -state.get("asked_count", 0),
        )

    if question_type in ["drilldown", "severity", "red_flag", "context"]:
        ranked = sorted(branches, key=sort_key, reverse=True)
        return ranked[0].get("id", "")

    available = [
        branch for branch in branches
        if (branch_states.get(branch.get("id"), {}) or {}).get("status") != "rejected"
    ] or branches
    ranked = sorted(
        available,
        key=lambda branch: (
            1 if (branch_states.get(branch.get("id"), {}) or {}).get("status") in ["drilldown", "suspected"] else 0,
            -(branch_states.get(branch.get("id"), {}) or {}).get("asked_count", 0),
        ),
        reverse=True
    )
    return ranked[0].get("id", "")


def unasked_branch(session: dict, cluster_rank: int) -> str:
    # Each branch holds different possible conditions; one that was never asked
    # about can't be scored on the results page.
    cluster = cluster_by_rank(session, cluster_rank)
    states = ((session.get("cluster_scores", {}) or {}).get(str(cluster_rank), {}) or {}).get("branches", {}) or {}
    for branch in (cluster.get("branches", []) if cluster else []):
        if (states.get(branch.get("id"), {}) or {}).get("asked_count", 0) == 0:
            return branch.get("id", "")
    return ""


def should_finish_early(session: dict) -> bool:
    scores = session.get("cluster_scores", {}) or {}
    turn_count = session.get("turn_count", 0)
    if turn_count < 6:
        return False
    for cluster_score in scores.values():
        if cluster_score.get("status") != "confirmed":
            continue
        if unasked_branch(session, cluster_score.get("rank", 0)):
            continue
        branch_id = cluster_score.get("dominant_branch", "")
        branch_state = (cluster_score.get("branches", {}) or {}).get(branch_id, {}) if branch_id else {}
        if not branch_state:
            continue
        followup_count = sum(
            1 for answer in cluster_score.get("answers", [])
            if answer.get("branch_id") == branch_id and answer.get("question_type") in ["severity", "red_flag", "context"]
        )
        if followup_count >= 2:
            return True
    return False


def build_safe_json_response(session: dict, meta: dict, reasoning: str = "") -> str:
    rank = meta.get("cluster_rank", 0) or 0
    question_type = meta.get("question_type", "context")
    branch_id = meta.get("branch_id", "")
    cluster = cluster_by_rank(session, rank) if rank else {}
    branch = branch_by_id(cluster, branch_id) if cluster and branch_id else {}

    if question_type == "final":
        message = (
            "Based on our conversation today, there is a strong enough pattern for wellness awareness. "
            "Please do consult a qualified physician and Ayurvedic practitioner for proper guidance. "
            "This is wellness awareness only and not a medical diagnosis."
        )
    else:
        question_bank = {
            "screening": branch.get("screening_questions", []),
            "drilldown": branch.get("drilldown_questions", []),
            "severity": branch.get("severity_questions", []),
            "red_flag": branch.get("red_flag_questions", []),
            "context": branch.get("severity_questions", []) + branch.get("drilldown_questions", []),
        }
        candidates = question_bank.get(question_type, []) or cluster.get("drilldown_questions", []) or cluster.get("screening_questions", [])
        question = candidates[0] if candidates else "Could you tell me a little more about how this has been feeling recently?"
        message = f"Thank you for sharing that. {question}"

    return json.dumps({
        "reasoning": reasoning or "Fallback JSON response created to preserve chat format.",
        "cluster_rank": rank,
        "question_type": question_type,
        "chat_message": message
    })


def update_cluster_support(cluster_score: dict, cluster: dict, branch_id: str, answer_label: str, user_message: str, question_type: str) -> None:
    cluster_score[answer_label] = cluster_score.get(answer_label, 0) + 1
    cluster_score["score"] = cluster_score.get("score", 0) + score_answer(answer_label)
    cluster_score["dominant_branch"] = branch_id or cluster_score.get("dominant_branch", "")

    if answer_label == "yes_strong":
        cluster_score["strong_yes_streak"] = cluster_score.get("strong_yes_streak", 0) + 1
        cluster_score["strong_no_streak"] = 0
    elif answer_label == "no_strong":
        cluster_score["strong_no_streak"] = cluster_score.get("strong_no_streak", 0) + 1
        cluster_score["strong_yes_streak"] = 0
    else:
        cluster_score["strong_yes_streak"] = 0
        cluster_score["strong_no_streak"] = 0

    branch = branch_by_id(cluster, branch_id) if cluster else {}
    branch_scores = cluster_score.get("branches", {}) or {}
    branch_score = branch_scores.get(branch_id, {}) if branch_id else {}
    organs = branch.get("organs", []) if branch else []
    diseases = branch.get("diseases", []) if branch else []

    if branch_score:
        branch_score[answer_label] = branch_score.get(answer_label, 0) + 1
        branch_score["asked_count"] = branch_score.get("asked_count", 0) + 1
        if answer_label == "yes_strong":
            branch_score["possible_organs"] = _dedupe_preserve(branch_score.get("possible_organs", []) + organs)
            branch_score["possible_diseases"] = _dedupe_preserve(branch_score.get("possible_diseases", []) + diseases)
        elif answer_label == "yes_but":
            branch_score["possible_organs"] = _dedupe_preserve(branch_score.get("possible_organs", []) + organs[:2])
            branch_score["possible_diseases"] = _dedupe_preserve(branch_score.get("possible_diseases", []) + diseases[:3])

        if branch_score.get("yes_strong", 0) >= 2 or (
            branch_score.get("yes_strong", 0) >= 1 and branch_score.get("yes_but", 0) >= 2
        ):
            branch_score["status"] = "confirmed"
            branch_score["confirmed_organs"] = _dedupe_preserve(branch_score.get("confirmed_organs", []) + organs)
            branch_score["confirmed_diseases"] = _dedupe_preserve(branch_score.get("confirmed_diseases", []) + diseases)
        elif branch_score.get("no_strong", 0) >= 2 or (
            branch_score.get("no_strong", 0) >= 1 and branch_score.get("no_but", 0) >= 2
        ):
            branch_score["status"] = "rejected"
        elif branch_score.get("yes_but", 0) > 0 or (
            branch_score.get("yes_strong", 0) > 0 and (
                branch_score.get("no_but", 0) > 0 or branch_score.get("no_strong", 0) > 0
            )
        ):
            branch_score["status"] = "drilldown"
        elif branch_score.get("yes_strong", 0) > 0:
            branch_score["status"] = "suspected"
        elif branch_score.get("no_strong", 0) > 0 or branch_score.get("no_but", 0) > 0:
            branch_score["status"] = "screening"

    confirmed_branches = [b for b in branch_scores.values() if b.get("status") == "confirmed"]
    suspected_branches = [b for b in branch_scores.values() if b.get("status") in ["suspected", "drilldown"]]
    rejected_branches = [b for b in branch_scores.values() if b.get("status") == "rejected"]

    cluster_score["confirmed_organs"] = _dedupe_preserve([
        item for b in confirmed_branches for item in b.get("confirmed_organs", [])
    ])
    cluster_score["confirmed_diseases"] = _dedupe_preserve([
        item for b in confirmed_branches for item in b.get("confirmed_diseases", [])
    ])
    cluster_score["possible_organs"] = _dedupe_preserve([
        item for b in suspected_branches for item in b.get("possible_organs", [])
    ])
    cluster_score["possible_diseases"] = _dedupe_preserve([
        item for b in suspected_branches for item in b.get("possible_diseases", [])
    ])

    if confirmed_branches:
        cluster_score["status"] = "confirmed"
    elif len(rejected_branches) >= max(1, min(2, len(branch_scores))) and not suspected_branches:
        cluster_score["status"] = "rejected"
    elif suspected_branches:
        cluster_score["status"] = "drilldown"
    elif cluster_score.get("yes_strong", 0) > 0 or cluster_score.get("yes_but", 0) > 0:
        cluster_score["status"] = "suspected"
    else:
        cluster_score["status"] = "screening"

    cluster_score["answers"].append({
        "answer": user_message,
        "result": answer_label,
        "question_type": question_type,
        "branch_id": branch_id,
        "branch_label": branch.get("label", "") if branch else "",
        "status_after_answer": cluster_score.get("status", "screening"),
    })


def build_runtime_state(session: dict, next_assistant_turn: int) -> str:
    return """

RUNTIME STATE FOR THIS TURN
Next assistant turn number: {turn}
Current cluster support scores:
{scores}

Use this state to choose the next question. If this is turn 15, provide the final summary and do not ask another question.
""".format(
        turn=next_assistant_turn,
        scores=json.dumps(session.get("cluster_scores", {}), indent=2)
    )


def fallback_question_meta(turn: int, session: dict) -> dict:
    if should_finish_early(session):
        return {"cluster_rank": 0, "question_type": "final"}

    cluster1 = (session.get("cluster_scores", {}) or {}).get("1", {}) or {}
    cluster1_positive = cluster1.get("yes_strong", 0)
    cluster1_negative = cluster1.get("no_strong", 0)
    cluster1_score = cluster1.get("score", 0)
    cluster1_status = cluster1.get("status", "screening")
    confirmed_rank = next(
        (
            score.get("rank", 1)
            for score in (session.get("cluster_scores", {}) or {}).values()
            if score.get("status") == "confirmed"
        ),
        None
    )

    if confirmed_rank and turn < 15:
        pending = unasked_branch(session, confirmed_rank)
        if pending:
            return {"cluster_rank": confirmed_rank, "question_type": "screening", "branch_id": pending}
        return {
            "cluster_rank": confirmed_rank,
            "question_type": "context",
            "branch_id": choose_branch_for_cluster(session, confirmed_rank, "context")
        }

    if turn == 1:
        return {"cluster_rank": 1, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, 1, "screening")}
    if turn == 2:
        if cluster1_positive > 0 and cluster1_score > 0:
            return {"cluster_rank": 1, "question_type": "drilldown", "branch_id": choose_branch_for_cluster(session, 1, "drilldown")}
        return {"cluster_rank": 1, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, 1, "screening")}
    if turn == 3:
        if cluster1_status == "rejected" or (cluster1_negative > 0 and cluster1_positive == 0 and cluster1_score <= 0):
            return {"cluster_rank": 2, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, 2, "screening")}
        if cluster1_status == "drilldown":
            return {"cluster_rank": 1, "question_type": "drilldown", "branch_id": choose_branch_for_cluster(session, 1, "drilldown")}
        return {"cluster_rank": 1, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, 1, "screening")}
    if turn == 4:
        if cluster1_status == "rejected" or (cluster1_negative > 0 and cluster1_positive == 0 and cluster1_score <= 0):
            strongest = strongest_cluster_rank(session)
            return {"cluster_rank": strongest, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, strongest, "screening")}
        if cluster1_status in ["drilldown", "suspected"]:
            return {"cluster_rank": 1, "question_type": "drilldown", "branch_id": choose_branch_for_cluster(session, 1, "drilldown")}
        return {"cluster_rank": 1, "question_type": "screening", "branch_id": choose_branch_for_cluster(session, 1, "screening")}
    if turn <= 8:
        strongest = strongest_cluster_rank(session)
        strongest_state = (session.get("cluster_scores", {}) or {}).get(str(strongest), {}) or {}
        if strongest_state.get("status") == "confirmed":
            return {"cluster_rank": strongest, "question_type": "context", "branch_id": choose_branch_for_cluster(session, strongest, "context")}
        return {"cluster_rank": strongest, "question_type": "drilldown", "branch_id": choose_branch_for_cluster(session, strongest, "drilldown")}
    if turn <= 11:
        if confirmed_rank:
            return {"cluster_rank": confirmed_rank, "question_type": "context", "branch_id": choose_branch_for_cluster(session, confirmed_rank, "context")}
        strongest = strongest_cluster_rank(session)
        return {"cluster_rank": strongest, "question_type": "severity", "branch_id": choose_branch_for_cluster(session, strongest, "severity")}
    if turn <= 13:
        if confirmed_rank:
            return {"cluster_rank": confirmed_rank, "question_type": "context", "branch_id": choose_branch_for_cluster(session, confirmed_rank, "context")}
        strongest = strongest_cluster_rank(session)
        return {"cluster_rank": strongest, "question_type": "red_flag", "branch_id": choose_branch_for_cluster(session, strongest, "red_flag")}
    if turn == 14:
        strongest = strongest_cluster_rank(session)
        return {"cluster_rank": strongest, "question_type": "context", "branch_id": choose_branch_for_cluster(session, strongest, "context")}
    return {"cluster_rank": 0, "question_type": "final"}


def strongest_cluster_rank(session: dict) -> int:
    scores = session.get("cluster_scores", {}) or {}
    if not scores:
        return 1
    ranked = sorted(
        scores.values(),
        key=lambda item: (
            item.get("score", 0),
            item.get("yes_strong", 0),
            item.get("yes_but", 0),
            -item.get("no_strong", 0),
            -int(item.get("rank", 99)),
        ),
        reverse=True
    )
    return ranked[0].get("rank", 1)


def cluster_by_rank(session: dict, rank: int) -> dict:
    for cluster in session.get("chat_plan", {}).get("clusters", []):
        if cluster.get("rank") == rank:
            return cluster
    return {}


def build_backend_reasoning(session: dict, turn: int, meta: dict, answer_label: str = None) -> dict:
    rank = meta.get("cluster_rank", 0) or 0
    question_type = meta.get("question_type", "")
    branch_id = meta.get("branch_id", "")
    cluster = cluster_by_rank(session, rank) if rank else {}
    branch = branch_by_id(cluster, branch_id) if cluster and branch_id else {}
    score_state = session.get("cluster_scores", {}).get(str(rank), {}) if rank else {}
    branch_state = (score_state.get("branches", {}) or {}).get(branch_id, {}) if rank and branch_id else {}

    stage_reason = {
        "screening": "This turn is checking whether the active cluster deserves deeper attention or should be set aside.",
        "drilldown": "This turn drills deeper because the cluster has partial or mixed support and still needs confirmation.",
        "severity": "This turn checks duration, frequency, or severity so we can tell a mild pattern from a stronger one.",
        "red_flag": "This turn checks warning signs for the strongest supported cluster.",
        "context": "This turn gathers simple lifestyle/context details because the cluster is already strong enough or confirmed.",
        "final": "This turn summarizes the clusters after the 15-question interview.",
    }.get(question_type, "This turn follows the active HitList interview plan.")

    if rank and turn <= 4:
        selection_reason = (
            f"Cluster {rank} is active now because its gate status is {score_state.get('status', 'screening')} "
            f"with yes_strong={score_state.get('yes_strong', 0)}, yes_but={score_state.get('yes_but', 0)}, "
            f"no_but={score_state.get('no_but', 0)}, no_strong={score_state.get('no_strong', 0)}."
        )
    elif rank:
        selection_reason = (
            f"Cluster {rank} is active now because its current support score is {score_state.get('score', 0)} "
            f"and gate status is {score_state.get('status', 'screening')} "
            f"(yes_strong={score_state.get('yes_strong', 0)}, yes_but={score_state.get('yes_but', 0)}, "
            f"no_but={score_state.get('no_but', 0)}, no_strong={score_state.get('no_strong', 0)})."
        )
    else:
        selection_reason = "No cluster is being asked because the interview has reached the summary turn."

    candidates = []
    if branch:
        question_bank = {
            "screening": branch.get("screening_questions", []),
            "drilldown": branch.get("drilldown_questions", []),
            "severity": branch.get("severity_questions", []),
            "red_flag": branch.get("red_flag_questions", []),
            "context": branch.get("severity_questions", []) + branch.get("drilldown_questions", []),
        }
        candidates = question_bank.get(question_type, [])[:4]

    why_this = []
    if cluster:
        terms = ", ".join((branch.get("organs", []) + branch.get("diseases", []))[:6]) if branch else ", ".join((cluster.get("organs", []) + cluster.get("diseases", []))[:6])
        terms = terms or cluster.get("system", "")
        why_this.append(f"Active cluster: #{rank} {cluster.get('system', '')}.")
        if branch:
            why_this.append(f"Active branch: {branch.get('label', '')}.")
        why_this.append(f"Raw HitList terms considered: {terms}.")
        symptom_targets = branch.get("symptom_targets", []) if branch else cluster.get("symptom_targets", [])
        if symptom_targets:
            why_this.append(
                "Symptom targets in this branch: " +
                ", ".join(symptom_targets[:6]) + "."
            )
        if candidates:
            why_this.append(
                "Question candidates for this stage: " +
                " | ".join(candidates) + "."
            )
        if branch_state:
            why_this.append(
                f"Branch gate status: {branch_state.get('status', 'screening')} "
                f"(yes_strong={branch_state.get('yes_strong', 0)}, yes_but={branch_state.get('yes_but', 0)}, "
                f"no_but={branch_state.get('no_but', 0)}, no_strong={branch_state.get('no_strong', 0)})."
            )
        why_this.append(selection_reason)
        why_this.append(stage_reason)
        if answer_label:
            why_this.append(f"The user's last answer was classified as {answer_label}, which affected the next branch.")
    else:
        why_this.append(stage_reason)

    return {
        "turn": turn,
        "cluster_rank": rank,
        "cluster_system": cluster.get("system", ""),
        "branch_id": branch_id,
        "branch_label": branch.get("label", ""),
        "question_type": question_type,
        "selection_reason": selection_reason,
        "stage_reason": stage_reason,
        "raw_terms": {
            "organs": branch.get("organs", []) if branch else (cluster.get("organs", []) if cluster else []),
            "diseases": branch.get("diseases", []) if branch else (cluster.get("diseases", []) if cluster else []),
        },
        "symptom_targets": (branch.get("symptom_targets", []) if branch else (cluster.get("symptom_targets", []) if cluster else []))[:8],
        "candidate_questions": candidates,
        "cluster_score_snapshot": score_state,
        "branch_score_snapshot": branch_state,
        "explanation_text": "\n".join(why_this),
    }


# ── Function 3: Build system prompt from patient context ─────────
def build_system_prompt(
    rule1_result,
    complete_analysis,
    dasha_result,
    disease_filter=None,
    most_probable=None,
    gender="Unknown",
    hitlist_result=None
) -> str:
    """
    Builds the Gemma system prompt using the patient's chart data.
    """
    from rules import find_dominant_theme

    if disease_filter is None:
        disease_filter = {}
    if most_probable is None:
        most_probable = []
    if hitlist_result is None:
        hitlist_result = {}

    def dedupe_keep_order(items):
        seen = set()
        output = []
        for item in items:
            clean = (item or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            output.append(clean)
        return output

    def build_hitlist_cluster_text(cluster, rank):
        label = cluster.get("label", "Unknown System")
        planet_terms = []
        nakshatra_terms = []
        rashi_terms = []

        for entry in cluster.get("trail", []):
            source = entry.get("source", "")
            term = entry.get("term", "")
            if "HL1" in source or "Planet" in source:
                planet_terms.append(term)
            elif "HL2" in source or "Nakshatra" in source:
                nakshatra_terms.append(term)
            elif "HL3" in source or "Rashi" in source:
                rashi_terms.append(term)

        planet_terms = dedupe_keep_order(planet_terms)
        nakshatra_terms = dedupe_keep_order(nakshatra_terms)
        rashi_terms = dedupe_keep_order(rashi_terms)
        probable_diseases = dedupe_keep_order(nakshatra_terms + planet_terms)
        organs = dedupe_keep_order(rashi_terms)

        return (
            f"CLUSTER {rank}: {label}\n"
            f"  Planet: {', '.join(planet_terms) if planet_terms else 'None'}\n"
            f"  Nakshatra: {', '.join(nakshatra_terms) if nakshatra_terms else 'None'}\n"
            f"  Rashi: {', '.join(rashi_terms) if rashi_terms else 'None'}\n"
            f"  Organs: {', '.join(organs) if organs else 'General'}\n"
            f"  Probable Diseases: {', '.join(probable_diseases) if probable_diseases else 'General'}\n"
        )

    top4 = hitlist_result.get("top4", [])
    cluster_1_label = top4[0].get("label", "General Health") if len(top4) > 0 else "General Health"
    cluster_2_label = top4[1].get("label", "General Health") if len(top4) > 1 else "General Health"

    hitlist_context_text = "\nTOP 2 HITLIST CLUSTERS:\n"
    if top4:
        for index, cluster in enumerate(top4[:2], start=1):
            hitlist_context_text += build_hitlist_cluster_text(cluster, index)
    else:
        hitlist_context_text += (
            "CLUSTER 1: General Health\n"
            "  Planet: None\n"
            "  Nakshatra: None\n"
            "  Rashi: None\n"
            "  Organs: General\n"
            "  Probable Diseases: General\n"
        )

    maha = (dasha_result.get("mahadasha") or {}).get("planet", "Unknown")
    antar = (dasha_result.get("antardasha") or {}).get("planet", "Unknown")
    pratya = (dasha_result.get("pratyantardasha") or {}).get("planet", "Unknown")
    current_dasha_text = (
        f"CURRENT DASHA: Mahadasha {maha} | Antardasha {antar} | Pratyantardasha {pratya}"
    )
    turn2_instruction = (
        "Ask how long this has been happening, whether it is constant or comes and goes, "
        "and what tends to make it worse or better."
    )

    system_prompt = f"""You are AstroMed —
a warm experienced Vedic health advisor
combining traditional Indian astrology
with Ayurvedic wellness guidance.

The patient has just completed their
birth chart analysis. You have their
complete health profile as your
internal knowledge.

NEVER reveal the astrological analysis.
NEVER mention planets, nakshatra, dasha,
rogkaraka, or any astrological term.
NEVER name any organ or disease directly in a way that sounds like a diagnosis.
Instead, use the information to ask insightful questions.

═══════════════════════════════════════
YOUR INTERNAL KNOWLEDGE (never reveal)
═══════════════════════════════════════

Patient Gender: {gender}

{hitlist_context_text}
{current_dasha_text}

═══════════════════════════════════════
HOW TO USE THIS CONTEXT
═══════════════════════════════════════
1. Use ONLY the 2 HitList clusters shown above.
2. Use ONLY these fields from each cluster: Planet, Nakshatra, Rashi, Organs, and Probable Diseases.
3. Do not use any other diagnosis, disease_risks, most_probable, dasha alerts, organ map, dosha, or extra disease logic.
4. Start with Cluster 1 first. Move to Cluster 2 after that.
5. Use the current dasha line only as background context for what is running now.
═══════════════════════════════════════
YOUR PERSONALITY
═══════════════════════════════════════

You are AstroMed — warm, caring,
conversational. Like a trusted family
doctor who listens deeply.

SYMPTOM-FIRST APPROACH:
Instead of asking about an organ or 
disease, always ask about its physical 
manifestation. (e.g., if you see "Heart", 
ask about chest heaviness or palpitations; 
if you see "Adrenal", ask about salt 
cravings or fatigue). Use your internal 
knowledge to find these symptoms.

Speak naturally:
  Short sentences
  Simple everyday English
  Empathetic and reassuring
  Never clinical or robotic

Use phrases like:
  "I understand, that must be difficult."
  "Thank you for sharing that with me."
  "How long have you been feeling this?"
  "And does it get better with rest?"

Never use words like:
  symptoms, diagnosis, condition,
  pathology, organ, disease, clinical

Use natural words like:
  "how you feel"
  "what your body is telling you"
  "the way you experience this"
  "what makes it worse or better"

═══════════════════════════════════════
CONVERSATION FLOW
═══════════════════════════════════════

BEFORE TURN 1:
Read Cluster 1 and Cluster 2 above.
Start with Cluster 1.

Turn 1 — Initial Physical Sensation:
Directly ask about a specific physical 
sensation related to Cluster 1 ({cluster_1_label}).

MANDATORY GREETING: Start with "Hello, I am AstroMedica, and I am here to assist you."

Example: "Hello, I am AstroMedica, and I am here to assist you. Have you noticed any particular cravings for salty foods lately, or perhaps feeling a bit lightheaded when you stand up quickly?"
(Adjust example to fit the actual diseases in CLUSTER 1).
Never name the disease or organ directly.

Turn 2 — Duration and Quality:
{turn2_instruction}
Keep the focus on Cluster 1.

Turn 3 — Specific Manifestation:
Now ask about ONE specific sensation related to the 
"Probable Diseases" or "Organs" listed in 
CLUSTER 1.

INSTRUCTION: You MUST use your internal knowledge to identify 
the actual physical symptoms of the diseases listed. 
For example, if "Liver" is the focus, don't ask about the liver; 
ask about bitterness in the mouth, yellowing of eyes, or 
unusual fatigue after eating.

Pick the most prominent symptom that fits the group pattern.
Frame it as a physical experience.
Never name the disease or organ directly.

Turn 4 — Move to Cluster 2:
Move to the second cluster ({cluster_2_label}).
Ask broadly about that body region first.
Same pattern — broad first then specific.

Turn 5-7 — Continue With Current Dasha In Mind:
Keep the currently running dasha as background context only.
Stay focused on Cluster 1 and Cluster 2.
Specific experience questions only.
Never name planets or organs directly.

Turn 8+ — Assessment:
Only after minimum 6 exchanges.
Your chat_message MUST start with:
"Based on our conversation today,"
Your chat_message MUST end with:
"Please do consult a qualified physician
and Ayurvedic practitioner for proper
guidance. What I have shared is for
wellness awareness only and not a
medical diagnosis."

═══════════════════════════════════════
STRICT RULES
═══════════════════════════════════════

1. CRITICAL OUTPUT FORMAT:
   You MUST respond ONLY with a valid 
   JSON object on every single turn.
   
   Do NOT include markdown formatting.
   Do NOT wrap in ```json blocks.
   Do NOT add any text outside the JSON.
   
   Your output must strictly follow 
   this exact schema every time:
   
   {{
     "reasoning": "1 sentence explaining which planet (RK level), which body system, whether drilling down or moving to new system, and whether dasha is active for this planet.",
     
     "chat_message": "Your warm empathetic reply and next question directed at the patient."
   }}
   
   Reasoning field examples:
   
   "Saturn RK1 in 6th house — asking about leg heaviness. Saturn Mahadasha active — highest urgency. First question."
   
   "Patient confirmed joint pain — drilling down on duration before moving to Jupiter RK2 liver symptoms. Saturn Mahadasha still active."
   
   "Drill-down complete on Saturn RK1 — moving to Jupiter RK2. Governs liver and fat. Not in active dasha — lifetime tendency only."
   
   "Venus Drishti aspecting 6th house — asking about eye discomfort. Venus Antardasha active — moderate urgency."
   
   The patient will ONLY see chat_message.
   The reasoning field is for internal tracking only.
   NEVER put astrological terms, planet names, or JSON tags inside chat_message.

2. ONE question per response — never two
3. Maximum 3 sentences per response
4. Minimum 6 exchanges before assessment
5. Forbidden words: kidney, liver, heart,
   lung, ovary, bladder, stomach, bowel,
   diabetes, cancer, any disease name,
   any organ name, any planet name,
   any astrological term
6. Always acknowledge before next question
7. Sound human — vary your responses
   Never repeat same phrases twice
8. If patient goes off topic — gently
   guide back to health
9. If patient seems worried — reassure
   before continuing
10. If patient gives a very short answer
    (yes, no, tired, bad, okay, fine) —
    apply the DRILL-DOWN RULE.
    Validate their feeling warmly in 
    one sentence, then ask ONE specific 
    detail about that sensation.
    
    EXAMPLE IF PATIENT SAYS "YES":
    {{
      "reasoning": "Patient said yes. Drilling down on sensation.",
      "chat_message": "Thank you for sharing that. How long have you been noticing this feeling?"
    }}
11. NEVER ask an umbrella or general health 
    question at any point in the conversation.
    Every single question must be about one 
    specific physical experience or sensation.
    
    SYMPTOM SEARCH RULE: For every disease or organ 
    listed in the context, you MUST internally 
    identify its primary symptoms and ask about 
    them instead of the organ itself.
    (Example: For 'Adrenal', ask about salt cravings, 
    dizziness when standing, or dark spots on skin).

12. Turn 1 question must come from the FIRST 
    body system in "Body systems of concern". 
    Turn 2 from the second system. Turn 3 from 
    the third. Work through your internal 
    knowledge in order — most urgent first.

13. CRITICAL GENDER FILTER: The patient is {gender}. You MUST completely ignore and filter out any organs or diseases from your internal knowledge that belong to the opposite sex (e.g., do not ask a male about ovaries/uterus, do not ask a female about prostate). Do not mention this filtering process to the user.

═══════════════════════════════════════
RESPONSE LENGTH
═══════════════════════════════════════

Each turn:
  1 sentence acknowledging their answer
  1-2 sentences for next question
  3 sentences maximum total

Final assessment:
  200-250 words
  Warm and practical
  No jargon whatsoever

CRITICAL OUTPUT REQUIREMENT: You are communicating directly with an API parser. You MUST return your response as a single, valid JSON object. DO NOT wrap the JSON in markdown formatting (e.g., ```json). DO NOT output any conversational text, greetings, or explanations outside of the JSON object. 
Example required format:
{{
  "reasoning": "Internal logic here.",
  "chat_message": "Your conversational reply here."
}}
"""
    return system_prompt


# ── Function 4: Start chat session ───────────────────────────────
def start_chat(
    session_id: str,
    patient_id: str,
    conversation_store: dict,
    call_gemini_fn  # the async call_gemini function from server.py
) -> str:
    """
    Loads patient context from files, builds system prompt,
    gets opening message from Gemma, saves to chat_log.json,
    stores session in conversation_store.
    Returns opening message text.
    """
    context       = load_patient_context(patient_id)
    chat_plan = build_hitlist_chat_plan(context.get("hitlist", {}), max_clusters=3)
    # Gemini-written questions per disease (question_generator.py); falls back
    # to symptom_mapper's built-in questions for anything it can't cover.
    from question_generator import improve_chat_plan
    chat_plan = improve_chat_plan(patient_id, chat_plan, context.get("gender", "Unknown"))
    system_prompt = build_hitlist_system_prompt(
        chat_plan,
        context.get("gender", "Unknown"),
        context.get("dasha_chat_priority", {})
    )

    debug_context = build_debug_context(context, system_prompt, chat_plan)

    initial_msg = {
        "role":    "user",
        "content": "Hello. I am ready to begin my health consultation."
    }

    conversation_store[session_id] = {
        "system_prompt": system_prompt,
        "debug_context": debug_context,
        "chat_plan": chat_plan,
        "cluster_scores": initial_cluster_scores(chat_plan),
        "last_question_meta": {},
        "messages":      [initial_msg],
        "turn_count":    0,
        "patient_id":    patient_id,
        "chat_plan_source": context.get("chat_plan_source", "hitlist_v1"),
    }

    # Get opening from Gemma
    runtime_prompt = system_prompt + build_runtime_state(
        conversation_store[session_id],
        1
    )
    opening = ask_gemini(call_gemini_fn, runtime_prompt, [initial_msg], 1)
    parsed_opening = parse_model_json(opening)
    if not parsed_opening:
        log_unreadable_reply(1, opening)
        opening_meta = fallback_question_meta(1, conversation_store[session_id])
        opening = build_safe_json_response(
            conversation_store[session_id],
            opening_meta,
            "Model returned non-JSON for opening turn. Using safe fallback response."
        )
        parsed_opening = parse_model_json(opening)
    conversation_store[session_id]["last_question_meta"] = {
        **fallback_question_meta(1, conversation_store[session_id]),
        **{
            key: value for key, value in {
                "cluster_rank": parsed_opening.get("cluster_rank"),
                "question_type": parsed_opening.get("question_type"),
                "branch_id": parsed_opening.get("branch_id")
            }.items() if value is not None
        }
    }
    # Same check as send_message: the opening answer must be credited to a real branch.
    opening_meta = conversation_store[session_id]["last_question_meta"]
    if opening_meta.get("branch_id") and not branch_by_id(
            cluster_by_rank(conversation_store[session_id], opening_meta.get("cluster_rank") or 0),
            opening_meta["branch_id"]):
        opening_meta["branch_id"] = choose_branch_for_cluster(
            conversation_store[session_id], opening_meta.get("cluster_rank") or 0,
            opening_meta.get("question_type", "screening"))
    conversation_store[session_id]["debug_context"]["latest_question_explanation"] = build_backend_reasoning(
        conversation_store[session_id],
        1,
        conversation_store[session_id]["last_question_meta"]
    )

    conversation_store[session_id]["messages"].append({
        "role":    "assistant",
        "content": opening
    })
    conversation_store[session_id]["turn_count"] += 1

    save_chat_turn(
        patient_id,
        1,
        "assistant",
        opening,
        {
            **conversation_store[session_id]["last_question_meta"],
            "session_id": session_id,
            "backend_reasoning": conversation_store[session_id]["debug_context"]["latest_question_explanation"]
        }
    )

    return {
        "response": opening,
        "debug_context": debug_context
    }


# ── Function 5: Handle one user message ──────────────────────────
def send_message(
    session_id: str,
    user_message: str,
    conversation_store: dict,
    call_gemini_fn  # the async call_gemini function from server.py
) -> dict:
    """
    Adds user message to history, gets Gemma response,
    saves both to chat_log.json, detects if final assessment.
    Returns: { response, turn, is_final, patient_id }
    """
    session    = conversation_store[session_id]
    patient_id = session["patient_id"]
    answer_label = classify_answer(user_message)
    last_meta = session.get("last_question_meta", {}) or {}
    cluster_rank = last_meta.get("cluster_rank")
    branch_id = last_meta.get("branch_id", "")
    if cluster_rank:
        score_key = str(cluster_rank)
        cluster_score = session.get("cluster_scores", {}).get(score_key)
        cluster = cluster_by_rank(session, cluster_rank)
        if cluster_score:
            update_cluster_support(
                cluster_score,
                cluster,
                branch_id,
                answer_label,
                user_message,
                last_meta.get("question_type", "")
            )
    session.setdefault("debug_context", {})["latest_answer_result"] = answer_label

    # Add user message and save to file
    session["messages"].append({
        "role":    "user",
        "content": user_message
    })

    turn = session["turn_count"] + 1
    save_chat_turn(patient_id, turn, "user", user_message, {
        "session_id": session_id,
        "answer_result": answer_label,
        "answered_cluster_rank": cluster_rank,
        "answered_branch_id": branch_id,
        "answered_question_type": last_meta.get("question_type", "")
    })

    # Get Gemma response
    next_assistant_turn = session["turn_count"] + 1
    # Decide BEFORE writing the reply whether this turn ends the chat - the
    # early-finish check used to run only after it, so the chat could close on
    # a question the patient never got to answer.
    finish_now = next_assistant_turn >= 15 or should_finish_early(
        {**session, "turn_count": next_assistant_turn})
    runtime_prompt = session["system_prompt"] + build_runtime_state(
        session,
        next_assistant_turn
    )
    if finish_now:
        runtime_prompt += ("\nTHIS IS THE FINAL TURN. Give the final summary now in the FINAL TURN FORMAT "
                           "(question_type \"final\"). Do not ask another question.\n")
    response = ask_gemini(call_gemini_fn, runtime_prompt, session["messages"], next_assistant_turn)
    parsed_response = parse_model_json(response)
    fallback_meta = ({"cluster_rank": 0, "question_type": "final"} if finish_now
                     else fallback_question_meta(next_assistant_turn, session))
    if parsed_response and finish_now and parsed_response.get("question_type") != "final":
        print(f"[Chat] Turn {next_assistant_turn}: final turn but Gemini asked another question - using the standard summary")
        parsed_response = {}
        response = ""
    if not parsed_response:
        log_unreadable_reply(next_assistant_turn, response)
        response = build_safe_json_response(
            session,
            fallback_meta,
            "Model returned non-JSON. Using safe fallback response."
        )
        parsed_response = parse_model_json(response)
    session["last_question_meta"] = {
        **fallback_meta,
        **{
            key: value for key, value in {
                "cluster_rank": parsed_response.get("cluster_rank"),
                "question_type": parsed_response.get("question_type"),
                "branch_id": parsed_response.get("branch_id")
            }.items() if value is not None
        }
    }
    # Keep the model's branch_id only if it names a real branch of that
    # cluster; otherwise the answer would be credited to nothing.
    meta = session["last_question_meta"]
    if meta.get("branch_id") and not branch_by_id(cluster_by_rank(session, meta.get("cluster_rank") or 0), meta["branch_id"]):
        meta["branch_id"] = choose_branch_for_cluster(session, meta.get("cluster_rank") or 0, meta.get("question_type", "screening"))
    session["debug_context"]["latest_question_explanation"] = build_backend_reasoning(
        session,
        next_assistant_turn,
        session["last_question_meta"],
        answer_label=answer_label
    )

    session["messages"].append({
        "role":    "assistant",
        "content": response
    })
    session["turn_count"] += 1
    turn = session["turn_count"]

    save_chat_turn(patient_id, turn, "assistant", response, {
        **session["last_question_meta"],
        "session_id": session_id,
        "cluster_scores": session.get("cluster_scores", {}),
        "backend_reasoning": session["debug_context"]["latest_question_explanation"]
    })

    # Detect final assessment
    is_final = (
        (turn >= 15 or should_finish_early(session)) and
        "based on our conversation today" in response.lower()
    )
    if finish_now or turn >= 15:
        is_final = True

    # Mark completed in log
    if is_final:
        folder   = f"patients/{patient_id}"
        log_file = f"{folder}/chat_log.json"
        with open(log_file) as f:
            log = json.load(f)
        log["completed"]        = True
        log["completed_at"]     = datetime.now().isoformat()
        log["final_assessment"] = response
        log["hitlist_cluster_scores"] = session.get("cluster_scores", {})
        log["chat_plan_source"] = session.get("chat_plan_source", "hitlist_v1")
        with open(log_file, "w") as f:
            json.dump(log, f, indent=2)
        sync_rl_report(patient_id, log)
        if session.get("skip_rl"):
            # Tester chat: not for RL training (set in server.start_consultation).
            print(f"[RL] Patient {patient_id}: tester chat - RL not queued")
        else:
            queue_rl(patient_id)
        print(f"✅ Chat completed — log saved for patient {patient_id}")

    session.setdefault("debug_context", {})["cluster_scores"] = session.get("cluster_scores", {})

    return {
        "response":   response,
        "turn":       turn,
        "is_final":   is_final,
        "patient_id": patient_id,
        "debug_context": session.get("debug_context", {})
    }
