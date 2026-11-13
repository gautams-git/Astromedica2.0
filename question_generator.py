"""
question_generator.py — Gemini writes the chat's question set at chat start.

Why: symptom_mapper.py has 14 hand-written symptom groups. Organs outside
them (nose, ears, eyes, throat...) got only generic fallback questions, and
every kidney disease got the same "swelling around the ankles" question.

What it does, once per patient and Combined Algo result:
  1. Takes the chat plan's top-3 organs and their Combined Algo diseases.
  2. Asks Gemini (Vertex AI, gemini_client.py) for one question branch per
     disease: plain single-symptom questions (screening / drilldown /
     severity / red flags), plus "telling apart" questions per organ that
     separate it from the other two.
  3. Checks every question (no disease names, no internal organ names, one
     short question each) and drops the ones that fail.
  4. Saves the result to patients/<id>/chat_questions.json - reused by later
     chats of the same patient and readable for review.
Any organ Gemini can't cover keeps symptom_mapper's built-in questions, and
if Gemini fails completely the chat runs on the built-in plan as before.

Branch ids are "gen_<disease>" (generated_branch_id), which combine_output.py
uses to score each disease on the results page.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
from datetime import datetime

import httpx

import gemini_client
from config import GEMINI_MODEL

QUESTIONS_FILE = "chat_questions.json"
# Bump when the prompt or the checks change, so saved question sets are rebuilt.
PROMPT_VERSION = 3

LIMITS = {"symptoms": 6, "screening": 2, "drilldown": 3, "severity": 2, "red_flags": 2}
TELL_APART_LIMIT = 2
MAX_QUESTION_CHARS = 170

# Internal organs a patient can't see or feel. Everyday body parts (nose,
# ears, eyes, throat, chest, stomach, skin, back, legs...) are allowed.
INTERNAL_ORGAN_WORDS = [
    "kidney", "kidneys", "renal", "liver", "hepatic", "lung", "lungs", "pulmonary",
    "thyroid", "pancreas", "pancreatic", "intestine", "intestines", "bowel", "colon",
    "bladder", "prostate", "uterus", "womb", "ovary", "ovaries", "spleen", "gallbladder",
    "gall bladder", "adrenal", "sinus", "sinuses", "tonsil", "tonsils", "larynx", "pharynx",
    "oesophagus", "esophagus", "trachea", "bronchi", "cervix",
]

PROMPT = """You write the questions for a friendly health check-up chat in India.
The people answering can be young or old, may have little schooling, and may not
speak English as a first language. Every question must be easy for ALL of them.

For each body area below and each condition under it, write questions that help
find out whether the person has that condition's symptoms.

PATIENT GENDER: {gender}

BODY AREAS AND CONDITIONS, as JSON. Each string in "conditions" is ONE condition,
even if it has a comma in it. The patient must never see these condition names.
{areas}

RULES FOR EVERY QUESTION
1. Ask about ONE symptom only. Never use "or" / "and" to join two symptoms,
   two body parts or two words that mean the same - pick the single best word.
   Bad:  "Do you have swelling in your ankles or puffy eyes?"
   Good: "In the morning, do your eyes look puffy?"
   Bad:  "Do you feel dizzy or lightheaded?"     Good: "Do you feel dizzy when you stand up?"
   Bad:  "Does your skin or eyes look yellow?"   Good: "Do your eyes look yellow?"
   Bad:  "Do you feel very weak and tired?"      Good: "Do you feel tired all day?"
2. Use simple everyday words a 12-year-old understands. No medical words.
   Say "pass urine", not "urinate"; "swelling", not "oedema"; "coughing up
   blood", not "haemoptysis"; "out of breath", not "dyspnoea".
3. Keep it short: one sentence, at most 18 words, ending with "?".
4. It must be answerable with yes / no / sometimes.
5. Never say a condition name and never say an internal organ (kidney, liver,
   lungs, thyroid, bladder, prostate, sinus, tonsils...). Everyday body parts
   people can see or feel are fine: nose, ears, eyes, throat, mouth, teeth,
   skin, chest, stomach, back, joints, legs, feet, head.
6. Neutral and calm: no scary words, do not suggest the answer
   (not "You surely have..."). Start with "Do you", "Have you", "Does your", "Is your".
7. Respect the gender: no questions that do not apply to it.
8. Ask about each condition's OWN typical symptoms, wherever in the body they
   are - not about the body area it is listed under. An eye condition listed
   under "Ears" gets eye questions; a cough illness listed under "Nose" gets
   cough, fever and weight questions.
9. Make each condition's questions specific to THAT condition, so a "yes"
   really points to it. Never give two conditions the same question.

FOR EACH CONDITION RETURN
- "symptoms": {symptoms} short plain symptom phrases (e.g. "blocked nose at night")
- "screening": {screening} questions about the most common, mild symptom first
- "drilldown": {drilldown} questions about more specific signs of this condition
- "severity": {severity} questions about how often / how long / effect on daily life
- "red_flags": {red_flags} questions about warning signs that need a doctor soon

FOR EACH BODY AREA ALSO RETURN
- "telling_apart": {tell_apart} questions about a symptom that is typical for this
  area but NOT for the other areas in the list, to tell them apart.

Return only JSON in exactly this shape, same areas and conditions, same spelling:
{{"areas": [{{"area": "...", "telling_apart": ["..."],
  "conditions": [{{"condition": "...", "symptoms": ["..."], "screening": ["..."],
    "drilldown": ["..."], "severity": ["..."], "red_flags": ["..."]}}]}}]}}
"""


def _norm(value) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def generated_branch_id(disease: str) -> str:
    """Branch id of a generated question branch; combine_output scores by it."""
    return "gen_" + _norm(disease).replace(" ", "_")


def _plan_input(chat_plan: dict) -> list:
    areas = []
    for cluster in chat_plan.get("clusters", []):
        organ = cluster.get("system", "")
        diseases = [d for d in (cluster.get("diseases") or []) if _norm(d) and _norm(d) != _norm(organ)]
        areas.append({"rank": cluster.get("rank"), "area": organ, "conditions": diseases or [organ]})
    return areas


def _cache_key(areas: list, gender: str) -> str:
    raw = json.dumps({"v": PROMPT_VERSION, "model": GEMINI_MODEL, "gender": gender, "areas": areas},
                     sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


async def _ask_gemini(prompt: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingBudget": 1024},
        },
    }
    headers = await gemini_client.auth_headers()
    # 429 = per-minute quota, 5xx = overloaded; both usually clear within seconds.
    waits = [3, 8]
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(len(waits) + 1):
            r = await client.post(gemini_client.generate_url(GEMINI_MODEL), json=payload, headers=headers)
            if r.status_code not in (429, 500, 502, 503, 504) or attempt == len(waits):
                break
            print(f"[Questions] Gemini {r.status_code}, retrying in {waits[attempt]}s")
            await asyncio.sleep(waits[attempt])
    if r.status_code in (401, 403):
        raise RuntimeError(gemini_client.auth_failure_message(r.status_code, r.text))
    r.raise_for_status()
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _bad_question(text: str, banned_names: list) -> str:
    """Why a generated question is rejected, or '' if it is fine."""
    norm = f" {_norm(text)} "
    if not text.endswith("?"):
        return "not a question"
    if len(text) > MAX_QUESTION_CHARS:
        return "too long"
    if text.count("?") > 1:
        return "more than one question"
    if " or " in norm or " and " in norm:
        return "joins two things with or/and"
    for word in INTERNAL_ORGAN_WORDS:
        if f" {_norm(word)} " in norm:
            return f"names an internal organ ({word})"
    for name in banned_names:
        if len(name) >= 5 and f" {name} " in norm:
            return f"names a condition ({name})"
    return ""


def _clean_list(items, limit: int, banned_names: list, rejected: list, questions: bool = True) -> list:
    out, seen = [], set()
    for item in items if isinstance(items, list) else []:
        text = " ".join(str(item or "").split())
        if not text or _norm(text) in seen:
            continue
        if questions:
            reason = _bad_question(text, banned_names)
            if reason:
                rejected.append({"question": text, "reason": reason})
                continue
        seen.add(_norm(text))
        out.append(text)
        if len(out) >= limit:
            break
    return out


# Condition names built from everyday words ("Eye sore", "Neck problems") are
# fine in a question; clinical names ("Tuberculosis") are not.
EVERYDAY_BODY_WORDS = {
    "eye", "eyes", "ear", "ears", "nose", "throat", "neck", "mouth", "teeth", "tooth", "skin",
    "head", "back", "chest", "stomach", "leg", "legs", "foot", "feet", "hand", "hands", "joint",
    "joints", "knee", "knees", "hair", "nail", "nails",
}


def _banned_names(areas: list) -> list:
    names = {_norm(c) for a in areas for c in a["conditions"]}
    return sorted(n for n in names if not (set(n.split()) & EVERYDAY_BODY_WORDS))


def _validate(raw: dict, areas: list) -> tuple[dict, list]:
    """{rank: {"telling_apart": [...], "conditions": {name: {...}}}} for the areas
    Gemini covered usably, plus the rejected questions."""
    rejected = []
    banned = _banned_names(areas)
    by_name = {_norm(a.get("area")): a for a in (raw.get("areas") or []) if isinstance(a, dict)}
    result = {}
    for area in areas:
        got = by_name.get(_norm(area["area"]))
        if not got:
            continue
        got_conditions = {_norm(c.get("condition")): c for c in (got.get("conditions") or []) if isinstance(c, dict)}
        conditions = {}
        used = set()   # a question already given to another condition in this area
        for name in area["conditions"]:
            c = got_conditions.get(_norm(name))
            if not c:
                continue
            cleaned = {"symptoms": _clean_list(c.get("symptoms"), LIMITS["symptoms"], banned, rejected, questions=False)}
            for key in ("screening", "drilldown", "severity", "red_flags"):
                fresh = [q for q in (c.get(key) if isinstance(c.get(key), list) else [])
                         if _norm(q) not in used]
                cleaned[key] = _clean_list(fresh, LIMITS[key], banned, rejected)
            if cleaned["screening"]:          # no screening question = can't open this condition
                conditions[name] = cleaned
                used |= {_norm(q) for key in ("screening", "drilldown", "severity", "red_flags")
                         for q in cleaned[key]}
        if conditions:
            result[area["rank"]] = {
                "telling_apart": _clean_list(got.get("telling_apart"), TELL_APART_LIMIT, banned, rejected),
                "conditions": conditions,
            }
    return result, rejected


def _apply(chat_plan: dict, generated: dict) -> dict:
    """The chat plan with generated branches for the covered organs."""
    plan = copy.deepcopy(chat_plan)
    for cluster in plan.get("clusters", []):
        area = generated.get(cluster.get("rank"))
        if not area:
            cluster["question_source"] = "built-in"
            continue
        organ = cluster.get("system", "")
        branches = []
        for disease, q in area["conditions"].items():
            branches.append({
                "id": generated_branch_id(disease),
                "label": disease,
                "matched_terms": [disease],
                "organs": [organ],
                "diseases": [disease],
                "symptom_targets": q["symptoms"],
                "screening_questions": q["screening"],
                "drilldown_questions": q["drilldown"],
                "severity_questions": q["severity"],
                "red_flag_questions": q["red_flags"],
            })

        def collect(key):
            return [x for b in branches for x in b[key]]

        cluster["branches"] = branches
        cluster["symptom_targets"] = collect("symptom_targets")[:10]
        cluster["screening_questions"] = collect("screening_questions")[:4]
        # Telling-apart questions first: they confirm the organ itself.
        cluster["drilldown_questions"] = (area["telling_apart"] + collect("drilldown_questions"))[:6]
        cluster["severity_questions"] = collect("severity_questions")[:3]
        cluster["red_flag_questions"] = collect("red_flag_questions")[:3]
        cluster["telling_apart_questions"] = area["telling_apart"]
        cluster["question_source"] = "gemini"
    return plan


def _count_questions(generated: dict) -> int:
    return sum(len(q[k]) for a in generated.values() for q in a["conditions"].values()
               for k in ("screening", "drilldown", "severity", "red_flags")) + \
        sum(len(a["telling_apart"]) for a in generated.values())


def improve_chat_plan(patient_id: str, chat_plan: dict, gender: str = "Unknown") -> dict:
    """The chat plan with Gemini-written questions, or the plan unchanged if
    that isn't possible. Never raises - the chat must always start."""
    areas = _plan_input(chat_plan)
    if not areas:
        return chat_plan
    folder = os.path.join("patients", patient_id)
    path = os.path.join(folder, QUESTIONS_FILE)
    key = _cache_key(areas, gender)
    names = ", ".join(a["area"] for a in areas)

    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("key") == key and saved.get("generated"):
            generated = {int(r): v for r, v in saved["generated"].items()}
            print(f"[Questions] Reusing saved question set for {names} -> {path}")
            return _apply(chat_plan, generated)
    except (OSError, ValueError):
        pass

    areas_text = json.dumps([{"area": a["area"], "conditions": a["conditions"]} for a in areas],
                            indent=1, ensure_ascii=False)
    # One extra of each kind, so the count stays full after the checks drop some.
    prompt = PROMPT.format(gender=gender or "Unknown", areas=areas_text, tell_apart=TELL_APART_LIMIT + 1,
                           symptoms=LIMITS["symptoms"], **{k: v + 1 for k, v in LIMITS.items() if k != "symptoms"})
    started = datetime.now()
    try:
        raw_text = asyncio.run(_ask_gemini(prompt))
        raw = json.loads(raw_text)
        generated, rejected = _validate(raw, areas)
    except Exception as e:
        print(f"[Questions] [FAIL] Gemini could not write the questions ({type(e).__name__}: {str(e)[:200]})"
              f" - using built-in questions")
        return chat_plan
    if not generated:
        print("[Questions] [FAIL] Gemini's questions did not pass the checks - using built-in questions")
        return chat_plan

    seconds = (datetime.now() - started).total_seconds()
    missing = [a["area"] for a in areas if a["rank"] not in generated]
    print(f"[Questions] Gemini wrote {_count_questions(generated)} questions for "
          f"{', '.join(a['area'] for a in areas if a['rank'] in generated)} in {seconds:.0f}s"
          f"{f' ({len(rejected)} dropped by the checks)' if rejected else ''}"
          f"{f' - built-in questions kept for {missing}' if missing else ''} -> {path}")
    try:
        os.makedirs(folder, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "patient_id": patient_id,
                "generated_at": datetime.now().isoformat(),
                "model": GEMINI_MODEL,
                "prompt_version": PROMPT_VERSION,
                "key": key,
                "input": areas,
                "generated": {str(r): v for r, v in generated.items()},
                "rejected": rejected,
            }, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[Questions] could not save {path}: {e}")
    return _apply(chat_plan, generated)
