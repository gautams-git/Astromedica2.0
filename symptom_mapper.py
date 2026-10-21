"""
HitList symptom translation layer for chat.

The disease and organ names remain unchanged in the analysis engine. This file
adds a chat-only interpretation layer so the model asks about lived symptoms
without saying organ or disease names to the patient.
"""

import re


TERM_SYMPTOM_RULES = [
    {
        "keys": ["kidney", "renal", "urinary", "urine", "bladder", "dropsy"],
        "symptoms": [
            "swelling around the ankles or feet",
            "puffiness around the eyes in the morning",
            "waking often at night to pass urine",
            "passing much less or much more urine than usual",
            "pressure or aching in the lower back",
        ],
        "screening": [
            "Have you noticed swelling around your ankles or puffiness around your eyes, especially in the morning?"
        ],
        "drilldown": [
            "When this happens, do you also notice changes in how often you pass urine, especially at night?",
            "Does the swelling or puffiness get worse by evening or after a salty meal?",
        ],
        "severity": [
            "Has this been happening repeatedly over weeks, or is it only occasional?"
        ],
        "red_flags": [
            "Have you had fever, burning while passing urine, or strong lower back discomfort recently?"
        ],
    },
    {
        "keys": ["liver", "bile", "jaundice", "fat in the body"],
        "symptoms": [
            "bitter taste in the mouth",
            "yellowing around the eyes",
            "heaviness or discomfort after oily food",
            "unusual tiredness after meals",
            "itching without a clear skin cause",
        ],
        "screening": [
            "After meals, especially oily or heavy food, do you feel unusually heavy, bitter in the mouth, or more tired than expected?"
        ],
        "drilldown": [
            "Have you noticed yellowing around the eyes, unusual itching, or darker urine along with that heaviness?",
            "Does the discomfort feel worse after rich food or late-night eating?",
        ],
        "severity": [
            "How often does this happen in a normal week?"
        ],
        "red_flags": [
            "Have you recently noticed yellow eyes, severe weakness, or persistent vomiting?"
        ],
    },
    {
        "keys": ["adrenal", "secretion", "postural hypotension"],
        "symptoms": [
            "dizziness when standing quickly",
            "salt cravings",
            "sudden weakness",
            "darkening on elbows, scars, or skin creases",
            "feeling faint after stress or exertion",
        ],
        "screening": [
            "Do you ever feel lightheaded when you stand up quickly, along with unusual weakness or craving salty foods?"
        ],
        "drilldown": [
            "Have you noticed darkening on your elbows, scars, or skin creases along with that weakness?",
            "Does the weakness get worse after stress, heat, or missing meals?",
        ],
        "severity": [
            "When it happens, does it pass quickly or leave you drained for hours?"
        ],
        "red_flags": [
            "Have you had fainting, severe dizziness, or sudden extreme weakness recently?"
        ],
    },
    {
        "keys": ["diabetes", "insulin", "hypoglycemia", "sugar"],
        "symptoms": [
            "shakiness if meals are delayed",
            "sudden sweating",
            "unusual thirst",
            "frequent urination",
            "blurred vision when hungry or tired",
        ],
        "screening": [
            "If meals are delayed, do you feel shaky, sweaty, unusually weak, or suddenly very hungry?"
        ],
        "drilldown": [
            "Do you also feel unusually thirsty or need to pass urine more often than before?",
            "Does eating something sweet or a meal quickly settle that shaky feeling?",
        ],
        "severity": [
            "How many times a week do you notice this shaky or weak feeling?"
        ],
        "red_flags": [
            "Have you had confusion, faintness, or blurred vision with these episodes?"
        ],
    },
    {
        "keys": ["heart", "circulation", "blood pressure", "palpitation"],
        "symptoms": [
            "chest pressure",
            "heart racing",
            "breathlessness on mild effort",
            "dizziness with exertion",
            "swelling in feet by evening",
        ],
        "screening": [
            "Do you ever feel chest pressure, racing beats, or breathlessness during simple activity?"
        ],
        "drilldown": [
            "Does it settle with rest, or does it continue even when you slow down?",
            "Do you feel dizzy or unusually tired along with it?",
        ],
        "severity": [
            "Has this limited walking, climbing stairs, or normal work?"
        ],
        "red_flags": [
            "Have you had chest pressure with sweating, faintness, or pain spreading to the arm or jaw?"
        ],
    },
    {
        "keys": ["lung", "breathing", "pulmonary", "respiratory", "sore throat"],
        "symptoms": [
            "shortness of breath",
            "dry cough",
            "wheezing",
            "tightness while breathing",
            "repeated throat irritation",
        ],
        "screening": [
            "Have you noticed shortness of breath, wheezing, or a dry cough that keeps returning?"
        ],
        "drilldown": [
            "Is it worse at night, after dust exposure, or during exertion?",
            "Do you feel tightness while breathing when this happens?",
        ],
        "severity": [
            "Does it interrupt sleep or daily activity?"
        ],
        "red_flags": [
            "Have you had severe breathlessness, bluish lips, or coughing blood?"
        ],
    },
    {
        "keys": ["bowel", "abdomen", "abdominal", "stomach", "digestive", "intestinal", "tumour"],
        "symptoms": [
            "bloating",
            "burning after meals",
            "change in bowel pattern",
            "abdominal cramps",
            "unexplained appetite change",
        ],
        "screening": [
            "Have you noticed bloating, burning after meals, or a change in your usual bathroom pattern?"
        ],
        "drilldown": [
            "Is this linked to certain foods, late meals, or stress?",
            "Do you get cramps, heaviness, or incomplete relief after going to the bathroom?",
        ],
        "severity": [
            "How long has this pattern been going on?"
        ],
        "red_flags": [
            "Have you noticed blood, unexplained weight loss, or persistent severe pain?"
        ],
    },
    {
        "keys": ["nerve", "neurological", "brain", "ataxia", "trembling"],
        "symptoms": [
            "tingling or numbness",
            "trembling",
            "unsteady walking",
            "head pressure",
            "poor concentration",
        ],
        "screening": [
            "Have you felt tingling, numbness, trembling, or unsteadiness while walking?"
        ],
        "drilldown": [
            "Is it more on one side, or does it affect both sides equally?",
            "Does it come with headaches, heaviness in the head, or trouble concentrating?",
        ],
        "severity": [
            "Is it getting more frequent or staying about the same?"
        ],
        "red_flags": [
            "Have you had sudden weakness on one side, slurred speech, or sudden severe headache?"
        ],
    },
    {
        "keys": ["joint", "hip", "thigh", "leg", "ankle", "calf", "locomotor", "rheumatism", "limb", "shank"],
        "symptoms": [
            "joint stiffness",
            "heaviness in the legs",
            "hip or thigh discomfort",
            "difficulty walking after rest",
            "swelling or warmth around joints",
        ],
        "screening": [
            "Do you feel stiffness, heaviness, or discomfort in your legs or joints when you start moving?"
        ],
        "drilldown": [
            "Is it worse in the morning, after sitting, or after walking for a while?",
            "Do you notice swelling, warmth, or reduced movement in the area?",
        ],
        "severity": [
            "Does this affect your walking, stairs, or daily work?"
        ],
        "red_flags": [
            "Have you had sudden swelling, severe pain, or inability to bear weight?"
        ],
    },
    {
        "keys": ["skin", "itch", "rash", "external", "leprosy"],
        "symptoms": [
            "itching",
            "slow healing marks",
            "skin darkening",
            "recurring rashes",
            "dryness or cracking",
        ],
        "screening": [
            "Have you noticed unusual itching, slow-healing marks, or darkening in skin folds or scars?"
        ],
        "drilldown": [
            "Does it come and go, or has it stayed for many weeks?",
            "Is it linked with sweating, food, stress, or dryness?",
        ],
        "severity": [
            "Is it spreading or becoming more frequent?"
        ],
        "red_flags": [
            "Have you had painful spreading redness, fever, or open wounds that are not healing?"
        ],
    },
    {
        "keys": ["prostate"],
        "symptoms": [
            "weak or slow urine stream",
            "straining to start passing urine",
            "dribbling after passing urine",
            "feeling the bladder is not fully empty",
        ],
        "screening": [
            "When you pass urine, is the stream weaker or slower than before, or do you need to strain to start?"
        ],
        "drilldown": [
            "Do you feel as though your bladder is not fully empty afterwards, or notice dribbling at the end?",
            "Do you need to get up at night to pass urine because of this?",
        ],
        "severity": [
            "Has this been getting gradually worse over months?"
        ],
        "red_flags": [
            "Have you been unable to pass urine at all, or seen blood in your urine?"
        ],
    },
    {
        "keys": ["micturition", "frequent urination"],
        "symptoms": [
            "passing urine more often during the day",
            "sudden urge to pass urine",
            "passing only small amounts each time",
        ],
        "screening": [
            "Do you find yourself needing to pass urine much more often during the day than you used to?"
        ],
        "drilldown": [
            "Does the urge come on suddenly, and is it hard to hold?",
            "Do you pass only small amounts each time?",
        ],
        "severity": [
            "Is this affecting your work, travel, or sleep?"
        ],
        "red_flags": [
            "Is there any burning, pain, or fever along with it?"
        ],
    },
    {
        "keys": ["piles", "fistula", "anus", "anal", "haemorrhoid", "hemorrhoid"],
        "symptoms": [
            "discomfort while passing stool",
            "itching around the back passage",
            "bleeding when passing stool",
            "a lump or swelling near the back passage",
        ],
        "screening": [
            "Have you had discomfort, itching, or any bleeding when passing stool?"
        ],
        "drilldown": [
            "Is it linked with constipation or straining?",
            "Have you noticed a lump, swelling, or discharge near the back passage?",
        ],
        "severity": [
            "How often does this happen in a normal month?"
        ],
        "red_flags": [
            "Have you had heavy bleeding, severe pain, or fever with it?"
        ],
    },
    {
        "keys": ["blood", "anaemia", "anemia"],
        "symptoms": [
            "unusual tiredness",
            "looking pale",
            "bruising easily",
            "breathless on small effort",
        ],
        "screening": [
            "Have you been feeling unusually tired, looking paler, or bruising more easily than before?"
        ],
        "drilldown": [
            "Do you get breathless or dizzy with small effort, like climbing a few stairs?",
            "Do small cuts take longer than usual to stop bleeding?",
        ],
        "severity": [
            "Has this tiredness stayed for more than a few weeks?"
        ],
        "red_flags": [
            "Have you fainted, or noticed black stools or blood when coughing?"
        ],
    },
]


FALLBACK = {
    "symptoms": [
        "unusual tiredness",
        "repeating discomfort",
        "changes from your normal routine",
    ],
    "screening": [
        "Have you noticed any repeating physical discomfort that feels different from your usual pattern?"
    ],
    "drilldown": [
        "When it appears, what makes it better or worse?"
    ],
    "severity": [
        "How long has this been happening, and how often does it return?"
    ],
    "red_flags": [
        "Has it ever felt sudden, severe, or difficult to manage without help?"
    ],
}


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _dedupe(items, limit=None):
    seen = set()
    out = []
    for item in items:
        text = str(item).strip()
        key = _norm(text)
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit and len(out) >= limit:
            break
    return out


def _rule_id(rule: dict) -> str:
    keys = rule.get("keys", [])
    base = keys[0] if keys else "general"
    return _norm(base).replace(" ", "_")


def map_term(term: str) -> dict | None:
    term_norm = _norm(term)
    for rule in TERM_SYMPTOM_RULES:
        if any(key in term_norm for key in rule["keys"]):
            return rule
    return None


def extract_cluster_terms(cluster: dict) -> dict:
    organs = []
    diseases = []
    systems = []

    for trail in cluster.get("trail", []):
        source = trail.get("source", "")
        term = trail.get("term", "")
        if not term:
            continue
        if "HL3" in source or "Rashi" in source:
            organs.append(term)
        elif "HL2" in source or "Nakshatra" in source:
            diseases.append(term)
        elif "HL1" in source or "Planet" in source:
            diseases.append(term)
        else:
            systems.append(term)

    return {
        "organs": _dedupe(organs, 6),
        "diseases": _dedupe(diseases, 8),
        "systems": _dedupe(systems, 4),
    }


def build_hitlist_chat_plan(hitlist_result: dict, max_clusters: int = 2) -> dict:
    clusters = []
    for index, cluster in enumerate(hitlist_result.get("top4", [])[:max_clusters], start=1):
        terms = extract_cluster_terms(cluster)
        branch_map = {}

        def ensure_branch(rule: dict) -> dict:
            branch_id = _rule_id(rule)
            if branch_id not in branch_map:
                branch_map[branch_id] = {
                    "id": branch_id,
                    "label": ", ".join(rule.get("keys", [])[:2]) or "general",
                    "matched_terms": [],
                    "organs": [],
                    "diseases": [],
                    "symptom_targets": [],
                    "screening_questions": [],
                    "drilldown_questions": [],
                    "severity_questions": [],
                    "red_flag_questions": [],
                }
            return branch_map[branch_id]

        for entry in cluster.get("trail", []):
            term = entry.get("term", "")
            if not term:
                continue
            rule = map_term(term)
            if not rule:
                continue
            branch = ensure_branch(rule)
            source = entry.get("source", "")
            branch["matched_terms"].append(term)
            if "HL3" in source or "Rashi" in source:
                branch["organs"].append(term)
            else:
                branch["diseases"].append(term)
            branch["symptom_targets"].extend(rule.get("symptoms", []))
            branch["screening_questions"].extend(rule.get("screening", []))
            branch["drilldown_questions"].extend(rule.get("drilldown", []))
            branch["severity_questions"].extend(rule.get("severity", []))
            branch["red_flag_questions"].extend(rule.get("red_flags", []))

        label_rule = map_term(cluster.get("label", ""))
        if label_rule:
            branch = ensure_branch(label_rule)
            branch["symptom_targets"].extend(label_rule.get("symptoms", []))
            branch["screening_questions"].extend(label_rule.get("screening", []))
            branch["drilldown_questions"].extend(label_rule.get("drilldown", []))
            branch["severity_questions"].extend(label_rule.get("severity", []))
            branch["red_flag_questions"].extend(label_rule.get("red_flags", []))

        branches = list(branch_map.values())
        if not branches:
            branches = [{
                "id": "general",
                "label": "general",
                "matched_terms": [],
                "organs": [],
                "diseases": [],
                "symptom_targets": FALLBACK.get("symptoms", []),
                "screening_questions": FALLBACK.get("screening", []),
                "drilldown_questions": FALLBACK.get("drilldown", []),
                "severity_questions": FALLBACK.get("severity", []),
                "red_flag_questions": FALLBACK.get("red_flags", []),
            }]

        for branch in branches:
            branch["matched_terms"] = _dedupe(branch["matched_terms"], 6)
            branch["organs"] = _dedupe(branch["organs"], 4)
            branch["diseases"] = _dedupe(branch["diseases"], 6)
            branch["symptom_targets"] = _dedupe(branch["symptom_targets"], 8)
            branch["screening_questions"] = _dedupe(branch["screening_questions"], 2)
            branch["drilldown_questions"] = _dedupe(branch["drilldown_questions"], 3)
            branch["severity_questions"] = _dedupe(branch["severity_questions"], 2)
            branch["red_flag_questions"] = _dedupe(branch["red_flag_questions"], 2)

        screening = []
        drilldown = []
        severity = []
        red_flags = []
        symptoms = []
        for branch in branches:
            screening.extend(branch.get("screening_questions", []))
            drilldown.extend(branch.get("drilldown_questions", []))
            severity.extend(branch.get("severity_questions", []))
            red_flags.extend(branch.get("red_flag_questions", []))
            symptoms.extend(branch.get("symptom_targets", []))

        clusters.append({
            "rank": index,
            "system": cluster.get("label", f"Priority Cluster {index}"),
            "score": cluster.get("score", 0),
            "convergence": cluster.get("convergence", 0),
            "hitlists": cluster.get("hitlists", []),
            "organs": terms["organs"],
            "diseases": terms["diseases"],
            "source_trail": cluster.get("trail", []),
            "symptom_targets": _dedupe(symptoms, 10),
            "screening_questions": _dedupe(screening, 3),
            "drilldown_questions": _dedupe(drilldown, 4),
            "severity_questions": _dedupe(severity, 2),
            "red_flag_questions": _dedupe(red_flags, 2),
            "branches": branches,
        })

    return {
        "max_questions": 15,
        "minimum_questions": 12,
        "strategy": [
            "Turns 1-2 focus on Cluster 1 screening first.",
            "Turn 3 stays on Cluster 1 unless the patient has been negative so far; only then check Cluster 2.",
            "Turn 4 continues Cluster 1 by default, but uses Cluster 2 only when Cluster 1 has weak or negative support.",
            "Turns 5-8 drill deeper into Cluster 1 by default, while still following any clearly stronger support from Cluster 2.",
            "Turns 9-11 ask duration, frequency, trigger, and severity checks.",
            "Turns 12-13 ask red-flag safety checks.",
            "Turns 14-15 ask lifestyle/context and then summarize.",
        ],
        "clusters": clusters,
    }


# ── Answer scoring ───────────────────────────────────────────────
# Whole words / phrases only (so "not daily" is not read as "daily"), in a
# fixed order. English, Hinglish and how-often answers ("3 times a week").
# Labels: yes_strong (yes) · yes_but (partly yes) · no_but (partly no / unsure)
# · no_strong (no).

_UNSURE = ["not sure", "dont know", "do not know", "no idea", "cant say", "can not say",
           "not certain", "pata nahi", "pata nahin", "maybe", "may be", "perhaps"]
_SOFT_NO = ["no but", "only sometimes", "not really", "not much", "not often", "not daily", "not every day",
            "not everyday", "not regularly", "rarely", "hardly", "almost never", "only once",
            "once in a while", "very rarely", "seldom", "bahut kam", "kam hi"]
_SOFT_YES = ["yes but", "not always", "sometimes", "sometime", "some times", "occasionally",
             "at times", "now and then", "on and off", "off and on", "a little", "a bit",
             "little bit", "mild", "mildly", "slightly", "kabhi kabhi", "kabhi kabhar",
             "thoda", "thodi", "thoda sa", "halka"]
_STRONG_NO = ["no never", "never", "not at all", "nothing like that", "absolutely not",
              "kabhi nahi", "kabhi nahin", "bilkul nahi", "bilkul nahin", "nope", "nah", "naa",
              "nahi", "nahin", "nai", "none", "no", "noo", "nooo", "nop", "n"]
_YES_WORDS = ["yes", "yess", "yesss", "ys", "yse", "yas", "y", "yea", "yeah", "yeh", "yep", "yup",
              "ya", "yaa", "yah", "haan", "haa",
              "han", "ha", "ji", "sure", "correct", "right", "true", "absolutely", "definitely",
              "indeed", "of course", "ofcourse", "exactly"]
_OFTEN = ["every day", "everyday", "daily", "always", "often", "very often", "frequently",
          "frequent", "all the time", "a lot", "very much", "most days", "regularly",
          "constantly", "many times", "every night", "every morning", "roz", "rozana",
          "hamesha", "bahut", "zyada"]
_SYMPTOM_WORDS = ["noticed", "have", "pain", "swelling", "dizzy", "weak", "burning", "tired",
                  "breath", "shaky", "sweat", "itch", "stiff", "numb", "hurts", "ache", "aches"]
_NEGATIONS = ["not", "dont", "doesnt", "didnt", "isnt", "wasnt", "havent", "cannot"]

# "3 times", "twice a week", "once a day", "for 2 years", "since a month"...
_FREQUENCY = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|once|twice|thrice)\s*(times?|x)\b"
    r"|\b(twice|thrice)\b"
    r"|\b(once|\d+)\s+(a|per|every)\s+(day|week|month|night)\b"
    r"|\b(for|since)\s+(\d+|a|an|one|two|three|few|many|several|last)\s+"
    r"(days?|weeks?|months?|years?)\b")


def _has(msg: str, phrases) -> bool:
    return any(re.search(r"\b" + re.escape(p) + r"\b", msg) for p in phrases)


def classify_answer(text: str) -> str:
    msg = _norm(str(text or "").replace("'", ""))   # "don't" -> "dont"
    if not msg:
        return "no_but"
    says_yes = _has(msg, _YES_WORDS)

    if _has(msg, _UNSURE):
        return "no_but"
    if says_yes and (_has(msg, _SOFT_NO) or _has(msg, _SOFT_YES) or _has(msg, _NEGATIONS)):
        return "yes_but"                       # "yes, but not daily", "yes sometimes"
    if _has(msg, _SOFT_NO):
        return "no_but"
    if _has(msg, _SOFT_YES):
        return "yes_but"
    if _has(msg, _STRONG_NO):
        return "no_strong"
    if says_yes or _has(msg, _OFTEN) or _FREQUENCY.search(msg):
        return "yes_strong"
    if _has(msg, _NEGATIONS):
        return "no_strong"                     # "I dont get that", "it doesnt happen"
    if _has(msg, _SYMPTOM_WORDS):
        return "yes_strong"                    # "my back hurts in the morning"
    print(f"[Chat] Answer not recognised: {str(text)[:80]!r} -> counted as 'yes, but'")
    return "yes_but"


def score_answer(label: str) -> int:
    return {
        "yes_strong": 3,
        "yes_but": 1,
        "no_but": -1,
        "no_strong": -3,
    }.get(label, 0)
