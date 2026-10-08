from datetime import date, timedelta
# pyre-ignore-all-errors
import re
from knowledge_base import (
    RASHI_ORGANS,
    PLANET_DISEASES,
    NAKSHATRA_DISEASES,
    PLANET_DOSHA,
    DOSHA_SYMPTOMS,
    get_nakshatra_diseases,
    GENDER_EXCLUSIVE_TERMS
)
from collections import defaultdict

# ── Procedural Filters ──────────────────────────────────────────
PROCEDURAL_TERMS = {"amputation", "surgery", "operation", "sex urge"}

# ── Semantic Body-Zone Mapping (Ayurvedic/Astrological) ──────────────────
SEMANTIC_GROUPS = [
    # 1. Hormonal System and Glands/Pancreatic areas
    (["pituitary", "hypothalamus", "thyroid", "adrenal secretion", "adrenal", "insulin", 
      "diabetes", "goitre", "iodine deficiency", "endocrine", "oestrogen", "estrogen", 
      "pancreas", "pancreatic", "bile", "hormonal", "gland"],
     "Hormonal System and Glands"),

    # 2. Excretion Systems
    (["anus", "anal", "rectum", "piles", "fistula", "haemorrhoids", "excretory organs", 
      "bleeding", "bowels"],
     "Excretion Systems"),

    # 3. Heart and Cardio
    (["blood pressure", "high bp", "low bp", "hypertension", "heart attack", "palpitation", 
      "thrombosis", "heart beat", "heart beats", "heart failure", "heart valve",
      "dilated heart", "sudden shock", "fainting", "faint", "nausea", "cardiac",
      "cardiovascular", "heart", "cardio", "rheumatic heart","rheumatism","chest"],
     "Heart and Cardio"),

    # 4. Pulmonary & Respiratory System
    (["lung", "pulmonary", "respiratory", "breathing", "asthma", "dry cough", "cough", 
      "eosinophilia", "pneumonia", "tuberculosis", "tb", "hiccup", "chest", "upper chest", 
      "bronchial", "pestilence"],
     "Pulmonary & Respiratory System"),

    # 5. Haematology and Blood
    (["haematology", "blood poisoning", "over heated blood", "anaemia", "anemia", 
      "corrupt blood", "infection","septic", "fever", "malaria", "filaria", "plague", "pox", "poison", 
      "hyperaemia", "blood"],
     "Haematology and Blood"),

    # 6. Reproductive Organs and System
    (["menses", "menstruation", "leucorrhoea", "fibroid", "testicle", "semen", 
      "prostate", "genital", "sex urge", "impotency", "veneral", "syphilis", "vagina", 
      "sperm", "sex organs", "ovary", "ovaries", "womb", "uterus", "reproductive", "breast", "breast pain",
      "female organs", "groin", "groins"],
     "Reproductive Organs and System"),

    # 7. Renal and Urinary
    (["kidney stone", "urinary tracts", "frequent micturition", "kidney", "bladder", 
      "urine", "urinary", "micturition", "nephritis", "renal", "adrenal", "adrenal secretion", 
      "diabetes", "blood pressure", "high bp", "low bp"],
     "Renal/Kidney and Urinary"),

    # 8. Head and Brain Region
    (["brain", "mind", "memory", "epilepsy", "coma", "haemorrhage", "stroke", 
      "meningitis", "spasms", "neuralgia", "mental instability", "congestion of brain", 
      "clotting in brain", "cerebral", "mental", "headache", "head", "fainting", "faint", 
      "anaemia", "anemia"],
     "Head and Brain Region"),

    # 9. Face Region
    (["eye sore", "reddish eyes", "vision", "sight", "hearing", "deafness", "speech", 
      "face", "nose", "ear", "tongue", "teeth", "mouth", "polyps", "facial"],
     "Face Region"),

    # 10. Abdominal Region & Digestive
    (["solar plexus", "solar plexes", "intestine", "liver", "spleen", "lower abdomen", 
      "naval", "navel", "waist", "digestion", "indigestion", "gastric", "ulcer", 
      "gallstones", "jaundice", "enlargement of spleen", "stomach", "abdominal", 
      "bowels", "pancreas", "pancreatic", "insulin", "bile", "nausea"],
     "Abdominal Region & Digestive"),

    # 11. Nervous System
    (["nerves problems", "nervous problems", "spinal meningitis", "nerve", "nervous", 
      "neural", "paralysis", "neuralgia", "cerebral"],
     "Nervous System"),

    # 12. Skin and Sensory
    (["skin diseases", "leprosy", "eczema", "psoriasis", "irritation", "itching", 
      "skin", "itch", "rash", "pimple", "wound", "pus", "pox"],
     "Skin and Sensory"),

    # 13. Musculoskeletal & Lower Body (Structural, Hips, Legs & Feet)
    (["swelling above the knees", "locomotor ataxia", "rheumatic pains", 
      "hip diseases", "leg injury", "hip joints", "rheumatism", "milk leg", 
      "cold foot", "perspiring feet", "pains in legs", 
      "long lasting", "hip", "thigh", "leg", "limb", "knee",
      "foot", "feet", "ankle", "calf", "shank", "Amputation",
      "lower back", "muscle", "flesh", "joints"],
     "Musculoskeletal & Lower Body"),

    # 14. Bones & Skeletal System
    (["bone", "bones", "marrow", "fracture", "skeletal", "spine", "spinal", 
      "curvature of spine", "vertebrae", "collar bone"],
     "Bones & Skeletal System"),
]

def get_semantic_groups(term: str) -> list:
    t = term.lower().strip()
    matched_groups = []
    for keywords, group_name in SEMANTIC_GROUPS:
        for kw in keywords:
            # Use regex to match whole words or specific substrings to avoid "ear" matching "heart"
            pattern = rf"\b{re.escape(kw.lower())}\b"
            if re.search(pattern, t):
                matched_groups.append(group_name)
                break # Match found in this group, move to next
    return matched_groups if matched_groups else []

# ── Keyword → canonical category ──────────────────────────────
KEYWORD_CATEGORIES = [
    (["hip joint","hip dis","hips joint","locomotor","ataxia","rheuma","rheumat","joint pain","joint"],
     "Hip Joints & Rheumatism"),
    (["leg ","legs","limb","thigh","lower body","leg inj", "milk leg", "feet", "foot"],
     "Legs / Limbs / Thighs"),
    (["blood", "over heated blood", "blood poisoning"],
     "Blood Disorders"),
    (["pulmon","lung","breath","respir","chest","dry cough", "eosinophilia", "hiccup"],
     "Lungs / Pulmonary"),
    (["bone","fractur","skeletal","spinal","spine", "hip bone", "stiffness"],
     "Bones / Spine"),
    (["knee","kne"],
     "Knees"),
    (["ankle","shank"],
     "Feet / Ankles"),
    (["nerve","neural","paralys", "nervous", "hysteria"],
     "Nerves / Paralysis"),
    (["skin","itch","rash","eczema","leprosy","lepros","wound","pus","septic","pimple", "pus formation"],
     "Skin / Wounds"),
    (["eye","vision","sight","optic", "affected eyes"],
     "Eyes"),
    (["heart","cardiac","cardiov","dropsy","apoplexy","heartbeat", "palpitation", "shock"],
     "Heart"),
    (["liver","hepat","gallston", "jaundice"],
     "Liver"),
    (["kidney","renal","nephrit", "micturition", "urine"],
     "Kidneys"),
    (["head","brain","migrain","epilep","faint","meningi","cerebr", "hyperaemia", "clotting"],
     "Head / Brain"),
    (["stomach","digest","gastric","bowel","colon","ulcer","indig","intestin","abdom", "dysentery", "worm", "tumour"],
     "Digestive / Stomach"),
    (["muscle","muscul","energy","marrow", "body pain", "weakness"],
     "Muscles / Energy"),
    (["impoten","sexual","genital","reproduct","vener","semen","menses","leucorr","fistula", "womb", "fibroid", "prostate"],
     "Reproductive / Genital"),
    (["throat","tonsil","diphther","goitr","thyroid","collar", "neck", "neck"],
     "Neck / Throat"),
    (["neck","face","facial","speech", "tongue", "teeth"],
     "Neck / Face"),
    (["shoulder","arm","elbow","wrist","hand"],
     "Shoulders / Arms"),
    (["fever","inflam","infect","malaria","pox","septic","viral","tubercul","filaria", "plague", "poison"],
     "Fever / Infections"),
    (["diabetes","insulin","adrenal","endocrin", "vitamin", "beri beri", "iodine"],
     "Metabolic / Diabetes / Vitamin"),
    (["mind","memory","emotion","anxiety","psychic","mental","insomn"],
     "Mind / Mental"),
    (["spleen","lymph","fluid"],
     "Spleen / Lymph"),
    (["fat","obesity","weight"],
     "Weight / Metabolism"),
    (["gas","acidity","flatulen","bloat"],
     "Gas / Acidity"),
    (["piles","haemorrhoid"],
     "Piles"),
    (["ear","deaf","hearing"],
     "Ears"),
    (["blood pressure","high bp","hypertens"],
     "Blood Pressure"),
]

def get_category(disease: str) -> str:
    d = disease.lower().strip()
    for keywords, category in KEYWORD_CATEGORIES:
        for kw in keywords:
            if kw in d:
                return category
    return disease.title()

def perform_tiered_diagnostic_analysis(rule1_result, complete_analysis, chart_data, patient_gender="Unknown"):
    # Step 1: Initialize Tracking
    theme_data = defaultdict(lambda: {
        "score": 0,
        "nak_count": 0,
        "points_log": [],
        "source_to_symptoms": defaultdict(set), # Maps "Source [Role]" -> set of diseases
        "symptoms": set(),
        "nakshatra_names": set(),
        "planet_names": set(),
        "rashi_names": set(),
        "processed_source_symptoms": set() # NEW: Tracks (source_type, source_name, disease_str) to avoid double counting
    })

    def process_hit(disease_str, source_type, source_name, weight, role_tag=""):
        if not disease_str: return
        
        disease_lower = disease_str.lower().strip()
        if disease_lower in PROCEDURAL_TERMS or disease_lower in ["—", "none", "unknown"]:
            return
        
        # --- GENDER FILTER ---
        if patient_gender in ["M", "F", "Male", "Female"]: 
            safe_gender = "M" if patient_gender.startswith("M") else "F" 
            exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in disease_lower), None)
            if exclusive_gender and exclusive_gender != safe_gender: 
                return 
        
        themes = get_semantic_groups(disease_str)
        if not themes:
            themes = [disease_str.title()]
            
        for theme in themes:
            data = theme_data[theme]

            # --- NEW: DOUBLE COUNTING PREVENTION ---
            # A specific source (like Venus) should only contribute its weight ONCE per theme,
            # regardless of how many symptoms it matches in that theme.
            # This prevents "symptom spamming" from lower-priority planets from 
            # overwhelming high-priority indicators.
            hit_key = (source_type, source_name, theme)
            
            # Create a clean label for the mapping
            clean_role = role_tag if role_tag else "6th House"
            mapped_label = f"{source_type} ({source_name} [{clean_role}])"
            
            if hit_key in data["processed_source_symptoms"]:
                # --- NEW: EVIDENCE BONUS ---
                # Each additional unique symptom from the same source in the same theme
                # adds a flat +1 point "Evidence Bonus" to the confidence score.
                if source_type == "Nakshatra":
                    data["symptoms"].add(disease_str)
                
                if disease_lower not in data["source_to_symptoms"][mapped_label]:
                    data["score"] += 1
                    data["points_log"].append(f"{mapped_label} (Bonus): +1")
                    data["source_to_symptoms"][mapped_label].add(disease_str)
                continue
                
            data["processed_source_symptoms"].add(hit_key)
            # ---------------------------------------

            # --- SYMPTOMS COLLECTION RESTRICTED TO NAKSHATRAS ---
            if source_type == "Nakshatra":
                data["symptoms"].add(disease_str)
            # --------------------------------------------------------

            data["score"] += weight  # Add dynamic weight
            
            # Add the exact disease to this specific source's list
            data["source_to_symptoms"][mapped_label].add(disease_str)

            # Log the math equation piece
            data["points_log"].append(f"{mapped_label}: +{weight}")
            
            # Tag the name with its astrological role
            tagged_name = f"{source_name} [{role_tag}]" if role_tag and source_name else source_name
            
            if source_type == "Nakshatra" and source_name:
                data["nakshatra_names"].add(tagged_name)
                data["nak_count"] += 1
            elif source_type == "Planet" and source_name:
                data["planet_names"].add(tagged_name)
            elif source_type == "Rashi" and source_name:
                data["rashi_names"].add(tagged_name)

    # 2. Build the Available Hierarchy based on what exists in the chart
    priority_queue = []
    
    # Priority 1: Occupants (RK1)
    if rule1_result.get("rogkaraka_1"):
        priority_queue.append(("Occupant", rule1_result["rogkaraka_1"]))
        
    # Priority 2: Dispositor (RK3)
    if rule1_result.get("rogkaraka_3"):
        priority_queue.append(("Dispositor", [rule1_result["rogkaraka_3"]]))
        
    # Priority 3: Aspecting Planets (Dhrishti)
    # Safely extract from complete_analysis nested structure
    drishti_data = complete_analysis.get("drishti", {})
    aspecting_planets = drishti_data.get("aspects_sixth_house", [])
    if aspecting_planets:
        # Convert aspecting_planets (which are simple dicts) to a format that looks like RK planets
        # We need name and nakshatra info
        all_planets = chart_data.get("planets", [])
        p_list = []
        for asp in aspecting_planets:
            pname = asp.get("planet")
            p_full = next((p for p in all_planets if p["name"] == pname), None)
            if p_full:
                p_list.append({
                    "name": p_full["name"],
                    "sign": p_full["sign"]["name"] if isinstance(p_full.get("sign"), dict) else p_full.get("sign", "—"),
                    "nakshatra": p_full.get("nakshatra", {}).get("name") if isinstance(p_full.get("nakshatra"), dict) else p_full.get("nakshatra"),
                    "nakshatra_pada": p_full.get("nakshatra", {}).get("pada") if isinstance(p_full.get("nakshatra"), dict) else p_full.get("nakshatra_pada")
                })
        if p_list:
            priority_queue.append(("Aspect", p_list))
        
    # Priority 4: Lord of 6th House (RK2)
    if rule1_result.get("rogkaraka_2"):
        priority_queue.append(("Lord", [rule1_result["rogkaraka_2"]]))

    # Priority 5: Ascendant (Final Fallback - Baseline weight)
    if chart_data.get("ascendant"):
        asc = chart_data["ascendant"]
        asc_lord = rule1_result.get("ascendant_lord", "Ascendant")
        priority_queue.append(("Ascendant", [{
            "name": asc_lord,
            "sign": asc["sign"]["name"] if isinstance(asc.get("sign"), dict) else asc.get("sign", "—"),
            "nakshatra": asc.get("nakshatra", {}).get("name") if isinstance(asc.get("nakshatra"), dict) else asc.get("nakshatra"),
            "nakshatra_pada": asc.get("nakshatra", {}).get("pada") if isinstance(asc.get("nakshatra"), dict) else asc.get("nakshatra_pada")
        }]))

    active_roles = [f"Rank {i+1}: {role}" for i, (role, _) in enumerate(priority_queue)]
    chart_hierarchy_str = " | ".join(active_roles) if active_roles else "No Active Planets"

    # 3. Dynamic Weight Tiers (Rank 1 is the most powerful active planet)
    weight_tiers = [
        {"nak": 10, "pla": 3,   "ras": 4},   # Rank 1 Power
        {"nak": 7,  "pla": 2,   "ras": 2},   # Rank 2 Power
        {"nak": 5,  "pla": 1.5, "ras": 2},   # Rank 3 Power
        {"nak": 2,  "pla": 1,   "ras": 1}    # Rank 4 Power
    ]

    # Build the Intelligent Logic Narrative
    narrative_lines = []
    if priority_queue:
        top_role = priority_queue[0][0]
        if top_role == "Occupant":
            narrative_lines.append("Standard Logic: Occupant (RK1) is present and holds 1st Priority.")
        elif top_role == "Dispositor":
            narrative_lines.append("Fallback Triggered: Since Occupant is missing, Dispositor slid up to take 1st Priority.")
        elif top_role == "Aspect":
            narrative_lines.append("Fallback Triggered: Since Occupant & Dispositor are missing, Aspect slid up to take 1st Priority.")
        elif top_role == "Lord":
            narrative_lines.append("Fallback Triggered: Since higher roles are missing, Lord slid up to take 1st Priority.")
        elif top_role == "Ascendant":
            narrative_lines.append("Fallback Triggered: No 6th house indicators found. Relying strictly on Ascendant baseline.")
            
        narrative_lines.append("") # Empty line for spacing
        narrative_lines.append("Applied Weights:")
        for i, (role, _) in enumerate(priority_queue):
            if role == "Ascendant":
                w_nak, w_pla, w_ras = 1, 0, 0.5 # Rank 5 weights
            else:
                w_nak = weight_tiers[i]["nak"] if i < len(weight_tiers) else 1
                w_pla = weight_tiers[i]["pla"] if i < len(weight_tiers) else 0.5
                w_ras = weight_tiers[i]["ras"] if i < len(weight_tiers) else 0.5
            narrative_lines.append(f"• Rank {i+1} ({role}): Nakshatra +{w_nak} | Planet +{w_pla} | Rashi +{w_ras}")
    else:
        narrative_lines.append("No active planets found.")
        
    logic_narrative_str = "\n".join(narrative_lines)

    # 4. Loop through the queue and apply the scaled weights
    for i, (role, planets_list) in enumerate(priority_queue):
        # Lock the weight for the Ascendant
        if role == "Ascendant":
            w_nak, w_pla, w_ras = 1, 0, 0.5
        else:
            w_nak = weight_tiers[i]["nak"] if i < len(weight_tiers) else 1
            w_pla = weight_tiers[i]["pla"] if i < len(weight_tiers) else 0.5
            w_ras = weight_tiers[i]["ras"] if i < len(weight_tiers) else 0.5
        
        for p in planets_list:
            # 4a. Process Rashi (Sign) hits
            rsign = p.get("sign")
            if rsign and rsign != "—":
                r_organs_raw = RASHI_ORGANS.get(rsign, "")
                if isinstance(r_organs_raw, str):
                    r_list = [o.strip() for o in r_organs_raw.split(",") if o.strip()]
                    for organ in r_list:
                        process_hit(organ, "Rashi", rsign, weight=w_ras, role_tag=role)

            # 4b. Process Planet hits
            pname = p.get("name")
            if pname:
                p_dis = PLANET_DISEASES.get(pname, [])
                if isinstance(p_dis, str):
                    p_dis = [d.strip() for d in p_dis.split(",") if d.strip()]
                for d in p_dis:
                    process_hit(d, "Planet", pname, weight=w_pla, role_tag=role)
            
            # 4c. Process Nakshatra hits
            nk = p.get("nakshatra")
            if nk and nk != "—":
                nk_dis = get_nakshatra_diseases(nk, p.get("nakshatra_pada"))
                if isinstance(nk_dis, str):
                    nk_dis = [d.strip() for d in nk_dis.split(",") if d.strip()]
                for d in nk_dis:
                    process_hit(d, "Nakshatra", nk, weight=w_nak, role_tag=role)

    # Step 5, 6: Calculation, Sorting, and Formatting
    final_output = []
    for name, data in theme_data.items():
        if not data["symptoms"]: continue # ONLY SHOW THEMES WITH NAKSHATRA SYMPTOMS
        
        score = data["score"]
        
        nk_n = ", ".join(sorted(list(data["nakshatra_names"]))) or "None"
        pl_n = ", ".join(sorted(list(data["planet_names"]))) or "None"
        rs_n = ", ".join(sorted(list(data["rashi_names"]))) or "None"

        # Build the explicit mapping text line by line
        mapping_lines = []
        # Sort alphabetically so the output is consistent
        for src, symps in sorted(data["source_to_symptoms"].items()):
            symp_str = ", ".join(sorted(list(symps)))
            mapping_lines.append(f"• {src}:\n   ↳ {symp_str}")
        mapping_display = "\n".join(mapping_lines)
        
        # Build the exact math equation string
        math_equation = " + ".join(data["points_log"]) + f" = {data['score']}"
        
        # Build the new transparent debug trail
        explanation = (
            f"🧠 LOGIC ENGINE NARRATIVE:\n"
            f"{logic_narrative_str}\n\n"
            f"👑 CHART PRIORITY STATUS:\n"
            f"{chart_hierarchy_str}\n\n"
            f"🎯 ACTIVE ROLES IN THIS THEME:\n"
            f"• NAKSHATRAS: {nk_n}\n"
            f"• PLANETS: {pl_n}\n"
            f"• RASHIS: {rs_n}\n\n"
            f"📋 KNOWLEDGEBASE MATCHES:\n"
            f"{mapping_display}\n\n"
            f"🧮 MATH EQUATION:\n"
            f"{math_equation}"
        )
        
        # Determine layers for tier classification
        layers = []
        if data["nakshatra_names"]: layers.append("Nakshatra")
        if data["planet_names"]: layers.append("Planet")
        if data["rashi_names"]: layers.append("Rashi")
        
        has_nak = "Nakshatra" in layers
        has_planet = "Planet" in layers
        has_rashi = "Rashi" in layers
        
        # Tier Classification Logic
        tier = 5 # Default
        label = "General Monitoring"
        
        if has_nak:
            if has_rashi and has_planet:
                tier = 1
                label = "CRITICAL / TRIPLE-LOCK"
            elif has_rashi or has_planet:
                tier = 2
                label = "HIGH RISK / DOUBLE-LOCK"
            else:
                tier = 3
                label = "MODERATE RISK / NAKSHATRA ONLY"
        elif has_rashi and has_planet:
            tier = 4
            label = "SECONDARY RISK / RASHI + PLANET"
        else:
            tier = 5
            label = "GENERAL TENDENCY"

        final_output.append({
            "condition": name,
            "score": score,
            "tier": tier,
            "label": label,
            "layers": layers,
            "explanation": explanation,
            "nak_count": data["nak_count"], # Temporary key for tie-breaking
            "contributing_symptoms": sorted(list(data["symptoms"])),
            "confirmed_by": ", ".join(sorted(list(data["nakshatra_names"] | data["rashi_names"]))),
            "backed_by": ", ".join(sorted(list(data["planet_names"]))) if data["planet_names"] else "Pending Confirmation"
        })

    # Sort DESC by score, then DESC by nak_count
    # We use a slight epsilon to handle floating point precision in sort
    final_output.sort(key=lambda x: (round(x["score"], 2), x["nak_count"]), reverse=True)

    # Remove internal sorting key
    for item in final_output:
        del item["nak_count"]

    return final_output[:15]


def get_top_diseases(rule1_result, chart_data, complete_analysis):
    """
    Returns top 3 MOST PROBABLE DISEASE CLUSTERS derived from the 
    detailed Tiered Diagnostic Analysis. This ensures consistency between 
    the summary and the math verification sections.
    """
    most_probable = complete_analysis.get("most_probable", [])
    if not most_probable:
        return []

    # Format the top 3 for the Summary boxes
    top3 = []
    for item in most_probable[:3]:
        top3.append({
            "category": item["condition"],
            "disease": ", ".join(item["contributing_symptoms"]),
            "score": item["score"],
            "confirmed_by": item.get("confirmed_by", "Unknown"),
            "backed_by": item.get("backed_by", "Pending Confirmation")
        })
    
    return top3


SIGN_LORDS = {
    "Aries":       "Mars",    "Mesha": "Mars",
    "Taurus":      "Venus",   "Vrishabha": "Venus",
    "Gemini":      "Mercury", "Mithuna": "Mercury",
    "Cancer":      "Moon",    "Karka": "Moon",
    "Leo":         "Sun",     "Simha": "Sun",
    "Virgo":       "Mercury", "Kanya": "Mercury",
    "Libra":       "Venus",   "Tula": "Venus",
    "Scorpio":     "Mars",    "Vrischika": "Mars",
    "Sagittarius": "Jupiter", "Dhanu": "Jupiter",
    "Capricorn":   "Saturn",  "Makara": "Saturn",
    "Aquarius":    "Saturn",  "Kumbha": "Saturn",
    "Pisces":      "Jupiter", "Meena": "Jupiter"
}


def _build_conditional_moon(chart_data, rogkaraka_1, rogkaraka_2, rogkaraka_3, drishti_on_6th=None):
    planets = chart_data.get("planets", [])
    houses = chart_data.get("houses", [])
    moon = next((p for p in planets if p.get("name") == "Moon"), None)
    if not moon:
        return {
            "include": False,
            "reason": "Moon not found in chart data.",
        }

    moon_house = moon.get("house", 0)
    moon_house_sign = ""
    for house in houses:
        if house.get("house") == moon_house:
            moon_house_sign = (house.get("sign") or {}).get("name", "")
            break

    moon_house_lord = SIGN_LORDS.get(moon_house_sign, "")
    connected_role = ""
    connected_planet = ""
    connected_drishti = {}

    occupant_names = {p.get("name", "") for p in (rogkaraka_1 or []) if p.get("name")}
    if moon_house_lord and moon_house_lord in occupant_names:
        connected_role = "occupant"
        connected_planet = moon_house_lord
    elif moon_house_lord and moon_house_lord == ((rogkaraka_3 or {}).get("name", "")):
        connected_role = "dispositor"
        connected_planet = moon_house_lord
    elif moon_house_lord and moon_house_lord == ((rogkaraka_2 or {}).get("name", "")):
        connected_role = "6th_house_lord"
        connected_planet = moon_house_lord
    else:
        drishti_list = drishti_on_6th or []
        drishti_planets = {d.get("planet", "") for d in drishti_list if d.get("planet")}
        if moon_house_lord and moon_house_lord in drishti_planets:
            connected_role = "drishti"
            connected_planet = moon_house_lord
            connected_drishti = next((d for d in drishti_list if d.get("planet") == moon_house_lord), {}) or {}

    nk = moon.get("nakshatra", {})
    if isinstance(nk, dict):
        nk_name = nk.get("name", "—")
        nk_pada = nk.get("pada", "")
    elif isinstance(nk, str):
        nk_name, nk_pada = nk, ""
    else:
        nk_name, nk_pada = "—", ""

    return {
        "include": bool(connected_role),
        "reason": (
            f"Moon house lord {moon_house_lord} matched Rule 1 {connected_role}."
            if connected_role else
            f"Moon house lord {moon_house_lord or '—'} did not match Rule 1 occupant/dispositor/6th-house-lord/drishti planets."
        ),
        "connected_role": connected_role,
        "connected_planet": connected_planet,
        "drishti_match": connected_drishti,
        "house_lord": moon_house_lord,
        "house_sign": moon_house_sign,
        "planet": {
            "name": "Moon",
            "house": moon_house,
            "sign": (moon.get("sign") or {}).get("name", "—") if isinstance(moon.get("sign"), dict) else moon.get("sign", "—"),
            "degree": moon.get("degree", 0),
            "minutes": moon.get("minutes", 0),
            "nakshatra": nk_name,
            "nakshatra_pada": nk_pada,
            "isRetrograde": moon.get("isRetrograde", False),
        }
    }

def _build_house_occupant_relations(chart_data, rogkaraka_1, rogkaraka_2, rogkaraka_3, drishti_on_6th, house_numbers):
    """
    Display-only. For each planet occupying house_numbers (2/8/11/12),
    checks if that planet is directly connected to Rule 1:
      - Occupant (RK1), Dispositor (RK3), 6th Lord (RK2), or Drishti.
    Does NOT feed into HitList, scoring, or Rashi Correlation.
    Returns dict: { house_num -> [ {name, house, connected_role, connection_reason, related} ] }
    """
    planets      = chart_data.get("planets", [])
    occupant_names  = {p.get("name", "") for p in (rogkaraka_1 or []) if p.get("name")}
    rk3_name     = (rogkaraka_3 or {}).get("name", "")
    rk2_name     = (rogkaraka_2 or {}).get("name", "")
    drishti_list = drishti_on_6th or []
    drishti_planets = {d.get("planet", "") for d in drishti_list if d.get("planet")}

    result = {}
    for house_num in house_numbers:
        occupant_relations = []
        for p in planets:
            if p.get("name") == "Ascendant":
                continue
            if p.get("house") == house_num:
                pname = p.get("name", "")
                connected_role    = ""
                connection_reason = ""

                if pname in occupant_names:
                    connected_role    = "occupant"
                    connection_reason = f"{pname} is directly sitting in the 6th house as an Occupant (RK1)."
                elif rk3_name and pname == rk3_name:
                    connected_role    = "dispositor"
                    connection_reason = f"{pname} is the Dispositor (RK3) of the 6th lord chain."
                elif rk2_name and pname == rk2_name:
                    connected_role    = "6th_house_lord"
                    connection_reason = f"{pname} is the 6th House Lord (RK2)."
                elif pname in drishti_planets:
                    connected_role    = "drishti"
                    d_detail     = next((d for d in drishti_list if d.get("planet") == pname), {})
                    aspect_type  = d_detail.get("aspect_type", "aspect")
                    from_house   = d_detail.get("from_house", house_num)
                    connection_reason = f"{pname} casts a {aspect_type} on the 6th house from House {from_house}."

                occupant_relations.append({
                    "name":             pname,
                    "house":            house_num,
                    "connected_role":   connected_role,
                    "connection_reason": connection_reason,
                    "related":          bool(connected_role)
                })
        result[house_num] = occupant_relations

    return result


def apply_rule_1(chart_data):

    asc_house = chart_data["ascendant"].get("house", 1)
    
    # The 6th house is NOT always chart house 6.
    # It is the 6th position counting FROM the ascendant. Ascendant = position 1.
    sixth_house_number = ((asc_house - 1) + 5) % 12 + 1
    eighth_house_number = ((asc_house - 1) + 7) % 12 + 1
    eleventh_house_number = ((asc_house - 1) + 10) % 12 + 1
    
    # Step 1 — Find 6th house sign:
    sixth_house_sign = ""
    for h in chart_data.get("houses", []):
        if h["house"] == sixth_house_number:
            sixth_house_sign = h["sign"]["name"]
            break

    # Find 8th house sign:
    eighth_house_sign = ""
    for h in chart_data.get("houses", []):
        if h["house"] == eighth_house_number:
            eighth_house_sign = h["sign"]["name"]
            break

    # Find 11th house sign:
    eleventh_house_sign = ""
    for h in chart_data.get("houses", []):
        if h["house"] == eleventh_house_number:
            eleventh_house_sign = h["sign"]["name"]
            break

    eighth_house_lord = SIGN_LORDS.get(eighth_house_sign, "")
    eleventh_house_lord = SIGN_LORDS.get(eleventh_house_sign, "")
            
    # Step 2 — Rogkaraka 1:
    rogkaraka_1 = []
    eighth_house_occupants = []
    eleventh_house_occupants = []
    for p in chart_data.get("planets", []):
        if p["name"] == "Ascendant":
            continue

        nk = p.get("nakshatra", {})
        if isinstance(nk, dict):
            nk_name = nk.get("name", "—")
            nk_pada = nk.get("pada", "")
        elif isinstance(nk, str):
            nk_name, nk_pada = nk, ""
        else:
            nk_name, nk_pada = "—", ""

        planet_details = {
            "name": p["name"],
            "house": p["house"],
            "sign": p["sign"]["name"] if isinstance(p.get("sign"), dict) else p.get("sign", "—"),
            "degree": p.get("degree", 0),
            "minutes": p.get("minutes", 0),
            "nakshatra": nk_name,
            "nakshatra_pada": nk_pada,
            "isRetrograde": p.get("isRetrograde", False)
        }

        if p["house"] == sixth_house_number:
            rogkaraka_1.append(planet_details)
        
        if p["house"] == eighth_house_number:
            eighth_house_occupants.append(planet_details)

        if p["house"] == eleventh_house_number:
            eleventh_house_occupants.append(planet_details)
            
    # Step 3 — Rogkaraka 2:
    rogkaraka_2 = {}
    sixth_house_lord = SIGN_LORDS.get(sixth_house_sign, "")
    for p in chart_data.get("planets", []):
        if p["name"] == sixth_house_lord:
            nk = p.get("nakshatra", {})
            if isinstance(nk, dict):
                nk_name = nk.get("name", "—")
                nk_pada = nk.get("pada", "")
            elif isinstance(nk, str):
                nk_name, nk_pada = nk, ""
            else:
                nk_name, nk_pada = "—", ""

            rogkaraka_2 = {
                "name": p["name"],
                "house": p["house"],
                "sign": p["sign"]["name"] if isinstance(p.get("sign"), dict) else p.get("sign", "—"),
                "degree": p.get("degree", 0),
                "minutes": p.get("minutes", 0),
                "nakshatra": nk_name,
                "nakshatra_pada": nk_pada
            }
            break

    # Find the 8th house lord planet object:
    eighth_house_lord_obj = {}
    for p in chart_data.get("planets", []):
        if p["name"] == eighth_house_lord:
            nk = p.get("nakshatra", {})
            if isinstance(nk, dict):
                nk_name = nk.get("name", "—")
                nk_pada = nk.get("pada", "")
            elif isinstance(nk, str):
                nk_name, nk_pada = nk, ""
            else:
                nk_name, nk_pada = "—", ""
            
            eighth_house_lord_obj = {
                "name": p["name"],
                "house": p["house"],
                "sign": p["sign"]["name"] if isinstance(p.get("sign"), dict) else p.get("sign", "—"),
                "degree": p.get("degree", 0),
                "minutes": p.get("minutes", 0),
                "nakshatra": nk_name,
                "nakshatra_pada": nk_pada
            }
            break
            
    # Step 4 — Rogkaraka 3:
    rogkaraka_3 = None
    if rogkaraka_2:
        rk2_house = rogkaraka_2["house"]
        rk2_sign = rogkaraka_2["sign"]
        
        if rk2_sign != sixth_house_sign:
            rk3_name = SIGN_LORDS.get(rk2_sign, "")
            for p in chart_data.get("planets", []):
                if p["name"] == rk3_name:
                    nk = p.get("nakshatra", {})
                    if isinstance(nk, dict):
                        nk_name = nk.get("name", "—")
                        nk_pada = nk.get("pada", "")
                    elif isinstance(nk, str):
                        nk_name, nk_pada = nk, ""
                    else:
                        nk_name, nk_pada = "—", ""

                    rogkaraka_3 = {
                        "name": p["name"],
                        "house": p["house"],
                        "sign": p["sign"]["name"] if isinstance(p.get("sign"), dict) else p.get("sign", "—"),
                        "degree": p.get("degree", 0),
                        "minutes": p.get("minutes", 0),
                        "nakshatra": nk_name,
                        "nakshatra_pada": nk_pada
                    }
                    break
                    
    # Step 5 — Ascendant Details:
    asc = chart_data.get("ascendant", {})
    nk = asc.get("nakshatra", {})
    if isinstance(nk, dict):
        asc_nk_name = nk.get("name", "—")
        asc_nk_pada = nk.get("pada", "")
    elif isinstance(nk, str):
        asc_nk_name, asc_nk_pada = nk, ""
    else:
        asc_nk_name, asc_nk_pada = "—", ""

    asc_sign = asc.get("name", "")
    asc_lord = SIGN_LORDS.get(asc_sign, "")

    # Pre-build result to pass to drishti analysis
    rule1_result = {
        "ascendant_house": asc_house,
        "ascendant_sign": asc_sign,
        "ascendant_lord": asc_lord,
        "ascendant_nakshatra": asc_nk_name,
        "ascendant_nakshatra_pada": asc_nk_pada,
        "sixth_house_number": sixth_house_number,
        "sixth_house_sign": sixth_house_sign,
        "sixth_house_lord": sixth_house_lord,
        "rogkaraka_1": rogkaraka_1,
        "rogkaraka_2": rogkaraka_2,
        "rogkaraka_3": rogkaraka_3,
        "eighth_house_number": eighth_house_number,
        "eighth_house_sign": eighth_house_sign,
        "eighth_house_lord": eighth_house_lord,
        "eighth_house_occupants": eighth_house_occupants,
        "eighth_house_lord_obj": eighth_house_lord_obj,
        "eleventh_house_number": eleventh_house_number,
        "eleventh_house_sign": eleventh_house_sign,
        "eleventh_house_lord": eleventh_house_lord,
        "eleventh_house_occupants": eleventh_house_occupants,
    }

    # Step 6 — Drishti on 6th house (required for HitList)
    planets_list = chart_data.get("planets", [])
    drishti_data = analyse_drishti(planets_list, rule1_result)
    rule1_result["drishti_on_6th"] = drishti_data.get("aspects_sixth_house", [])
    rule1_result["conditional_moon"] = _build_conditional_moon(
        chart_data,
        rogkaraka_1,
        rogkaraka_2,
        rogkaraka_3,
        rule1_result.get("drishti_on_6th"),
    )
    rule1_result["house_occupant_relations"] = _build_house_occupant_relations(
        chart_data,
        rogkaraka_1,
        rogkaraka_2,
        rogkaraka_3,
        rule1_result.get("drishti_on_6th"),
        [2, 8, 11, 12]
    )

    return rule1_result


def generate_chat_questions(rule1_result: dict) -> dict:
    """
    Generates personalised health questions
    based on Rogkaraka analysis output.
    Uses knowledge_base data to map planets
    and signs to health indicators.
    """

    questions = []
    context   = {}

    # ── Step 1: Collect all Rogkaraka planets ──
    relevant_planets = []

    rk1_list = rule1_result.get("rogkaraka_1", [])
    for p in rk1_list:
        if p["name"] not in relevant_planets:
            relevant_planets.append(p["name"])

    rk2 = rule1_result.get("rogkaraka_2")
    if rk2 and rk2["name"] not in relevant_planets:
        relevant_planets.append(rk2["name"])

    rk3 = rule1_result.get("rogkaraka_3")
    if rk3 and rk3["name"] not in relevant_planets:
        relevant_planets.append(rk3["name"])

    # Add Ascendant Lord
    asc_lord = rule1_result.get("ascendant_lord")
    if asc_lord and asc_lord not in relevant_planets:
        relevant_planets.append(asc_lord)

    # ── Step 2: Get 6th house context ──────────
    sixth_sign   = rule1_result.get("sixth_house_sign", "")
    sixth_organs = RASHI_ORGANS.get(sixth_sign, "")
    context["sixth_house_sign"]   = sixth_sign
    context["sixth_house_organs"] = sixth_organs

    # Add Ascendant context
    asc_sign = rule1_result.get("ascendant_sign", "")
    asc_organs = RASHI_ORGANS.get(asc_sign, "")
    context["ascendant_sign"] = asc_sign
    context["ascendant_organs"] = asc_organs

    # ── Step 3: Get planet body systems ────────
    planet_systems = {}
    for pname in relevant_planets:
        system = PLANET_DISEASES.get(pname, "")
        if system:
            planet_systems[pname] = system
    context["planet_systems"] = planet_systems

    # ── Step 4: Get dosha types ───────────────
    doshas = []
    for pname in relevant_planets:
        dosha = PLANET_DOSHA.get(pname, "")
        if dosha and dosha not in doshas:
            doshas.append(dosha)
    context["doshas"] = doshas

    # ── Step 5: Get nakshatra diseases ─────────
    nakshatra_diseases = []
    all_rk = rk1_list + ([rk2] if rk2 else []) + ([rk3] if rk3 else [])
    
    # Add Ascendant Nakshatra to the pool
    asc_nk = rule1_result.get("ascendant_nakshatra")
    if asc_nk and asc_nk != "—":
        all_rk.append({
            "name": "Ascendant",
            "nakshatra": asc_nk,
            "nakshatra_pada": rule1_result.get("ascendant_nakshatra_pada")
        })

    for p in all_rk:
        nk_name = p.get("nakshatra", "")
        if nk_name and nk_name != "—":
            diseases = get_nakshatra_diseases(nk_name, p.get("nakshatra_pada"))
            for d in diseases[:2]:
                if d not in nakshatra_diseases:
                    nakshatra_diseases.append(d)
    context["nakshatra_diseases"] = nakshatra_diseases

    # ── Step 6: Build questions ─────────────────

    # Q0: Based on Ascendant organs (Baseline)
    if asc_organs:
        organ_list = [o.strip() for o in asc_organs.split(",")]
        for organ in organ_list[:1]:
            questions.append({
                "id":       len(questions) + 1,
                "category": "Baseline Health",
                "question": f"Have you generally felt that your "
                            f"{organ.lower()} — or your overall "
                            f"physical strength — needs more "
                            f"care than usual?",
                "options":  ["Yes", "No", "Sometimes"],
                "context":  f"Ascendant ({asc_sign}) "
                            f"governs {organ}"
            })

    # Q1 & Q2: Based on 6th house organs
    if sixth_organs:
        organ_list = [o.strip() for o in sixth_organs.split(",")]
        for organ in organ_list[:2]:
            questions.append({
                "id":       len(questions) + 1,
                "category": "Current Symptoms",
                "question": f"Do you currently experience "
                            f"any issues related to your "
                            f"{organ.lower()}?",
                "options":  ["Yes", "No", "Sometimes"],
                "context":  f"6th house ({sixth_sign}) "
                            f"governs {organ}"
            })

    # Q3: Based on Rogkaraka planet body system
    for pname in relevant_planets[:1]:
        system = PLANET_DISEASES.get(pname, "")
        if system:
            first = system.split(",")[0].strip()
            questions.append({
                "id":       len(questions) + 1,
                "category": "Body Systems",
                "question": f"Have you noticed any "
                            f"{first.lower()} related "
                            f"problems recently?",
                "options":  ["Yes", "No", "Sometimes"],
                "context":  f"{pname} governs {first}"
            })

    # Q4: Nakshatra disease tendency
    if nakshatra_diseases:
        disease = nakshatra_diseases[0]
        questions.append({
            "id":       len(questions) + 1,
            "category": "Medical History",
            "question": f"Have you or a close family "
                        f"member ever experienced "
                        f"{disease.lower()}?",
            "options":  ["Yes, myself",
                         "Yes, family member",
                         "No", "Not sure"],
            "context":  f"Nakshatra indicates tendency "
                        f"towards {disease}"
        })

    # Q5: Dosha symptoms
    if doshas:
        dosha    = doshas[0]
        symptoms = DOSHA_SYMPTOMS.get(dosha, "")
        questions.append({
            "id":       len(questions) + 1,
            "category": "Ayurvedic Constitution",
            "question": f"Do you frequently experience "
                        f"any of these: {symptoms}?",
            "options":  ["Frequently", "Sometimes",
                         "Rarely", "Never"],
            "context":  f"{dosha} dosha imbalance indicated"
        })

    # Q6: Mental stress
    questions.append({
        "id":       len(questions) + 1,
        "category": "Mental Peace",
        "question": "How would you rate your mental "
                    "stress level currently?",
        "options":  ["Very High", "High",
                     "Moderate", "Low", "Very Low"],
        "context":  "Mental state affects physical health"
    })

    # Q7: Energy levels
    questions.append({
        "id":       len(questions) + 1,
        "category": "Energy Levels",
        "question": "How are your overall energy "
                    "levels through the day?",
        "options":  ["Excellent", "Good",
                     "Average", "Poor", "Very Poor"],
        "context":  "Energy indicates overall vitality"
    })

    # Q8: Sleep quality
    questions.append({
        "id":       len(questions) + 1,
        "category": "Lifestyle",
        "question": "How is your sleep quality "
                    "on most nights?",
        "options":  ["Very Good", "Good",
                     "Fair", "Poor", "Very Poor"],
        "context":  "Sleep quality affects all body systems"
    })

    return {
        "questions": questions,
        "context":   context,
        "total":     len(questions)
    }



def score_health_risks(rule1_result: dict,
                       user_answers: list) -> list:
    """
    Combines Rogkaraka chart data with user answers
    to produce probability-scored disease risk list.
    Returns risks sorted highest first.
    """

    risks = []

    sixth_sign   = rule1_result.get("sixth_house_sign", "")
    sixth_organs = RASHI_ORGANS.get(sixth_sign, "")

    # ── Collect all Rogkaraka planets ─────────────────────
    rk_planets = []
    for p in rule1_result.get("rogkaraka_1", []):
        rk_planets.append(p)
    rk2 = rule1_result.get("rogkaraka_2")
    if rk2:
        rk_planets.append(rk2)
    rk3 = rule1_result.get("rogkaraka_3")
    if rk3:
        rk_planets.append(rk3)

    # Add Ascendant Lord
    asc_lord_name = rule1_result.get("ascendant_lord")
    if asc_lord_name:
        rk_planets.append({"name": asc_lord_name})

    # ── Build answer score map ─────────────────────────────
    answer_scores    = {}
    lifestyle_penalty = 0

    for ans in user_answers:
        question = ans.get("question", "").lower()
        answer   = ans.get("answer",   "").lower()
        category = ans.get("category", "").lower()

        if answer in ["yes", "yes, myself"]:
            ans_score = 3
        elif answer in ["sometimes", "yes, family member", "frequently"]:
            ans_score = 2
        elif answer in ["rarely", "not sure", "moderate", "average",
                        "fair", "high"]:
            ans_score = 1
        elif answer in ["very high"]:
            ans_score = 2
        elif answer in ["no", "never", "good", "very good",
                        "excellent", "low", "very low"]:
            ans_score = 0
        elif answer in ["poor"]:
            ans_score = 1
        elif answer in ["very poor"]:
            ans_score = 2
        else:
            ans_score = 0

        answer_scores[category] = answer_scores.get(category, 0) + ans_score

        if category in ["lifestyle", "energy levels", "mental peace"]:
            lifestyle_penalty += ans_score

    # ── Base chart score per Rogkaraka ─────────────────────
    rk1_count       = len(rule1_result.get("rogkaraka_1", []))
    base_chart_score = 3 + rk1_count

    # ── Score each organ from 6th house ───────────────────
    if sixth_organs:
        organs = [o.strip() for o in sixth_organs.split(",")]
        for i, organ in enumerate(organs):
            organ_lower = organ.lower()
            chart_score = max(1, base_chart_score - i)

            symptom_score = 0
            for ans in user_answers:
                q_lower = ans.get("question", "").lower()
                a_lower = ans.get("answer",   "").lower()
                if organ_lower in q_lower:
                    if a_lower in ["yes", "yes, myself"]:
                        symptom_score = 3
                    elif a_lower in ["sometimes", "frequently"]:
                        symptom_score = 2
                    elif a_lower in ["rarely"]:
                        symptom_score = 1

            planet_match = 0
            for p in rk_planets:
                pname   = p.get("name", "")
                systems = PLANET_DISEASES.get(pname, "").lower()
                if organ_lower in systems:
                    planet_match = 2

            total_score = chart_score + symptom_score + planet_match + lifestyle_penalty
            risks.append({
                "condition":     organ,
                "category":      "Organ System",
                "chart_score":   chart_score,
                "symptom_score": symptom_score,
                "total_score":   total_score,
                "source":        f"6th house ({sixth_sign})"
            })

    # ── Score each organ from Ascendant (Baseline) ────────
    asc_sign   = rule1_result.get("ascendant_sign", "")
    asc_organs = RASHI_ORGANS.get(asc_sign, "")
    if asc_organs:
        organs = [o.strip() for o in asc_organs.split(",")]
        for i, organ in enumerate(organs[:3]):
            organ_lower = organ.lower()
            # Baseline chart score for Ascendant is 1
            chart_score = 1

            symptom_score = 0
            for ans in user_answers:
                q_lower = ans.get("question", "").lower()
                a_lower = ans.get("answer",   "").lower()
                if organ_lower in q_lower:
                    if a_lower in ["yes", "yes, myself"]:
                        symptom_score = 3
                    elif a_lower in ["sometimes", "frequently"]:
                        symptom_score = 2

            total_score = chart_score + symptom_score + lifestyle_penalty
            risks.append({
                "condition":     organ,
                "category":      "Organ System",
                "chart_score":   chart_score,
                "symptom_score": symptom_score,
                "total_score":   total_score,
                "source":        f"Ascendant ({asc_sign})"
            })

    # ── Score nakshatra diseases ───────────────────────────
    nakshatra_done = []
    
    # Add Ascendant Nakshatra to the pool
    asc_nk = rule1_result.get("ascendant_nakshatra")
    if asc_nk and asc_nk != "—":
        rk_planets.append({
            "name": "Ascendant",
            "nakshatra": asc_nk,
            "nakshatra_pada": rule1_result.get("ascendant_nakshatra_pada")
        })

    for p in rk_planets:
        nk_name = p.get("nakshatra", "")
        if not nk_name or nk_name == "—" or nk_name in nakshatra_done:
            continue
        nakshatra_done.append(nk_name)

        # Baseline score for Ascendant Nakshatra is 1, others are 2
        chart_score = 1 if p.get("name") == "Ascendant" else 2
        
        for disease in get_nakshatra_diseases(nk_name, p.get("nakshatra_pada"))[:3]:
            disease_lower = disease.lower()
            symptom_score = 0

            for ans in user_answers:
                q_lower = ans.get("question", "").lower()
                a_lower = ans.get("answer",   "").lower()
                disease_words = disease_lower.split()
                matched = any(w in q_lower for w in disease_words if len(w) > 4)
                if matched:
                    if a_lower in ["yes", "yes, myself"]:
                        symptom_score = 3
                    elif a_lower in ["sometimes", "frequently"]:
                        symptom_score = 2

            for ans in user_answers:
                if ans.get("category", "") == "Medical History":
                    a_lower = ans.get("answer", "").lower()
                    if a_lower in ["yes, myself", "yes, family member"]:
                        symptom_score += 1

            total_score = chart_score + symptom_score + int(lifestyle_penalty / 2)
            risks.append({
                "condition":     disease,
                "category":      "Nakshatra Tendency",
                "chart_score":   chart_score,
                "symptom_score": symptom_score,
                "total_score":   total_score,
                "source":        f"{nk_name} nakshatra"
            })

    # ── Score dosha imbalances ─────────────────────────────
    doshas_done = []
    for p in rk_planets:
        pname = p.get("name", "")
        dosha = PLANET_DOSHA.get(pname, "")
        if not dosha or dosha in doshas_done:
            continue
        doshas_done.append(dosha)

        chart_score  = 2
        dosha_score  = 0
        for ans in user_answers:
            if ans.get("category", "") == "Ayurvedic Constitution":
                a_lower = ans.get("answer", "").lower()
                if a_lower == "frequently":
                    dosha_score = 3
                elif a_lower == "sometimes":
                    dosha_score = 2
                elif a_lower == "rarely":
                    dosha_score = 1

        total_score = chart_score + dosha_score + lifestyle_penalty
        risks.append({
            "condition":     f"{dosha} Dosha Imbalance",
            "category":      "Ayurvedic",
            "chart_score":   chart_score,
            "symptom_score": dosha_score,
            "total_score":   total_score,
            "source":        f"{pname} planet"
        })

    # ── Assign risk levels ─────────────────────────────────
    for risk in risks:
        score = risk["total_score"]
        if score >= 7:
            risk["risk_level"]  = "HIGH"
            risk["risk_color"]  = "#ef4444"
            risk["risk_bg"]     = "rgba(239,68,68,0.08)"
            risk["risk_border"] = "rgba(239,68,68,0.25)"
        elif score >= 4:
            risk["risk_level"]  = "MODERATE"
            risk["risk_color"]  = "#f59e0b"
            risk["risk_bg"]     = "rgba(245,158,11,0.08)"
            risk["risk_border"] = "rgba(245,158,11,0.25)"
        else:
            risk["risk_level"]  = "LOW"
            risk["risk_color"]  = "#22c55e"
            risk["risk_bg"]     = "rgba(34,197,94,0.08)"
            risk["risk_border"] = "rgba(34,197,94,0.25)"

    # Sort HIGH → MODERATE → LOW, then by score desc
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    risks.sort(key=lambda x: (order[x["risk_level"]], -x["total_score"]))

    # Deduplicate keeping highest score
    seen, unique_risks = [], []
    for r in risks:
        cond_lower = r["condition"].lower()
        if cond_lower not in seen:
            seen.append(cond_lower)
            unique_risks.append(r)

    return unique_risks


def analyse_dasha(dasha_raw: dict,
                  rule1_result: dict) -> dict:
    """
    Finds current Mahadasha, Antardasha
    and Pratyantardasha from Prokerala data.
    Checks if any match Rogkaraka planets.
    """

    result = {
        "mahadasha":       None,
        "antardasha":      None,
        "pratyantardasha": None,
        "mahadasha_periods": [],
        "antardasha_groups": [],
        "past_5_years":    [],
        "expanded_history": [],
        "alerts":          [],
        "is_peak_risk":    False,
        "match_count":     0,
        "upcoming_alerts": []
    }

    today = date.today()
    five_years_ago = today - timedelta(days=365 * 5)
    six_months_out = today + timedelta(days=180)

    # Collect Rogkaraka planet names
    rk_planets = set()
    for p in rule1_result.get("rogkaraka_1",[]):
        rk_planets.add(p["name"])
    rk2 = rule1_result.get("rogkaraka_2")
    if rk2:
        rk_planets.add(rk2["name"])
    rk3 = rule1_result.get("rogkaraka_3")
    if rk3:
        rk_planets.add(rk3["name"])

    print(f"   RK planets for dasha check: "
          f"{rk_planets}")

    # Get dasha periods list
    # Try different key names Prokerala may use
    periods = (
        dasha_raw.get("dasha_periods") or
        dasha_raw.get("mahadasha")     or
        dasha_raw.get("dashas")        or
        dasha_raw.get("vimshottari_dasha") or
        []
    )

    if not periods:
        print("   No dasha periods in response")
        print(f"   Available keys: "
              f"{list(dasha_raw.keys())}")
        return result

    def _planet_name(period):
        return (
            (period or {}).get("planet", {}).get("name", "")
            or (period or {}).get("lord", "")
            or (period or {}).get("name", "")
        )

    def _period_start(period):
        return str((period or {}).get("start") or (period or {}).get("start_date", ""))[:10]

    def _period_end(period):
        return str((period or {}).get("end") or (period or {}).get("end_date", ""))[:10]

    def _antardashas(period):
        return (
            (period or {}).get("antardasha")
            or (period or {}).get("sub_periods")
            or (period or {}).get("bhukti")
            or []
        )

    result["mahadasha_periods"] = [
        {
            "mahadasha": _planet_name(maha),
            "start_date": _period_start(maha),
            "end_date": _period_end(maha),
        }
        for maha in periods
        if _planet_name(maha)
    ]

    result["antardasha_groups"] = []
    for maha in periods:
        maha_planet = _planet_name(maha)
        rows = []
        for antar in _antardashas(maha):
            antar_planet = _planet_name(antar)
            if not antar_planet:
                continue
            rows.append({
                "mahadasha": maha_planet,
                "antardasha": antar_planet,
                "start_date": _period_start(antar),
                "end_date": _period_end(antar),
            })
        if maha_planet and rows:
            result["antardasha_groups"].append({
                "mahadasha": maha_planet,
                "start_date": _period_start(maha),
                "end_date": _period_end(maha),
                "rows": rows,
            })

    # Find current Mahadasha
    for maha in periods:
        try:
            m_start_str = str(maha.get("start") or maha.get("start_date", "2000-01-01"))[:10]
            m_end_str   = str(maha.get("end") or maha.get("end_date", "2100-01-01"))[:10]
            m_start = date.fromisoformat(m_start_str)
            m_end   = date.fromisoformat(m_end_str)
        except Exception as e:
            print(f"   Date parse error: {e}")
            continue

        if not (m_start <= today <= m_end):
            continue

        # Found current Mahadasha
        maha_planet = (
            maha.get("planet",{}).get("name","")
            or maha.get("lord","")
            or maha.get("name","")
        )

        result["mahadasha"] = {
            "planet":     maha_planet,
            "start_date": m_start_str,
            "end_date":   m_end_str,
            "duration":   str(maha.get("duration", ""))
        }

        # Find current Antardasha
        antardashas = (
            maha.get("antardasha")   or
            maha.get("sub_periods")  or
            maha.get("bhukti")       or
            []
        )

        for antar in antardashas:
            try:
                a_start_str = str(antar.get("start") or antar.get("start_date", "2000-01-01"))[:10]
                a_end_str   = str(antar.get("end") or antar.get("end_date", "2100-01-01"))[:10]
                a_start = date.fromisoformat(a_start_str)
                a_end   = date.fromisoformat(a_end_str)
            except:
                continue

            if not (a_start <= today <= a_end):
                continue

            # Found current Antardasha
            antar_planet = (
                antar.get("planet",{}).get("name","")
                or antar.get("lord","")
                or antar.get("name","")
            )

            result["antardasha"] = {
                "planet":     antar_planet,
                "start_date": a_start_str,
                "end_date":   a_end_str,
                "duration":   str(antar.get("duration", ""))
            }

            # Find current Pratyantardasha
            pratyas = (
                antar.get("pratyantardasha") or
                antar.get("sub_periods")     or
                antar.get("antardasha")      or
                []
            )

            for pratya in pratyas:
                try:
                    p_start_str = str(pratya.get("start") or pratya.get("start_date", "2000-01-01"))[:10]
                    p_end_str   = str(pratya.get("end") or pratya.get("end_date", "2100-01-01"))[:10]
                    p_start = date.fromisoformat(p_start_str)
                    p_end   = date.fromisoformat(p_end_str)
                except:
                    continue

                if not (p_start <= today <= p_end):
                    continue

                pratya_planet = (
                    pratya.get("planet",{}).get("name","")
                    or pratya.get("lord","")
                    or pratya.get("name","")
                )

                result["pratyantardasha"] = {
                    "planet":     pratya_planet,
                    "start_date": p_start_str,
                    "end_date":   p_end_str,
                    "duration":   str(pratya.get("duration", ""))
                }
                break
            break
        break

    # Check for upcoming dasha periods in the next 6 months
    for maha in periods:
        try:
            m_start_str = str(maha.get("start") or maha.get("start_date", "2000-01-01"))[:10]
            m_start = date.fromisoformat(m_start_str)
        except:
            continue
            
        maha_planet = maha.get("planet",{}).get("name","") or maha.get("lord","") or maha.get("name","")
        if today < m_start <= six_months_out and maha_planet in rk_planets:
            result["upcoming_alerts"].append({
                "level": "Mahadasha",
                "planet": maha_planet,
                "start_date": m_start_str,
                "message": f"Watchful period approaching: {maha_planet} activates as Mahadasha on {m_start_str}."
            })
            
        antardashas = maha.get("antardasha") or maha.get("sub_periods") or maha.get("bhukti") or []
        for antar in antardashas:
            try:
                a_start_str = str(antar.get("start") or antar.get("start_date", "2000-01-01"))[:10]
                a_start = date.fromisoformat(a_start_str)
            except:
                continue
                
            antar_planet = antar.get("planet",{}).get("name","") or antar.get("lord","") or antar.get("name","")
            if today < a_start <= six_months_out and antar_planet in rk_planets:
                result["upcoming_alerts"].append({
                    "level": "Antardasha",
                    "planet": antar_planet,
                    "start_date": a_start_str,
                    "message": f"Watchful period approaching: {antar_planet} activates as Antardasha on {a_start_str}."
                })
                
            pratyas = antar.get("pratyantardasha") or antar.get("sub_periods") or antar.get("antardasha") or []
            for pratya in pratyas:
                try:
                    p_start_str = str(pratya.get("start") or pratya.get("start_date", "2000-01-01"))[:10]
                    p_start = date.fromisoformat(p_start_str)
                except:
                    continue
                    
                pratya_planet = pratya.get("planet",{}).get("name","") or pratya.get("lord","") or pratya.get("name","")
                if today < p_start <= six_months_out and pratya_planet in rk_planets:
                    result["upcoming_alerts"].append({
                        "level": "Pratyantardasha",
                        "planet": pratya_planet,
                        "start_date": p_start_str,
                        "message": f"Watchful period approaching: {pratya_planet} activates as Pratyantardasha on {p_start_str}."
                    })

    # Check Rogkaraka matches
    levels = [
        ("Mahadasha",
         result["mahadasha"],       "HIGH"),
        ("Antardasha",
         result["antardasha"],      "CRITICAL"),
        ("Pratyantardasha",
         result["pratyantardasha"], "CRITICAL")
    ]

    for level_name, level_data, severity in levels:
        if not level_data:
            continue
        planet = level_data.get("planet","")
        if planet in rk_planets:
            result["alerts"].append({
                "level":    level_name,
                "planet":   planet,
                "end_date": level_data.get(
                    "end_date",""),
                "severity": severity,
                "message":  (
                    f"{planet} is your disease "
                    f"indicator AND currently "
                    f"running as {level_name}. "
                    f"This risk is ACTIVE NOW."
                )
            })

    result["match_count"]  = len(result["alerts"])
    result["is_peak_risk"] = (
        result["match_count"] >= 2)

    # Build a readable history for the last 5 years using the finest
    # available level (Pratyantardasha) and its parent periods.
    past_five_year_rows = []
    for maha in periods:
        maha_planet = (
            maha.get("planet", {}).get("name", "")
            or maha.get("lord", "")
            or maha.get("name", "")
        )
        antardashas = (
            maha.get("antardasha")
            or maha.get("sub_periods")
            or maha.get("bhukti")
            or []
        )

        for antar in antardashas:
            antar_planet = (
                antar.get("planet", {}).get("name", "")
                or antar.get("lord", "")
                or antar.get("name", "")
            )
            pratyas = (
                antar.get("pratyantardasha")
                or antar.get("sub_periods")
                or antar.get("antardasha")
                or []
            )

            if pratyas:
                for pratya in pratyas:
                    try:
                        p_start_str = str(pratya.get("start") or pratya.get("start_date", ""))[:10]
                        p_end_str = str(pratya.get("end") or pratya.get("end_date", ""))[:10]
                        if not p_start_str or not p_end_str:
                            continue
                        p_start = date.fromisoformat(p_start_str)
                        p_end = date.fromisoformat(p_end_str)
                    except Exception:
                        continue

                    if p_end < five_years_ago or p_start > today:
                        continue

                    pratya_planet = (
                        pratya.get("planet", {}).get("name", "")
                        or pratya.get("lord", "")
                        or pratya.get("name", "")
                    )
                    past_five_year_rows.append({
                        "mahadasha": maha_planet,
                        "antardasha": antar_planet,
                        "pratyantardasha": pratya_planet,
                        "start_date": p_start_str,
                        "end_date": p_end_str
                    })
            else:
                try:
                    a_start_str = str(antar.get("start") or antar.get("start_date", ""))[:10]
                    a_end_str = str(antar.get("end") or antar.get("end_date", ""))[:10]
                    if not a_start_str or not a_end_str:
                        continue
                    a_start = date.fromisoformat(a_start_str)
                    a_end = date.fromisoformat(a_end_str)
                except Exception:
                    continue

                if a_end < five_years_ago or a_start > today:
                    continue

                past_five_year_rows.append({
                    "mahadasha": maha_planet,
                    "antardasha": antar_planet,
                    "pratyantardasha": "—",
                    "start_date": a_start_str,
                    "end_date": a_end_str
                })

    result["past_5_years"] = sorted(
        past_five_year_rows,
        key=lambda item: item["start_date"],
        reverse=True
    )

    current_mahadasha_planet = (result.get("mahadasha") or {}).get("planet", "")
    current_mahadasha_start = (result.get("mahadasha") or {}).get("start_date", "")
    expanded_history_start = None

    if current_mahadasha_start:
        for index, maha in enumerate(periods):
            maha_planet = (
                maha.get("planet", {}).get("name", "")
                or maha.get("lord", "")
                or maha.get("name", "")
            )
            maha_start_str = str(maha.get("start") or maha.get("start_date", ""))[:10]
            if maha_planet == current_mahadasha_planet and maha_start_str == current_mahadasha_start:
                if index > 0:
                    expanded_history_start = str(
                        periods[index - 1].get("start") or periods[index - 1].get("start_date", "")
                    )[:10]
                else:
                    expanded_history_start = maha_start_str
                break

    if expanded_history_start:
        try:
            expanded_history_cutoff = date.fromisoformat(expanded_history_start)
        except Exception:
            expanded_history_cutoff = None

        if expanded_history_cutoff:
            expanded_rows = []
            for maha in periods:
                maha_planet = (
                    maha.get("planet", {}).get("name", "")
                    or maha.get("lord", "")
                    or maha.get("name", "")
                )
                antardashas = (
                    maha.get("antardasha")
                    or maha.get("sub_periods")
                    or maha.get("bhukti")
                    or []
                )

                for antar in antardashas:
                    antar_planet = (
                        antar.get("planet", {}).get("name", "")
                        or antar.get("lord", "")
                        or antar.get("name", "")
                    )
                    pratyas = (
                        antar.get("pratyantardasha")
                        or antar.get("sub_periods")
                        or antar.get("antardasha")
                        or []
                    )

                    if pratyas:
                        for pratya in pratyas:
                            try:
                                p_start_str = str(pratya.get("start") or pratya.get("start_date", ""))[:10]
                                p_end_str = str(pratya.get("end") or pratya.get("end_date", ""))[:10]
                                if not p_start_str or not p_end_str:
                                    continue
                                p_start = date.fromisoformat(p_start_str)
                                p_end = date.fromisoformat(p_end_str)
                            except Exception:
                                continue

                            if p_end < expanded_history_cutoff or p_start > today:
                                continue

                            pratya_planet = (
                                pratya.get("planet", {}).get("name", "")
                                or pratya.get("lord", "")
                                or pratya.get("name", "")
                            )
                            expanded_rows.append({
                                "mahadasha": maha_planet,
                                "antardasha": antar_planet,
                                "pratyantardasha": pratya_planet,
                                "start_date": p_start_str,
                                "end_date": p_end_str
                            })
                    else:
                        try:
                            a_start_str = str(antar.get("start") or antar.get("start_date", ""))[:10]
                            a_end_str = str(antar.get("end") or antar.get("end_date", ""))[:10]
                            if not a_start_str or not a_end_str:
                                continue
                            a_start = date.fromisoformat(a_start_str)
                            a_end = date.fromisoformat(a_end_str)
                        except Exception:
                            continue

                        if a_end < expanded_history_cutoff or a_start > today:
                            continue

                        expanded_rows.append({
                            "mahadasha": maha_planet,
                            "antardasha": antar_planet,
                            "pratyantardasha": "—",
                            "start_date": a_start_str,
                            "end_date": a_end_str
                        })

            result["expanded_history"] = sorted(
                expanded_rows,
                key=lambda item: item["start_date"],
                reverse=True
            )

    print(f"   Mahadasha:  "
          f"{result['mahadasha']['planet'] if result['mahadasha'] else 'Not found'}")
    print(f"   Antardasha: "
          f"{result['antardasha']['planet'] if result['antardasha'] else 'Not found'}")
    print(f"   Pratyantar: "
          f"{result['pratyantardasha']['planet'] if result['pratyantardasha'] else 'Not found'}")
    print(f"   Dasha alerts: "
          f"{result['match_count']}")

    return result


def get_aspected_houses(planet_name: str,
                        planet_house: int) -> list:
    """
    Returns all house numbers aspected by a planet.
    """
    aspects = []

    # Standard 7th aspect
    h7 = ((planet_house - 1 + 6) % 12) + 1
    aspects.append(h7)

    if planet_name == "Saturn":
        h3  = ((planet_house - 1 + 2) % 12) + 1
        h10 = ((planet_house - 1 + 9) % 12) + 1
        aspects.extend([h3, h10])

    elif planet_name in ["Jupiter", "Rahu"]:
        h5 = ((planet_house - 1 + 4) % 12) + 1
        h9 = ((planet_house - 1 + 8) % 12) + 1
        aspects.extend([h5, h9])

    elif planet_name == "Mars":
        h4 = ((planet_house - 1 + 3) % 12) + 1
        h8 = ((planet_house - 1 + 7) % 12) + 1
        aspects.extend([h4, h8])

    return list(set(aspects))


def get_aspect_type(planet_name: str,
                    from_house:  int,
                    to_house:    int) -> str:
    """Returns human readable aspect description."""
    diff = ((to_house - from_house) % 12) + 1

    if diff == 7:
        return "7th aspect (standard)"
    elif planet_name == "Saturn" and diff == 3:
        return "3rd aspect (Saturn special)"
    elif planet_name == "Saturn" and diff == 10:
        return "10th aspect (Saturn special)"
    elif planet_name in ["Jupiter","Rahu"] \
         and diff == 5:
        return "5th aspect (Jupiter/Rahu special)"
    elif planet_name in ["Jupiter","Rahu"] \
         and diff == 9:
        return "9th aspect (Jupiter/Rahu special)"
    elif planet_name == "Mars" and diff == 4:
        return "4th aspect (Mars special)"
    elif planet_name == "Mars" and diff == 8:
        return "8th aspect (Mars special)"
    return f"{diff}th aspect"


def analyse_drishti(planets: list,
                    rule1_result: dict) -> dict:
    """
    Finds all planets that aspect either:
    A. The 6th house directly
    B. Any Rogkaraka planet's house
    """

    sixth_house = rule1_result.get(
        "sixth_house_number", 6)

    # Rogkaraka houses and names
    rk_houses = set()
    rk_names  = set()

    for p in rule1_result.get("rogkaraka_1",[]):
        rk_houses.add(p.get("house", 0))
        rk_names.add(p["name"])

    rk2 = rule1_result.get("rogkaraka_2")
    if rk2:
        rk_houses.add(rk2.get("house", 0))
        rk_names.add(rk2["name"])

    # Step 1 — Build a house occupants map first:
    house_occupants = {}
    for planet in planets:
        h = planet.get("house", 0)
        if h not in house_occupants:
            house_occupants[h] = []
        house_occupants[h].append({
            "name":      planet.get("name",""),
            "sign":      planet.get("sign",{}).get("name",""),
            "house":     h,
            "nakshatra": planet.get("nakshatra",{}).get("name","—"),
            "nakshatra_pada": planet.get("nakshatra",{}).get("pada","")
        })

    aspects_sixth = []
    aspects_rk    = []

    for planet in planets:
        pname  = planet.get("name","")
        phouse = planet.get("house", 0)

        # Skip Rogkaraka planets themselves
        if pname in rk_names:
            continue

        # Skip these from aspect calculation
        if pname in ["Ascendant", "Ketu"]:
            continue

        aspected = get_aspected_houses(pname, phouse)

        # Aspects 6th house directly
        if sixth_house in aspected:
            aspects_sixth.append({
                "planet":      pname,
                "from_house":  phouse,
                "aspects_house": sixth_house,
                "aspect_type": get_aspect_type(pname, phouse, sixth_house),
                "occupants":   house_occupants.get(sixth_house, [])
            })

        # Aspects a Rogkaraka house
        for rk_house in rk_houses:
            if rk_house in aspected and rk_house != sixth_house:
                aspects_rk.append({
                    "planet":      pname,
                    "from_house":  phouse,
                    "aspects_house": rk_house,
                    "aspect_type": get_aspect_type(pname, phouse, rk_house),
                    "occupants":   house_occupants.get(rk_house, [])
                })

    all_drishti = list(set([p["planet"] for p in aspects_sixth] + [p["planet"] for p in aspects_rk]))

    drishti_occupants = []
    for entry in aspects_sixth + aspects_rk:
        for occ in entry.get("occupants", []):
            occ_name = occ["name"]
            
            # Skip if already a Rogkaraka
            if occ_name in rk_names:
                continue
            
            # Skip Ascendant
            if occ_name == "Ascendant":
                continue
            
            # Add if not already in list
            already = any(d["name"] == occ_name for d in drishti_occupants)
            if not already:
                drishti_occupants.append(occ)

    print(f"   Drishti — aspects 6th: {[p['planet'] for p in aspects_sixth]}")
    print(f"   Drishti — aspects RK:  {[p['planet'] for p in aspects_rk]}")
    print(f"   Drishti occupants: {[p['name'] for p in drishti_occupants]}")

    return {
        "aspects_sixth_house":  aspects_sixth,
        "aspects_rogkaraka":    aspects_rk,
        "all_drishti_planets":  all_drishti,
        "drishti_occupants":    drishti_occupants,
        "house_occupants":      house_occupants
    }


def build_organ_disease_map(rule1_result, complete_analysis, chart_data=None):
    rk1_list   = rule1_result.get("rogkaraka_1", [])
    rk2        = rule1_result.get("rogkaraka_2") or {}
    rk3        = rule1_result.get("rogkaraka_3") or {}
    sixth_sign = rule1_result.get("sixth_house_sign", "")

    result = []
    already_added = defaultdict(set) # sign -> set of diseases (Change 4)

    # Change 3: Get confirmed categories from the tiered diagnostic output
    confirmed_categories = set()
    for item in complete_analysis.get("disease_risks", []):
        grps = get_semantic_groups(item.get("condition", ""))
        for grp in grps:
            confirmed_categories.add(grp)

    def process_and_add_to_map(p_obj, sign, level_label):
        pname = p_obj.get("name", "")
        # Get organs from Rashi
        rashi_organs_raw = RASHI_ORGANS.get(sign, "")
        organs_list = [o.strip() for o in rashi_organs_raw.split(",") if o.strip()] if rashi_organs_raw else []

        # Change 3: Narrow the organ heading
        filtered_organs = []
        for organ in organs_list:
            grps = get_semantic_groups(organ.strip())
            if any(grp in confirmed_categories for grp in grps):
                filtered_organs.append(organ.strip())
        
        if not filtered_organs:
            filtered_organs = organs_list[:3]
        organ_heading = " / ".join(filtered_organs)

        # Change 2: Remove planet diseases, keep ONLY nakshatra diseases
        narrow_diseases = []
        nk = p_obj.get("nakshatra", "")
        pada = p_obj.get("nakshatra_pada", 0)
        nk_diseases = get_nakshatra_diseases(nk, pada)
        
        for d in nk_diseases:
            d_clean = d.strip()
            if not d_clean: continue
            
            # Change 4: P3 skip diseases that already exist in P1/P2 for this sign
            if level_label == "P3" and d_clean.lower() in already_added[sign]:
                continue
                
            narrow_diseases.append({
                "disease": d_clean,
                "source":  f"{level_label} Nakshatra ({nk})",
                "level":   level_label
            })
            
            # Track P1/P2 diseases for this sign
            if level_label in ["P1", "P2"]:
                already_added[sign].add(d_clean.lower())

        # If no unique diseases are found for this level/sign combo (common in P3), skip the entry
        if not narrow_diseases:
            return

        # Elaborate Sub-Categorization: Group diseases by their semantic categories
        sections_map = defaultdict(list)
        for nd in narrow_diseases:
            grps = get_semantic_groups(nd["disease"])
            if not grps:
                grps = ["Other Related Risks"]
            for grp in grps:
                sections_map[grp].append(nd)
        
        sub_sections = [{"category": k, "diseases": v} for k, v in sections_map.items()]

        # Change 1: Merge same-sign planets into one entry
        existing = next((e for e in result if e["level"] == level_label and e["rashi_sign"] == sign), None)
        if existing:
            if pname and pname not in existing["planet"]:
                existing["planet"] += f" / {pname}"
            
            # Merge sub-sections
            for new_sec in sub_sections:
                ext_sec = next((s for s in existing["sub_sections"] if s["category"] == new_sec["category"]), None)
                if ext_sec:
                    for nd in new_sec["diseases"]:
                        if not any(e["disease"].lower() == nd["disease"].lower() for e in ext_sec["diseases"]):
                            ext_sec["diseases"].append(nd)
                else:
                    existing["sub_sections"].append(new_sec)
        else:
            result.append({
                "level":           level_label,
                "planet":          pname,
                "rashi_sign":      sign,
                "organ_heading":   organ_heading,
                "sub_sections":    sub_sections,
                # Keep full organs for downstream mapping (Change 1/3)
                "rashi_organs":    organs_list,
                "nakshatra_name":  nk
            })

    # ── COLLECTION ───────────────────────────────────────────
    for p in rk1_list:
        process_and_add_to_map(p, sixth_sign, "P1")

    if rk3:
        rk3_sign = rk3.get("sign", "")
        if isinstance(rk3_sign, dict): rk3_sign = rk3_sign.get("name", "")
        process_and_add_to_map(rk3, rk3_sign, "P2")

    if rk1_list and rk3:
        drishti_data = complete_analysis.get("drishti", {})
        aspects_6 = drishti_data.get("aspects_sixth_house", [])
        all_planets = chart_data.get("planets", []) if chart_data else []
        for asp in aspects_6:
            pname = asp.get("planet", "")
            p_data = next((p for p in all_planets if p.get("name") == pname), None)
            if p_data:
                p_sign = p_data.get("sign", {}).get("name", "") if isinstance(p_data.get("sign"), dict) else p_data.get("sign", "")
                nk_info = p_data.get("nakshatra", {})
                if isinstance(nk_info, dict):
                    nk_name, nk_pada = nk_info.get("name", ""), nk_info.get("pada", 0)
                else:
                    nk_name, nk_pada = nk_info, p_data.get("nakshatra_pada", 0)
                process_and_add_to_map({"name": pname, "nakshatra": nk_name, "nakshatra_pada": nk_pada}, p_sign, "P3")
    elif rk2:
        rk2_sign = rk2.get("sign", "")
        if isinstance(rk2_sign, dict): rk2_sign = rk2_sign.get("name", "")
        process_and_add_to_map(rk2, rk2_sign, "P3")

    return result




def find_dominant_theme(organ_disease_map):
    """
    Scans rashi_organs across all priority
    levels P1 P2 P3 and finds which body
    region appears most. Returns dominant
    theme dict for use in chat_module.py
    """

    REGION_KEYWORDS = {
        "Lower Body": [
            "hip", "thigh", "limb", "knee",
            "leg", "ankle", "foot", "feet",
            "bone", "flesh"
        ],
        "Upper Body": [
            "neck", "throat", "collar",
            "shoulder", "chest", "breast",
            "heart", "lung", "back", "spine"
        ],
        "Head and Mind": [
            "head", "brain", "mind", "eye",
            "ear", "nose", "face", "tongue",
            "teeth"
        ],
        "Blood and Organs": [
            "blood", "liver", "kidney",
            "genital", "bladder", "urinary",
            "bowel", "stomach", "abdomen"
        ],
        "Nervous System": [
            "nerve", "memory", "emotion",
            "psychic", "breathing", "skin"
        ]
    }

    region_scores   = {r: 0  for r in REGION_KEYWORDS}
    region_evidence = {r: [] for r in REGION_KEYWORDS}
    planet_signatures = []

    for item in organ_disease_map:
        level   = item.get("level",  "")
        planet  = item.get("planet", "")
        organs  = item.get("rashi_organs", [])
        weight  = 3 if level == "P1" else (
                  2 if level == "P2" else 1)

        for organ in organs:
            organ_lower = organ.lower()
            for region, keywords in REGION_KEYWORDS.items():
                for kw in keywords:
                    if kw in organ_lower:
                        region_scores[region] += weight
                        ev = f"{organ} ({level} — {planet})"
                        if ev not in region_evidence[region]:
                            region_evidence[region].append(ev)

        if planet == "Saturn":
            planet_signatures.append("CHRONIC")
        if planet == "Mars":
            planet_signatures.append("ACUTE_EPISODES")
        if planet == "Rahu":
            planet_signatures.append("UNPREDICTABLE")

    dominant_region = max(
        region_scores,
        key=lambda r: region_scores[r]
    )

    levels_present = []
    for item in organ_disease_map:
        organs = item.get("rashi_organs", [])
        level  = item.get("level", "")
        for organ in organs:
            organ_lower = organ.lower()
            for kw in REGION_KEYWORDS[dominant_region]:
                if kw in organ_lower and level not in levels_present:
                    levels_present.append(level)

    return {
        "dominant_region":    dominant_region,
        "evidence":           region_evidence[dominant_region],
        "score":              region_scores[dominant_region],
        "multi_level":        len(levels_present) > 1,
        "levels_present":     levels_present,
        "is_chronic":         "CHRONIC" in planet_signatures,
        "has_acute_episodes": "ACUTE_EPISODES" in planet_signatures,
        "planet_signatures":  planet_signatures,
        "all_region_scores":  region_scores
    }


def complete_risk_analysis(
    chart_data:   dict,
    rule1_result: dict,
    dasha_result: dict,
    patient_gender: str = "Unknown"
) -> dict:
    """
    Master function combining all 3 steps:
    Step A: Rogkaraka (base risk)
    Step B: Drishti (aspecting planets)
    Step C: Dasha matching (active period)
    """

    planets    = chart_data.get("planets", [])
    sixth_sign = rule1_result.get("sixth_house_sign","")

    # ══ STEP A ═════════════════════════════════
    rk1_list = rule1_result.get("rogkaraka_1",[])
    rk2      = rule1_result.get("rogkaraka_2")
    rk3      = rule1_result.get("rogkaraka_3")

    rk_names = set()
    for p in rk1_list:
        rk_names.add(p["name"])
    if rk2:
        rk_names.add(rk2["name"])
    if rk3:
        rk_names.add(rk3["name"])

    # Add Ascendant Lord to the mix of disease planets
    asc_lord = rule1_result.get("ascendant_lord")
    if asc_lord:
        rk_names.add(asc_lord)

    # Weighted Scoring System:
    # RK1 (Priority 1) = 6 points
    # RK3 (Priority 2) = 3 points
    # RK2 (Priority 3) = 1 point
    # Ascendant (Baseline) = 1 point
    
    step_a_score  = len(rk1_list) * 6
    if rk3:
        step_a_score += 3
    if rk2:
        step_a_score += 1
    step_a_score += 1 # Base Ascendant point

    # ══ STEP B ═════════════════════════════════
    drishti      = analyse_drishti(planets, rule1_result)
    drishti_names = set(drishti["all_drishti_planets"])

    step_b_score  = 0
    step_b_score += len(drishti["aspects_sixth_house"]) * 3
    step_b_score += len(drishti["aspects_rogkaraka"]) * 2

    step_b_score += len(drishti.get("drishti_occupants", [])) * 1

    # ══ STEP C ═════════════════════════════════
    # Add occupant planets from Drishti
    drishti_occupants = drishti.get("drishti_occupants", [])
    occupant_names = set(p["name"] for p in drishti_occupants)

    all_disease_planets = (
        rk_names |
        drishti_names |
        occupant_names
    )

    active_dasha  = []
    step_c_score  = 0

    dasha_weights = {
        "mahadasha":       3,
        "antardasha":      4,
        "pratyantardasha": 5
    }

    for level, weight in dasha_weights.items():
        level_data = dasha_result.get(level)
        if not level_data:
            continue
        planet = level_data.get("planet","")
        if planet in all_disease_planets:
            active_dasha.append({
                "planet":     planet,
                "level":      level,
                "weight":     weight,
                "end_date":   level_data.get("end_date",""),
                "is_rk":      planet in rk_names,
                "is_drishti": planet in drishti_names
            })
            step_c_score += weight

    # ══ TOTAL ══════════════════════════════════
    total = (step_a_score + step_b_score + step_c_score)

    if total >= 12:
        overall = "CRITICAL"
        color   = "#ef4444"
    elif total >= 8:
        overall = "HIGH"
        color   = "#ef4444"
    elif total >= 5:
        overall = "MODERATE"
        color   = "#f59e0b"
    else:
        overall = "LOW"
        color   = "#22c55e"

    is_peak = any(p["is_rk"] for p in active_dasha)

    # ══ DISEASE RISKS ══════════════════════════
    risks = []

    # From 6th house organs
    organs = RASHI_ORGANS.get(sixth_sign,"")
    if organs:
        for i, organ in enumerate([o.strip() for o in organs.split(",")][:5]):
            # --- GENDER FILTER FOR ORGANS ---
            organ_lower = organ.lower().strip()
            if patient_gender in ["M", "F", "Male", "Female"]:
                safe_gender = "M" if patient_gender.startswith("M") else "F"
                exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in organ_lower), None)
                if exclusive_gender and exclusive_gender != safe_gender:
                    continue
            # -------------------------------
            base   = max(1, step_a_score - i)
            dri    = len(drishti["aspects_sixth_house"]) * 2
            total_r = base + dri + step_c_score
            risks.append({
                "condition":     organ,
                "category":      "Organ System",
                "source":        f"6th house ({sixth_sign})",
                "base_score":    base,
                "drishti_bonus": dri,
                "dasha_bonus":   step_c_score,
                "total_score":   total_r
            })

    # From Ascendant organs
    asc_sign = rule1_result.get("ascendant_sign")
    if asc_sign:
        asc_organs = RASHI_ORGANS.get(asc_sign, "")
        if asc_organs:
            for i, organ in enumerate([o.strip() for o in asc_organs.split(",")][:3]):
                organ_lower = organ.lower().strip()
                if patient_gender in ["M", "F", "Male", "Female"]:
                    safe_gender = "M" if patient_gender.startswith("M") else "F"
                    exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in organ_lower), None)
                    if exclusive_gender and exclusive_gender != safe_gender:
                        continue
                # Base score for Ascendant is 1
                total_r = 1 + step_c_score
                risks.append({
                    "condition":     organ,
                    "category":      "Organ System",
                    "source":        f"Ascendant ({asc_sign})",
                    "base_score":    1,
                    "drishti_bonus": 0,
                    "dasha_bonus":   step_c_score,
                    "total_score":   total_r
                })

    # From nakshatra diseases (Weighted)
    # Priority 1 (RK1) gets 5 base points
    # Priority 2 (RK3) gets 3 base points
    # Priority 3 (RK2) gets 1 base point
    
    all_rk_weighted = []
    for p in rk1_list:
        all_rk_weighted.append((p, 5, "Priority 1"))
    if rk3:
        all_rk_weighted.append((rk3, 3, "Priority 2"))
    if rk2:
        all_rk_weighted.append((rk2, 1, "Priority 3"))
    
    # Add Ascendant Nakshatra as Priority 4 (Baseline)
    asc_nk = rule1_result.get("ascendant_nakshatra")
    if asc_nk and asc_nk != "—":
        all_rk_weighted.append(({
            "name": "Ascendant",
            "nakshatra": asc_nk,
            "nakshatra_pada": rule1_result.get("ascendant_nakshatra_pada")
        }, 1, "Priority 4 (Ascendant)"))

    for p, base_w, p_label in all_rk_weighted:
        nk = p.get("nakshatra","")
        if not nk or nk == "—":
            continue
        for disease in get_nakshatra_diseases(nk, p.get("nakshatra_pada"))[:3]:
            # --- GENDER FILTER FOR NAKSHATRA DISEASES ---
            disease_lower = disease.lower().strip()
            if patient_gender in ["M", "F", "Male", "Female"]:
                safe_gender = "M" if patient_gender.startswith("M") else "F"
                exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in disease_lower), None)
                if exclusive_gender and exclusive_gender != safe_gender:
                    continue
            # --------------------------------------------
            # Score = base_w + dasha_bonus
            total_r = base_w + step_c_score
            risks.append({
                "condition":     disease,
                "category":      "Nakshatra",
                "source":        f"{nk} nakshatra ({p_label})",
                "base_score":    base_w,
                "drishti_bonus": 0,
                "dasha_bonus":   step_c_score,
                "total_score":   total_r
            })

    # From Drishti occupant planets
    for occ in drishti_occupants:
        occ_name = occ["name"]
        
        # Body systems
        system = PLANET_DISEASES.get(occ_name,"")
        if system:
            cond = system.split(",")[0].strip()
            # --- GENDER FILTER FOR DRISHTI OCCUPANT SYSTEMS ---
            cond_lower = cond.lower().strip()
            if patient_gender in ["M", "F", "Male", "Female"]:
                safe_gender = "M" if patient_gender.startswith("M") else "F"
                exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in cond_lower), None)
                if exclusive_gender and exclusive_gender != safe_gender:
                    pass # We skip adding this risk
                else:
                    occ_score = 1 + step_c_score
                    risks.append({
                        "condition":     cond,
                        "category":      "Drishti Occupant",
                        "source":        f"{occ_name} in aspected house (via Drishti)",
                        "base_score":    1,
                        "drishti_bonus": 2,
                        "dasha_bonus":   step_c_score,
                        "total_score":   occ_score
                    })
            else:
                occ_score = 1 + step_c_score
                risks.append({
                    "condition":     cond,
                    "category":      "Drishti Occupant",
                    "source":        f"{occ_name} in aspected house (via Drishti)",
                    "base_score":    1,
                    "drishti_bonus": 2,
                    "dasha_bonus":   step_c_score,
                    "total_score":   occ_score
                })
        
        # Nakshatra diseases
        nk = occ.get("nakshatra","")
        if nk and nk != "—":
            pada = occ.get("nakshatra_pada", 1)
            nk_diseases = get_nakshatra_diseases(nk, pada)
            for disease in nk_diseases[:2]:
                # --- GENDER FILTER FOR DRISHTI NAKSHATRA DISEASES ---
                disease_lower = disease.lower().strip()
                if patient_gender in ["M", "F", "Male", "Female"]:
                    safe_gender = "M" if patient_gender.startswith("M") else "F"
                    exclusive_gender = next((g for kw, g in GENDER_EXCLUSIVE_TERMS.items() if kw in disease_lower), None)
                    if exclusive_gender and exclusive_gender != safe_gender:
                        continue
                # ---------------------------------------------------
                occ_score = 1 + step_c_score
                risks.append({
                    "condition":     disease,
                    "category":      "Drishti Nakshatra",
                    "source":        f"{occ_name} nakshatra {nk} (via Drishti)",
                    "base_score":    1,
                    "drishti_bonus": 1,
                    "dasha_bonus":   step_c_score,
                    "total_score":   occ_score
                })

    # Dosha risks
    for pname in list(rk_names)[:2]:
        dosha = PLANET_DOSHA.get(pname,"")
        if dosha:
            total_r = 2 + step_c_score
            risks.append({
                "condition":     f"{dosha} Dosha Imbalance",
                "category":      "Ayurvedic",
                "source":        f"{pname} planet",
                "base_score":    2,
                "drishti_bonus": 0,
                "dasha_bonus":   step_c_score,
                "total_score":   total_r
            })

    # Assign risk levels
    for r in risks:
        s = r["total_score"]
        if s >= 12:
            r["risk_level"]  = "CRITICAL"
            r["risk_color"]  = "#ef4444"
            r["risk_border"] = "rgba(239,68,68,0.3)"
            r["risk_bg"]     = "rgba(239,68,68,0.08)"
        elif s >= 8:
            r["risk_level"]  = "HIGH"
            r["risk_color"]  = "#ef4444"
            r["risk_border"] = "rgba(239,68,68,0.25)"
            r["risk_bg"]     = "rgba(239,68,68,0.06)"
        elif s >= 4:
            r["risk_level"]  = "MODERATE"
            r["risk_color"]  = "#f59e0b"
            r["risk_border"] = "rgba(245,158,11,0.25)"
            r["risk_bg"]     = "rgba(245,158,11,0.06)"
        else:
            r["risk_level"]  = "LOW"
            r["risk_color"]  = "#22c55e"
            r["risk_border"] = "rgba(34,197,94,0.25)"
            r["risk_bg"]     = "rgba(34,197,94,0.06)"

    # Sort and deduplicate
    risks.sort(key=lambda x: -x["total_score"])
    seen  = []
    final = []
    for r in risks:
        k = r["condition"].lower()
        if k not in seen:
            seen.append(k)
            final.append(r)

    print(f"\n   == COMPLETE RISK ANALYSIS ==")
    print(f"   Step A (Rogkaraka): {step_a_score}")
    print(f"   Step B (Drishti):   {step_b_score}")
    print(f"   Step C (Dasha):     {step_c_score}")
    print(f"   Total:              {total}")
    print(f"   Overall Risk:       {overall}")
    print(f"   Peak Risk:          {is_peak}")
    print(f"   Active Dasha:       {[p['planet'] for p in active_dasha]}")

    complete_analysis = {
        "step_a_score":          step_a_score,
        "step_b_score":          step_b_score,
        "step_c_score":          step_c_score,
        "total_score":           total,
        "overall_risk":          overall,
        "risk_color":            color,
        "is_peak_risk":          is_peak,
        "rogkaraka_planets":     list(rk_names),
        "drishti":               drishti,
        "active_dasha_planets":  active_dasha,
        "disease_risks":         final,
        "high_count":    sum(1 for r in final if r["risk_level"] in ["HIGH","CRITICAL"]),
        "moderate_count":sum(1 for r in final if r["risk_level"] == "MODERATE"),
        "low_count":     sum(1 for r in final if r["risk_level"] == "LOW")
    }

    complete_analysis["organ_disease_map"] = build_organ_disease_map(
        rule1_result, complete_analysis, chart_data
    )

    complete_analysis["dominant_theme"] = find_dominant_theme(
        complete_analysis["organ_disease_map"]
    )

    complete_analysis["most_probable"] = perform_tiered_diagnostic_analysis(
        rule1_result, complete_analysis, chart_data, patient_gender=patient_gender
    )

    complete_analysis["top_diseases"] = get_top_diseases(
        rule1_result, chart_data, complete_analysis
    )

    return complete_analysis

def apply_disease_filter(rule1_result, dasha_result, gender="Unknown"):
    from knowledge_base import (
        PLANET_DISEASES,
        PLANET_FRIENDS,
        filter_terms_by_gender,
    )
    # SIGN_LORDS is already available in the global scope of rules.py
    
    
    # ── Step 1: Score all RK planets ──
    scores = {}
    rk1_list = rule1_result.get("rogkaraka_1", [])
    rk2      = rule1_result.get("rogkaraka_2") or {}
    rk3      = rule1_result.get("rogkaraka_3") or {}

    for p in rk1_list:
        name = p.get("name", "")
        if name:
            scores[name] = scores.get(name, 0) + 3

    if rk2:
        name = rk2.get("name", "")
        if name:
            scores[name] = scores.get(name, 0) + 1

    if rk3:
        name = rk3.get("name", "")
        if name:
            scores[name] = scores.get(name, 0) + 2

    if not scores:
        return {
            "primary_rk":       None,
            "combined_planets": [],
            "secondary_active": [],
            "filter_summary":   "No RK planets found"
        }

    primary_name = max(scores, key=lambda x: scores[x])

    # ── Step 2: Dasha activation check ──
    maha   = dasha_result.get("mahadasha", {})
    antar  = dasha_result.get("antardasha", {})
    pratya = dasha_result.get("pratyantardasha", {})

    maha_planet   = maha.get("planet", "")
    antar_planet  = antar.get("planet", "")
    pratya_planet = pratya.get("planet", "")

    active_in_maha   = (primary_name == maha_planet)
    active_in_antar  = (primary_name == antar_planet)
    active_in_pratya = (primary_name == pratya_planet)

    active_count = sum([
        active_in_maha,
        active_in_antar,
        active_in_pratya
    ])

    if active_count >= 3:
        severity = "CRITICAL"
    elif active_count == 2:
        severity = "HIGH"
    elif active_count == 1:
        severity = "MODERATE"
    else:
        severity = "LOW"

    primary_diseases = PLANET_DISEASES.get(primary_name, [])
    primary_diseases = filter_terms_by_gender(primary_diseases, gender)

    # ── Step 3: Friend/Owner check ──
    combined_planets = []
    all_rk_names = (
        [p.get("name") for p in rk1_list]
        + ([rk2.get("name")] if rk2 else [])
        + ([rk3.get("name")] if rk3 else [])
    )
    all_rk_names = [n for n in all_rk_names if n]

    primary_rk_obj = None
    for p in rk1_list:
        if p.get("name") == primary_name:
            primary_rk_obj = p
            break
    if not primary_rk_obj and rk2.get("name") == primary_name:
        primary_rk_obj = rk2
    if not primary_rk_obj and rk3.get("name") == primary_name:
        primary_rk_obj = rk3
        
    if primary_rk_obj:
        nakshatra = primary_rk_obj.get("nakshatra", "")
        pada = primary_rk_obj.get("nakshatra_pada", 1)
        nak_diseases = filter_terms_by_gender(get_nakshatra_diseases(nakshatra, pada), gender)
        for nd in nak_diseases:
             if nd not in primary_diseases:
                  primary_diseases.append(nd)

    primary_house_sign = ""
    if primary_rk_obj:
        primary_house_sign = primary_rk_obj.get("sign", {}).get("name", "") if isinstance(primary_rk_obj.get("sign"), dict) else primary_rk_obj.get("sign", "")

    sign_lord = SIGN_LORDS.get(primary_house_sign, "")

    active_dasha_planets = list(set(filter(None, [
        maha_planet,
        antar_planet,
        pratya_planet
    ])))

    friends_equation = {
        "Sun": ["Mars", "Jupiter", "Moon"],
        "Saturn": ["Mercury", "Venus"]
    }
    friends_of_primary = friends_equation.get(primary_name, PLANET_FRIENDS.get(primary_name, []))
    friends_of_primary = [p for p in friends_of_primary if p not in ["Rahu", "Ketu"]]
    planet_order = [
        "Sun", "Moon", "Mars", "Mercury", "Jupiter", "Venus", "Saturn", "Rahu", "Ketu"
    ]
    all_known_planets = list(planet_order)
    enemies_of_primary = [p for p in all_known_planets if p != primary_name and p not in friends_of_primary]
    if primary_name not in ["Rahu", "Ketu"]:
        for p in ["Rahu", "Ketu"]:
            if p in all_known_planets and p not in enemies_of_primary:
                enemies_of_primary.append(p)

    rk_chain_relationships = []
    for rk_name in all_rk_names:
        rk_friends = friends_equation.get(rk_name, PLANET_FRIENDS.get(rk_name, []))
        rk_friends = [p for p in rk_friends if p not in ["Rahu", "Ketu"]]
        rk_enemies = [p for p in all_known_planets if p != rk_name and p not in rk_friends]
        if rk_name not in ["Rahu", "Ketu"]:
            for p in ["Rahu", "Ketu"]:
                if p in all_known_planets and p not in rk_enemies:
                    rk_enemies.append(p)
        rk_chain_relationships.append({
            "planet": rk_name,
            "friends": rk_friends,
            "enemies": rk_enemies,
        })

    for dp in active_dasha_planets:
        if dp == primary_name:
            continue
        is_friend = dp in friends_of_primary
        is_owner  = dp == sign_lord
        is_rk     = dp in all_rk_names

        if (is_friend or is_owner) and is_rk:
            dp_diseases = PLANET_DISEASES.get(dp, [])
            dp_diseases = filter_terms_by_gender(dp_diseases, gender)

            reason = []
            if is_friend:
                reason.append("friend_of_primary")
            if is_owner:
                reason.append("house_owner")

            combined_planets.append({
                "planet":    dp,
                "diseases":  dp_diseases,
                "reason":    "_and_".join(reason),
                "severity":  severity
            })

            # Upgrade PRIMARY severity
            if severity == "MODERATE":
                severity = "HIGH"
            elif severity == "HIGH":
                severity = "CRITICAL"

    # ── Step 4: Secondary active RK ──
    severity_order = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

    def downgrade(s):
        idx = severity_order.index(s)
        return severity_order[max(0, idx - 1)]

    secondary_active = []
    for rk_name in all_rk_names:
        if rk_name == primary_name:
            continue
        if rk_name in active_dasha_planets:
            sec_diseases = PLANET_DISEASES.get(rk_name, [])
            sec_diseases = filter_terms_by_gender(sec_diseases, gender)
            secondary_active.append({
                "planet":   rk_name,
                "diseases": sec_diseases,
                "severity": downgrade(severity)
            })

    # ── Step 5: Build summary ──
    active_levels = []
    if active_in_maha:
        active_levels.append("Mahadasha")
    if active_in_antar:
        active_levels.append("Antardasha")
    if active_in_pratya:
        active_levels.append("Pratyantardasha")

    if active_levels:
        levels_text = f"{primary_name} active in {', '.join(active_levels)}."
    else:
        levels_text = f"{primary_name} not in current Dasha — lifetime tendency only."

    combined_text = ""
    if combined_planets:
        names = [c["planet"] for c in combined_planets]
        combined_text = f" {', '.join(names)} combining as friend/owner."

    active_friends_background = [p for p in active_dasha_planets if p in friends_of_primary and p != primary_name]
    active_enemies_background = [p for p in active_dasha_planets if p in enemies_of_primary and p != primary_name]
    active_owner_background = [sign_lord] if sign_lord and sign_lord in active_dasha_planets and sign_lord != primary_name else []
    active_rk_overlap = [p for p in active_dasha_planets if p in all_rk_names]
    active_friend_rk_overlap = [p for p in active_rk_overlap if p in friends_of_primary and p != primary_name]
    active_owner_rk_overlap = [p for p in active_rk_overlap if p == sign_lord and p != primary_name]
    active_enemy_rk_overlap = [p for p in active_rk_overlap if p in enemies_of_primary and p != primary_name]

    return {
        "primary_rk": {
            "planet":              primary_name,
            "score":               scores[primary_name],
            "diseases":            primary_diseases,
            "dasha_levels_active": active_levels,
            "severity":            severity,
            "active_in_maha":      active_in_maha,
            "active_in_antar":     active_in_antar,
            "active_in_pratya":    active_in_pratya
        },
        "combined_planets": combined_planets,
        "secondary_active": secondary_active,
        "relation_equation": {
            "primary_planet": primary_name,
            "primary_sign": primary_house_sign,
            "sign_lord": sign_lord,
            "friends": friends_of_primary,
            "enemies": enemies_of_primary,
            "active_dasha_planets": active_dasha_planets,
            "rk_chain": all_rk_names,
            "active_friends_background": active_friends_background,
            "active_enemies_background": active_enemies_background,
            "active_owner_background": active_owner_background,
            "active_rk_overlap": active_rk_overlap,
            "active_friend_rk_overlap": active_friend_rk_overlap,
            "active_owner_rk_overlap": active_owner_rk_overlap,
            "active_enemy_rk_overlap": active_enemy_rk_overlap,
            "rk_chain_relationships": rk_chain_relationships,
        },
        "filter_summary":   (levels_text + combined_text)
    }

def generate_1_year_health_forecast(dasha_data, chart_data, today_date=None):
    """
    Forecasts health risks for the next 365 days based on Dasha periods.
    Only planets in the Rogkaraka chain (RK1, RK2, RK3) trigger High/Critical Likelihood.
    """
    from datetime import date, timedelta
    
    if today_date is None:
        today = date.today()
    else:
        if isinstance(today_date, str):
            today = date.fromisoformat(today_date[:10])
        else:
            today = today_date

    end_date_limit = today + timedelta(days=365)
    
    # Get Rogkaraka Analysis
    rule1_result = apply_rule_1(chart_data)
    
    # 1. Build danger_planets from Rogkarakas ONLY
    danger_planets = set()
    
    # RK1 — planets sitting in 6th house
    for p in rule1_result.get("rogkaraka_1", []):
        pname = p.get("name", "")
        if pname:
            danger_planets.add(pname)
    
    # RK2 — 6th house lord
    rk2 = rule1_result.get("rogkaraka_2") or {}
    rk2_name = rk2.get("name", "")
    if rk2_name:
        danger_planets.add(rk2_name)
    
    # RK3 — dispositor of RK2
    rk3 = rule1_result.get("rogkaraka_3") or {}
    rk3_name = rk3.get("name", "")
    if rk3_name:
        danger_planets.add(rk3_name)

    # 8th house danger planets
    eighth_danger_planets = set()
    
    # 8th house occupants
    for p in rule1_result.get("eighth_house_occupants", []):
        pname = p.get("name", "")
        if pname:
            eighth_danger_planets.add(pname)
    
    # 8th house lord
    eighth_lord = rule1_result.get("eighth_house_lord", "")
    if eighth_lord:
        eighth_danger_planets.add(eighth_lord)
    
    # Remove overlap with primary danger_planets
    # If a planet is in BOTH sets it is already 
    # handled as High Likelihood — remove from 
    # eighth_danger_planets to avoid duplication
    eighth_danger_planets -= danger_planets
            
    planet_info = {p["name"]: p for p in chart_data.get("planets", [])}
    sixth_house = rule1_result.get("sixth_house_number", 6)
    
    # 2. Filter Dasha Data for the next 365 days
    periods = (
        dasha_data.get("dasha_periods") or
        dasha_data.get("mahadasha")     or
        dasha_data.get("dashas")        or
        dasha_data.get("vimshottari_dasha") or
        []
    )
    
    forecast = []
    
    for maha in periods:
        try:
            m_start_str = str(maha.get("start") or maha.get("start_date"))[:10]
            m_end_str = str(maha.get("end") or maha.get("end_date"))[:10]
            m_start = date.fromisoformat(m_start_str)
            m_end = date.fromisoformat(m_end_str)
        except: continue
        
        if m_end < today or m_start > end_date_limit: continue
        
        maha_planet = maha.get("planet", {}).get("name") or maha.get("lord") or maha.get("name")
        
        antars = maha.get("antardasha") or maha.get("sub_periods") or maha.get("bhukti") or []
        for antar in antars:
            try:
                a_start_str = str(antar.get("start") or antar.get("start_date"))[:10]
                a_end_str = str(antar.get("end") or antar.get("end_date"))[:10]
                a_start = date.fromisoformat(a_start_str)
                a_end = date.fromisoformat(a_end_str)
            except: continue
            
            if a_end < today or a_start > end_date_limit: continue
            
            antar_planet = antar.get("planet", {}).get("name") or antar.get("lord") or antar.get("name")
            
            pratyas = antar.get("pratyantardasha") or antar.get("sub_periods") or antar.get("antardasha") or []
            for pratya in pratyas:
                try:
                    p_start_str = str(pratya.get("start") or pratya.get("start_date"))[:10]
                    p_end_str = str(pratya.get("end") or pratya.get("end_date"))[:10]
                    p_start = date.fromisoformat(p_start_str)
                    p_end = date.fromisoformat(p_end_str)
                except: continue
                
                if p_end < today or p_start > end_date_limit: continue
                
                pratya_planet = pratya.get("planet", {}).get("name") or pratya.get("lord") or pratya.get("name")
                
                # Risk Assessment
                active_lords = [maha_planet, antar_planet, pratya_planet]
                triggering_planets = []
                seen_triggers = set()
                for p in active_lords:
                    if p in danger_planets and p not in seen_triggers:
                        triggering_planets.append(p)
                        seen_triggers.add(p)
                
                eighth_triggering = []
                for p in active_lords:
                    if p in eighth_danger_planets:
                        eighth_triggering.append(p)

                # Set Risk Level based on triggering count
                if (len(triggering_planets) >= 2 or 
                    (len(triggering_planets) >= 1 and len(eighth_triggering) >= 1)):
                    risk_level = "CRITICAL — Multiple Indicators Active"
                elif len(triggering_planets) == 1:
                    risk_level = "High Likelihood"
                elif len(eighth_triggering) >= 1:
                    risk_level = "Moderate — Chronic Tendency"
                else:
                    risk_level = "Low Likelihood (Protected)"
                
                # Potential Symptoms
                symptoms = []
                secondary_found = False
                
                # A. Handle triggering Rogkarakas
                if risk_level in ["High Likelihood", "CRITICAL — Multiple Indicators Active"]:
                    for tp in triggering_planets:
                        # Determine roles for display
                        roles = []
                        rk1_names = [p.get("name") for p in (rule1_result.get("rogkaraka_1") or []) if p]
                        if tp in rk1_names: roles.append("Occupant")
                        
                        rk2_val = rule1_result.get("rogkaraka_2")
                        if rk2_val and isinstance(rk2_val, dict) and tp == rk2_val.get("name"):
                            roles.append("Lord")
                            
                        rk3_val = rule1_result.get("rogkaraka_3")
                        if rk3_val and isinstance(rk3_val, dict) and tp == rk3_val.get("name"):
                            roles.append("Dispositor")
                        
                        roles_str = " & ".join(roles)

                        p_data = planet_info.get(tp, {})
                        
                        # Extract Nakshatra details
                        nk_info = p_data.get("nakshatra", {})
                        if isinstance(nk_info, dict):
                            nk_name = nk_info.get("name", "—")
                            nk_pada = nk_info.get("pada", 1)
                        else:
                            nk_name = nk_info
                            nk_pada = p_data.get("nakshatra_pada", 1)
                            
                        # Fetch precise diseases from Nakshatra logic
                        nk_diseases = get_nakshatra_diseases(nk_name, nk_pada)
                        if isinstance(nk_diseases, str):
                            nk_diseases = [d.strip() for d in nk_diseases.split(",") if d.strip()]
                        
                        # Format detailed string for UI
                        disease_summary = ", ".join(nk_diseases[:3]) if nk_diseases else "General planetary tendency"
                        
                        # Add Rashi context if available
                        rasi_name = p_data.get("sign", {}).get("name", "—") if isinstance(p_data.get("sign"), dict) else p_data.get("sign", "—")
                        rasi_organ = RASHI_ORGANS.get(rasi_name, "")
                        if rasi_organ:
                            # Just pick the first 2 organs to keep it concise
                            organs = [o.strip() for o in rasi_organ.split(",")[:2]]
                            disease_summary += f" (affecting {' & '.join(organs)})"
                            
                        detailed_msg = f"⚠️ High Risk Trigger: {tp} ({roles_str}) - Triggering {nk_name} tendencies: {disease_summary}"
                        symptoms.append(detailed_msg)

                # A2. Handle triggering 8th House planets
                for tp in eighth_triggering:
                    roles_str = "8th Lord" if tp == eighth_lord else "8th House Occupant"
                    
                    p_data = planet_info.get(tp, {})
                    nk_info = p_data.get("nakshatra", {})
                    if isinstance(nk_info, dict):
                        nk_name = nk_info.get("name", "—")
                        nk_pada = nk_info.get("pada", 1)
                    else:
                        nk_name = nk_info
                        nk_pada = p_data.get("nakshatra_pada", 1)
                    
                    nk_diseases = get_nakshatra_diseases(nk_name, nk_pada)
                    disease_summary = ", ".join(nk_diseases[:3]) if nk_diseases else "General chronic tendency"
                    
                    rasi_name = p_data.get("sign", {}).get("name", "—") if isinstance(p_data.get("sign"), dict) else p_data.get("sign", "—")
                    rasi_organ = RASHI_ORGANS.get(rasi_name, "")
                    if rasi_organ:
                        organs = [o.strip() for o in rasi_organ.split(",")[:2]]
                        disease_summary += f" (affecting {' & '.join(organs)})"
                    
                    detailed_msg = (
                        f"🔵 Chronic/Surgery Risk: {tp} ({roles_str}) - "
                        f"Triggering {nk_name} tendencies: {disease_summary}"
                    )
                    symptoms.append(detailed_msg)

                # B. Handle Secondary Watch level
                for p in set(active_lords):
                    if p not in danger_planets:
                        p_data = planet_info.get(p, {})
                        p_house = p_data.get("house", 0)
                        aspected = get_aspected_houses(p, p_house)
                        
                        if sixth_house in aspected:
                            secondary_found = True
                            detailed_msg = f"ℹ️ Secondary Influence: {p} (Aspecting {sixth_house}th House) — Aspecting disease house. Watch level only."
                            symptoms.append(detailed_msg)
                
                # C. Protected Period Fallback
                if not triggering_planets and not secondary_found:
                    symptoms.append("🛡️ Protected Period: No active disease indicators or secondary influences for this period.")
                
                forecast.append({
                    "start_date": p_start.isoformat(),
                    "end_date": p_end.isoformat(),
                    "mahadasha": maha_planet,
                    "antardasha": antar_planet,
                    "pratyantardasha": pratya_planet,
                    "risk_level": risk_level,
                    "triggering_planets": triggering_planets,
                    "potential_symptoms": symptoms
                })
                
    return forecast

if __name__ == "__main__":
    import json
    with open("chart_data.json") as f:
        chart_data = json.load(f)

    result = apply_rule_1(chart_data)

    print("\n" + "="*55)
    print("  RULE 1 — ROGKARAKA ANALYSIS")
    print("="*55)
    print(f"  Ascendant        : {result['ascendant_sign']} (House {result['ascendant_house']})")
    print(f"  6th from Ascendant: House {result['sixth_house_number']} ({result['sixth_house_sign']})")
    print(f"  6th House Lord   : {result['sixth_house_lord']}")
    print()

    if result["rogkaraka_1"]:
        for p in result["rogkaraka_1"]:
            print(f"  Rogkaraka-1 : {p['name']} in House {p['house']} ({p['sign']}) {p['degree']}° {p['minutes']}' — {p['nakshatra']}")
    else:
        print("  Rogkaraka-1 : No planet in 6th house from ascendant")

    rk2 = result["rogkaraka_2"]
    if rk2:
        print(f"  Rogkaraka-2 : {rk2['name']} in House {rk2['house']} ({rk2['sign']}) {rk2['degree']}° {rk2['minutes']}' — {rk2['nakshatra']}")

    if result["rogkaraka_3"]:
        rk3 = result["rogkaraka_3"]
        print(f"  Rogkaraka-3 : {rk3['name']} in House {rk3['house']} ({rk3['sign']}) {rk3['degree']}° {rk3['minutes']}' — {rk3['nakshatra']}")
    else:
        print("  Rogkaraka-3 : Not applicable — lord is in its own house")

    print("="*55)

    with open("rule1_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("  Saved → rule1_result.json")
