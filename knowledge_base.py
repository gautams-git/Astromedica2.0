# ── Gender-Specific Terms ─────────────────────
GENDER_EXCLUSIVE_TERMS = {
    "ovaries": "F",
    "ovary": "F",
    "uterus": "F",
    "uterine": "F",
    "menstruation": "F",
    "menstrual": "F",
    "menses": "F",
    "female organs": "F",
    "womb": "F",
    "leucorrhoea": "F",
    "cervix": "F",
    "cervical": "F",
    "fallopian": "F",
    "pregnancy": "F",
    "pregnant": "F",
    "lactation": "F",
    "prostate": "M",
    "testicles": "M",
    "testicle": "M",
    "testicular": "M",
    "semen": "M",
    "seminal": "M",
    "penis": "M",
    "penile": "M",
    "breast": "F",
    "breasts": "F",
    "vagina": "F",
    "vaginal": "F",
    "sperm": "M",
}

def normalize_gender(gender):
    value = str(gender or "").strip().lower()
    if value.startswith("m"):
        return "M"
    if value.startswith("f"):
        return "F"
    return None

def is_term_allowed_for_gender(term, gender):
    gender_code = normalize_gender(gender)
    if not gender_code:
        return True
    term_lower = str(term or "").lower()
    for keyword, allowed_gender in GENDER_EXCLUSIVE_TERMS.items():
        if keyword in term_lower and allowed_gender != gender_code:
            return False
    return True

def filter_terms_by_gender(terms, gender):
    if isinstance(terms, str):
        terms = [item.strip() for item in terms.split(",")]
    return [
        term for term in (terms or [])
        if str(term or "").strip() and is_term_allowed_for_gender(term, gender)
    ]

def filter_nakshatra_diseases_by_gender(nakshatra_diseases, gender):
    filtered = {}
    for nakshatra, pada_data in (nakshatra_diseases or {}).items():
        if isinstance(pada_data, dict):
            filtered[nakshatra] = {
                pada: filter_terms_by_gender(diseases, gender)
                for pada, diseases in pada_data.items()
            }
        else:
            filtered[nakshatra] = filter_terms_by_gender(pada_data, gender)
    return filtered

def filter_planet_diseases_by_gender(planet_diseases, gender):
    return {
        planet: filter_terms_by_gender(diseases, gender)
        for planet, diseases in (planet_diseases or {}).items()
    }

def filter_rashi_organs_by_gender(rashi_organs, gender):
    return {
        rashi: ", ".join(filter_terms_by_gender(organs, gender))
        for rashi, organs in (rashi_organs or {}).items()
    }

# ── Rashi → Organs/Body Parts ─────────────
RASHI_ORGANS = {
    "Aries": "Head, Brain, Mind, Face",
    "Mesha": "Head, Brain, Mind, Face",
    "Taurus": "Face, Eye, Nose, Tongue, Ears, Fingers, Navels, Teeth, Bones, Flesh, Mouth, Throat, Neck",
    "Vrishabha": "Face, Eye, Nose, Tongue, Ears, Fingers, Navels, Teeth, Bones, Flesh, Mouth, Throat, Neck",
    "Rishabha": "Face, Eye, Nose, Tongue, Ears, Fingers, Navels, Teeth, Bones, Flesh, Mouth, Throat, Neck",
    "Gemini": "Neck, Throat, Collar, Breathings, Hands, Ears, Body Growth, Shoulders, Upper Chest, Lungs, Arms",
    "Mithuna": "Neck, Throat, Collar, Breathings, Hands, Ears, Body Growth, Shoulders, Upper Chest, Lungs, Arms",
    "Mithun": "Neck, Throat, Collar, Breathings, Hands, Ears, Body Growth, Shoulders, Upper Chest, Lungs, Arms",
    "Cancer": "Breast, Stomach, Heart, Lungs, Chest, Blood",
    "Karka": "Breast, Stomach, Heart, Lungs, Chest, Blood",
    "Kataka": "Breast, Stomach, Heart, Lungs, Chest, Blood",
    "Leo": "Heart, Upper Back, Abdomen, Mind, Upper Stomach, Spine",
    "Simha": "Heart, Upper Back, Abdomen, Mind, Upper Stomach, Spine",
    "Virgo": "Solar Plexes, Bowels, Lower Abdomen, Naval, Flesh, Mental Faculties, Lower Stomach, Intestines, Waist",
    "Kanya": "Solar Plexes, Bowels, Lower Abdomen, Naval, Flesh, Mental Faculties, Lower Stomach, Intestines, Waist",
    "Libra": "Kidney, Ovaries, Groins, Semen, Female Organs, Loins",
    "Tula": "Kidney, Ovaries, Groins, Semen, Female Organs, Loins",
    "Scorpio": "Bladder, Sex Organs, Urinary Tracts, Blood, Genitals, Anus, Excretory Organs",
    "Vrischika": "Bladder, Sex Organs, Urinary Tracts, Blood, Genitals, Anus, Excretory Organs",
    "Vrishchika": "Bladder, Sex Organs, Urinary Tracts, Blood, Genitals, Anus, Excretory Organs",
    "Sagittarius": "Hips, Joints, Thighs, Liver",
    "Dhanu": "Hips ,Joints, Thighs, Liver",
    "Dhanur": "Hips ,Joints, Thighs, Liver",
    "Capricorn": "Knees, Spleens, Bones, Flesh, Joints, Heart",
    "Makara": "Knees, Spleens, Bones, Flesh, Joints, Heart",
    "Aquarius": "Ankle, Distributary Systems of Body Fluids, Shanks, Breathing, Calves",
    "Kumbha": "Ankle, Distributary Systems of Body Fluids, Shanks, Breathing, Calves",
    "Pisces": "Feet, Blood, Psychic Faculty, Lymphatic System, Toes",
    "Meena": "Feet, Blood, Psychic Faculty, Lymphatic System, Toes"
}

# ── Rashi → Degree Spans ────────────────────
RASHI_DEGREES = {
    "Aries": (0, 30), "Mesha": (0, 30),
    "Taurus": (30, 60), "Vrishabha": (30, 60), "Rishabha": (30, 60),
    "Gemini": (60, 90), "Mithuna": (60, 90), "Mithun": (60, 90),
    "Cancer": (90, 120), "Karka": (90, 120), "Kataka": (90, 120),
    "Leo": (120, 150), "Simha": (120, 150),
    "Virgo": (150, 180), "Kanya": (150, 180),
    "Libra": (180, 210), "Tula": (180, 210),
    "Scorpio": (210, 240), "Vrischika": (210, 240), "Vrishchika": (210, 240),
    "Sagittarius": (240, 270), "Dhanu": (240, 270), "Dhanur": (240, 270),
    "Capricorn": (270, 300), "Makara": (270, 300),
    "Aquarius": (300, 330), "Kumbha": (300, 330),
    "Pisces": (330, 360), "Meena": (330, 360)
}

# ── Planet → Body Systems ──────────────────
PLANET_DISEASES = {
    "Sun":     "Eyes, Chest, Mind, Bones, Heart, Head",
    "Moon":    "Blood, Water in the body, Lungs, Chest, Memory, Emotions and Desires, Heart",
    "Mars":    "Blood, Energy, Muscle, Head, Testicles, Injury, Bone Marrow",
    "Mercury": "Speech, Skin, Respiratory Canal, Lungs",
    "Jupiter": "Fat in the body, Liver, Feet, Hips, Stomach, Intestines, Anus, Excretory organs",
    "Venus":   "Face, Neck, Genital, Semen, Sex urge, Eyes, Kidney",
    "Saturn":  "Long lasting diseases, Pestilence, Leg diseases, Impotency, Nerves problems, Bones, Anus, Excretory organs", 
    "Rahu":    "Gas troubles, Accute pain, Lung trouble, Difficulty in breathing, Leprosy, Genitals, Excretory organs, Further everything that apply to Saturn (Shanivat Rahu)",
    "Ketu":    "Enlargement of spleen, Lung trouble, Fever, Stomach pain, Body pain, Unknown diseases, Genitals, Further all diseases apply to Mars (Kujavat Ketu)"
}

# ── Nakshatra → Diseases ───────────────────
NAKSHATRA_LORDS = {
    "Ashwini": "Ketu",
    "Bharani": "Venus",
    "Krithika": "Sun",
    "Krittika": "Sun",
    "Rohini": "Moon",
    "Mrigashira": "Mars",
    "Mrigashirsha": "Mars",
    "Aridra": "Rahu",
    "Ardra": "Rahu",
    "Punarvasu": "Jupiter",
    "Pushya": "Saturn",
    "Ashlesha": "Mercury",
    "Magha": "Ketu",
    "Purvaphalguni": "Venus",
    "Purva Phalguni": "Venus",
    "Uttara Phalguni": "Sun",
    "Hasta": "Moon",
    "Chitha": "Mars",
    "Chitra": "Mars",
    "Swathi": "Rahu",
    "Swati": "Rahu",
    "Vishaka": "Jupiter",
    "Vishakha": "Jupiter",
    "Anuradha": "Saturn",
    "Jyesta": "Mercury",
    "Jyeshtha": "Mercury",
    "Moola": "Ketu",
    "Mula": "Ketu",
    "Poorvashada": "Venus",
    "Purva Ashadha": "Venus",
    "Uttaraashada": "Sun",
    "Uttara Ashadha": "Sun",
    "Shravana": "Moon",
    "Dhanishta": "Mars",
    "Shathabhisha": "Rahu",
    "Shatabhisha": "Rahu",
    "Poorvabhadra": "Jupiter",
    "Purva Bhadrapada": "Jupiter",
    "Uttara Bhadrapada": "Saturn",
    "Revathi": "Mercury",
    "Revati": "Mercury",
}

NAKSHATRA_DISEASES = {
    "Ashwini": {
        "1,2,3,4": [
            "Injury in the head",
            "Congestion in the brain",
            "Cerebral anaemia",
            "Fainting",
            "Epilepsy",
            "Violence",
            "Spasms",
            "Severe headache on any one side",
            "Neuralgia, Coma",
            "Cerebral haemorrhages",
            "Paralytic stroke",
            "Smallpox"
        ]
    },
    "Bharani": {
        "1,2,3,4": [
            "Injury in the head mostly in the forehead, just around the eyes",
            "Cold",
            "Venereal distemper",
            "Syphilis affecting tube and vision",
            "Weakness"
        ]
    },
    "Krithika": {
        "1": [
            "Sharp fever",
            "Malaria",
            "Filaria",
            "Plague",
            "Small pox",
            "Wounds",
            "Brain fever",
            "Accident including fire accidents"
        ],
        "2,3,4": [
            "Pimples",
            "Cuts",
            "Reddish eyes",
            "Eye sore",
            "Throat infections",
            "Tumours in the knees",
            "Neck problems",
            "Polyps of nose"
        ]
    },
    "Rohini": {
        "1,2,3,4": [
            "Sore throat",
            "Cold & Cough",
            "Irregular menses",
            "Pain in the breasts"
        ]
    },
    "Mrigashira": {
        "1,2": [
            "Pimples",
            "Injury in the face",
            "Pain in the throat",
            "Diphtheria",
            "Weakness",
            "Venereal distemper",
            "Polypus"
        ],
        "3,4": [
            "Corrupted blood",
            "Itches",
            "Wounds & Fracture of arms",
            "Collar bone fracture",
            "Pains in the shoulders"
        ]
    },
    "Aridra": {
        "1,2,3,4": [
            "Asthma",
            "Dry coughs",
            "Diphtheria",
            "Ear infections",
            "Eosinophilia"
        ]
    },
    "Punarvasu": {
        "1,2,3": [
            "Swelling and pain in the ear",
            "Goitre due to Iodine deficiency",
            "Pulmonary apoplexy"
        ],
        "4": [
            "Dropsy",
            "Beri beri",
            "Stomach upset",
            "Tuberculosis",
            "Pneumonia",
            "Liver complaint & problems"
        ]
    },
    "Pushya": {
        "1,2,3,4": [
            "Tuberculosis",
            "Ulceratives in the respiratory system",
            "Gastric ulcer",
            "Gallstones",
            "Cancer",
            "Cough & hiccups",
            "Eczema"
        ]
    },
    "Ashlesha": {
        "1,2,3,4": [
            "Vitamin 'B' deficiency",
            "Stomach pain",
            "Pains in knee & legs",
            "Nervous problems",
            "Indigestion"
        ]
    },
    "Magha": {
        "1,2,3,4": [
            "Heart affected by sudden shock",
            "Grief of poison",
            "Pain in the back",
            "Kidney stone",
            "Spinal Meningitis"
        ]
    },
    "Purvaphalguni": {
        "1,2,3,4": [
            "Heart affected due to disappointment of love affairs",
            "Heart affects due to loss of children",
            "Swelling of ankle",
            "Blood pressure",
            "Heart Valves affected",
            "Curvature of spine"
        ]
    },
    "Uttara Phalguni": {
        "1": [
            "Pains in the back & head",
            "Spotted fever",
            "Plague",
            "Hyperaemia",
            "Blood pressure",
            "Temporary mental instability due to blood clotting",
            "Palpitation",
        ],
        "2,3,4": [
            "Tumours in the bowels",
            "Stomach disorder",
            "Sore throat",
            "Swelling in the neck"
        ]
    },
    "Hasta": {
        "1,2,3,4": [
            "Vitamin 'B' deficiency",
            "Gas formation",
            "Loose bowel",
            "Pain & disorder in the bowels",
            "Weakness of arms and shoulder",
            "Breathing problems",
            "Worm infestations",
            "Typhoid",
            "Bacillary dysentery",
            "Hysteria"
        ]
    },
    "Chitha": {
        "1,2": [
            "Ulcers",
            "Acute pains in the body",
            "Choleric humours",
            "Irritation & itching",
            "Wounds from insects"
        ],
        "3,4": [
            "Problems in kidney & bladder",
            "Excess of urine and kidney problem",
            "Brain fever"
        ]
    },
    "Swathi": {
        "1,2,3,4": [
            "Leprosy",
            "Pus formation",
            "Skin problems",
            "Urinary infection",
            "Eczema"
        ]
    },
    "Vishaka": {
        "1,2,3": [
            "Deficiency of adrenal secretion",
            "Diabetes, Insulin deficiency",
            "Congestion of brain"
        ],
        "4": [
            "Diseases of womb",
            "Fibroid tumour",
            "Prostate gland enlargement",
            "Urinary trouble",
            "Frequent micturition",
            "Abnormal bleeding during menses"
        ]
    },
    "Anuradha": {
        "1,2,3,4": [
            "Suppression of menses, poor bleeding & severe pain",
            "Piles",
            "Phlegm",
            "Sore throat",
            "Fracture of hip bone"
        ]
    },
    "Jyesta": {
        "1,2,3,4": [
            "Leucorrhoea",
            "Piles, Fistula",
            "Affliction of bowels",
            "Pains in arms and shoulder"
        ]
    },
    "Moola": {
        "1,2,3,4": [
            "Locomotor ataxia",
            "Rheumatism",
            "Hip diseases",
            "Pulmonary troubles"
        ]
    },
    "Poorvashada": {
        "1,2,3,4": [
            "Diabetes",
            "Respiratory diseases",
            "Swelling above the knees"
        ]
    },
    "Uttaraashada": {
        "1": [
            "Paralysis of limbs",
            "Pulmonary diseases",
            "Affected eyes"
        ],
        "2,3,4": [
            "Skin diseases",
            "Leprosy",
            "Digestive disorders",
            "Gas in stomach",
            "Palpitation of heart",
            "Cardiac thrombosis"
        ]
    },
    "Shravana": {
        "1,2,3,4": [
            "Filaria",
            "Eczema",
            "Skin diseases",
            "Tuberculosis",
            "Poor digestion",
            "Palpitation of heart"
        ]
    },
    "Dhanishta": {
        "1,2": [
            "Leg injury",
            "Eosinophilia",
            "Hiccups",
            "Amputation"
        ],
        "3,4": [
            "Fracture of legs",
            "Blood poisoning",
            "Heart failure",
            "Cardiac thrombosis",
            "High Blood pressure",
            "Over heated blood",
            "Palpitation"
        ]
    },
    "Shathabhisha": {
        "1,2,3,4": [
            "Rheumatism",
            "Rheumatic heart diseases",
            "Leprosy",
            "High Blood pressure",
            "Fracture",
            "Worm infestation"
        ]
    },
    "Poorvabhadra": {
        "1,2,3": [
            "Apoplexy",
            "Irregular heart beats",
            "Dropsy",
            "Milk leg",
            "Swollen ankles",
            "Dilated heart",
            "Low blood pressure"
        ],
        "4": [
            "Swelling in body",
            "Perspiring feet",
            "Enlarged Liver",
            "Abdominal tumour",
            "Intestinal disorders",
            "Jaundice"
        ]
    },
    "Uttara Bhadrapada": {
        "1,2,3,4": [
            "Rheumatic pains",
            "Indigestion",
            "Constipation",
            "Hernia",
            "Cold foot",
            "Fracture in foot",
            "Tuberculosis",
            "Dropsy"
        ]
    },
    "Revathi": {
        "1,2,3,4": [
            "Abdominal disorders",
            "Intestinal Ulcers mostly due to alcohol & drugs",
            "Nephritis"
        ]
    }
}

def get_nakshatra_diseases(nk_name, pada=None):
    """
    Returns the list of diseases for a given Nakshatra and Pada.
    Handles grouped pada keys (e.g., '1,2,3,4') and legacy list formats.
    """
    if not nk_name or nk_name == "—":
        return []
        
    # Standardize name for lookup
    data = NAKSHATRA_DISEASES.get(nk_name)
    if not data:
        # Case-insensitive fallback
        name_lower = nk_name.lower().strip()
        for k, v in NAKSHATRA_DISEASES.items():
            if k.lower() == name_lower:
                data = v
                break
    
    if not data:
        return []
        
    # Handle legacy list format
    if isinstance(data, list):
        return data
        
    # Handle nested dictionary format (with support for grouped pada keys)
    if isinstance(data, dict):
        p_str = str(pada) if pada is not None else "1"
        
        # Check for direct match (e.g., "1")
        if p_str in data:
            return data[p_str]
            
        # Check for grouped match (e.g., "1,2,3,4")
        for key, diseases in data.items():
            if p_str in [p.strip() for p in key.split(",")]:
                return diseases
        
        # Fallback to "1" or first available key
        return data.get("1") or next(iter(data.values()), [])
        
    return []

# ── Planet → Tridosha ──────────────────────
PLANET_DOSHA = {
    "Moon":    "Vatha",
    "Mercury": "Vatha",
    "Saturn":  "Vatha",
    "Venus":   "Vatha",
    "Sun":     "Pittha",
    "Mars":    "Pittha",
    "Jupiter": "Kapha",
    "Rahu":    "Vatha",
    "Ketu":    "Pittha"
}

# ── Dosha → Symptoms to watch ─────────────
DOSHA_SYMPTOMS = {
    "Vatha":  "dryness, gas, anxiety, nerve issues, insomnia, joint pain",
    "Pittha": "inflammation, fever, acidity, anger, skin rashes, burning sensation",
    "Kapha":  "heaviness, mucus, lethargy, weight gain, congestion, slow digestion"
}

# ── Planet Friends ──────────────────────────
PLANET_FRIENDS = {
    "Sun":     ["Moon", "Mars", "Jupiter"],
    "Moon":    ["Sun", "Mercury"],
    "Mars":    ["Sun", "Moon", "Jupiter"],
    "Mercury": ["Sun", "Venus"],
    "Jupiter": ["Sun", "Moon", "Mars"],
    "Venus":   ["Mercury", "Saturn"],
    "Saturn":  ["Mercury", "Venus"],
    "Rahu":    ["Venus", "Saturn", "Mercury"],
    "Ketu":    ["Mars", "Venus", "Saturn"]
}
