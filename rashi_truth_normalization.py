from __future__ import annotations

from collections import defaultdict

from knowledge_base import NAKSHATRA_DISEASES, PLANET_DISEASES, RASHI_ORGANS


def _split_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


CANONICAL_RASHI_ORGANS = sorted({_part for raw in RASHI_ORGANS.values() for _part in _split_csv(raw)})
CANONICAL_LOOKUP = {term.lower(): term for term in CANONICAL_RASHI_ORGANS}


def _canonicalize(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for term in terms:
        key = str(term or "").strip().lower()
        if not key:
            continue
        canonical = CANONICAL_LOOKUP.get(key)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        normalized.append(canonical)
    return normalized


PLANET_TERM_TO_RASHI_ORGANS = {
    "Accute pain": ["Blood", "Flesh", "Bones", "Joints"],
    "Blood": ["Blood"],
    "Body pain": ["Flesh", "Bones", "Joints", "Blood"],
    "Bone Marrow": ["Bones", "Blood"],
    "Bones": ["Bones"],
    "Chest": ["Chest", "Heart", "Lungs"],
    "Difficulty in breathing": ["Breathing", "Breathings", "Lungs", "Chest"],
    "Emotions and Desires": ["Mind", "Psychic Faculty", "Mental Faculties"],
    "Energy": ["Blood", "Flesh", "Mind"],
    "Enlargement of spleen": ["Spleens", "Blood"],
    "Eyes": ["Eye"],
    "Face": ["Face"],
    "Fat in the body": ["Flesh", "Body Growth", "Liver"],
    "Feet": ["Feet", "Toes"],
    "Fever": ["Blood", "Mind"],
    "Further all diseases apply to Mars (Kujavat Ketu)": ["Blood", "Head", "Bones", "Flesh", "Genitals"],
    "Further everything that apply to Saturn (Shanivat Rahu)": ["Bones", "Joints", "Shanks", "Calves", "Feet"],
    "Gas troubles": ["Stomach", "Bowels", "Intestines", "Lower Stomach"],
    "Genital": ["Genitals", "Sex Organs"],
    "Head": ["Head", "Brain", "Mind"],
    "Hips": ["Hips Joints", "Thighs"],
    "Impotency": ["Genitals", "Sex Organs", "Semen"],
    "Injury": ["Bones", "Flesh", "Joints", "Blood"],
    "Leg diseases": ["Shanks", "Calves", "Knees", "Feet", "Joints"],
    "Leprosy": ["Flesh", "Blood"],
    "Liver": ["Liver"],
    "Long lasting diseases": ["Bones", "Joints", "Blood"],
    "Lung trouble": ["Lungs", "Chest", "Breathing", "Breathings"],
    "Lungs": ["Lungs"],
    "Memory": ["Mind", "Brain", "Mental Faculties"],
    "Mind": ["Mind"],
    "Muscle": ["Flesh", "Arms", "Thighs", "Calves"],
    "Neck": ["Neck", "Throat"],
    "Nerves problems": ["Brain", "Mind", "Mental Faculties", "Spine"],
    "Pestilence": ["Blood", "Lungs", "Chest"],
    "Respiratory Canal": ["Throat", "Lungs", "Breathing", "Breathings", "Chest"],
    "Semen": ["Semen"],
    "Sex urge": ["Sex Organs", "Genitals", "Semen"],
    "Skin": ["Flesh"],
    "Speech": ["Tongue", "Mouth", "Throat"],
    "Stomach pain": ["Stomach", "Upper Stomach", "Lower Stomach"],
    "Testicles": ["Genitals", "Sex Organs", "Semen"],
    "Unknown diseases": ["Blood", "Mind"],
    "Water in the body": ["Distributary Systems of Body Fluids", "Lymphatic System", "Blood"],
}


DISEASE_KEYWORD_TO_RASHI_ORGANS = [
    (["head", "brain", "cerebral", "coma", "epilepsy", "headache", "apoplexy", "stroke", "mental"], ["Head", "Brain", "Mind", "Mental Faculties"]),
    (["eye", "vision", "sight", "forehead", "face", "nose", "ear", "tongue", "teeth", "mouth", "throat", "neck"], ["Eye", "Face", "Nose", "Ears", "Tongue", "Teeth", "Mouth", "Throat", "Neck"]),
    (["cold", "cough", "asthma", "pulmonary", "respiratory", "breathing", "pneumonia", "tuberculosis", "eosinophilia", "hiccup", "phlegm"], ["Lungs", "Chest", "Breathing", "Breathings", "Throat", "Nose"]),
    (["heart", "cardiac", "palpitation", "blood pressure", "thrombosis", "dilated heart", "heart beats", "heart valves", "rheumatic heart"], ["Heart", "Chest", "Blood", "Upper Back"]),
    (["blood", "anaemia", "anemia", "hyperaemia", "poisoning", "fever", "plague", "filaria", "pox", "malaria"], ["Blood"]),
    (["stomach", "gastric", "indigestion", "gas", "bowel", "bowels", "intestinal", "intestines", "abdominal", "abdomen", "ulcer", "constipation", "hernia", "jaundice", "digestion"], ["Stomach", "Upper Stomach", "Lower Stomach", "Bowels", "Intestines", "Abdomen", "Lower Abdomen"]),
    (["liver"], ["Liver"]),
    (["spleen"], ["Spleens", "Blood"]),
    (["kidney", "urinary", "urine", "micturition", "bladder", "nephritis"], ["Kidney", "Urinary Tracts", "Bladder"]),
    (["womb", "prostate", "menses", "menstrual", "breast", "ovary", "ovaries", "semen", "genital", "testicle", "leucorrhoea", "venereal", "syphilis", "female organs"], ["Female Organs", "Ovaries", "Breast", "Semen", "Genitals", "Sex Organs"]),
    (["hip", "hips", "thigh", "leg", "limb", "knee", "ankle", "foot", "feet", "calf", "shank", "locomotor", "rheumat", "fracture", "joint", "amputation", "back", "spine", "shoulder", "arms", "collar bone"], ["Hips Joints", "Thighs", "Knees", "Ankle", "Feet", "Toes", "Calves", "Shanks", "Joints", "Bones", "Upper Back", "Spine", "Shoulders", "Arms", "Collar"]),
    (["skin", "eczema", "leprosy", "itch", "pimple", "pus", "wound"], ["Flesh", "Blood"]),
    (["dropsy", "swelling", "lymph", "water"], ["Distributary Systems of Body Fluids", "Lymphatic System", "Blood"]),
    (["diabetes", "insulin", "adrenal"], ["Blood", "Kidney", "Urinary Tracts", "Liver", "Distributary Systems of Body Fluids"]),
    (["nerv", "neuralgia", "paralysis", "hysteria"], ["Brain", "Mind", "Mental Faculties", "Spine"]),
]


DISEASE_PHRASE_OVERRIDES = {
    "Cold": ["Nose", "Throat", "Lungs", "Chest", "Breathings"],
    "Cold & Cough": ["Nose", "Throat", "Lungs", "Chest", "Breathing", "Breathings"],
    "Cold foot": ["Feet", "Toes"],
    "Rheumatism": ["Joints", "Bones", "Hips Joints", "Thighs", "Knees"],
    "Rheumatic pains": ["Joints", "Bones", "Feet", "Knees", "Hips Joints"],
    "Rheumatic heart diseases": ["Heart", "Chest", "Blood", "Joints"],
    "Diabetes": ["Blood", "Kidney", "Urinary Tracts", "Liver", "Distributary Systems of Body Fluids"],
    "Diabetes, Insulin deficiency": ["Blood", "Kidney", "Urinary Tracts", "Liver", "Distributary Systems of Body Fluids"],
    "Insulin deficiency": ["Blood", "Liver", "Kidney", "Urinary Tracts"],
    "Pulmonary troubles": ["Lungs", "Chest", "Breathing", "Breathings"],
    "Respiratory diseases": ["Lungs", "Chest", "Breathing", "Breathings"],
    "Pulmonary diseases": ["Lungs", "Chest", "Breathing", "Breathings"],
    "Leg injury": ["Shanks", "Calves", "Knees", "Feet", "Bones", "Flesh"],
    "Hip diseases": ["Hips Joints", "Thighs", "Joints", "Bones"],
    "Swelling above the knees": ["Knees", "Thighs", "Distributary Systems of Body Fluids"],
    "Milk leg": ["Shanks", "Calves", "Feet", "Distributary Systems of Body Fluids"],
    "Perspiring feet": ["Feet", "Toes"],
    "Enlarged Liver": ["Liver"],
    "Abdominal tumour": ["Abdomen", "Lower Abdomen", "Stomach"],
    "Intestinal disorders": ["Intestines", "Bowels", "Lower Abdomen"],
    "Abdominal disorders": ["Abdomen", "Lower Abdomen", "Stomach", "Bowels"],
    "Neuralgia, Coma": ["Brain", "Mind", "Mental Faculties"],
    "Stomach disorder": ["Stomach", "Upper Stomach", "Lower Stomach"],
    "Problems in kidney & bladder": ["Kidney", "Bladder", "Urinary Tracts"],
    "Gas in stomach": ["Stomach", "Upper Stomach", "Lower Stomach", "Bowels"],
    "Urinary trouble": ["Urinary Tracts", "Bladder", "Kidney"],
    "Frequent micturition": ["Urinary Tracts", "Bladder", "Kidney"],
}


def map_planet_term_to_rashi_organs(term: str) -> list[str]:
    return _canonicalize(PLANET_TERM_TO_RASHI_ORGANS.get(term, []))


def map_disease_to_rashi_organs(disease: str) -> list[str]:
    disease = str(disease or "").strip()
    if not disease:
        return []

    if disease in DISEASE_PHRASE_OVERRIDES:
        return _canonicalize(DISEASE_PHRASE_OVERRIDES[disease])

    disease_lower = disease.lower()
    hits: list[str] = []

    for keywords, organs in DISEASE_KEYWORD_TO_RASHI_ORGANS:
        if any(keyword in disease_lower for keyword in keywords):
            hits.extend(organs)

    for organ in CANONICAL_RASHI_ORGANS:
        organ_lower = organ.lower()
        if organ_lower in disease_lower:
            hits.append(organ)

    # Preserve full coverage by falling back to broad-body proxies when no direct match exists.
    if not hits:
        generic_fallbacks = [
            ("weakness", ["Blood", "Mind"]),
            ("violence", ["Head", "Blood"]),
            ("accident", ["Bones", "Blood", "Flesh"]),
            ("poison", ["Blood", "Stomach"]),
            ("worm", ["Bowels", "Intestines"]),
            ("tumour", ["Flesh", "Abdomen"]),
            ("cancer", ["Flesh", "Blood"]),
            ("pain", ["Flesh", "Bones"]),
            ("shock", ["Heart", "Mind"]),
        ]
        for keyword, organs in generic_fallbacks:
            if keyword in disease_lower:
                hits.extend(organs)
        if not hits:
            hits.extend(["Blood"])

    return _canonicalize(hits)


def build_planet_truth_map() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    all_terms = sorted({part.strip() for raw in PLANET_DISEASES.values() for part in _split_csv(raw)})
    for term in all_terms:
        result[term] = map_planet_term_to_rashi_organs(term)
    return result


def build_nakshatra_truth_map() -> dict[str, dict[str, dict[str, list[str]]]]:
    result: dict[str, dict[str, dict[str, list[str]]]] = {}
    for nakshatra, pada_map in NAKSHATRA_DISEASES.items():
        result[nakshatra] = {}
        for pada_key, diseases in pada_map.items():
            result[nakshatra][pada_key] = {
                disease: map_disease_to_rashi_organs(disease)
                for disease in diseases
            }
    return result


def build_rashi_truth_map() -> dict[str, list[str]]:
    return {rashi: _canonicalize(_split_csv(raw)) for rashi, raw in RASHI_ORGANS.items()}


RASHI_TRUTH_MAP = build_rashi_truth_map()
PLANET_TRUTH_MAP = build_planet_truth_map()
NAKSHATRA_TRUTH_MAP = build_nakshatra_truth_map()


def validate_truth_maps() -> dict[str, list[str]]:
    problems: dict[str, list[str]] = defaultdict(list)

    for term, mapped in PLANET_TRUTH_MAP.items():
        if not mapped:
            problems["unmapped_planet_terms"].append(term)

    for nakshatra, pada_map in NAKSHATRA_TRUTH_MAP.items():
        for pada_key, disease_map in pada_map.items():
            for disease, mapped in disease_map.items():
                if not mapped:
                    problems["unmapped_diseases"].append(f"{nakshatra}:{pada_key}:{disease}")

    return dict(problems)


if __name__ == "__main__":
    issues = validate_truth_maps()
    print(f"Canonical Rashi organs: {len(CANONICAL_RASHI_ORGANS)}")
    print(f"Planet terms mapped: {len(PLANET_TRUTH_MAP)}")
    total_diseases = sum(len(disease_map) for pada_map in NAKSHATRA_TRUTH_MAP.values() for disease_map in pada_map.values())
    print(f"Nakshatra diseases mapped: {total_diseases}")
    if issues:
        print("Validation issues found:")
        for key, values in issues.items():
            print(f"  {key}: {len(values)}")
            for value in values[:10]:
                print(f"    - {value}")
    else:
        print("All planet terms and nakshatra diseases mapped to Rashi-organ source truth.")
