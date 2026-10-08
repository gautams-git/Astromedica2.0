import unittest
from unittest.mock import patch

import disease_first_logic
from eleventh_lord_confirmation import (
    confirmation_boost,
    get_eleventh_lord_confirmation_source,
    match_disease,
)


class EleventhLordConfirmationTests(unittest.TestCase):
    def test_reads_only_eleventh_lord_nakshatra_diseases(self):
        chart = {
            "planets": [{
                "name": "Venus",
                "nakshatra": {"name": "Test Nakshatra", "pada": 2},
                "sign": {"name": "Tula"},
            }]
        }
        rule1 = {"eleventh_house_lord": "Venus"}
        with patch(
            "eleventh_lord_confirmation.NAKSHATRA_DISEASES",
            {"Test Nakshatra": {"2": ["Skin disease"]}},
        ):
            source = get_eleventh_lord_confirmation_source(chart, rule1)

        self.assertEqual(source["planet"], "Venus")
        self.assertEqual(source["diseases"], ["Skin disease"])
        self.assertNotIn("planet_diseases", source)
        self.assertNotIn("rashi_organs", source)

    def test_unmatched_disease_is_ignored(self):
        with patch(
            "eleventh_lord_confirmation.NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS",
            {
                "Heart disease": ["Heart"],
                "Skin disease": ["Skin"],
            },
        ):
            self.assertIsNone(match_disease("Heart disease", "Skin disease"))

    def test_matching_candidate_receives_five_percent_only(self):
        source = {
            "planet": "Venus",
            "nakshatra": "Test Nakshatra",
            "pada": 2,
            "diseases": ["Skin disease"],
        }
        built_source = {
            "priority": "P4",
            "label": "6th House Lord",
            "planet": "Saturn",
            "sign": "",
            "nakshatra": "",
            "pada": 1,
            "diseases": ["Skin disease"],
            "planet_organs": [],
            "rashi_organs": [],
        }
        chart = {"planets": []}
        rule1 = {
            "rogkaraka_1": [],
            "eleventh_house_occupants": [],
            "rogkaraka_3": {},
            "drishti_on_6th": [],
            "sixth_house_lord": "Saturn",
            "ascendant_lord": "",
            "conditional_moon": {},
            "eleventh_house_sign": "",
        }

        with (
            patch.object(
                disease_first_logic,
                "_build_source",
                side_effect=lambda priority, label, planet, gender, include_rashi=False:
                    built_source if priority == "P4" else None,
            ),
            patch(
                "eleventh_lord_confirmation.get_eleventh_lord_confirmation_source",
                return_value=source,
            ),
            patch(
                "eleventh_lord_confirmation.NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS",
                {"Skin disease": ["Skin"]},
            ),
            patch.object(
                disease_first_logic,
                "NAKSHATRA_DISEASE_TO_SUPERSET_ORGANS",
                {"Skin disease": ["Skin"]},
            ),
        ):
            result = disease_first_logic.run_disease_first_logic(chart, rule1)

        disease = result["top_diseases"][0]
        base_score = disease_first_logic.SOURCE_WEIGHTS["P4"]
        self.assertEqual(disease["score"], round(base_score + confirmation_boost(base_score), 2))
        self.assertEqual(
            disease["eleventh_lord_confirmation"]["label"],
            "11th Lord Nakshatra Confirmation",
        )
        self.assertEqual(len(result["top_diseases"]), 1)


if __name__ == "__main__":
    unittest.main()
