import json
import os
import tempfile
import unittest

from algorithm_comparison import ALGORITHM_NAMES, write_algorithm_comparison


def sample_processed(hitlist_score=10):
    supporting_organ = {
        "organ": "Heart",
        "score": 8,
        "supports": [],
    }
    return {
        "hitlist": {
            "top4": [{
                "label": "Heart / Blood",
                "score": hitlist_score,
                "matched_organs": ["Heart"],
                "matched_diseases": ["Hypertension"],
                "trail": [],
            }]
        },
        "rashi_correlation": {
            "top3": [{
                "label": "Heart",
                "score": 9,
                "matched_organs": ["Heart"],
                "matched_diseases": ["Hypertension"],
            }]
        },
        "organ_truth_correlation": {
            "top_organs": [{
                "organ": "Heart",
                "score": 8,
                "related_diseases": [{"disease": "Hypertension", "score": 8}],
            }],
            "top_diseases": [{"disease": "Hypertension", "score": 8}],
        },
        "disease_first_logic": {
            "top_diseases": [{
                "disease": "Hypertension",
                "score": 7,
                "supporting_organs": [supporting_organ],
            }]
        },
        "disease_compare_logic": {
            "top_diseases": [{
                "disease": "Hypertension",
                "score": 6,
                "supporting_organs": [supporting_organ],
            }]
        },
    }


class AlgorithmComparisonTests(unittest.TestCase):
    def test_writes_named_algorithms_and_combined_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = write_algorithm_comparison(folder, "021", sample_processed())

            self.assertEqual(set(output["algorithms"]), set(ALGORITHM_NAMES.values()))
            self.assertIn("Combined Algo Summary", output)
            top = output["Combined Algo Summary"]["top_organs"][0]
            self.assertEqual(top["organ"], "Heart")
            self.assertEqual(top["algorithm_support_count"], 5)
            self.assertEqual(len(top["supporting_algorithms"]), 5)

    def test_rerun_overwrites_same_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "algorithm_comparison.json")
            write_algorithm_comparison(folder, "021", sample_processed(hitlist_score=10))
            with open(path) as file:
                first = json.load(file)

            write_algorithm_comparison(folder, "021", sample_processed(hitlist_score=100))
            with open(path) as file:
                second = json.load(file)

            self.assertEqual(first["patient_id"], second["patient_id"])
            first_score = first["algorithms"][ALGORITHM_NAMES["hitlist"]][
                "top_organs_and_systems"
            ][0]["score"]
            second_score = second["algorithms"][ALGORITHM_NAMES["hitlist"]][
                "top_organs_and_systems"
            ][0]["score"]
            self.assertEqual(first_score, 10)
            self.assertEqual(second_score, 100)


if __name__ == "__main__":
    unittest.main()
