import json
import os
import tempfile
import unittest
from unittest.mock import patch

import jataka_fetch


class PatientStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        self.patients_dir = os.path.join(self.temp_dir.name, "patients")
        self.patches = [
            patch.object(jataka_fetch, "PATIENTS_DIR", self.patients_dir),
            patch.object(
                jataka_fetch,
                "PATIENT_INDEX_FILE",
                os.path.join(self.patients_dir, "index.json"),
            ),
        ]
        for active_patch in self.patches:
            active_patch.start()

    def tearDown(self):
        for active_patch in reversed(self.patches):
            active_patch.stop()
        os.chdir(self.original_cwd)
        self.temp_dir.cleanup()

    def test_same_chart_reuses_patient_folder(self):
        chart_key = jataka_fetch.get_request_key(
            "Test Patient", "1990-01-02", "08:30", 12.9716, 77.5946
        )

        first_id, first_folder, first_created = jataka_fetch.get_or_create_patient(
            chart_key
        )
        second_id, second_folder, second_created = jataka_fetch.get_or_create_patient(
            chart_key
        )

        self.assertEqual(first_id, "001")
        self.assertEqual(first_id, second_id)
        self.assertEqual(first_folder, second_folder)
        self.assertTrue(first_created)
        self.assertFalse(second_created)
        folder_count = sum(entry.is_dir() for entry in os.scandir(self.patients_dir))
        self.assertEqual(folder_count, 1)

    def test_profile_updates_preserve_blank_fields(self):
        chart_key = "chart-key"
        patient_id, folder, created = jataka_fetch.get_or_create_patient(chart_key)

        jataka_fetch.update_patient_records(
            folder,
            patient_id,
            chart_key,
            {"name": "Test Patient", "father_name": "First Father"},
            "Prokerala API",
            created,
        )
        jataka_fetch.update_patient_records(
            folder,
            patient_id,
            chart_key,
            {
                "name": "Test Patient Updated",
                "father_name": "",
                "mother_name": "Mother",
            },
            "Local Cache",
            False,
        )

        with open(os.path.join(folder, "profile.json")) as profile_file:
            profile = json.load(profile_file)
        with open(os.path.join(folder, "run_history.json")) as history_file:
            history = json.load(history_file)

        self.assertEqual(profile["name"], "Test Patient Updated")
        self.assertEqual(profile["father_name"], "First Father")
        self.assertEqual(profile["mother_name"], "Mother")
        self.assertEqual(len(history), 2)


if __name__ == "__main__":
    unittest.main()
