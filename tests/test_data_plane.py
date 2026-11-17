"""Data-plane, knowledge-base, migration, and chat durability tests.

These tests use the in-memory document store. They do not contact production.
"""

from __future__ import annotations

import copy
import os
import unittest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("KB_SOURCE", "fixtures")
os.environ.pop("KB_STRICT", None)

from app.chat import chat_mode, rl_failure_blocks_reply, rl_hint_mode
from app.chat.sessions import configure_session_backend, persist_session, recover_session
from app.core.config import DataSettings
from app.core.errors import ForgedIdentityError, MigrationRefusedError, MissingDatasetError, OwnershipError
from app.infrastructure.firestore.memory import MemoryDocumentStore
from app.infrastructure.firestore.repositories import DataPlane
from app.knowledge.bindings import publish
from app.knowledge.service import KnowledgeBaseService
from migrations.migrate_firestore import main as migrate_main
from migrations.plan import apply_migration, plan_migration


def _settings(**overrides) -> DataSettings:
    base = dict(
        app_env="test",
        project_id="astromedica-local",
        emulator_host="",
        kb_source="fixtures",
        kb_strict=False,
        kb_cache_ttl_seconds=60,
        chat_mode="classic",
        rl_hint_mode="off",
        chat_durable=True,
        rl_config_strict=False,
        identity_encryption_key="",
        identity_encryption_required=False,
        model_provider="vertex",
    )
    base.update(overrides)
    return DataSettings(**base)


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.plane = DataPlane(MemoryDocumentStore(), encryption_key="test-key")

    def test_user_lookup_is_idempotent_and_collision_resistant(self):
        first = self.plane.ensure_user("user_clerk_a", email="a@example.com")
        second = self.plane.ensure_user("user_clerk_a", email="other@example.com")
        self.assertEqual(first["internal_user_id"], second["internal_user_id"])
        self.assertEqual(second["email"], "a@example.com")
        with self.assertRaises(ForgedIdentityError):
            self.plane.ensure_user("user_clerk_a", internal_user_id="different-user")
        with self.assertRaises(ForgedIdentityError):
            self.plane.ensure_user("bad/id")

    def test_patient_identity_source_run_chat_audit_and_isolation(self):
        owner = self.plane.ensure_user("user_owner")["internal_user_id"]
        other = self.plane.ensure_user("user_other")["internal_user_id"]
        self.plane.upsert_patient(owner, "p1", {"algo_version": 1})
        saved = self.plane.save_identity(owner, "p1", {
            "name": "Hidden Name",
            "date_of_birth": "1990-01-01",
            "birth_time": "08:00",
            "birth_place": "Pune",
            "latitude": 18.5,
            "longitude": 73.8,
            "gender": "F",
        })
        self.assertTrue(saved["encrypted"])
        stored = self.plane.store.get(("users", owner, "patients", "p1", "private", "identity"))
        self.assertNotIn("Hidden Name", str(stored))
        self.assertEqual(self.plane.read_identity(owner, owner, "p1")["name"], "Hidden Name")
        with self.assertRaises(OwnershipError):
            self.plane.read_identity(other, owner, "p1")
        self.plane.save_source(owner, "p1", {"prokerala_chart": {"planets": [{"name": "Sun"}]}, "prokerala_dasha": {"maha": "Sun"}})
        self.plane.save_run(
            owner, "p1", "run-1",
            algorithm_version="legacy-rules-1",
            status="complete",
            input_reference="run-1",
            input_hash="abc",
            combined_result={"top_organs": ["Heart"]},
            result_summary={"top_organs": ["Heart"]},
        )
        nested = [[1, 2], [3]]
        self.plane.save_algorithm_output(owner, "p1", "run-1", "hitlist", {"score": nested}, algorithm_version="legacy-rules-1")
        self.assertEqual(self.plane.read_algorithm_output(owner, owner, "p1", "run-1", "hitlist")["score"], nested)
        big = "x" * 1_000_000
        self.plane.save_algorithm_output(owner, "p1", "run-1", "dasha", {"blob": big}, algorithm_version="legacy-rules-1")
        self.assertEqual(self.plane.read_algorithm_output(owner, owner, "p1", "run-1", "dasha")["blob"], big)
        self.plane.append_turn(owner, "p1", "chat-1", {"role": "assistant", "content": "How are you?", "turn_id": "000001"})
        self.plane.append_turn(owner, "p1", "chat-1", {"role": "user", "content": "Tired", "turn_id": "000002"})
        turns = self.plane.list_turns(owner, owner, "p1", "chat-1")
        self.assertEqual([turn["sequence_number"] for turn in turns], [1, 2])
        self.plane.complete_chat(owner, "p1", "chat-1", {"organ": "Heart"})
        self.plane.save_rl_output(owner, "p1", "chat-1", "rl-1", [{"Heart": []}], model_version="v2")
        self.assertEqual(self.plane.read_rl_output(owner, owner, "p1", "chat-1", "rl-1"), [{"Heart": []}])
        event = self.plane.append_audit(owner, "p1", "chart_created", {"name": "Hidden Name", "run_id": "run-1"})
        self.assertNotIn("name", event["metadata"])
        self.assertIsNone(self.plane.get_patient(other, other, "p1"))
        with self.assertRaises(OwnershipError):
            self.plane.list_turns(other, owner, "p1", "chat-1")

    def test_system_config_rejects_secrets(self):
        with self.assertRaises(ForgedIdentityError):
            self.plane.save_system_config("runtime", {"api_key": "nope"})


class KnowledgeTests(unittest.TestCase):
    def test_fixture_cache_and_missing_dataset(self):
        service = KnowledgeBaseService(DataPlane(MemoryDocumentStore(), encryption_key=""), _settings())
        first = service.require("rashi_organs")
        second = service.require("rashi_organs")
        self.assertEqual(first, second)
        self.assertGreaterEqual(service.hits, 1)
        with self.assertRaises(MissingDatasetError):
            service.require("not_a_dataset")
        with self.assertRaises(MissingDatasetError):
            service.require("symptom_library")

    def test_firestore_source_does_not_fall_back(self):
        plane = DataPlane(MemoryDocumentStore(), encryption_key="")
        service = KnowledgeBaseService(plane, _settings(kb_source="firestore", kb_strict=True, app_env="production"))
        with self.assertRaises(MissingDatasetError):
            service.require("rashi_organs")
        self.assertEqual(service.stats()["cached"], [])

    def test_publish_preserves_algorithm_lookup(self):
        import knowledge_base

        before_organs = copy.deepcopy(knowledge_base.RASHI_ORGANS)
        before_degrees = copy.deepcopy(knowledge_base.RASHI_DEGREES)
        key = next(iter(knowledge_base.NAKSHATRA_DISEASES))
        before_lookup = copy.deepcopy(knowledge_base.get_nakshatra_diseases(key, 1))
        plane = DataPlane(MemoryDocumentStore(), encryption_key="")
        from app.knowledge.fixtures import load_fixture_records
        from app.knowledge.registry import KNOWLEDGE_SPECS

        fixtures = load_fixture_records()["knowledge"]
        for spec in KNOWLEDGE_SPECS:
            if spec.category != "knowledge":
                continue
            status = plane.write_dataset_if_absent(
                spec.dataset_id,
                {
                    "dataset_name": spec.dataset_name,
                    "description": spec.description,
                    "source_files": list(spec.source_files),
                    "record_mode": spec.record_mode,
                    "status": spec.status,
                    "category": spec.category,
                },
                fixtures[spec.dataset_id],
            )
            self.assertEqual(status, "created")
            self.assertEqual(
                plane.write_dataset_if_absent(spec.dataset_id, {"dataset_name": spec.dataset_name}, fixtures[spec.dataset_id]),
                "unchanged",
            )
        service = KnowledgeBaseService(plane, _settings(kb_source="firestore", kb_cache_ttl_seconds=0))
        publish(service)
        self.assertEqual(knowledge_base.RASHI_ORGANS, before_organs)
        self.assertEqual(knowledge_base.RASHI_DEGREES, before_degrees)
        self.assertEqual(knowledge_base.get_nakshatra_diseases(key, 1), before_lookup)

    def test_unavailable_store_is_not_an_empty_dataset(self):
        from app.core.errors import FirestoreUnavailableError

        class Down:
            def get(self, path):
                raise FirestoreUnavailableError("down")

            def list_docs(self, path, limit=100, start_after=None):
                raise FirestoreUnavailableError("down")

        service = KnowledgeBaseService(Down(), _settings(kb_source="firestore"))
        with self.assertRaises(FirestoreUnavailableError):
            service.require("planet_diseases")


class MigrationTests(unittest.TestCase):
    def test_dry_run_writes_nothing_and_apply_is_idempotent(self):
        store = MemoryDocumentStore()
        store.set(("users", "u1"), {"internal_user_id": "u1", "clerk_user_id": "user_u1"})
        store.set(("users", "u1", "patients", "p1"), {"patient_id": "p1"})
        store.set(("users", "u1", "identity_vault", "p1"), {
            "patient_id": "p1", "name": "Kept", "dob": "2001-02-03", "birth_place": "Delhi",
        })
        store.set(("users", "u1", "algorithm_results", "run1_hitlist"), {
            "patient_id": "p1", "input_id": "run1", "algorithm": "hitlist",
            "output": {"top4": [{"label": "Heart"}]}, "algorithm_version": "1", "status": "complete",
        })
        store.set(("chat_logs", "p1__start"), {
            "patient_id": "p1", "turns": [{"role": "user", "message": "yes", "timestamp": "t"}],
        })
        before = len(store.docs)
        report = plan_migration(store)
        self.assertEqual(report["writes"], 0)
        self.assertEqual(len(store.docs), before)
        self.assertEqual(report["mode"], "dry-run")
        plane = DataPlane(store, encryption_key="")
        applied = apply_migration(plane, store)
        self.assertGreater(applied["created"]["knowledge"], 0)
        self.assertGreater(applied["created"]["rl_config"], 0)
        self.assertEqual(applied["created"]["identity"], 1)
        self.assertEqual(applied["created"]["runs"], 1)
        self.assertEqual(applied["created"]["chats"], 1)
        self.assertIsNotNone(store.get(("users", "u1", "identity_vault", "p1")))
        self.assertTrue(store.get(("knowledge_base", "rashi_organs"))["checksum"])
        identity = plane.read_identity("u1", "u1", "p1")
        self.assertEqual(identity["name"], "Kept")
        output = plane.read_algorithm_output("u1", "u1", "p1", "run1", "hitlist")
        self.assertEqual(output["top4"][0]["label"], "Heart")
        again = apply_migration(plane, store)
        self.assertEqual(again["created"]["knowledge"], 0)
        self.assertEqual(again["created"]["identity"], 0)
        self.assertEqual(again["created"]["runs"], 0)
        self.assertEqual(again["reconciled"]["checksums_matched"], again["reconciled"]["datasets_checked"])

    def test_cli_refuses_production_project(self):
        os.environ["APP_ENV"] = "production"
        os.environ["FIRESTORE_PROJECT_ID"] = "astrohealth-509609"
        os.environ["FIRESTORE_BACKEND"] = "firestore"
        try:
            with self.assertRaises(MigrationRefusedError):
                migrate_main(["--apply"])
        finally:
            os.environ["APP_ENV"] = "test"
            os.environ.pop("FIRESTORE_PROJECT_ID", None)
            os.environ.pop("FIRESTORE_BACKEND", None)


class ChatDurabilityTests(unittest.TestCase):
    def test_second_worker_recovers_session_and_flags_stay_classic(self):
        self.assertEqual(chat_mode(), "classic")
        self.assertEqual(rl_hint_mode(), "off")
        self.assertFalse(rl_failure_blocks_reply())
        plane = DataPlane(MemoryDocumentStore(), encryption_key="")
        configure_session_backend(plane)
        owner = plane.ensure_user("user_chat")["internal_user_id"]
        plane.upsert_patient(owner, "p9")
        persist_session(owner, "session-1", {
            "patient_id": "p9",
            "system_prompt": "Never diagnose.",
            "messages": [{"role": "assistant", "content": "Opening question"}],
            "turn_count": 1,
            "cluster_scores": {"Heart": 1},
        })
        first = recover_session(owner, "session-1")
        second = recover_session(owner, "session-1")
        self.assertEqual(first["messages"][0]["content"], "Opening question")
        self.assertEqual(second["system_prompt"], "Never diagnose.")
        self.assertIsNone(recover_session("someone-else", "session-1"))


if __name__ == "__main__":
    unittest.main()
