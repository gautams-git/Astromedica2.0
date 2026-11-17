"""Idempotent copy from the current collections onto the target paths.

Dry-run is the default. Existing target documents are left in place.
Source documents are never deleted.
"""

from __future__ import annotations

from typing import Any

from app.core.clock import utc_now
from app.infrastructure.firestore.codec import canonical_json, sha256_text
from app.knowledge.fixtures import load_fixture_records
from app.knowledge.registry import KNOWLEDGE_SPECS, RL_SPECS


LEGACY_USER_COLLECTIONS = (
    "patients",
    "inputs",
    "chart_inputs",
    "prokerala_results",
    "algorithm_results",
    "combined_results",
    "identity_vault",
    "chart_sources",
)


def _users(store) -> list[tuple[str, dict]]:
    rows, cursor = store.list_docs(("users",), limit=200)
    collected = list(rows)
    while cursor:
        page, cursor = store.list_docs(("users",), limit=200, start_after=cursor)
        collected.extend(page)
    return collected


def _child_docs(store, path: tuple[str, ...]) -> list[tuple[str, dict]]:
    rows, cursor = store.list_docs(path, limit=200)
    collected = list(rows)
    while cursor:
        page, cursor = store.list_docs(path, limit=200, start_after=cursor)
        collected.extend(page)
    return collected


def plan_migration(store) -> dict:
    """Read source documents and describe the copy. Does not write."""
    datasets = []
    fixtures = load_fixture_records()
    for spec in KNOWLEDGE_SPECS:
        records = fixtures["knowledge"][spec.dataset_id]
        checksum = sha256_text(canonical_json(records))
        existing = store.get(("knowledge_base", spec.dataset_id))
        action = "create"
        if existing and existing.get("checksum") == checksum:
            action = "unchanged"
        elif existing:
            action = "drift_skip"
        datasets.append({
            "dataset_id": spec.dataset_id,
            "category": spec.category,
            "records": len(records),
            "checksum": checksum,
            "action": action,
        })
    rl = []
    for spec in RL_SPECS:
        records = fixtures["rl_config"][spec.dataset_id]
        checksum = sha256_text(canonical_json(records))
        existing = store.get(("rl_config", spec.dataset_id))
        action = "create" if not existing else ("unchanged" if existing.get("checksum") == checksum else "drift_skip")
        rl.append({"dataset_id": spec.dataset_id, "records": len(records), "checksum": checksum, "action": action})

    user_actions = []
    for user_id, user in _users(store):
        patients = _child_docs(store, ("users", user_id, "patients"))
        for patient_id, patient in patients:
            user_actions.append(_plan_patient(store, user_id, patient_id, patient))
    legacy_chats = _child_docs(store, ("chat_logs",))
    return {
        "generated_at": utc_now(),
        "mode": "dry-run",
        "writes": 0,
        "knowledge": datasets,
        "rl_config": rl,
        "patients": user_actions,
        "legacy_chat_logs": len(legacy_chats),
        "notes": [
            "knowledge_base/datasets/{id}/records/{id} is not a valid Firestore path (odd segment count).",
            "Target dataset path is knowledge_base/{datasetId} plus records/{recordId}.",
            "RL tables move to rl_config and are not merged into knowledge_base.",
            "Source collections are retained.",
        ],
    }


def _plan_patient(store, user_id: str, patient_id: str, patient: dict) -> dict:
    identity = store.get(("users", user_id, "identity_vault", patient_id))
    source = store.get(("users", user_id, "chart_sources", patient_id))
    private_identity = store.get(("users", user_id, "patients", patient_id, "private", "identity"))
    private_source = store.get(("users", user_id, "patients", patient_id, "private", "source"))
    algorithms = _child_docs(store, ("users", user_id, "algorithm_results"))
    owned_algos = [(doc_id, doc) for doc_id, doc in algorithms if doc.get("patient_id") == patient_id]
    return {
        "internal_user_id": user_id,
        "patient_id": patient_id,
        "hidden": bool(patient.get("hidden")),
        "identity": "create" if identity and not private_identity else ("unchanged" if private_identity else "absent"),
        "source": "create" if source and not private_source else ("unchanged" if private_source else "absent"),
        "algorithm_documents": len(owned_algos),
        "overwrite": False,
    }


def apply_migration(plane, store) -> dict:
    """Copy missing targets. Drift and existing documents are skipped."""
    report = plan_migration(store)
    created = {"knowledge": 0, "rl_config": 0, "identity": 0, "source": 0, "runs": 0, "chats": 0}
    fixtures = load_fixture_records()
    for spec in KNOWLEDGE_SPECS:
        status = plane.write_dataset_if_absent(
            spec.dataset_id,
            {
                "dataset_name": spec.dataset_name,
                "description": spec.description,
                "status": spec.status,
                "source_files": list(spec.source_files),
                "record_mode": spec.record_mode,
                "category": spec.category,
            },
            fixtures["knowledge"][spec.dataset_id],
        )
        if status == "created":
            created["knowledge"] += 1
    for spec in RL_SPECS:
        status = plane.write_rl_config_if_absent(
            spec.dataset_id,
            fixtures["rl_config"][spec.dataset_id],
            spec.description,
        )
        if status == "created":
            created["rl_config"] += 1
    for user_id, _user in _users(store):
        for patient_id, _patient in _child_docs(store, ("users", user_id, "patients")):
            _copy_patient(plane, store, user_id, patient_id, created)
        _copy_chat_logs(plane, store, user_id, created)
    report["mode"] = "apply"
    report["created"] = created
    report["writes"] = sum(created.values())
    report["reconciled"] = _reconcile(store, fixtures)
    return report


def _copy_patient(plane, store, user_id: str, patient_id: str, created: dict) -> None:
    identity = store.get(("users", user_id, "identity_vault", patient_id))
    if identity and not store.get(("users", user_id, "patients", patient_id, "private", "identity")):
        plane.save_identity(user_id, patient_id, {
            "name": identity.get("name"),
            "date_of_birth": identity.get("dob") or identity.get("date_of_birth"),
            "birth_place": identity.get("birth_place"),
            "father_name": identity.get("father_name"),
            "mother_name": identity.get("mother_name"),
        })
        created["identity"] += 1
    source = store.get(("users", user_id, "chart_sources", patient_id))
    if source and not store.get(("users", user_id, "patients", patient_id, "private", "source")):
        plane.save_source(user_id, patient_id, {
            "raw_source": {
                "raw_chart_json": source.get("raw_chart_json"),
                "raw_dasha_json": source.get("raw_dasha_json"),
            },
            "source_provider": "prokerala",
            "source_version": "legacy_chart_sources",
            "input_hash": sha256_text(canonical_json(source)),
        })
        created["source"] += 1
    grouped: dict[str, list] = {}
    for doc_id, doc in _child_docs(store, ("users", user_id, "algorithm_results")):
        if doc.get("patient_id") != patient_id:
            continue
        run_id = str(doc.get("input_id") or "legacy")
        grouped.setdefault(run_id, []).append((doc_id, doc))
    for run_id, docs in grouped.items():
        target = store.get(("users", user_id, "patients", patient_id, "runs", run_id))
        if target:
            continue
        plane.save_run(
            user_id,
            patient_id,
            run_id,
            algorithm_version=str(docs[0][1].get("algorithm_version") or "legacy"),
            status="complete",
            input_reference=run_id,
            input_hash=sha256_text(run_id),
            execution_metadata={"migrated_from": "algorithm_results"},
        )
        for _doc_id, doc in docs:
            algorithm_id = str(doc.get("algorithm") or "unknown")
            plane.save_algorithm_output(
                user_id,
                patient_id,
                run_id,
                algorithm_id,
                doc.get("output"),
                algorithm_version=str(doc.get("algorithm_version") or "legacy"),
                status=str(doc.get("status") or "complete"),
            )
        created["runs"] += 1


def _copy_chat_logs(plane, store, user_id: str, created: dict) -> None:
    owned = {patient_id for patient_id, _doc in _child_docs(store, ("users", user_id, "patients"))}
    for doc_id, doc in _child_docs(store, ("chat_logs",)):
        patient_id = str(doc.get("patient_id") or "")
        if patient_id not in owned or doc_id == "meta_marker":
            continue
        chat_id = f"legacy_{doc_id}"[:700]
        if store.get(("users", user_id, "patients", patient_id, "chats", chat_id)):
            continue
        plane.open_chat(user_id, patient_id, chat_id, {
            "status": "completed" if doc.get("completed") else "active",
            "completed": bool(doc.get("completed")),
            "chat_algorithm_version": "legacy_chat_logs",
        })
        for index, turn in enumerate(doc.get("turns") or [], start=1):
            plane.append_turn(user_id, patient_id, chat_id, {
                "turn_id": f"{index:06d}",
                "role": turn.get("role") or "user",
                "content": turn.get("message") or turn.get("content") or "",
                "timestamp": turn.get("timestamp"),
                "metadata": turn.get("metadata") or {},
            })
        created["chats"] += 1


def _reconcile(store, fixtures: dict) -> dict:
    checked = 0
    matched = 0
    for spec in KNOWLEDGE_SPECS:
        meta = store.get(("knowledge_base", spec.dataset_id))
        expected = sha256_text(canonical_json(fixtures["knowledge"][spec.dataset_id]))
        checked += 1
        if meta and meta.get("checksum") == expected:
            matched += 1
    return {"datasets_checked": checked, "checksums_matched": matched}
