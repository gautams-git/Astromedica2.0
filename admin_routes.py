"""
admin_routes.py — the admin page (/admin): every account, its charts, and
three actions. Only Clerk ids in ADMIN_CLERK_USER_IDS get any data; everyone
else gets "Not found", even when typing the URL.

  Delete chart      removes one chart and everything linked to it; the user
                    can create a new chart.
  Reset allowance   deletes nothing; the user can create a new chart again.
  Delete all data   every chart of the account plus its user record. The
                    Clerk login stays, so signing in again starts fresh.

What "everything linked to a chart" is (Firestore has no cascade delete):
  users/{uid}/patients/{pid}  (+ its subcollections)
  users/{uid}/chart_sources/{pid}, users/{uid}/identity_vault/{pid}
  users/{uid}/{inputs, chart_inputs, prokerala_results, algorithm_results,
              combined_results, tester_charts} documents with patient_id == pid
  users/{uid}/account/chart_allowance  (fixed by chart_access.get_allowance)
  chat_logs/{pid}__...               RL chat logs (Abhishek's collection)
  patients/{pid}/ on the server disk  chat log, RAG/RL reports, question set
With 001-style ids two accounts that typed the same birth details share one
patient; then the disk folder and RL logs are kept while another account uses it.
Every action is recorded in users/{admin}/admin_audit and printed as [Admin].

Testers (tester_access.py): an account's row shows a Tester badge with today's
charts and total runs; "Remove tester" ends the role (their charts stay). The
"Tester accuracy" panel sums up the feedback testers gave. Tester charts are kept
until deleted here: "Delete chart" removes that chart's tester runs too.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone

from flask import Blueprint, g, jsonify, request, send_from_directory

import chart_access
import tester_access
from app_auth import AUTH_ENABLED, is_admin, login_required
from app_core.errors import AuthorizationError, RequestValidationError
from app_core.runtime import get_services
from app_core.validation import require_patient_id

admin_bp = Blueprint("admin", __name__)

PATIENTS_DIR = "patients"
PER_RUN_COLLECTIONS = ("inputs", "chart_inputs", "prokerala_results", "algorithm_results", "combined_results",
                       tester_access.RUNS)
PER_PATIENT_DOCS = ("patients", "chart_sources", "identity_vault")


def _repo():
    return get_services().repository


def _require_admin():
    if not is_admin():
        raise AuthorizationError()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


# ── Emails from Clerk (cached briefly; the page still works without them) ──
_CLERK_CACHE = {"at": 0.0, "users": {}}


def _clerk_profiles(clerk_ids: list[str]) -> dict:
    settings = get_services().settings
    if not AUTH_ENABLED or not settings.clerk_secret_key or not clerk_ids:
        return {}
    if time.time() - _CLERK_CACHE["at"] < 120 and all(c in _CLERK_CACHE["users"] for c in clerk_ids):
        return _CLERK_CACHE["users"]
    found = {}
    try:
        from clerk_backend_api import Clerk
        with Clerk(bearer_auth=settings.clerk_secret_key) as clerk:
            for start in range(0, len(clerk_ids), 100):
                for user in clerk.users.list(request={"user_id": clerk_ids[start:start + 100], "limit": 100}) or []:
                    primary = next((e.email_address for e in (user.email_addresses or [])
                                    if e.id == user.primary_email_address_id), None)
                    if not primary and user.email_addresses:
                        primary = user.email_addresses[0].email_address
                    found[user.id] = {
                        "email": primary or "",
                        "name": " ".join(p for p in (user.first_name, user.last_name) if p),
                        "last_sign_in_at": user.last_sign_in_at,
                    }
    except Exception as e:
        print(f"[Admin] could not load emails from Clerk: {type(e).__name__}: {str(e)[:150]}")
        return _CLERK_CACHE["users"]
    _CLERK_CACHE.update(at=time.time(), users=found)
    return found


# ── Reading the accounts ────────────────────────────────────────────────────

def _chart_row(pid: str, patient_doc: dict, active_pid: str | None) -> dict:
    folder = os.path.join(PATIENTS_DIR, pid)
    profile = _read_json(os.path.join(folder, "profile.json")) or {}
    chat = _read_json(os.path.join(folder, "chat_log.json")) or {}
    turns = chat.get("turns") or []
    last_chat = turns[-1].get("timestamp") if turns else None
    if pid == active_pid:
        status = "active"
    elif patient_doc.get("hidden"):
        status = "replaced"
    else:
        status = "saved"
    return {
        "patient_id": pid,
        "name": profile.get("name", ""),
        "dob": profile.get("dob", ""),
        "birth_place": profile.get("birth_place", ""),
        "gender": profile.get("gender", ""),
        "created_at": patient_doc.get("created_at"),
        "status": status,
        "last_chat_at": last_chat,
        "chat_completed": bool(chat.get("completed")),
        "rag_report": os.path.exists(os.path.join(folder, "final_output.json")),
        "rl_report": os.path.exists(os.path.join(folder, "RL Output.json")),
        "on_disk": os.path.isdir(folder),
    }


def _account(user: dict, profiles: dict) -> dict:
    uid = user["internal_user_id"]
    clerk_id = user.get("clerk_user_id") or ""
    allowance = chart_access.get_allowance(uid, heal=False)
    charts = [_chart_row(pid, doc, allowance["active_patient_id"])
              for pid, doc in _repo().list_docs(uid, "patients")]
    charts.sort(key=lambda c: str(c.get("created_at") or ""), reverse=True)
    profile = profiles.get(clerk_id, {})
    is_tester = tester_access.has_role(uid)
    runs = _repo().list_docs(uid, tester_access.RUNS) if is_tester else []
    return {
        "internal_user_id": uid,
        "clerk_user_id": clerk_id,
        "email": profile.get("email", ""),
        "name": profile.get("name", ""),
        "created_at": user.get("created_at"),
        "last_sign_in_at": profile.get("last_sign_in_at"),
        "is_admin": clerk_id in _admin_ids(),
        "charts_used": allowance["charts_used"],
        "max_charts": chart_access.MAX_CHARTS,
        "active_patient_id": allowance["active_patient_id"],
        "charts": charts,
        "is_tester": is_tester,
        "tester_charts_today": tester_access.charts_today(uid) if is_tester else 0,
        "tester_daily_cap": tester_access.daily_cap(),
        "tester_runs": len(runs),
    }


def _admin_ids() -> set:
    from app_auth import ADMIN_CLERK_USER_IDS
    return ADMIN_CLERK_USER_IDS


def _find_user(uid: str) -> dict:
    user = next((u for u in _repo().list_users() if u["internal_user_id"] == uid), None)
    if not user:
        raise RequestValidationError("No such account.")
    return user


def _email_of(user: dict) -> str:
    clerk_id = user.get("clerk_user_id") or ""
    return _clerk_profiles([clerk_id]).get(clerk_id, {}).get("email", "") if clerk_id else ""


# ── The cascade delete ──────────────────────────────────────────────────────

def _other_owners(uid: str, pid: str) -> list[str]:
    return [u["internal_user_id"] for u in _repo().list_users()
            if u["internal_user_id"] != uid and _repo().get(u["internal_user_id"], "patients", pid)]


def _delete_rl_logs(pid: str) -> int:
    from rl_enhancer import config_db
    deleted = 0
    for snap in config_db._patient_chat_docs(pid):
        snap.reference.delete(**config_db.CALL)
        deleted += 1
    return deleted


def _forget_on_disk(pid: str) -> bool:
    folder = os.path.join(PATIENTS_DIR, pid)
    existed = os.path.isdir(folder)
    if existed:
        shutil.rmtree(folder)
    # patients/index.json maps birth details -> patient id; drop the stale link.
    index_path = os.path.join(PATIENTS_DIR, "index.json")
    index = _read_json(index_path)
    if isinstance(index, dict) and pid in index.values():
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in index.items() if v != pid}, f, indent=2)
    return existed


def delete_chart(uid: str, pid: str) -> dict:
    """Every step runs even if an earlier one fails; the report says what happened."""
    report = {"patient_id": pid, "removed": [], "kept": [], "failed": []}

    def step(label, fn):
        try:
            result = fn()
            if result:
                report["removed"].append(label if result is True else f"{label} ({result})")
        except Exception as e:
            report["failed"].append(f"{label}: {type(e).__name__}: {str(e)[:120]}")

    for coll in PER_PATIENT_DOCS:
        step(f"{coll}/{pid}", lambda c=coll: _repo().delete(uid, c, pid))
    for coll in PER_RUN_COLLECTIONS:
        step(f"{coll} records", lambda c=coll: sum(_repo().delete(uid, c, doc_id)
                                                   for doc_id, _ in _repo().list_docs(uid, c, pid)))
    def fix_allowance():
        before = _repo().get(uid, "account", "chart_allowance") or {}
        after = chart_access.get_allowance(uid, heal=True)
        return bool(before) and (before.get("charts_used"), before.get("active_patient_id")) != \
            (after["charts_used"], after["active_patient_id"])
    step("chart allowance updated", fix_allowance)

    others = _other_owners(uid, pid)
    if others:
        report["kept"].append(f"disk folder and RL chat logs - patient {pid} is also used by "
                              f"{len(others)} other account(s)")
    else:
        step("RL chat logs", lambda: _delete_rl_logs(pid))
        step(f"disk folder patients/{pid}", lambda: _forget_on_disk(pid))
    return report


def _audit(action: str, target: dict, detail: dict) -> None:
    admin_uid = g.user["internal_user_id"]
    record = {"record_type": "admin_audit", "action": action, "at": _now(),
              "admin_clerk_user_id": getattr(getattr(g, "principal", None), "clerk_user_id", "local-dev"),
              "target_internal_user_id": target.get("internal_user_id"),
              "target_email": target.get("email", ""), **detail}
    try:
        _repo().save(admin_uid, "admin_audit", uuid.uuid4().hex, record)
    except Exception as e:
        print(f"[Admin] audit record not saved: {e}")
    print(f"[Admin] {record['admin_clerk_user_id']} -> {action} for "
          f"{record['target_email'] or record['target_internal_user_id'][:8] + '...'} {detail}")


# ── Routes ──────────────────────────────────────────────────────────────────

@admin_bp.get("/admin")
def admin_page():
    # The page itself holds no data; every API below checks the admin list.
    return send_from_directory(".", "admin.html")


@admin_bp.get("/admin/api/accounts")
@login_required(rate_limit="read")
def admin_accounts():
    _require_admin()
    users = _repo().list_users()
    profiles = _clerk_profiles([u.get("clerk_user_id") for u in users if u.get("clerk_user_id")])
    accounts = [_account(u, profiles) for u in users]
    accounts.sort(key=lambda a: str(a.get("created_at") or ""), reverse=True)
    return jsonify({
        "success": True,
        "emails_available": bool(profiles),
        "accounts": accounts,
        "totals": {"accounts": len(accounts), "charts": sum(len(a["charts"]) for a in accounts)},
    })


def _confirm_matches(*expected) -> bool:
    typed = str((request.get_json(silent=True) or {}).get("confirm", "")).strip().lower()
    return bool(typed) and typed in {str(e).strip().lower() for e in expected if e}


@admin_bp.post("/admin/api/accounts/<uid>/charts/<pid>/delete")
@login_required(rate_limit="read")
def admin_delete_chart(uid, pid):
    _require_admin()
    require_patient_id(pid)
    user = _find_user(uid)
    if not _repo().get(uid, "patients", pid):
        raise RequestValidationError("This account has no such chart.")
    if not _confirm_matches(pid):
        raise RequestValidationError("Type the patient id to confirm.")
    report = delete_chart(uid, pid)
    _audit("delete_chart", {**user, "email": _email_of(user)},
           {"patient_id": pid, "removed": len(report["removed"]), "failed": len(report["failed"])})
    return jsonify({"success": not report["failed"], "report": report})


@admin_bp.post("/admin/api/accounts/<uid>/reset-allowance")
@login_required(rate_limit="read")
def admin_reset_allowance(uid):
    _require_admin()
    user = _find_user(uid)
    if not _confirm_matches("reset"):
        raise RequestValidationError("Confirm the reset.")
    had = chart_access.reset_allowance(uid)
    _audit("reset_allowance", {**user, "email": _email_of(user)}, {"had_allowance": had})
    return jsonify({"success": True, "report": {
        "removed": ["chart allowance (now 0 of %d used)" % chart_access.MAX_CHARTS] if had else [],
        "kept": ["all charts"], "failed": []}})


@admin_bp.post("/admin/api/accounts/<uid>/remove-tester")
@login_required(rate_limit="read")
def admin_remove_tester(uid):
    _require_admin()
    user = _find_user(uid)
    if not _confirm_matches("remove"):
        raise RequestValidationError("Confirm removing the tester role.")
    had = tester_access.remove_tester(uid)
    _audit("remove_tester", {**user, "email": _email_of(user)}, {"was_tester": had})
    return jsonify({"success": True, "report": {
        "removed": ["tester role (back to the normal chart limit)"] if had else [],
        "kept": ["all charts and tester feedback"], "failed": []}})


@admin_bp.get("/admin/api/tester-accuracy")
@login_required(rate_limit="read")
def admin_tester_accuracy():
    _require_admin()
    return jsonify({"success": True, **tester_access.accuracy_summary()})


@admin_bp.post("/admin/api/accounts/<uid>/delete-all")
@login_required(rate_limit="read")
def admin_delete_all(uid):
    _require_admin()
    user = _find_user(uid)
    email = _email_of(user)
    if not _confirm_matches(email, uid):
        raise RequestValidationError("Type the account's email (or its id) to confirm.")
    report = {"removed": [], "kept": ["Clerk login (signing in again starts fresh)"], "failed": []}
    for pid, _doc in _repo().list_docs(uid, "patients"):
        one = delete_chart(uid, pid)
        report["removed"] += [f"[{pid}] {r}" for r in one["removed"]]
        report["kept"] += [f"[{pid}] {k}" for k in one["kept"]]
        report["failed"] += [f"[{pid}] {f}" for f in one["failed"]]
    try:
        count = _repo().delete_user(uid, user.get("clerk_user_id"))
        report["removed"].append(f"user record and its remaining documents ({count})")
    except Exception as e:
        report["failed"].append(f"user record: {type(e).__name__}: {str(e)[:120]}")
    _audit("delete_all_data", {**user, "email": email},
           {"removed": len(report["removed"]), "failed": len(report["failed"])})
    return jsonify({"success": not report["failed"], "report": report})
