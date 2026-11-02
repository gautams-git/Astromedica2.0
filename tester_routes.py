"""
tester_routes.py — the Tester button and the tester pages (see tester_access.py
for the rules).

  GET  /tester                        the "My test charts" page (no data in it)
  POST /tester/unlock                 password -> this account becomes a tester
  GET  /tester/api/status             is the account a tester, charts used today
  GET  /tester/api/charts             the tester's saved runs, newest first
  GET  /tester/api/feedback/<pid>     what the feedback panel on results.html needs
  POST /tester/api/feedback/<pid>     save the feedback for the latest run of <pid>

Everything except the page needs a signed-in account; the data routes answer
"Not found" to accounts that are not testers.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request, send_from_directory

import tester_access
from app_auth import check_patient_access, login_required
from app_core.errors import AuthorizationError
from app_core.validation import require_patient_id

tester_bp = Blueprint("tester", __name__)


def _require_tester():
    if not tester_access.is_tester():
        raise AuthorizationError()


def _status() -> dict:
    tester = tester_access.is_tester()
    used = tester_access.charts_today() if tester else 0
    cap = tester_access.daily_cap()
    return {"is_tester": tester, "charts_today": used, "daily_cap": cap,
            "remaining": max(0, cap - used) if tester else 0,
            "cap_applies": tester and tester_access.cap_applies()}


@tester_bp.get("/tester")
def tester_page():
    return send_from_directory(".", "tester.html")


@tester_bp.post("/tester/unlock")
@login_required(rate_limit="unlock")
def tester_unlock():
    if not tester_access.password():
        print("[Tester] Unlock refused - TESTER_PASSWORD is not set in .env")
        return jsonify({"success": False, "error": "Tester access is not set up on this server."}), 503
    if tester_access.is_tester():
        return jsonify({"success": True, **_status()})
    typed = (request.get_json(silent=True) or {}).get("password", "")
    if not tester_access.password_matches(typed):
        print("[Tester] Wrong password")
        return jsonify({"success": False, "error": "Wrong password."}), 403
    tester_access.make_tester()
    print("[Tester] Unlocked - this account is now a tester")
    return jsonify({"success": True, **_status()})


@tester_bp.get("/tester/api/status")
@login_required(rate_limit="read")
def tester_status():
    return jsonify({"success": True, **_status()})


@tester_bp.get("/tester/api/charts")
@login_required(rate_limit="read")
def tester_charts():
    _require_tester()
    return jsonify({"success": True, **_status(), "runs": tester_access.list_runs()})


@tester_bp.get("/tester/api/feedback/<patient_id>")
@login_required(rate_limit="read")
def tester_feedback_get(patient_id):
    require_patient_id(patient_id)
    if not tester_access.is_tester():
        return jsonify({"success": True, "is_tester": False})
    check_patient_access(patient_id)
    found = tester_access.latest_run(patient_id)
    if not found:
        return jsonify({"success": True, "is_tester": True, "run": None})
    run_id, run = found
    return jsonify({"success": True, "is_tester": True, "run": {
        "run_id": run_id,
        "top_organs": [item.get("organ") for item in run.get("combined_top") or []],
        # each organ with the diseases the Combined Algo lists for it (all can be rated)
        "organs": [{"organ": item.get("organ"), "diseases": item.get("diseases") or []}
                   for item in run.get("combined_top") or []],
        "feedback": run.get("feedback") or {},
    }})


@tester_bp.post("/tester/api/feedback/<patient_id>")
@login_required(rate_limit="read")
def tester_feedback_save(patient_id):
    _require_tester()
    check_patient_access(patient_id)
    saved = tester_access.save_feedback(patient_id, request.get_json(silent=True) or {})
    if not saved:
        return jsonify({"success": False, "error": "This chart was not made in tester mode, so it has no feedback panel."}), 404
    return jsonify({"success": True, **saved})
