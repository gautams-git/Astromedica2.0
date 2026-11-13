"""
chart_access.py — one chart per account, the saved chart in the main DB, and
the algorithm version.

Rules (login on, non-admin users):
  - An account can generate MAX_CHARTS charts in total: its first chart plus
    one correction ("I typed my birth details wrong"). Chatting is unlimited.
  - After a correction the previous chart is hidden from the user (kept for
    admins, chats included) and the new one becomes the active chart.
  - Accounts with no allowance record yet start fresh (existing users too).
  - Admins (ADMIN_CLERK_USER_IDS) and login-off local testing are unlimited.
  - Testers (tester_access.py) have no 2-chart limit; they have a daily cap of
    new Prokerala charts instead.

Firestore (per user, all under users/{internal_user_id}/):
  account/chart_allowance      {charts_used, active_patient_id, history[]}
  chart_sources/{patient_id}   raw Prokerala chart + dasha, profile, the planet
                               positions of generation day and algo_version.
                               Everything needed to rebuild the patient folder
                               and re-run the algorithms without Prokerala.

ALGO_VERSION: bump it whenever the astrology algorithms change. A returning
user whose saved results are older is re-run from the saved chart when they
open it (server._prepare_patient_chart).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from flask import g

import tester_access
from app_auth import AUTH_ENABLED, is_admin
from app_core.runtime import get_services

ALGO_VERSION = 1
MAX_CHARTS = 2

_ALLOWANCE = ("account", "chart_allowance")
_SOURCES = "chart_sources"


class ChartLimitError(Exception):
    """The account has used its first chart and its one correction."""


def _repo():
    return get_services().repository


def _uid() -> str:
    return g.user["internal_user_id"]


def _short_uid() -> str:
    return _uid()[:8] + "..."


def is_limited() -> bool:
    """Only signed-in users who are neither admins nor testers have the one-chart rule."""
    return AUTH_ENABLED and not is_admin() and not tester_access.is_tester()


# ── Allowance ──────────────────────────────────────────────────────────────

def get_allowance(uid: str | None = None, heal: bool = True) -> dict:
    """The account's allowance. Charts deleted since (admin page, or by hand in
    the Firestore console) no longer count and can't stay the active chart, so
    the user can create a chart again. heal=False only reports, never saves."""
    uid = uid or _uid()
    doc = _repo().get(uid, *_ALLOWANCE) or {}
    stored_used = int(doc.get("charts_used", 0))
    active = doc.get("active_patient_id")
    history = list(doc.get("history", []))

    def exists(pid):
        return bool(pid) and _repo().get(uid, "patients", pid) is not None

    live = [h for h in history if exists(h.get("patient_id"))]
    used = len(live) if history else stored_used
    live_active = active if exists(active) else None

    if heal and doc and (used != stored_used or live_active != active):
        _repo().save(uid, *_ALLOWANCE, {"charts_used": used, "active_patient_id": live_active, "history": live})
        print(f"[Allowance] user {uid[:8]}...: deleted chart(s) no longer count -> "
              f"{used}/{MAX_CHARTS} used, active chart {live_active or 'none'}")
    return {"charts_used": used, "active_patient_id": live_active, "history": live}


def chart_status() -> dict:
    """What the home and results pages need to decide what to show."""
    limited = is_limited()
    allowance = get_allowance() if limited else {"charts_used": 0, "active_patient_id": None}
    used = allowance["charts_used"]
    return {
        "limited": limited,
        "charts_used": used,
        "max_charts": MAX_CHARTS,
        "active_patient_id": allowance["active_patient_id"],
        "has_chart": bool(allowance["active_patient_id"]),
        "can_generate": (not limited) or used < MAX_CHARTS,
        # The "I typed my details wrong" option: only after the first chart.
        "can_correct": limited and used == 1 and bool(allowance["active_patient_id"]),
        "is_admin": is_admin(),          # the home page shows the Admin link
        "is_tester": tester_access.is_tester(),   # ... and the "My test charts" link
    }


def reset_allowance(uid: str) -> bool:
    """Admin page: the account can create charts again; nothing else changes."""
    return _repo().delete(uid, *_ALLOWANCE)


def check_can_generate() -> None:
    """Raises ChartLimitError before any Prokerala call when the allowance is used up."""
    if not is_limited():
        return
    used = get_allowance()["charts_used"]
    if used >= MAX_CHARTS:
        print(f"[Allowance] user {_short_uid()} has used {used}/{MAX_CHARTS} charts "
              f"-> Generate blocked (no Prokerala call)")
        raise ChartLimitError()
    kind = "first chart" if used == 0 else "correction (last chance)"
    print(f"[Allowance] user {_short_uid()} charts used {used}/{MAX_CHARTS} -> Generate allowed: {kind}")


def record_new_chart(patient_id: str) -> None:
    """Count a successful Generate; on a correction, hide the previous chart."""
    if not is_limited():
        return
    allowance = get_allowance()
    previous = allowance["active_patient_id"]
    used = allowance["charts_used"] + 1
    kind = "first" if used == 1 else "correction"
    history = allowance["history"] + [{
        "patient_id": patient_id,
        "kind": kind,
        "at": datetime.now(timezone.utc).isoformat(),
    }]
    _repo().save(_uid(), *_ALLOWANCE, {
        "charts_used": used,
        "active_patient_id": patient_id,
        "history": history,
    })
    print(f"[Allowance] user {_short_uid()} now {used}/{MAX_CHARTS} charts used "
          f"({kind}); active chart -> patient {patient_id}")

    if previous and previous != patient_id:
        doc = _repo().get(_uid(), "patients", previous)
        if doc:
            _repo().save(_uid(), "patients", previous, {**doc, "hidden": True,
                                                        "hidden_reason": "replaced by correction"})
            print(f"[Allowance] previous chart {previous} hidden from the user "
                  f"(kept for admins, chats included)")


# ── Saved chart (the source the algorithms run on) ─────────────────────────

def save_chart_source(patient_id: str, raw_chart: dict, raw_dasha: dict,
                      profile: dict, current_positions: dict) -> int:
    """Store what is needed to rebuild the patient folder. Returns its size in bytes."""
    record = {
        "patient_id": patient_id,
        # JSON text: Prokerala data can hold nested lists, which Firestore can't store.
        "raw_chart_json": json.dumps(raw_chart, ensure_ascii=False),
        "raw_dasha_json": json.dumps(raw_dasha, ensure_ascii=False),
        "current_positions_json": json.dumps(current_positions or {}, ensure_ascii=False),
        "profile": {k: v for k, v in (profile or {}).items() if v not in (None, "")},
        "algo_version": ALGO_VERSION,
    }
    _repo().save(_uid(), _SOURCES, patient_id, record)
    return sum(len(record[k]) for k in ("raw_chart_json", "raw_dasha_json", "current_positions_json"))


def load_chart_source(patient_id: str) -> dict | None:
    """The signed-in user's saved chart for this patient, or None."""
    doc = _repo().get(_uid(), _SOURCES, patient_id)
    if not doc or not doc.get("raw_chart_json"):
        return None
    return {
        "raw_chart": json.loads(doc["raw_chart_json"]),
        "raw_dasha": json.loads(doc.get("raw_dasha_json") or "{}"),
        "current_positions": json.loads(doc.get("current_positions_json") or "{}"),
        "profile": doc.get("profile") or {},
        "algo_version": int(doc.get("algo_version", 1)),
        "bytes": sum(len(doc.get(k) or "") for k in
                     ("raw_chart_json", "raw_dasha_json", "current_positions_json")),
        "path": f"users/{_short_uid()}/{_SOURCES}/{patient_id}",
    }


def mark_source_version(patient_id: str) -> None:
    doc = _repo().get(_uid(), _SOURCES, patient_id)
    if doc:
        _repo().save(_uid(), _SOURCES, patient_id, {**doc, "algo_version": ALGO_VERSION})
