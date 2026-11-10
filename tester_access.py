"""
tester_access.py — the Tester role: early-access people who test new charts and
the algorithms / chatbot, without the 2-chart limit.

How someone becomes a tester: they sign in, press "Tester" at the bottom of the
home page and type TESTER_PASSWORD (.env). That marks their account as a tester
(users/{id}/account/tester). An admin can remove the role on the admin page.

Rules for a tester:
  - No 2-chart limit; instead TESTER_DAILY_CAP (default 30) NEW charts per day,
    counted from midnight India time. A new chart costs 5 Prokerala calls; birth
    details already in charts_cache cost only 1 (today's planet positions, which
    every Generate fetches), so those are never counted or blocked. Admins have no cap.
  - Every chart run is kept in Firestore until an admin deletes it:
      users/{id}/tester_charts/{patient_id}_{timestamp}
        who/what was entered, the Combined Algo top organs, what each of the 5
        algorithms said, the chat, the final report, and the tester's feedback.
    Running the same birth details again later is a new run: the cached chart
    is reused (free) and all algorithms run again.
  - Tester chats are not queued for RL, so they never reach Abhishek's
    chat_logs training data (server.py sets session["skip_rl"]).
"""

from __future__ import annotations

import hmac
import json
import os
from datetime import datetime, timedelta, timezone

from flask import g

from algorithm_comparison import _display_organ, _normalize, _split_label
from app_auth import AUTH_ENABLED, is_admin
from app_core.runtime import get_services

IST = timezone(timedelta(hours=5, minutes=30))
RUNS = "tester_charts"
_ROLE = ("account", "tester")
_USAGE = ("account", "tester_usage")
MAX_JSON_CHARS = 600_000        # Firestore documents are limited to 1 MB

FEEDBACK_ORGAN_VALUES = ("correct", "partly", "wrong")
FEEDBACK_ANSWER_VALUES = ("yes", "partly", "no")


class TesterCapError(Exception):
    """The tester already made today's maximum of new charts."""


def _repo():
    return get_services().repository


def _uid() -> str:
    return g.user["internal_user_id"]


def daily_cap() -> int:
    try:
        return max(1, int(os.environ.get("TESTER_DAILY_CAP", "30")))
    except ValueError:
        return 30


def password() -> str:
    return os.environ.get("TESTER_PASSWORD", "").strip()


def password_matches(typed: str) -> bool:
    expected = password()
    return bool(expected) and hmac.compare_digest(str(typed or "").strip().encode(), expected.encode())


def today() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d")


# ── The role ───────────────────────────────────────────────────────────────

def has_role(uid: str) -> bool:
    return bool((_repo().get(uid, *_ROLE) or {}).get("is_tester"))


def is_tester() -> bool:
    """The signed-in account is a tester (looked up once per request)."""
    if not hasattr(g, "_is_tester"):
        g._is_tester = bool(getattr(g, "user", None)) and has_role(_uid())
    return g._is_tester


def make_tester(uid: str | None = None) -> None:
    uid = uid or _uid()
    _repo().save(uid, *_ROLE, {"is_tester": True, "since": datetime.now(timezone.utc).isoformat()})
    if hasattr(g, "_is_tester") and uid == _uid():
        g._is_tester = True


def remove_tester(uid: str) -> bool:
    return _repo().delete(uid, *_ROLE)


# ── The daily cap ──────────────────────────────────────────────────────────

def charts_today(uid: str | None = None) -> int:
    doc = _repo().get(uid or _uid(), *_USAGE) or {}
    return int(doc.get("charts", 0)) if doc.get("date") == today() else 0


def cap_applies() -> bool:
    """Admins have no cap. With login off everyone counts as admin, so the cap
    is then applied to testers anyway - that is what makes it testable locally."""
    return is_tester() and not (AUTH_ENABLED and is_admin())


def _would_call_prokerala(name, dob, birth_time, lat, lng) -> bool:
    from jataka_fetch import get_cached_data, get_request_key
    return get_cached_data(get_request_key(name, dob, birth_time, lat, lng)) is None


def check_daily_cap(name, dob, birth_time, lat, lng) -> None:
    """Before any Prokerala call: raises TesterCapError when a NEW chart would
    go over today's cap. A chart already in charts_cache is always allowed."""
    if not cap_applies():
        return
    used, cap = charts_today(), daily_cap()
    if used < cap:
        print(f"[Tester] {_uid()[:8]}... new charts today {used}/{cap} -> Generate allowed")
        return
    if not _would_call_prokerala(name, dob, birth_time, lat, lng):
        print(f"[Tester] {_uid()[:8]}... at {used}/{cap} today, but this chart is in charts_cache -> allowed (not a new chart)")
        return
    print(f"[Tester] {_uid()[:8]}... has made {used}/{cap} new charts today -> Generate blocked (no Prokerala call)")
    raise TesterCapError()


def _count_new_chart() -> int:
    used = charts_today() + 1
    _repo().save(_uid(), *_USAGE, {"date": today(), "charts": used})
    return used


# ── Saving a run ───────────────────────────────────────────────────────────

def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _algo_organs(section: dict) -> list[str]:
    """Organ names (same wording as the Combined Algo) one algorithm points to,
    best first, at most 3."""
    names: list[str] = []

    def add(value):
        name = _display_organ(value)
        if name and name not in names:
            names.append(name)

    for item in section.get("top_organs_and_systems") or []:
        for part in _split_label(item.get("label")):
            add(part)
    for item in section.get("top_organs") or []:
        add(item.get("organ") if isinstance(item, dict) else item)
    return names[:3]


def _algo_diseases(section: dict) -> list[str]:
    """Disease names one algorithm lists (strings or {"disease": ...} dicts), best first, at most 8."""
    names: list[str] = []

    def add(value):
        value = str(value or "").strip()
        if value and value not in names:
            names.append(value)

    for item in section.get("top_diseases") or []:
        add(item if isinstance(item, str) else (item.get("disease") or item.get("name")))
    # HitList keeps its diseases in the trail (Nakshatra rows), as the Combined Algo reads them.
    for item in section.get("top_organs_and_systems") or []:
        for step in item.get("trail") or []:
            if "nakshatra" in str(step.get("source", "")).lower():
                add(step.get("term"))
    return names[:8]


def _compact_algorithms(comparison: dict) -> dict:
    """Each algorithm's own output without its raw_output (which is huge)."""
    return {name: {k: v for k, v in section.items() if k != "raw_output"}
            for name, section in (comparison.get("algorithms") or {}).items()}


def record_run(patient_id: str, birth: dict, source: str) -> str:
    """Save this chart run under the tester's account. Returns the run id."""
    folder = os.path.join("patients", patient_id)
    comparison = _read_json(os.path.join(folder, "algorithm_comparison.json")) or {}
    combined = (comparison.get("Combined Algo Summary") or {}).get("top_organs") or []
    algorithms = _compact_algorithms(comparison)
    snapshot = json.dumps({"algorithms": algorithms, "combined": comparison.get("Combined Algo Summary") or {}},
                          ensure_ascii=False)
    if len(snapshot) > MAX_JSON_CHARS:
        snapshot = json.dumps({"combined": comparison.get("Combined Algo Summary") or {}}, ensure_ascii=False)

    processed = _read_json(os.path.join(folder, "processed_logic.json")) or {}
    now = datetime.now(IST)
    run_id = f"{patient_id}_{now:%Y%m%d%H%M%S}"
    fresh = source != "Local Cache"
    record = {
        "record_type": "tester_chart",
        "patient_id": patient_id,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "day": now.strftime("%Y-%m-%d"),
        "name": birth.get("name", ""),
        "dob": birth.get("dob", ""),
        "birth_time": birth.get("birth_time", ""),
        "birth_place": birth.get("birth_place", ""),
        "gender": birth.get("gender", ""),
        "chart_source": "prokerala" if fresh else "cache",
        "algo_version": processed.get("algo_version", 1),
        "combined_top": [{
            "organ": item.get("organ"),
            "score": item.get("consensus_score"),
            "support": item.get("algorithm_support_count"),
            "diseases": [d.get("disease") for d in (item.get("top_diseases") or [])][:3],
        } for item in combined[:3]],
        "algo_organs": {name: _algo_organs(section) for name, section in algorithms.items()},
        "algo_diseases": {name: _algo_diseases(section) for name, section in algorithms.items()},
        "snapshot_json": snapshot,
        "chat_turns": 0,
        "chat_completed": False,
    }
    _repo().save(_uid(), RUNS, run_id, record)
    used = _count_new_chart() if fresh else charts_today()
    print(f"[Tester] run {run_id} saved to {RUNS} ({'new Prokerala chart' if fresh else 'reused cached chart, algorithms re-run'}); "
          f"new charts today {used}/{daily_cap()}")
    return run_id


def latest_run(patient_id: str, uid: str | None = None) -> tuple[str, dict] | None:
    runs = _repo().list_docs(uid or _uid(), RUNS, patient_id)
    if not runs:
        return None
    return max(runs, key=lambda pair: str(pair[1].get("run_at") or ""))


def sync_chat(patient_id: str) -> None:
    """Copy the patient's latest chat (and final report, once it exists) into the
    tester's latest run of that chart, so it can be reviewed later."""
    found = latest_run(patient_id)
    if not found:
        return
    run_id, run = found
    folder = os.path.join("patients", patient_id)
    log = _read_json(os.path.join(folder, "chat_log.json")) or {}
    from combine_output import _latest_session_turns
    turns = _latest_session_turns(log)
    chat = {
        "turns": turns,
        "completed": bool(log.get("completed")),
        "final_assessment": log.get("final_assessment"),
        "chat_plan_source": log.get("chat_plan_source"),
    }
    chat_json = json.dumps(chat, ensure_ascii=False)
    update = {**run, "chat_turns": len(turns), "chat_completed": bool(log.get("completed")),
              "chat_json": chat_json[:MAX_JSON_CHARS] if len(chat_json) <= MAX_JSON_CHARS else
              json.dumps({**chat, "turns": turns[-40:]}, ensure_ascii=False)[:MAX_JSON_CHARS]}
    final = _read_json(os.path.join(folder, "final_output.json"))
    if final and log.get("completed"):
        final_json = json.dumps(final, ensure_ascii=False)
        if len(final_json) <= MAX_JSON_CHARS:
            update["final_output_json"] = final_json
    _repo().save(_uid(), RUNS, run_id, update)
    print(f"[Tester] chat of run {run_id} saved ({len(turns)} turns, completed={update['chat_completed']})")


# ── Feedback ───────────────────────────────────────────────────────────────

def disease_key(organ, disease) -> str:
    return f"{organ}::{disease}"


def clean_feedback(raw: dict, run: dict) -> dict:
    """Only known organs and fixed answers are kept; free text is trimmed."""
    allowed = {item["organ"]: item["organ"] for item in run.get("combined_top") or [] if item.get("organ")}
    organs = {}
    for organ, verdict in (raw.get("organs") or {}).items():
        if organ in allowed and verdict in FEEDBACK_ORGAN_VALUES:
            organs[organ] = verdict
    # Diseases are rated per organ: the key is disease_key(organ, disease).
    allowed_diseases = {disease_key(item["organ"], d) for item in run.get("combined_top") or []
                        if item.get("organ") for d in item.get("diseases") or []}
    diseases = {}
    for key, verdict in (raw.get("diseases") or {}).items():
        if key in allowed_diseases and verdict in FEEDBACK_ORGAN_VALUES:
            diseases[key] = verdict

    def answer(key):
        value = str(raw.get(key) or "").strip().lower()
        return value if value in FEEDBACK_ANSWER_VALUES else ""

    return {
        "organs": organs,
        "diseases": diseases,
        "missed_organ": str(raw.get("missed_organ") or "").strip()[:80],
        "chat_relevant": answer("chat_relevant"),
        "plan_matches": answer("plan_matches"),
        "note": str(raw.get("note") or "").strip()[:1000],
        "at": datetime.now(timezone.utc).isoformat(),
    }


def save_feedback(patient_id: str, raw: dict) -> dict | None:
    found = latest_run(patient_id)
    if not found:
        return None
    run_id, run = found
    feedback = clean_feedback(raw, run)
    _repo().save(_uid(), RUNS, run_id, {**run, "feedback": feedback})
    print(f"[Tester] feedback saved for run {run_id}: {feedback['organs']} "
          f"missed={feedback['missed_organ'] or '-'} chat_relevant={feedback['chat_relevant'] or '-'}")
    return {"run_id": run_id, "feedback": feedback}


def rating_of(run: dict) -> str:
    """not_rated | correct | partly | wrong, from the organ verdicts."""
    verdicts = list((run.get("feedback") or {}).get("organs", {}).values())   # the rating follows the organs
    if not verdicts:
        return "not_rated"
    if "correct" in verdicts:
        return "correct"
    if "partly" in verdicts:
        return "partly"
    return "wrong"


def run_summary(run_id: str, run: dict) -> dict:
    """One history row (no big JSON)."""
    return {
        "run_id": run_id,
        "patient_id": run.get("patient_id"),
        "run_at": run.get("run_at"),
        "day": run.get("day"),
        "name": run.get("name", ""),
        "dob": run.get("dob", ""),
        "birth_time": run.get("birth_time", ""),
        "birth_place": run.get("birth_place", ""),
        "chart_source": run.get("chart_source"),
        "top_organs": [item.get("organ") for item in run.get("combined_top") or []],
        "rating": rating_of(run),
        "chat_turns": run.get("chat_turns", 0),
        "chat_completed": bool(run.get("chat_completed")),
        "has_feedback": bool(run.get("feedback")),
        "chat_relevant": (run.get("feedback") or {}).get("chat_relevant", ""),
    }


def list_runs(uid: str | None = None) -> list[dict]:
    rows = [run_summary(run_id, run) for run_id, run in _repo().list_docs(uid or _uid(), RUNS)]
    rows.sort(key=lambda row: str(row.get("run_at") or ""), reverse=True)
    return rows


# ── Accuracy summary for the admin page ────────────────────────────────────

def accuracy_summary() -> dict:
    """How often each algorithm was right, from every tester's rated runs.
    Truth for a run = the organs the tester marked Correct + the organ they say
    we missed. An algorithm 'hits' when its top-3 holds a true organ, and 'top-1'
    when its first organ is one."""
    algo_stats: dict[str, dict] = {}
    totals = {"testers": 0, "runs": 0, "rated": 0, "organ_correct": 0, "organ_partly": 0, "organ_wrong": 0,
              "disease_correct": 0, "disease_partly": 0, "disease_wrong": 0,
              "chat_rated": 0, "chat_relevant_yes": 0, "chat_relevant_partly": 0, "chat_relevant_no": 0}
    for user in _repo().list_users():
        uid = user["internal_user_id"]
        runs = _repo().list_docs(uid, RUNS)
        if not runs and not has_role(uid):
            continue
        totals["testers"] += 1
        for _run_id, run in runs:
            totals["runs"] += 1
            feedback = run.get("feedback") or {}
            if not feedback:
                continue
            verdicts = feedback.get("organs") or {}
            if verdicts:
                totals["rated"] += 1
                for verdict in verdicts.values():
                    totals["organ_" + verdict] = totals.get("organ_" + verdict, 0) + 1
            disease_verdicts = feedback.get("diseases") or {}
            for verdict in disease_verdicts.values():
                totals["disease_" + verdict] += 1
            if disease_verdicts:
                correct_diseases = {_normalize(key.split("::", 1)[-1])
                                    for key, v in disease_verdicts.items() if v == "correct"}
                lists = dict(run.get("algo_diseases") or {})
                lists["Combined Algo"] = [d for item in run.get("combined_top") or [] for d in item.get("diseases") or []]
                for name, listed in lists.items():
                    stat = algo_stats.setdefault(name, {"algorithm": name, "rated": 0, "top1": 0, "top3": 0})
                    stat["disease_rated"] = stat.get("disease_rated", 0) + 1
                    if any(_normalize(d) in correct_diseases for d in listed):
                        stat["disease_hit"] = stat.get("disease_hit", 0) + 1
            if feedback.get("chat_relevant"):
                totals["chat_rated"] += 1
                totals["chat_relevant_" + feedback["chat_relevant"]] += 1
            truth = {_normalize(o) for o, v in verdicts.items() if v == "correct"}
            if feedback.get("missed_organ"):
                truth.add(_normalize(_display_organ(feedback["missed_organ"])))
            if not truth:
                continue
            for name, organs in (run.get("algo_organs") or {}).items():
                stat = algo_stats.setdefault(name, {"algorithm": name, "rated": 0, "top1": 0, "top3": 0})
                stat["rated"] += 1
                normalised = [_normalize(o) for o in organs]
                if normalised and normalised[0] in truth:
                    stat["top1"] += 1
                if any(o in truth for o in normalised):
                    stat["top3"] += 1
            combined = [_normalize(item.get("organ")) for item in run.get("combined_top") or []]
            stat = algo_stats.setdefault("Combined Algo", {"algorithm": "Combined Algo", "rated": 0, "top1": 0, "top3": 0})
            stat["rated"] += 1
            if combined and combined[0] in truth:
                stat["top1"] += 1
            if any(o in truth for o in combined):
                stat["top3"] += 1
    ordered = sorted(algo_stats.values(), key=lambda s: (s["algorithm"] != "Combined Algo", -s["top3"]))
    return {"totals": totals, "algorithms": ordered}
