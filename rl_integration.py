"""
rl_integration.py — runs Abhishek's Medical Report Enhancer (rl_enhancer/)
inside our Flask app, in place of his FastAPI main.py.

- Same URLs as his app (/api/upload, /api/jobs/..., /api/training/...,
  /healthz), so his page works unchanged; it's served at /rl.
- Same background worker: one thread takes queued jobs and runs his
  parse_input -> parse_chat_log_list -> build_output.
- Our addition: queue_patient_rl(patient_id) queues a patient's
  "For RL Report.json" when their chat finishes; the result is saved as
  patients/<id>/RL Output.json.
- Firestore (his, same project as the app: FIRESTORE_PROJECT_ID): stores his
  lookup tables and every chat log RL receives, so each RL run uses the
  patient's whole chat history. Without Firestore his code falls back to
  built-in tables and inline chats only.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify, render_template_string, request, send_file, send_from_directory

from rl_enhancer import config_loader
from rl_enhancer.chat_log_parser import parse_chat_log_list, persist_and_merge_chat_logs
from rl_enhancer.enhancer import build_output
from rl_enhancer.jobs import STORE
from rl_enhancer.parser import ParseError, parse_input, to_preview

logger = logging.getLogger("rl_integration")

BASE_DIR = Path(__file__).resolve().parent
RL_DIR = BASE_DIR / "rl_enhancer"
PATIENTS_DIR = BASE_DIR / "patients"
UPLOAD_DIR = RL_DIR / "data" / "uploads"
OUTPUT_DIR = RL_DIR / "data" / "outputs"
RL_REPORT_FILE = "For RL Report.json"
RL_OUTPUT_FILE = "RL Output.json"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
BUILD_ID = f"b{int(time.time() * 1000):x}"

rl_bp = Blueprint("rl", __name__)

_patient_jobs: dict[str, str] = {}  # job_id -> patient_id
_start_lock = threading.Lock()
_started = False


def _detail(message: str, status: int):
    # His page reads errors from "detail" (FastAPI's error shape).
    return jsonify({"detail": message}), status


def _training():
    # Imported lazily: training needs torch, and the rest of the app (and his
    # enhancer's library fallback) must keep working without it.
    # training2 = V2, which also learns from the chat logs in Firestore.
    # Abhishek's V1 (training.py, library only) is kept; to switch back,
    # import it here and in enhancer.symptoms_for_disease.
    from rl_enhancer import training2
    return training2


def _model_name() -> str:
    try:
        status = _training().training_status()
    except Exception:
        return "symptom library (torch not available)"
    if status.get("ppo_trained"):
        return "CustomANN (PPO-trained)"
    if status.get("pretrained"):
        return "CustomANN (pretrained)"
    return "symptom library (no trained model)"


# ── Worker (replaces his main.py _worker_loop) ─────────────────────────────

def _inline_chat_logs(wrapper: dict) -> list:
    chat_log = wrapper.get("chat_log") or []
    if isinstance(chat_log, dict):
        chat_log = [chat_log]
    return [d for d in chat_log if isinstance(d, dict)]


def _chat_key(doc: dict):
    return doc.get("session_id") or doc.get("started_at")


def _log_db_save(patient_id, inline: list) -> None:
    # His persist step only prints "Upserting" and swallows errors, so check
    # Firestore afterwards and say plainly what was (or wasn't) stored.
    if not patient_id or not inline:
        return
    if config_loader.status().get("source") != "firestore":
        print(f"[Firestore] NOT saved - Firestore not connected; chat for patient {patient_id} used for this run only")
        return
    from rl_enhancer import config_db
    stored = config_loader.get_chat_logs_for_patient(patient_id)
    stored_keys = {_chat_key(d) for d in stored}
    for doc in inline:
        key = _chat_key(doc)
        if key in stored_keys:
            print(f"[Firestore] Saved chat {key} for patient {patient_id} "
                  f"({len(doc.get('turns') or [])} turns, completed={bool(doc.get('completed'))}) "
                  f"-> {config_db.COLL_CHAT_LOGS} | patient {patient_id} now has {len(stored)} chat(s) stored")
        else:
            print(f"[Firestore] NOT saved - chat {key} for patient {patient_id} is not in Firestore after saving")


def _read_input(path: str):
    parsed, wrapper = parse_input(Path(path).read_bytes())
    inline = _inline_chat_logs(wrapper)
    # His DB step: saves the chats sent here into Firestore, and returns them
    # together with every earlier chat stored for this patient — so RL works
    # from the patient's whole history. Without Firestore it returns just the
    # inline chats.
    combined = persist_and_merge_chat_logs(wrapper.get("patient_id"), inline)
    _log_db_save(wrapper.get("patient_id"), inline)
    chat_summary = parse_chat_log_list(combined) if combined else None
    counts = {"chat_logs_used": len(combined), "chat_logs_inline": len(inline)}
    return parsed, wrapper, chat_summary, counts


def _save_patient_output(patient_id: str, wrapper: dict, output: list, counts: dict) -> Path:
    session = (_inline_chat_logs(wrapper) or [{}])[0]
    out_path = PATIENTS_DIR / patient_id / RL_OUTPUT_FILE
    out_path.write_text(json.dumps({
        "patient_id": patient_id,
        "generated_at": datetime.now().isoformat(),
        "model": _model_name(),
        "lookup_tables": config_loader.status().get("source"),
        "chat_session_id": session.get("session_id"),
        "chat_started_at": session.get("started_at"),
        "chat_completed": bool(session.get("completed")),
        # chats RL used: this one plus earlier ones stored in Firestore
        "chat_logs_used": counts["chat_logs_used"],
        "enhanced": output,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _run_job(job) -> None:
    patient_id = _patient_jobs.get(job.id)
    STORE.update(job.id, status="running", progress=0.02, message="Reading input…")
    if patient_id:
        print(f"[RL] Job {job.id} running for patient {patient_id}...")

    parsed, wrapper, chat_summary, counts = _read_input(job.input_path)
    total = sum(len(ds) for _, ds in parsed)
    extra = (f" (chat logs used: {counts['chat_logs_used']}, inline: {counts['chat_logs_inline']})"
             if counts["chat_logs_used"] else "")

    def progress_cb(p, msg):
        STORE.update(job.id, progress=min(max(p, 0.0), 0.99), message=msg)

    output = build_output(parsed, progress_cb=progress_cb, chat_summary=chat_summary)

    if patient_id:
        out_path = _save_patient_output(patient_id, wrapper, output, counts)
    else:
        out_path = OUTPUT_DIR / f"{job.id}_enhanced.json"
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    STORE.update(job.id, status="done", progress=1.0,
                 message=f"Done — {len(parsed)} organ(s), {total} disease(s){extra}",
                 output_path=str(out_path), result=output)
    if patient_id:
        print(f"[RL] Job {job.id} done ({_model_name()}) -> patients/{patient_id}/{RL_OUTPUT_FILE}, "
              f"{len(parsed)} organs / {total} diseases, chat logs used: {counts['chat_logs_used']}")


def _worker_loop() -> None:
    logger.info("RL enhancement worker started.")
    while True:
        job = STORE.next_queued()
        if job is None or job.input_path is None:
            # Training jobs have no input_path; their own thread moves them on.
            time.sleep(0.4)
            continue
        try:
            _run_job(job)
        except Exception as e:
            logger.exception("RL job %s failed", job.id)
            STORE.update(job.id, status="error", message="Enhancement failed",
                         error=f"{type(e).__name__}: {e}")
            if job.id in _patient_jobs:
                print(f"[RL] Job {job.id} failed for patient {_patient_jobs[job.id]}: {type(e).__name__}: {e}")


_RL_COLLECTIONS = ("symptom_library", "organ_aliases", "branch_hints", "chat_maps",
                   "stop_words", "organ_branch_families", "chat_logs")


def _log_db_contents(source: str) -> None:
    from rl_enhancer import config_db
    project = config_db.project_id() or "default project"
    if source != "firestore":
        print(f"[Firestore] RL not connected ({project}) - using built-in tables; chats won't be saved for RL")
        return
    try:
        db = config_db.get_db()
        counts = []
        for name in _RL_COLLECTIONS:
            docs = [d for d in db.collection(name).stream(**config_db.CALL)
                    if (d.to_dict() or {}).get("kind") != "meta"]
            counts.append(f"{name} ({len(docs)})")
        print(f"[Firestore] RL connected to {project}: {', '.join(counts)}")
    except Exception as e:
        print(f"[Firestore] RL connected, but could not count collections: {e}")


# Logger names used in his rl_enhancer/ files.
_HIS_LOGGERS = ("training", "config_db", "config_loader", "chat_log_parser", "medical_api", "rl_integration")


def _show_his_logs() -> None:
    """His main.py turned on INFO logging (logging.basicConfig); our Flask
    server doesn't, so his info lines (e.g. 'Corpus built: ... chat-log')
    were hidden. Turn them on for his loggers only, in his format."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    for name in _HIS_LOGGERS:
        log = logging.getLogger(name)
        log.setLevel(logging.INFO)
        log.addHandler(handler)
        log.propagate = False


def init_rl() -> None:
    """Start the worker and load his trained model — once per process (what his
    main.py lifespan did at startup)."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    _show_his_logs()
    for d in (UPLOAD_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # His lookup tables (symptom library, organ maps...) from Firestore, with
    # his built-in values if Firestore can't be reached; refreshed every 30 min.
    cfg = config_loader.reload_all()
    print(f"[RL] Lookup tables loaded from {cfg['source']} "
          f"(symptom library: {cfg['symptom_library_size']} diseases)")
    _log_db_contents(cfg["source"])
    try:
        from rl_enhancer import config_db
        config_db.ensure_chat_log_indexes()
    except Exception as e:
        print(f"[RL] Chat-log index not created: {e}")
    config_loader.start_refresher(interval_seconds=config_loader.REFRESH_SECONDS)

    try:
        _training().load_existing_weights()
    except Exception as e:
        print(f"[RL] Trained model not loaded ({type(e).__name__}: {e}) — using symptom library")
    threading.Thread(target=_worker_loop, name="rl-worker", daemon=True).start()
    print(f"[RL] Enhancer ready — model: {_model_name()}, training: V2 (learns from chat logs)")

    if SCHED_ENABLED:
        _start_scheduler()
        print(f"[RL Training] Scheduler on - first cycle in {SCHED_LEADIN_SECONDS // 60} min, "
              f"then every {SCHED_INTERVAL_SECONDS // 60} min (set TRAIN_INTERVAL_SECONDS / "
              f"TRAIN_SCHEDULER_ENABLED=false to change)")
    else:
        print("[RL Training] Scheduler off (TRAIN_SCHEDULER_ENABLED=false)")


def queue_patient_rl(patient_id: str) -> str | None:
    """Queue a patient's For RL Report.json for enhancement (called when their
    chat finishes). Returns the job id."""
    report = PATIENTS_DIR / patient_id / RL_REPORT_FILE
    if not report.exists():
        print(f"[RL] No {RL_REPORT_FILE} for patient {patient_id} — RL not run")
        return None
    # input_path is set last so the worker never picks the job up before it
    # knows which patient it belongs to.
    job = STORE.create(input_path=None)
    _patient_jobs[job.id] = patient_id
    STORE.update(job.id, input_path=str(report), message=f"Queued — patient {patient_id}")
    print(f"[RL] Chat completed for {patient_id} -> RL job {job.id} queued")
    return job.id


# ── RL Report on results.html (dev comparison with the LLM report) ─────────

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _norm(text) -> str:
    return " ".join(str(text or "").lower().replace("_", " ").split())


def _compare_with_llm(folder: Path, enhanced: list) -> dict:
    """LLM confirmed organ + diseases (final_output.json) vs the RL organs and
    the diseases RL lists under that same organ."""
    cf = _read_json(folder / "final_output.json").get("confirmed_findings") or {}
    rl = {}
    for organ_block in enhanced or []:
        for organ, diseases in organ_block.items():
            rl[organ] = [name for d in diseases for name in d if name != "evidence"]
    llm_organ = cf.get("organ") if cf.get("confirmed") else None
    rl_match = next((o for o in rl if _norm(o) == _norm(llm_organ)), None) if llm_organ else None
    llm_diseases = [d.get("name") for d in cf.get("diseases") or [] if d.get("name")]
    rl_diseases = rl.get(rl_match, [])
    rl_keys = {_norm(d) for d in rl_diseases}
    llm_keys = {_norm(d) for d in llm_diseases}
    return {
        "llm_organ": llm_organ,
        "rl_organs": list(rl),
        "organ_agrees": rl_match is not None,
        "diseases_in_both": [d for d in llm_diseases if _norm(d) in rl_keys],
        "only_llm": [d for d in llm_diseases if _norm(d) not in rl_keys],
        "only_rl": [d for d in rl_diseases if _norm(d) not in llm_keys],
    }


def patient_rl_report(patient_id: str) -> dict:
    """status: ready | pending | failed | unavailable. 'ready' only when
    RL Output.json is for the patient's latest chat, not an older one."""
    folder = PATIENTS_DIR / patient_id
    chat = (_inline_chat_logs(_read_json(folder / RL_REPORT_FILE)) or [{}])[0]
    output = _read_json(folder / RL_OUTPUT_FILE)
    current = bool(output) and output.get("chat_started_at") == chat.get("started_at")

    if current:
        return {"status": "ready", "report": output,
                "comparison": _compare_with_llm(folder, output.get("enhanced"))}

    jobs = [STORE.get(job_id) for job_id, pid in list(_patient_jobs.items()) if pid == patient_id]
    latest = max((j for j in jobs if j), key=lambda j: j.created_at, default=None)
    if latest and latest.status in ("queued", "running"):
        return {"status": "pending", "message": latest.message}
    if latest and latest.status == "error":
        return {"status": "failed", "message": latest.error or latest.message}
    return {"status": "unavailable",
            "message": "No RL report for this chat (RL did not run - e.g. the server restarted)."}


# ── Training (V2) — console lines show what it learns from ────────────────

def _chat_training_data() -> dict:
    """What V2 will learn from: stored chat logs + the symptom library."""
    t = _training()
    chat_logs = t._iter_all_chat_logs()
    chat_diseases = [disease for disease, _ in t._chat_log_corpus()] if chat_logs else []
    return {"chat_logs": len(chat_logs), "chat_diseases": chat_diseases,
            "library_diseases": len(config_loader.SYMPTOM_LIBRARY)}


def _watch_training_job(label: str, job_id: str) -> None:
    while True:
        job = STORE.get(job_id)
        if job is None or job.status in ("done", "error", "cancelled"):
            break
        time.sleep(2)
    if job is None:
        return
    if job.status == "done":
        print(f"[RL Training] {label} done - {job.message}")
    else:
        print(f"[RL Training] {label} {job.status} - {job.error or job.message}")


def start_training(kind: str, epochs: int | None = None) -> tuple[str, int]:
    """Start pretraining or PPO (V2). Raises RuntimeError if training is busy."""
    t = _training()
    start, default_epochs = {
        "pretrain": (t.start_pretraining, t.DEFAULT_PRETRAIN_EPOCHS),
        "ppo": (t.start_ppo, t.DEFAULT_PPO_EPOCHS),
    }[kind]
    epochs = epochs or default_epochs
    data = _chat_training_data()
    job_id = start(epochs=epochs)

    label = "Pretraining" if kind == "pretrain" else "PPO"
    chats = data["chat_diseases"]
    shown = ", ".join(chats[:5]) + (" ..." if len(chats) > 5 else "")
    print(f"[RL Training] {label} started ({epochs} epochs) - learning from "
          f"{data['chat_logs']} chat log(s) in Firestore "
          f"({len(chats)} disease(s) from patient chats{': ' + shown if chats else ''}) "
          f"+ {data['library_diseases']} library diseases")
    threading.Thread(target=_watch_training_job, args=(label, job_id),
                     name=f"watch-{kind}", daemon=True).start()
    return job_id, epochs


# ── His training scheduler (from his main.py) ─────────────────────────────
# Runs pretraining then PPO on a timer. His default is every 360 s (6 min) —
# a full retrain each time, too heavy next to live chats — so ours defaults
# to hourly, first run 5 min after start. Override with the same env vars.

SCHED_INTERVAL_SECONDS = int(os.getenv("TRAIN_INTERVAL_SECONDS", "3600"))
SCHED_LEADIN_SECONDS = int(os.getenv("TRAIN_LEADIN_SECONDS", "300"))
SCHED_ENABLED = os.getenv("TRAIN_SCHEDULER_ENABLED", "true").lower() == "true"

_sched_stop = threading.Event()
_sched_lock = threading.RLock()
_sched_thread: threading.Thread | None = None
_sched_state: dict = {
    "enabled": SCHED_ENABLED,
    "interval_seconds": SCHED_INTERVAL_SECONDS,
    "leadin_seconds": SCHED_LEADIN_SECONDS,
    "runs_completed": 0,
    "runs_skipped": 0,
    "last_started_at": None,
    "last_finished_at": None,
    "last_outcome": None,
    "next_run_estimate": None,
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wait_for_job(job_id: str, timeout_seconds: float = 3600.0) -> str:
    started = time.time()
    while time.time() - started < timeout_seconds:
        if _sched_stop.is_set():
            return "aborted"
        job = STORE.get(job_id)
        if job is None:
            return "missing"
        if job.status in ("done", "error", "cancelled"):
            return job.status
        time.sleep(2.0)
    return "timeout"


def _run_one_training_cycle() -> dict:
    # His main.py checked V1's training.STATE here while starting V2 jobs, so
    # it never saw V2 busy; check the module that actually trains.
    t = _training()
    with t.STATE.lock:
        if t.STATE.busy:
            return {"outcome": "skipped", "reason": "training_in_progress"}

    outcome = {"outcome": "started", "started_at": _iso_now()}
    try:
        pre_id, _ = start_training("pretrain")
    except RuntimeError as e:
        return {**outcome, "outcome": "skipped", "reason": f"pretrain_start_failed: {e}"}
    outcome["pretrain_job_id"] = pre_id
    outcome["pretrain_status"] = _wait_for_job(pre_id)
    if outcome["pretrain_status"] != "done":
        return {**outcome, "outcome": "pretrain_failed"}

    try:
        ppo_id, _ = start_training("ppo")
    except RuntimeError as e:
        return {**outcome, "outcome": "ppo_start_failed", "reason": str(e)}
    outcome["ppo_job_id"] = ppo_id
    outcome["ppo_status"] = _wait_for_job(ppo_id)
    outcome["outcome"] = "done" if outcome["ppo_status"] == "done" else "ppo_failed"
    outcome["finished_at"] = _iso_now()
    return outcome


def _record_cycle(result: dict) -> None:
    with _sched_lock:
        _sched_state["last_finished_at"] = _iso_now()
        _sched_state["last_outcome"] = result
        if result.get("outcome") == "skipped":
            _sched_state["runs_skipped"] += 1
        else:
            _sched_state["runs_completed"] += 1
    print(f"[RL Training] Scheduled cycle {result.get('outcome')}"
          + (f" ({result['reason']})" if result.get("reason") else ""))


def _scheduler_loop() -> None:
    if _sched_stop.wait(SCHED_LEADIN_SECONDS):
        return
    while not _sched_stop.is_set():
        with _sched_lock:
            enabled = _sched_state["enabled"]
        if enabled:
            with _sched_lock:
                _sched_state["last_started_at"] = _iso_now()
            print("[RL Training] Scheduled cycle starting (pretrain, then PPO)")
            try:
                result = _run_one_training_cycle()
            except Exception as e:
                logger.exception("Scheduled training cycle failed")
                result = {"outcome": "error", "error": str(e)}
            _record_cycle(result)
        with _sched_lock:
            interval = _sched_state["interval_seconds"]
            _sched_state["next_run_estimate"] = datetime.fromtimestamp(
                time.time() + interval, tz=timezone.utc).replace(microsecond=0).isoformat()
        if _sched_stop.wait(interval):
            return


def _start_scheduler() -> None:
    global _sched_thread
    with _sched_lock:
        if _sched_thread is not None and _sched_thread.is_alive():
            return
        _sched_stop.clear()
        _sched_thread = threading.Thread(target=_scheduler_loop, name="train-scheduler", daemon=True)
        _sched_thread.start()


# ── His page ────────────────────────────────────────────────────────────────

@rl_bp.get("/rl")
def rl_index():
    html = (RL_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    # His page links its assets at /static/; ours are served under /rl/static/.
    html = html.replace("/static/", "/rl/static/")
    return render_template_string(html, build_id=BUILD_ID[-8:])


@rl_bp.get("/rl/static/<path:path>")
def rl_static(path):
    return send_from_directory(RL_DIR / "static", path, max_age=0)


# ── His API (same URLs and response shapes as his main.py) ─────────────────

@rl_bp.post("/api/upload")
def rl_upload():
    upload = request.files.get("file")
    if upload is not None:
        if not (upload.filename or "").lower().endswith(".json"):
            return _detail("Only .json files are accepted.", 400)
        raw = upload.read()
    elif "application/json" in (request.content_type or "").lower():
        raw = request.get_data()
    else:
        return _detail("Send application/json or a .json file.", 415)

    if not raw:
        return _detail("Payload is empty.", 400)
    if len(raw) > MAX_UPLOAD_BYTES:
        return _detail("Payload too large.", 413)
    try:
        parsed, wrapper = parse_input(raw)
    except ParseError as e:
        return _detail(str(e), 400)

    chat_log = wrapper.get("chat_log") or []
    chat_log = [chat_log] if isinstance(chat_log, dict) else chat_log
    chat_summary = None
    if chat_log:
        try:
            chat_summary = parse_chat_log_list(chat_log)
        except Exception as e:
            logger.warning("chat_log parse failed: %s", e)

    src_path = UPLOAD_DIR / f"{time.time_ns():x}.json"
    src_path.write_bytes(raw)
    job = STORE.create(input_path=str(src_path))

    total = sum(len(ds) for _, ds in parsed)
    resp = {
        "job_id": job.id,
        "status": job.status,
        "message": f"Queued — {len(parsed)} organ(s), {total} disease(s)"
                   + (" (chat log attached)" if chat_summary else ""),
        "preview": to_preview(parsed),
    }
    if chat_summary:
        resp["chat_summary_preview"] = {
            "patient_id": chat_summary.get("patient_id"),
            "organs": chat_summary.get("organs"),
            "diseases_detected": [d["canonical"] for d in chat_summary.get("diseases", [])],
            "endorsed_symptoms": chat_summary.get("endorsed_symptoms", []),
        }
    return jsonify(resp)


@rl_bp.get("/api/jobs/<job_id>")
def rl_job_status(job_id):
    job = STORE.get(job_id)
    if not job:
        return _detail("Job not found.", 404)
    d = job.to_dict()
    d.pop("result", None)
    return jsonify(d)


@rl_bp.get("/api/jobs/<job_id>/result")
def rl_job_result(job_id):
    job = STORE.get(job_id)
    if not job:
        return _detail("Job not found.", 404)
    if job.status != "done":
        return _detail(f"Not finished ({job.status}).", 409)
    return jsonify({"job_id": job.id, "result": job.result})


@rl_bp.get("/api/jobs/<job_id>/download")
def rl_job_download(job_id):
    job = STORE.get(job_id)
    if not job:
        return _detail("Job not found.", 404)
    if job.status != "done":
        return _detail(f"Not finished ({job.status}).", 409)
    if not job.output_path or not Path(job.output_path).exists():
        return _detail("Output missing.", 410)
    return send_file(job.output_path, mimetype="application/json",
                     as_attachment=True, download_name=f"enhanced_{job_id}.json")


@rl_bp.get("/api/jobs")
def rl_list_jobs():
    limit = request.args.get("limit", 50, type=int)
    return jsonify([j.to_dict() for j in STORE.list(limit=limit)])


@rl_bp.post("/api/jobs/<job_id>/cancel")
def rl_cancel_job(job_id):
    job = STORE.cancel(job_id)
    if not job:
        return _detail("Job not found.", 404)
    return jsonify(job.to_dict())


@rl_bp.post("/api/chat_logs")
def rl_store_chat_log():
    doc = request.get_json(silent=True)
    if not isinstance(doc, dict) or not doc.get("patient_id") or not doc.get("session_id"):
        return _detail("Body must be a JSON object with patient_id and session_id.", 422)
    doc = {k: v for k, v in doc.items() if v is not None}
    try:
        config_loader.upsert_chat_log(doc)
    except Exception as e:
        return _detail(f"Could not store chat log: {e}", 500)
    return jsonify({"ok": True, "patient_id": doc["patient_id"], "session_id": doc["session_id"]})


@rl_bp.post("/api/chat_logs/batch")
def rl_store_chat_logs():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("chat_logs"), list):
        return _detail("Body must be {\"chat_logs\": [...]}.", 422)
    stored, errors = 0, []
    for doc in body["chat_logs"]:
        try:
            config_loader.upsert_chat_log(doc)
            stored += 1
        except Exception as e:
            errors.append({"session_id": (doc or {}).get("session_id"), "error": str(e)})
    return jsonify({"stored": stored, "errors": errors})


@rl_bp.get("/api/chat_logs/<patient_id>")
def rl_list_chat_logs(patient_id):
    limit = request.args.get("limit", 50, type=int)
    docs = config_loader.get_chat_logs_for_patient(patient_id, limit=limit)
    return jsonify({
        "patient_id": patient_id,
        "count": len(docs),
        "sessions": [
            {"session_id": d.get("session_id"), "started_at": d.get("started_at"),
             "completed": d.get("completed"), "turns": len(d.get("turns") or [])}
            for d in docs
        ],
    })


@rl_bp.delete("/api/chat_logs/<patient_id>/<session_id>")
def rl_delete_chat_log(patient_id, session_id):
    try:
        from rl_enhancer import config_db
        return jsonify({"deleted": config_db.delete_chat_log(patient_id, session_id)})
    except Exception as e:
        return _detail(f"Could not delete chat log: {e}", 500)


@rl_bp.get("/api/config/status")
def rl_config_status():
    return jsonify(config_loader.status())


@rl_bp.post("/api/config/reload")
def rl_config_reload():
    return jsonify(config_loader.reload_all())


@rl_bp.get("/api/training/status")
def rl_training_status():
    try:
        return jsonify(_training().training_status())
    except Exception as e:
        return _detail(f"Training unavailable: {type(e).__name__}: {e}", 503)


@rl_bp.post("/api/training/<kind>")
def rl_start_training(kind):
    if kind not in ("pretrain", "ppo"):
        return _detail("Unknown training kind.", 404)
    try:
        job_id, epochs = start_training(kind, request.args.get("epochs", type=int))
    except RuntimeError as e:
        return _detail(str(e), 409)
    except Exception as e:
        return _detail(f"Training unavailable: {type(e).__name__}: {e}", 503)
    label = "Pretraining" if kind == "pretrain" else "PPO training"
    return jsonify({"job_id": job_id, "kind": kind, "status": "queued", "epochs": epochs,
                    "message": f"{label} queued ({epochs} epochs)."})


@rl_bp.get("/healthz")
def rl_healthz():
    try:
        status = _training().training_status()
    except Exception as e:
        status = {"available": False, "error": f"{type(e).__name__}: {e}"}
    with _sched_lock:
        sched = dict(_sched_state)
    return jsonify({"ok": True, "build": BUILD_ID, "config": config_loader.status(),
                    "training": status, "scheduler": sched})


@rl_bp.get("/api/symptom_library/<disease>")
def rl_symptom_library(disease):
    key = disease.strip().lower()
    library = config_loader.SYMPTOM_LIBRARY
    symptoms = library.get(key, [])
    if not symptoms:
        for k, v in library.items():
            if key in k or k in key:
                return jsonify({"disease": k, "symptoms": v})
    return jsonify({"disease": key, "symptoms": symptoms})


@rl_bp.get("/api/scheduler/status")
def rl_scheduler_status():
    with _sched_lock:
        return jsonify(dict(_sched_state))


@rl_bp.post("/api/scheduler/config")
def rl_scheduler_config():
    body = request.get_json(silent=True) or {}
    with _sched_lock:
        if body.get("enabled") is not None:
            _sched_state["enabled"] = bool(body["enabled"])
        if body.get("interval_seconds") is not None:
            if int(body["interval_seconds"]) < 60:
                return _detail("interval_seconds must be >= 60", 400)
            _sched_state["interval_seconds"] = int(body["interval_seconds"])
        return jsonify(dict(_sched_state))


@rl_bp.post("/api/scheduler/run-now")
def rl_scheduler_run_now():
    t = _training()
    with t.STATE.lock:
        if t.STATE.busy:
            return _detail("A training job is already running.", 409)

    def _bg():
        with _sched_lock:
            _sched_state["last_started_at"] = _iso_now()
        print("[RL Training] Manual cycle starting (pretrain, then PPO)")
        try:
            result = _run_one_training_cycle()
        except Exception as e:
            logger.exception("Manual training cycle failed")
            result = {"outcome": "error", "error": str(e)}
        _record_cycle(result)

    threading.Thread(target=_bg, name="manual-train", daemon=True).start()
    return jsonify({"ok": True, "message": "Cycle queued."})
