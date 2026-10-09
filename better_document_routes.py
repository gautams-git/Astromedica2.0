#!/usr/bin/env python3
"""
better_document_routes.py
-------------------------
Flask blueprint that:
  * loads  patients/<id>/RL Output.json  +  patients/<id>/final_output.json
  * builds patients/<id>/BETTER Output.json  (auto, on demand and on file change)
  * returns it as JSON  ->  GET /patients/<id>/better.json
  * renders it as HTML  ->  GET /patients/<id>/better
  * can be triggered automatically at the end of the RL worker job.

Add to your Flask app:

    from better_document_routes import better_bp, init_better
    app.register_blueprint(better_bp)
    init_better(app)          # starts the background watcher
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, render_template_string, send_file, abort

logger = logging.getLogger("better_document")

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
PATIENTS_DIR   = BASE_DIR / "patients"
RL_OUTPUT_FILE = "RL Output.json"
AGG_OUTPUT_FILE = "final_output.json"
BETTER_JSON    = "BETTER Output.json"

GRADE_ORDER = ["VERY_LOW", "LOW", "MODERATE", "HIGH", "VERY_HIGH"]

better_bp = Blueprint("better", __name__)

# In-process cache: patient_id -> (mtime_rl, mtime_agg, better_doc)
_CACHE: dict[str, tuple[float, float, dict]] = {}
_CACHE_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────
def _norm(text: str) -> str:
    return " ".join(str(text).lower().replace("&", " and ").split())


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index_rl_records(rl_output):
    """Flatten RL's [{organ, diseases:[...]}] into a keyed dict."""
    if isinstance(rl_output, dict):
        rl_output = rl_output.get("enhanced") or rl_output.get("result") or []
    by_key = {}
    organ_order = []
    for block in rl_output or []:
        organ = (block.get("organ") or block.get("Organ") or "").strip()
        if organ and _norm(organ) not in {_norm(o) for o in organ_order}:
            organ_order.append(organ)
        for dis in block.get("diseases") or block.get("Diseases") or []:
            name = dis.get("disease") or dis.get("name") or ""
            if not name:
                continue
            by_key[(_norm(organ), _norm(name))] = {**dis, "organ": organ, "disease": name}
    return {"by_key": by_key, "organ_order": organ_order}


def _fuse(patient_id: str, rl_output: dict, agg: dict) -> dict:
    rl = _index_rl_records(rl_output)

    agg_disease_prob = {
        _norm(d["condition"]): d for d in agg.get("disease_probability", [])
    }
    agg_confirmed = agg.get("confirmed_findings", {}) or {}
    confirmed_names = {
        _norm(d["name"]) for d in agg_confirmed.get("diseases", []) if d.get("asked")
    }

    merged, used = [], set()
    for (organ_norm, disease_norm), rec in rl["by_key"].items():
        agg_rec = agg_disease_prob.get(disease_norm)
        if agg_rec:
            used.add(disease_norm)
        risk = None
        if agg_rec:
            risk = {
                "final_score":      agg_rec.get("final_score"),
                "grade":            agg_rec.get("grade"),
                "grade_label":      agg_rec.get("grade_label"),
                "grade_color":      agg_rec.get("grade_color"),
                "action":           agg_rec.get("action"),
                "dasha_multiplier": agg_rec.get("dasha_multiplier"),
                "base_score":       agg_rec.get("base_score"),
                "chat_score":       agg_rec.get("chat_score"),
                "matched_keywords": agg_rec.get("matched_keywords", []),
            }
        merged.append({
            "organ":  rec["organ"],
            "disease": rec["disease"],
            "symptoms":         rec.get("symptoms") or rec.get("enhanced_symptoms") or [],
            "natural_remedies": rec.get("natural_remedies") or rec.get("remedies") or [],
            "dietary_plan":     rec.get("dietary_plan") or rec.get("diet"),
            "lifestyle_plan":   rec.get("lifestyle_plan") or rec.get("lifestyle"),
            "yoga":             rec.get("yoga") or rec.get("yoga_plan"),
            "precautions":      rec.get("precautions") or [],
            "severity":         rec.get("severity"),
            "risk":             risk,
            "chat_confirmed":   disease_norm in confirmed_names,
        })

    for disease_norm, agg_rec in agg_disease_prob.items():
        if disease_norm in used:
            continue
        merged.append({
            "organ": "", "disease": agg_rec["condition"],
            "symptoms": [], "natural_remedies": [], "dietary_plan": None,
            "lifestyle_plan": None, "yoga": None, "precautions": [], "severity": None,
            "risk": {
                "final_score":      agg_rec.get("final_score"),
                "grade":            agg_rec.get("grade"),
                "grade_label":      agg_rec.get("grade_label"),
                "grade_color":      agg_rec.get("grade_color"),
                "action":           agg_rec.get("action"),
                "dasha_multiplier": agg_rec.get("dasha_multiplier"),
                "base_score":       agg_rec.get("base_score"),
                "chat_score":       agg_rec.get("chat_score"),
                "matched_keywords": agg_rec.get("matched_keywords", []),
            },
            "chat_confirmed": disease_norm in confirmed_names,
        })

    def sort_key(d):
        g = (d.get("risk") or {}).get("grade", "VERY_LOW")
        return (
            d.get("chat_confirmed", False),
            GRADE_ORDER.index(g) if g in GRADE_ORDER else 0,
            (d.get("risk") or {}).get("final_score") or 0,
        )
    merged.sort(key=sort_key, reverse=True)

    overall_grade = agg.get("overall_grade") or {}
    return {
        "patient_id":   patient_id,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "overall_grade":    overall_grade.get("label"),
            "overall_score":    overall_grade.get("score"),
            "overall_color":    overall_grade.get("color"),
            "see_doctor":       (agg.get("doctor") or {}).get("see_doctor"),
            "doctor_urgency":   (agg.get("doctor") or {}).get("urgency"),
            "doctor_reasoning": (agg.get("doctor") or {}).get("reasoning"),
            "specialist":       (agg.get("doctor") or {}).get("specialist"),
            "care_until":       (agg.get("care_period") or {}).get("primary_care_until"),
            "review_by":        (agg.get("care_period") or {}).get("review_date"),
            "disease_count":    len(merged),
            "confirmed_count":  sum(1 for d in merged if d.get("chat_confirmed")),
        },
        "top_priorities": [
            {
                "disease": d["disease"], "organ": d["organ"],
                "grade":   (d.get("risk") or {}).get("grade"),
                "score":   (d.get("risk") or {}).get("final_score"),
                "action":  (d.get("risk") or {}).get("action"),
                "chat_confirmed": d.get("chat_confirmed", False),
            }
            for d in merged[:5]
        ],
        "diseases":          merged,
        "confirmed_findings": agg.get("confirmed_findings", {"confirmed": False}),
        "care_period":       agg.get("care_period"),
        "food":              agg.get("food"),
        "doctor":            agg.get("doctor"),
        "lifestyle":         agg.get("lifestyle"),
        "wellness":          agg.get("wellness"),
        "raw_scores":        agg.get("raw_scores"),
    }


# ── Build / cache ─────────────────────────────────────────────────────────
def _source_mtimes(folder: Path) -> tuple[float, float]:
    rl  = folder / RL_OUTPUT_FILE
    agg = folder / AGG_OUTPUT_FILE
    return (
        rl.stat().st_mtime  if rl.exists()  else 0.0,
        agg.stat().st_mtime if agg.exists() else 0.0,
    )


def build_better_document(patient_id: str, force: bool = False) -> dict | None:
    """
    Build (or return cached) BETTER document. Returns None if either source
    is missing — callers can treat that as "not ready yet".
    """
    folder = PATIENTS_DIR / patient_id
    if not folder.exists():
        return None

    rl_path  = folder / RL_OUTPUT_FILE
    agg_path = folder / AGG_OUTPUT_FILE
    if not (rl_path.exists() and agg_path.exists()):
        return None

    mtime_rl, mtime_agg = _source_mtimes(folder)
    with _CACHE_LOCK:
        cached = _CACHE.get(patient_id)
        if cached and not force and cached[0] == mtime_rl and cached[1] == mtime_agg:
            return cached[2]

    rl_output = _load_json(rl_path)
    agg       = _load_json(agg_path)
    better    = _fuse(patient_id, rl_output, agg)

    out_path = folder / BETTER_JSON
    out_path.write_text(json.dumps(better, indent=2, ensure_ascii=False), encoding="utf-8")

    with _CACHE_LOCK:
        _CACHE[patient_id] = (mtime_rl, mtime_agg, better)

    logger.info("BETTER document built for %s (%d diseases, %d confirmed)",
                patient_id, better["summary"]["disease_count"],
                better["summary"]["confirmed_count"])
    return better


def trigger_better_build(patient_id: str) -> None:
    """Called from the RL worker when its job finishes. Non-blocking, safe."""
    try:
        build_better_document(patient_id)
    except Exception:
        logger.exception("BETTER build failed for %s", patient_id)


# ── Optional: background watcher ──────────────────────────────────────────
_watcher_stop = threading.Event()
_watcher_thread: threading.Thread | None = None


def _watch_loop(interval: float = 10.0) -> None:
    """
    Safety net: if a patient folder gains both sources outside the RL worker
    (e.g. combine_output.py run standalone), build the BETTER doc anyway.
    """
    while not _watcher_stop.is_set():
        try:
            if PATIENTS_DIR.exists():
                for folder in PATIENTS_DIR.iterdir():
                    if not folder.is_dir():
                        continue
                    if (folder / RL_OUTPUT_FILE).exists() and (folder / AGG_OUTPUT_FILE).exists():
                        build_better_document(folder.name)
        except Exception:
            logger.exception("Watcher iteration failed")
        _watcher_stop.wait(interval)


def init_better(app=None, watch_interval: float = 10.0) -> None:
    global _watcher_thread
    if _watcher_thread is not None and _watcher_thread.is_alive():
        return
    _watcher_stop.clear()
    _watcher_thread = threading.Thread(
        target=_watch_loop, args=(watch_interval,),
        name="better-watcher", daemon=True,
    )
    _watcher_thread.start()
    logger.info("BETTER document watcher started (every %.0fs)", watch_interval)


# ── HTML template ─────────────────────────────────────────────────────────
_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Better Report — {{ doc.patient_id }}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         max-width: 980px; margin: 0 auto; padding: 24px; }
  h1 { margin: 0 0 4px; }
  h2 { border-bottom: 1px solid #ccc5; padding-bottom: 4px; margin-top: 32px; }
  h3 { margin: 20px 0 6px; }
  .muted { color: #888; font-size: 13px; }
  .card { border: 1px solid #ccc5; border-radius: 10px; padding: 14px 16px; margin: 12px 0; }
  .grade { display: inline-block; padding: 2px 10px; border-radius: 999px;
           color: #fff; font-weight: 600; font-size: 12px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px;
          background:#0001; font-size:12px; margin:2px 4px 2px 0; }
  .confirm { color: #15803d; font-weight: 600; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  ul { margin: 6px 0 6px 18px; padding: 0; }
  li { margin: 2px 0; }
  .priority { display:flex; justify-content: space-between; align-items:center;
              padding: 8px 12px; border:1px solid #ccc5; border-radius:8px; margin:6px 0; }
  .priority .left { font-weight:600; }
  @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }
</style>
</head>
<body>

<h1>Better Health Report</h1>
<p class="muted">
  Patient <strong>{{ doc.patient_id }}</strong> ·
  generated {{ doc.generated_at[:19].replace('T',' ') }}
  · <a href="?format=json">view JSON</a>
</p>

{% set s = doc.summary %}
<div class="card">
  <div style="display:flex; align-items:center; gap:12px; flex-wrap:wrap">
    <span class="grade" style="background: {{ s.overall_color or '#666' }}">
      {{ s.overall_grade }} — {{ s.overall_score }}
    </span>
    <span>
      Doctor: <strong>{{ 'Yes' if s.see_doctor else 'No' }}</strong>
      — {{ s.doctor_urgency }}
    </span>
    <span class="muted">
      {{ s.disease_count }} conditions · {{ s.confirmed_count }} confirmed in chat
    </span>
  </div>
  <p style="margin:10px 0 0">{{ s.doctor_reasoning }}</p>
  <p class="muted" style="margin:6px 0 0">
    Care until <strong>{{ s.care_until }}</strong>,
    review by <strong>{{ s.review_by }}</strong>
    {% if s.specialist %} · {{ s.specialist }}{% endif %}
  </p>
</div>

<h2>Top priorities</h2>
{% for p in doc.top_priorities %}
  <div class="priority">
    <span class="left">
      {{ loop.index }}. {{ p.disease }}{% if p.organ %} <span class="muted">({{ p.organ }})</span>{% endif %}
      {% if p.chat_confirmed %}<span class="confirm">✓ chat-confirmed</span>{% endif %}
    </span>
    <span>
      <span class="grade" style="background: {{ (p.grade and {'VERY_HIGH':'#DC2626','HIGH':'#EF4444','MODERATE':'#F59E0B','LOW':'#22C55E','VERY_LOW':'#4ADE80'}.get(p.grade)) or '#666' }}">
        {{ p.grade }} · {{ p.score }}
      </span>
    </span>
  </div>
  <div class="muted" style="margin:-2px 0 8px 12px">{{ p.action }}</div>
{% endfor %}

<h2>Diseases ({{ doc.diseases|length }})</h2>
{% for d in doc.diseases %}
  {% set r = d.risk or {} %}
  <div class="card">
    <h3 style="margin-top:0">
      {{ d.disease }}
      {% if d.organ %}<span class="muted">— {{ d.organ }}</span>{% endif %}
      {% if d.chat_confirmed %}<span class="confirm">✓ chat-confirmed</span>{% endif %}
    </h3>
    {% if r %}
      <p style="margin:4px 0">
        <span class="grade" style="background: {{ r.grade_color or '#666' }}">
          {{ r.grade_label }} · {{ r.final_score }}
        </span>
        <span class="muted">
          (base {{ r.base_score }} + chat {{ r.chat_score }}{% if r.dasha_multiplier and r.dasha_multiplier != 1 %} × {{ r.dasha_multiplier }} dasha{% endif %})
        </span>
      </p>
      <p style="margin:4px 0"><em>{{ r.action }}</em></p>
      {% if r.matched_keywords %}
        <p class="muted" style="margin:4px 0">
          Chat matches:
          {% for k in r.matched_keywords %}<span class="pill">{{ k }}</span>{% endfor %}
        </p>
      {% endif %}
    {% endif %}
    <div class="grid">
      {% if d.symptoms %}
        <div><strong>Symptoms</strong><ul>{% for x in d.symptoms %}<li>{{ x }}</li>{% endfor %}</ul></div>
      {% endif %}
      {% if d.natural_remedies %}
        <div><strong>Natural remedies</strong><ul>{% for x in d.natural_remedies %}<li>{{ x }}</li>{% endfor %}</ul></div>
      {% endif %}
    </div>
    {% if d.dietary_plan %}<p><strong>Diet:</strong> {{ d.dietary_plan }}</p>{% endif %}
    {% if d.lifestyle_plan %}<p><strong>Lifestyle:</strong> {{ d.lifestyle_plan }}</p>{% endif %}
    {% if d.yoga %}<p><strong>Yoga:</strong> {{ d.yoga }}</p>{% endif %}
    {% if d.precautions %}
      <p><strong>Precautions:</strong>
        {% for x in d.precautions %}<span class="pill">{{ x }}</span>{% endfor %}
      </p>
    {% endif %}
  </div>
{% endfor %}

{% if doc.confirmed_findings and doc.confirmed_findings.confirmed %}
  <h2>Chat-confirmed findings</h2>
  <div class="card">
    <p><strong>Organ:</strong> {{ doc.confirmed_findings.organ }}</p>
    <ul>
      {% for d in doc.confirmed_findings.diseases %}
        <li>
          <strong>{{ d.name }}</strong>
          {% if d.asked %}
            — score {{ d.score }}/10
            ({{ d.yes_count }}/{{ d.question_count }} yes)
          {% else %}
            — <span class="muted">not asked in chat</span>
          {% endif %}
        </li>
      {% endfor %}
    </ul>
  </div>
{% endif %}

{% if doc.food %}
  <h2>Diet by dosha</h2>
  <div class="card">
    <p class="muted">Doshas: {{ doc.food.doshas|join(', ') }}</p>
    {% if doc.food.avoid %}
      <p><strong>Avoid:</strong> {{ doc.food.avoid|join(', ') }}</p>
    {% endif %}
    {% set veg = (doc.food.vegetarian or {}).get('eat') or [] %}
    {% set nonveg = (doc.food.non_vegetarian or {}).get('eat') or [] %}
    {% if veg %}<p><strong>Vegetarian:</strong> {{ veg|join(', ') }}</p>{% endif %}
    {% if nonveg %}<p><strong>Non-vegetarian:</strong> {{ nonveg|join(', ') }}</p>{% endif %}
  </div>
{% endif %}

{% if doc.lifestyle %}
  <h2>Lifestyle</h2>
  <ul>{% for x in doc.lifestyle %}<li>{{ x }}</li>{% endfor %}</ul>
{% endif %}

{% if doc.wellness %}
  <h2>Wellness</h2>
  <ul>{% for x in doc.wellness %}<li>{{ x }}</li>{% endfor %}</ul>
{% endif %}

</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────
@better_bp.get("/patients/<patient_id>/better")
def better_view(patient_id: str):
    if patient_id in ("..", ".") or "/" in patient_id or "\\" in patient_id:
        abort(400)
    doc = build_better_document(patient_id)
    if doc is None:
        return (
            f"<h1>Better report not ready</h1>"
            f"<p>Patient <code>{patient_id}</code> is missing "
            f"<code>{RL_OUTPUT_FILE}</code> or <code>{AGG_OUTPUT_FILE}</code>.</p>",
            202,
        )
    from flask import request
    if request.args.get("format") == "json":
        return jsonify(doc)
    return render_template_string(_TEMPLATE, doc=doc)


@better_bp.get("/patients/<patient_id>/better.json")
def better_json(patient_id: str):
    if patient_id in ("..", ".") or "/" in patient_id or "\\" in patient_id:
        abort(400)
    doc = build_better_document(patient_id)
    if doc is None:
        return jsonify({"detail": "Better report not ready"}), 202
    return jsonify(doc)


@better_bp.get("/patients/<patient_id>/better/download")
def better_download(patient_id: str):
    if patient_id in ("..", ".") or "/" in patient_id or "\\" in patient_id:
        abort(400)
    doc = build_better_document(patient_id)
    if doc is None:
        return jsonify({"detail": "Better report not ready"}), 202
    path = PATIENTS_DIR / patient_id / BETTER_JSON
    return send_file(path, mimetype="application/json",
                     as_attachment=True,
                     download_name=f"better_{patient_id}.json")


@better_bp.post("/patients/<patient_id>/better/rebuild")
def better_rebuild(patient_id: str):
    if patient_id in ("..", ".") or "/" in patient_id or "\\" in patient_id:
        abort(400)
    doc = build_better_document(patient_id, force=True)
    if doc is None:
        return jsonify({"detail": "Sources missing"}), 202
    return jsonify({"ok": True, "generated_at": doc["generated_at"]})