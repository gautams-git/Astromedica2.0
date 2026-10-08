"""Parse chat_log documents into a structured clinical summary."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from . import config_loader

logger = logging.getLogger("chat_log_parser")


# ---------------------------------------------------------------------------
# Canonicalisation helpers (backed by config_loader)
# ---------------------------------------------------------------------------

def _organ_map() -> Dict[str, str]:
    return config_loader.ORGAN_MAP


def _disease_map() -> List[Any]:
    return config_loader.DISEASE_MAP


def _canonical_organ(raw: str) -> Optional[str]:
    if not raw:
        return None
    low = raw.lower()
    for k, v in _organ_map().items():
        if k in low:
            return v
    return None


def _canonical_disease(raw: str) -> Optional[str]:
    if not raw:
        return None
    for pat, canonical in _disease_map():
        if pat.search(raw):
            return canonical
    return None


def _assistant_question(msg: str) -> str:
    try:
        obj = json.loads(msg)
        if isinstance(obj, dict):
            q = obj.get("chat_message") or obj.get("message") or ""
            if q:
                return q.strip()
    except Exception:
        pass
    return (msg or "").strip()


# ---------------------------------------------------------------------------
# Merge / persistence
# ---------------------------------------------------------------------------

def _session_key(doc: Dict[str, Any]) -> str:
    """Stable identity for a chat-log document."""
    sid = (doc.get("session_id") or "").strip()
    if sid:
        return sid
    # Fall back to a compound key when session_id is missing
    started = (doc.get("started_at") or "").strip()
    return f"_nostart_{started}" if started else "_anonymous"


def persist_and_merge_chat_logs(
    patient_id: Optional[str],
    inline_logs: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    Combine stored chat logs for a patient with any inline logs supplied by
    the request. Inline logs are persisted to DB first so they become
    part of the patient's history for future requests.

    Returns the deduplicated union (stored + inline), sorted by started_at
    when available.
    """
    stored: List[Dict[str, Any]] = []
    if patient_id:
        try:
            stored = config_loader.get_chat_logs_for_patient(patient_id)
        except Exception as e:
            logger.warning("Could not load stored chat logs for %s: %s",
                           patient_id, e)
            stored = []

    # Persist inline logs so they survive for subsequent requests.
    if inline_logs and patient_id:
        for doc in inline_logs:
            if not isinstance(doc, dict):
                continue
            try:
                doc_to_store = dict(doc)
                doc_to_store.setdefault("patient_id", patient_id)
                print("Upserting")
                config_loader.upsert_chat_log(doc_to_store)
            except Exception as e:
                logger.warning("Could not persist inline chat log: %s", e)

    # Merge by session_id, inline last (so its fields override stored ones).
    merged: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    for doc in stored + (inline_logs or []):
        if not isinstance(doc, dict):
            continue
        key = _session_key(doc)
        if key not in merged:
            order.append(key)
        merged[key] = {**merged.get(key, {}), **doc}

    combined = [merged[k] for k in order]

    def sort_key(d):
        return d.get("started_at") or ""

    combined.sort(key=sort_key)
    return combined


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_chat_log(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("chat_log entry must be a JSON object")

    patient_id = data.get("patient_id") or "unknown"
    turns: List[Dict[str, Any]] = data.get("turns") or []
    cluster_scores: Dict[str, Any] = data.get("hitlist_cluster_scores") or {}

    organs: set[str] = set()
    raw_diseases: List[Dict[str, Any]] = []
    symptom_evidence: Dict[str, Dict[str, Any]] = {}

    if turns:
        pending_question: Optional[str] = None
        pending_qtype: Optional[str] = None
        pending_branch: Optional[str] = None

        for t in turns:
            role = t.get("role")
            md = t.get("metadata") or {}
            if role == "assistant":
                pending_question = _assistant_question(t.get("message", ""))
                pending_qtype = md.get("question_type")
                pending_branch = md.get("branch_id")
            elif role == "user":
                text = pending_question or (t.get("message") or "").strip()
                if not text:
                    continue
                symptom_evidence[text] = {
                    "status": md.get("answer_result") or "",
                    "question_type": md.get("answered_question_type")
                                    or pending_qtype or "",
                    "branch": md.get("answered_branch_id")
                              or pending_branch or "",
                }
                pending_question = None
                pending_qtype = None
                pending_branch = None

    for rank_str, cluster in cluster_scores.items():
        if not isinstance(cluster, dict):
            continue
        status = cluster.get("status", "")
        if status not in ("suspected", "confirmed", "drilldown"):
            continue

        for organ_raw in cluster.get("confirmed_organs") or []:
            o = _canonical_organ(organ_raw)
            if o:
                organs.add(o)
        for organ_raw in cluster.get("possible_organs") or []:
            o = _canonical_organ(organ_raw)
            if o:
                organs.add(o)

        organ_for_disease: Optional[str] = None
        for cand in (cluster.get("possible_organs") or
                     cluster.get("confirmed_organs") or []):
            organ_for_disease = _canonical_organ(cand)
            if organ_for_disease:
                break
        if organ_for_disease is None:
            organ_for_disease = _canonical_organ(
                cluster.get("dominant_branch", "")
            )

        diseases = (cluster.get("confirmed_diseases") or
                    cluster.get("possible_diseases") or [])
        for d_raw in diseases:
            canonical = _canonical_disease(d_raw) or d_raw
            raw_diseases.append({
                "raw": d_raw,
                "canonical": canonical,
                "source_branch": cluster.get("dominant_branch", ""),
                "confidence": "high" if status == "confirmed" else "medium",
                "organ": organ_for_disease,
            })

        for branch_id, branch in (cluster.get("branches") or {}).items():
            if not isinstance(branch, dict):
                continue
            for organ_raw in (branch.get("confirmed_organs") or
                              branch.get("possible_organs") or []):
                o = _canonical_organ(organ_raw)
                if o:
                    organs.add(o)

            if not turns:
                for ans in branch.get("answers") or []:
                    ans_text = (ans.get("answer") or "").strip()
                    if not ans_text:
                        continue
                    q = ans.get("question") or ans_text
                    symptom_evidence[q] = {
                        "status": ans.get("result") or "",
                        "question_type": ans.get("question_type") or "",
                        "branch": ans.get("branch_id") or branch_id,
                    }

    seen: set[str] = set()
    diseases_clean: List[Dict[str, Any]] = []
    for d in raw_diseases:
        key = (d["canonical"] or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        diseases_clean.append(d)

    endorsed = [k for k, v in symptom_evidence.items()
                if v.get("status") in ("yes_strong", "yes_but")]
    denied = [k for k, v in symptom_evidence.items()
              if v.get("status") in ("no_strong", "no_but")]

    if not organs:
        for d in diseases_clean:
            if d.get("organ"):
                organs.add(d["organ"])

    return {
        "patient_id": patient_id,
        "organs": sorted(organs),
        "diseases": diseases_clean,
        "symptom_evidence": symptom_evidence,
        "endorsed_symptoms": endorsed,
        "denied_symptoms": denied,
    }


_CONF_RANK = {"low": 0, "medium": 1, "high": 2}


def _merge_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not summaries:
        return {
            "patient_id": "unknown",
            "all_patient_ids": [],
            "organs": [],
            "diseases": [],
            "symptom_evidence": {},
            "endorsed_symptoms": [],
            "denied_symptoms": [],
            "raw_summary": {"sessions": 0},
        }

    organs: set[str] = set()
    symptom_evidence: Dict[str, Dict[str, Any]] = {}
    disease_best: Dict[str, Dict[str, Any]] = {}
    patient_ids: List[str] = []

    for s in summaries:
        if s.get("patient_id"):
            patient_ids.append(s["patient_id"])
        organs.update(s.get("organs") or [])

        for text, meta in (s.get("symptom_evidence") or {}).items():
            prev = symptom_evidence.get(text)
            if prev is None:
                symptom_evidence[text] = dict(meta)
            else:
                if _CONF_RANK.get(meta.get("status", ""), 0) > \
                   _CONF_RANK.get(prev.get("status", ""), 0):
                    symptom_evidence[text] = dict(meta)

        for d in s.get("diseases") or []:
            key = (d.get("canonical") or d.get("raw") or "").lower()
            if not key:
                continue
            prev = disease_best.get(key)
            if prev is None:
                disease_best[key] = dict(d)
            else:
                if _CONF_RANK.get(d.get("confidence"), 0) > \
                   _CONF_RANK.get(prev.get("confidence"), 0):
                    disease_best[key] = dict(d)

    endorsed = [k for k, v in symptom_evidence.items()
                if v.get("status") in ("yes_strong", "yes_but")]
    denied = [k for k, v in symptom_evidence.items()
              if v.get("status") in ("no_strong", "no_but")]

    return {
        "patient_id": patient_ids[0] if patient_ids else "unknown",
        "all_patient_ids": patient_ids,
        "organs": sorted(organs),
        "diseases": list(disease_best.values()),
        "symptom_evidence": symptom_evidence,
        "endorsed_symptoms": endorsed,
        "denied_symptoms": denied,
        "raw_summary": {"sessions": len(summaries)},
    }


def parse_chat_log_list(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(logs, list):
        raise ValueError("chat_log must be a JSON array")
    summaries: List[Dict[str, Any]] = []
    for i, entry in enumerate(logs):
        if not isinstance(entry, dict):
            logger.warning("Skipping chat_log[%d]: not an object", i)
            continue
        try:
            summaries.append(parse_chat_log(entry))
        except Exception as e:
            logger.warning("Failed to parse chat_log[%d]: %s", i, e)
    return _merge_summaries(summaries)