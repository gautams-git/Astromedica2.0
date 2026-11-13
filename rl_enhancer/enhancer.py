"""
Build the enhanced report from conditions + optional chat summary.

All evidence computation reads from:
  - config_loader.SYMPTOM_LIBRARY
  - config_loader.ORGAN_ALIASES
  - config_loader.BRANCH_HINTS
  - config_loader.STOP_WORDS
  - config_loader.ORGAN_BRANCH_FAMILIES
  - the chat_summary passed in

No new DB collections or bootstrap data are introduced.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from . import config_loader


# ---------------------------------------------------------------------------
# Live config accessors
# ---------------------------------------------------------------------------

def _symptom_library() -> Dict[str, List[str]]:
    return config_loader.SYMPTOM_LIBRARY


def _organ_aliases() -> Dict[str, str]:
    return config_loader.ORGAN_ALIASES


def _branch_hints() -> Dict[str, str]:
    return config_loader.BRANCH_HINTS


def _stop_words() -> Set[str]:
    return config_loader.STOP_WORDS


def _organ_branch_families() -> Dict[str, List[str]]:
    return config_loader.ORGAN_BRANCH_FAMILIES


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

def canonical_organ(raw: str) -> str:
    if not raw:
        return "General"
    low = raw.strip().lower()
    aliases = _organ_aliases()
    if low in aliases:
        return aliases[low]
    for alias, canonical in aliases.items():
        if alias in low:
            return canonical
    return raw.strip().title()


def canonical_disease(raw: str) -> str:
    return raw.strip().title() if raw else "Unspecified Condition"


def _normalise_key(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _stem(w: str) -> str:
    """Light suffix stripper so 'nerve' ~ 'nervous' ~ 'neurological'."""
    w = w.lower()
    for suffix in ("ological", "ologic", "ical", "ous", "al", "ic", "e"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[:-len(suffix)]
    return w


# ---------------------------------------------------------------------------
# Keyword helpers
# ---------------------------------------------------------------------------

def _keywords(s: str) -> Set[str]:
    stop = _stop_words()
    return {w for w in re.findall(r"[a-z]{4,}", s.lower()) if w not in stop}


def _words(s: str) -> Set[str]:
    return set(re.findall(r"[a-z]{3,}", s.lower()))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Rarity weights (computed once from the existing symptom library)
# ---------------------------------------------------------------------------

_DF_CACHE: Optional[Dict[str, int]] = None
_DF_SIZE: int = 0


def _build_df_table() -> Tuple[Dict[str, int], int]:
    library = _symptom_library()
    df: Dict[str, int] = {}
    for symptoms in library.values():
        seen: Set[str] = set()
        for s in symptoms:
            for w in _keywords(s):
                seen.add(w)
        for w in seen:
            df[w] = df.get(w, 0) + 1
    return df, max(1, len(library))


def _df() -> Tuple[Dict[str, int], int]:
    global _DF_CACHE, _DF_SIZE
    if _DF_CACHE is None:
        _DF_CACHE, _DF_SIZE = _build_df_table()
    return _DF_CACHE, _DF_SIZE


def _word_weight(w: str) -> float:
    table, _ = _df()
    freq = table.get(w, 0) + 1
    return 1.0 / math.log(2 + freq)


def _weighted_overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    shared = a & b
    if not shared:
        return 0.0
    num = sum(_word_weight(w) for w in shared)
    den = sum(_word_weight(w) for w in b)
    if den <= 0:
        return 0.0
    return min(1.0, num / den)


# ---------------------------------------------------------------------------
# Branch resolution
# ---------------------------------------------------------------------------

def _branches_for_organ(organ: Optional[str]) -> Set[str]:
    if not organ:
        return set()
    families = _organ_branch_families()
    organ_low = organ.lower()
    organ_stem = _stem(organ_low)
    out: Set[str] = set()
    for canonical, branches in families.items():
        canon_low = canonical.lower()
        if (canon_low in organ_low
                or organ_low in canon_low
                or _stem(canon_low) == organ_stem):
            out |= {str(b).lower() for b in branches}
    return out


def _branches_from_hints(disease: str) -> Set[str]:
    disease_key = _normalise_key(disease)
    out: Set[str] = set()
    for hint_key, branch in _branch_hints().items():
        hint_key_n = _normalise_key(hint_key)
        if not hint_key_n or not branch:
            continue
        if (disease_key == hint_key_n
                or hint_key_n in disease_key
                or disease_key in hint_key_n):
            out.add(branch.lower())
    return out


def _all_branches_in_summary(chat_summary: Dict[str, Any]) -> Set[str]:
    branches: Set[str] = set()
    for meta in chat_summary.get("symptom_evidence", {}).values():
        b = (meta.get("branch") or "").lower().strip()
        if b:
            branches.add(b)
    for d in chat_summary.get("diseases", []):
        b = (d.get("source_branch") or "").lower().strip()
        if b:
            branches.add(b)
    return branches


def _stemmed_branch_match(candidate: str, in_scope: Set[str]) -> bool:
    if not candidate:
        return False
    if candidate in in_scope:
        return True
    cand_stem = _stem(candidate)
    return any(_stem(b) == cand_stem for b in in_scope)


# ---------------------------------------------------------------------------
# Disease matching
# ---------------------------------------------------------------------------

_MATCH_MIN_SCORE = 0.30


def _candidate_score(
    disease_key: str,
    target_words: Set[str],
    entry: Dict[str, Any],
    organ_branches: Set[str],
) -> Tuple[float, List[str]]:
    matched_by: List[str] = []
    score = 0.0

    canonical = _normalise_key(entry.get("canonical") or "")
    raw = _normalise_key(entry.get("raw") or "")
    candidates = [c for c in (canonical, raw) if c]

    for c in candidates:
        if c == disease_key:
            score += 1.0
            matched_by.append("exact")
            break
        if disease_key in c or c in disease_key:
            score += 0.7
            matched_by.append("substring")
            break

    for c in candidates:
        j = _jaccard(target_words, _words(c))
        if j > 0.2:
            score += 0.5 * j * 2.0
            matched_by.append("jaccard")
            break

    src_branch = (entry.get("source_branch") or "").lower()
    if src_branch and organ_branches and _stemmed_branch_match(src_branch, organ_branches):
        score += 0.35
        matched_by.append("branch")

    entry_organ = entry.get("organ")
    if entry_organ and organ_branches:
        if _stem(str(entry_organ).lower()) in {_stem(b) for b in organ_branches}:
            score += 0.2
            matched_by.append("organ")

    return score, matched_by


def _entry_tiebreak(entry: Dict[str, Any]) -> Tuple[int, int, str]:
    return (
        len(entry.get("confirmed_diseases") or []),
        len(entry.get("confirmed_organs") or []),
        _normalise_key(entry.get("raw") or ""),
    )


def _match_disease(
    disease: str,
    organ: Optional[str],
    chat_summary: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float, List[str]]:
    if not chat_summary:
        return None, 0.0, []

    disease_key = _normalise_key(disease)
    target_words = _words(disease_key)
    organ_branches = _branches_for_organ(organ)

    scored: List[Tuple[float, Tuple[int, int, str], Dict[str, Any], List[str]]] = []
    for entry in chat_summary.get("diseases", []):
        score, matched_by = _candidate_score(
            disease_key, target_words, entry, organ_branches
        )
        if score >= _MATCH_MIN_SCORE:
            scored.append((score, _entry_tiebreak(entry), entry, matched_by))

    if not scored:
        return None, 0.0, []

    # Highest score, then tiebreak higher-is-better, then lexicographic
    scored.sort(key=lambda t: (t[0], t[1][0], t[1][1], -ord(t[1][2][:1] or "\x00")),
                reverse=True)
    best_score, _, best_entry, best_matched_by = scored[0]
    return best_entry, best_score, best_matched_by


# ---------------------------------------------------------------------------
# Branch scope resolution
# ---------------------------------------------------------------------------

def _branches_for_disease(
    disease: str,
    organ: Optional[str],
    chat_summary: Dict[str, Any],
    matched: Optional[Dict[str, Any]],
) -> Set[str]:
    branches: Set[str] = set()
    if matched and matched.get("source_branch"):
        branches.add(matched["source_branch"].lower())
    branches |= _branches_from_hints(disease)
    if organ:
        branches |= _branches_for_organ(organ)
    if not branches:
        branches |= _all_branches_in_summary(chat_summary)
    return branches


# ---------------------------------------------------------------------------
# Status classification
# ---------------------------------------------------------------------------

def _status_is_endorsed(status: str) -> bool:
    return status in ("yes_strong", "yes_but")


def _status_is_denied(status: str) -> bool:
    return status in ("no_strong", "no_but")


def _status_rank(status: str) -> int:
    return {"yes_strong": 4, "no_strong": 3, "yes_but": 2, "no_but": 1}.get(status, 0)


def _question_type_weight(qtype: str) -> float:
    # screening questions are the most direct signal; drilldowns refine;
    # context questions are the weakest
    return {"screening": 1.0, "drilldown": 0.8, "context": 0.4}.get(qtype, 0.5)


# ---------------------------------------------------------------------------
# Question canonicalisation and per-question scoring
# ---------------------------------------------------------------------------

_GREETING_PREFIX = re.compile(
    r"^(hello|hi|welcome|to start|let's begin|i'm ready to begin|"
    r"thank you for sharing that|thank you for confirming|"
    r"that's helpful)[!,. ]+",
    re.I,
)
_INTERROGATIVE_PREFIX = re.compile(
    r"^(have you (felt|noticed|been (experiencing|feeling|having))|"
    r"do you (feel|experience|have)|did you (feel|notice)|"
    r"has this been|does this|does it|is it|are you)[ ,]+",
    re.I,
)


def _canonical_question(text: str) -> str:
    """
    Reduce a question to a stable key that collapses across phrasings.
    Two phrasings of the same clinical fact should produce the same key.
    """
    t = text.strip().lower()
    t = _GREETING_PREFIX.sub("", t)
    t = _INTERROGATIVE_PREFIX.sub("", t)
    t = t.rstrip("?.").strip()
    tokens = _keywords(t)
    if not tokens:
        tokens = _words(t) - _stop_words()
    stems = sorted(_stem(w) for w in tokens)
    return " ".join(stems)


def _question_score(
    branch: str,
    text: str,
    qtype: str,
    in_scope: Set[str],
    library_words: Set[str],
) -> float:
    branch_part = 1.0 if _stemmed_branch_match(branch, in_scope) else 0.0
    keyword_part = _weighted_overlap(_words(text), library_words)
    type_part = _question_type_weight(qtype)
    return 0.55 * branch_part + 0.30 * keyword_part + 0.15 * type_part


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

_MIN_QUESTION_SCORE = 0.40


def _evidence_for(
    disease: str,
    organ: Optional[str],
    chat_summary: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Return (endorsed, denied, evidence_block) for a disease.

    endorsed / denied are deduplicated by canonical form; the strongest
    phrasing of each is kept.
    """
    if not chat_summary:
        return [], [], {}

    matched, match_score, matched_by = _match_disease(disease, organ, chat_summary)
    in_scope = _branches_for_disease(disease, organ, chat_summary, matched)

    library_words: Set[str] = set()
    for s in _symptom_library().get(_normalise_key(disease), []):
        library_words |= _words(s)

    # If no curated library exists for this disease, derive keywords from
    # the chat log itself — this is data we already have, not new static data.
    if not library_words:
        for text in chat_summary.get("symptom_evidence", {}).keys():
            library_words |= _words(text)

    # Collect candidates with their raw score and canonical key
    candidates: List[Dict[str, Any]] = []
    for text, meta in chat_summary.get("symptom_evidence", {}).items():
        if not text:
            continue
        branch = (meta.get("branch") or "").lower().strip()
        qtype = meta.get("question_type", "")
        status = meta.get("status", "")

        score = _question_score(branch, text, qtype, in_scope, library_words)
        if score < _MIN_QUESTION_SCORE:
            continue
        if not (_status_is_endorsed(status) or _status_is_denied(status)):
            continue

        candidates.append({
            "text": text,
            "canonical": _canonical_question(text),
            "status": status,
            "rank": _status_rank(status),
            "score": score,
            "qtype": qtype,
        })

    # ---- Deduplicate by canonical form, keep the strongest answer ----------
    best_by_canonical: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        key = c["canonical"]
        if not key:
            continue
        prev = best_by_canonical.get(key)
        if prev is None:
            best_by_canonical[key] = c
            continue
        # Prefer higher status rank; break ties on higher score
        if (c["rank"], c["score"]) > (prev["rank"], prev["score"]):
            best_by_canonical[key] = c

    endorsed_pairs: List[Tuple[float, str]] = []
    denied_pairs: List[Tuple[float, str]] = []
    for c in best_by_canonical.values():
        if _status_is_endorsed(c["status"]):
            endorsed_pairs.append((c["score"], c["text"]))
        elif _status_is_denied(c["status"]):
            denied_pairs.append((c["score"], c["text"]))

    # Rank by score, keep original phrasing
    endorsed_pairs.sort(key=lambda t: t[0], reverse=True)
    denied_pairs.sort(key=lambda t: t[0], reverse=True)

    endorsed = [t for _, t in endorsed_pairs]
    denied = [t for _, t in denied_pairs]

    # ---- Confidence derived from the block's own content ------------------
    evidence_count = len(endorsed) + len(denied)
    avg_score = (
        (sum(s for s, _ in endorsed_pairs) + sum(s for s, _ in denied_pairs))
        / max(1, evidence_count)
    )

    if matched:
        match_component = 1.0 if match_score >= 0.8 else (
            0.7 if match_score >= 0.5 else 0.4)
    elif evidence_count:
        match_component = 0.4
    else:
        match_component = 0.0

    volume_component = min(1.0, evidence_count / 3.0)

    combined = 0.5 * match_component + 0.3 * volume_component + 0.2 * avg_score
    if combined >= 0.7:
        confidence = "high"
    elif combined >= 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    # ---- Assemble block ---------------------------------------------------
    evidence: Dict[str, Any] = {}

    if matched:
        evidence["confidence"] = confidence
        if matched.get("raw"):
            evidence["raw_label"] = matched["raw"]
        evidence["organ"] = matched.get("organ") or organ
        evidence["match_score"] = round(match_score, 3)
        evidence["matched_by"] = matched_by
    elif endorsed or denied:
        evidence["confidence"] = confidence
        if organ:
            evidence["organ"] = organ
        evidence["match_method"] = "branch_or_keyword"
    else:
        evidence["confidence"] = "low"
        if organ:
            evidence["organ"] = organ
        evidence["note"] = (
            "No chat-log evidence matched this disease. "
            "The output above is generated from the symptom library only."
        )
        evidence["attempted"] = {
            "branches_in_scope": sorted(in_scope),
            "keyword_count": len(library_words),
            "sessions_scanned": chat_summary.get("raw_summary", {}).get("sessions", 0),
        }

    evidence["endorsed"] = endorsed
    evidence["denied"] = denied
    return endorsed, denied, evidence


# ---------------------------------------------------------------------------
# Symptom list assembly
# ---------------------------------------------------------------------------

# Minimum weighted overlap for a library symptom to be promoted
_PROMOTION_THRESHOLD = 0.35


def _build_symptom_list(library: List[str], endorsed: List[str]) -> List[str]:
    promoted: List[str] = []
    rest: List[str] = []
    seen: Set[str] = set()

    endorsed_words: List[Set[str]] = [_words(q) for q in endorsed]

    for s in library:
        if not isinstance(s, str):
            continue
        ss = s.strip()
        if not ss:
            continue
        key = ss.lower()
        if key in seen:
            continue
        seen.add(key)

        sym_words = _words(ss)
        best_overlap = 0.0
        for qw in endorsed_words:
            ov = _weighted_overlap(sym_words, qw)
            if ov > best_overlap:
                best_overlap = ov
                if best_overlap >= _PROMOTION_THRESHOLD:
                    break
        if best_overlap >= _PROMOTION_THRESHOLD:
            promoted.append(ss)
        else:
            rest.append(ss)

    return promoted + rest


def _symptoms_from_chat(chat_summary: Dict[str, Any],
                        in_scope: Set[str]) -> List[str]:
    """
    Synthesize a symptom list from the chat log's own target phrases when
    the requested disease has no curated library entry.
    """
    targets: List[str] = []
    seen_targets: Set[str] = set()
    for meta in chat_summary.get("symptom_evidence", {}).values():
        for t in (meta.get("symptom_targets") or []):
            t = t.strip()
            if not t:
                continue
            key = t.lower()
            if key in seen_targets:
                continue
            seen_targets.add(key)
            targets.append(t[0].upper() + t[1:])

    if targets:
        return targets[:8]

    out: List[str] = []
    seen: Set[str] = set()
    for text, meta in chat_summary.get("symptom_evidence", {}).items():
        if not text:
            continue
        branch = (meta.get("branch") or "").lower()
        if in_scope and branch and not _stemmed_branch_match(branch, in_scope):
            continue
        if not _status_is_endorsed(meta.get("status", "")):
            continue

        t = _GREETING_PREFIX.sub("", text.strip())
        t = _INTERROGATIVE_PREFIX.sub("", t).strip().rstrip("?.").strip()
        if not t:
            continue
        sentence = t[0].upper() + t[1:]
        if sentence.lower() in seen:
            continue
        seen.add(sentence.lower())
        out.append(sentence)
        if len(out) >= 6:
            break
    return out


def symptoms_for_disease(
    disease: str,
    organ: Optional[str] = None,
    chat_summary: Optional[Dict[str, Any]] = None,
) -> List[str]:
    try:
        from .training2 import generate_symptoms  # V2: same model that learns from chat logs
        generated = generate_symptoms(disease)
        if generated:
            return generated
    except Exception:
        pass

    key = _normalise_key(disease)
    library_data = _symptom_library()
    library: List[str] = []
    if key in library_data:
        library = list(library_data[key])
    else:
        for lib_key, symptoms in library_data.items():
            if key and (key in lib_key or lib_key in key):
                library = list(symptoms)
                break

    endorsed, _, _ = _evidence_for(disease, organ, chat_summary)
    merged = _build_symptom_list(library, endorsed)

    if merged:
        return merged

    if chat_summary:
        matched, _, _ = _match_disease(disease, organ, chat_summary)
        in_scope = _branches_for_disease(disease, organ, chat_summary, matched)
        synthesized = _symptoms_from_chat(chat_summary, in_scope)
        if synthesized:
            return synthesized

    return [
        f"Clinical features consistent with {disease}",
        "Generalised fatigue",
        "Reduced functional capacity",
        "Symptoms warranting clinical evaluation",
    ]


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_output(
    parsed: List[Tuple[str, List[str]]],
    progress_cb=None,
    chat_summary: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Assemble the enhanced report.

    A per-request cache ensures evidence is computed exactly once per
    (disease, organ) pair and reused for both symptom assembly and the
    evidence block.
    """
    total = sum(len(ds) for _, ds in parsed)
    done = 0
    output: List[Dict[str, Any]] = []

    cache: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    for organ_raw, diseases in parsed:
        organ = canonical_organ(organ_raw)
        blocks: List[Dict[str, Any]] = []

        for disease_raw in diseases:
            disease = canonical_disease(disease_raw)

            cache_key = (_normalise_key(disease), organ)
            if cache_key not in cache:
                _, _, ev = _evidence_for(disease, organ, chat_summary)
                cache[cache_key] = ev

            evidence = cache[cache_key]

            symptoms = symptoms_for_disease(
                disease, organ=organ, chat_summary=chat_summary
            )

            cleaned: List[str] = []
            seen_symptoms: Set[str] = set()
            for s in symptoms:
                if not isinstance(s, str):
                    continue
                ss = s.strip()
                if not ss:
                    continue
                if ss.lower() in seen_symptoms:
                    continue
                seen_symptoms.add(ss.lower())
                cleaned.append(ss)

            block: Dict[str, Any] = {disease: cleaned}
            if evidence:
                block["evidence"] = evidence

            blocks.append(block)
            done += 1
            if progress_cb and total > 0:
                progress_cb(
                    0.1 + 0.85 * (done / total),
                    f"Enhanced {done}/{total}: {organ} — {disease}",
                )

        output.append({organ: blocks})

    return output