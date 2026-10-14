"""
Background training — CustomANN pretraining + PPO fine-tuning.

Training corpus = curated + API-enriched symptom library
                ∪ clean chat-log symptom phrases
                ∪ clinical prose mined from chat logs (concept-aligned).

Defaults: 20 epochs each. Weights are saved under weights/.

This module derives its notion of "clinical concept" from the data it
already has (library co-occurrence, chat logs, and the model's own
embeddings). There is no hand-authored medical ontology.
"""
from __future__ import annotations

import copy
import logging
import math
import random
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .jobs import STORE
from .enhancer import _symptom_library, _normalise_key

from .medical_api import fetch_symptoms_for_disease
from . import config_db
from . import config_loader


logger = logging.getLogger("training")

WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

PRETRAIN_WEIGHTS = WEIGHTS_DIR / "customann_pretrained.pt"
PPO_WEIGHTS = WEIGHTS_DIR / "customann_ppo.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_PRETRAIN_EPOCHS = 20
DEFAULT_PPO_EPOCHS = 20

PAD = "<pad>"; UNK = "<unk>"; BOS = "<bos>"; EOS = "<eos>"
SEP = "<sep>"; STOP = "<stop>"
SPECIAL_TOKENS = [PAD, UNK, BOS, EOS, SEP, STOP]

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Question / boilerplate detection
# ---------------------------------------------------------------------------

_QUESTION_MARKERS = (
    "have you",
    "do you",
    "does it",
    "did you",
    "are you",
    "can you",
    "could you",
    "would you",
    "when you",
    "how often",
    "how many",
    "how long",
    "how would",
    "tell me",
    "describe",
    "i'd like to ask",
    "i would like to ask",
    "to start",
    "i'm ready to begin",
    "i am ready to begin",
    "thank you for",
    "welcome",
    "hello",
    "good to connect",
    "to explore a bit further",
    "i'm wondering",
    "i am wondering",
    "to help me understand",
    "to understand the pattern",
    "could you tell me",
    "what do you",
    "where do you",
    "any swelling",
    "any fever",
)

_BOILERPLATE_MARKERS = (
    "clinical features consistent with",
    "symptoms warranting clinical evaluation",
    "reduced functional capacity",
    "generalised fatigue",
    "generalized fatigue",
)


def _looks_like_question(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    if not t:
        return True
    if t.endswith("?"):
        return True
    for m in _QUESTION_MARKERS:
        if m in t:
            return True
    return False


def _looks_like_boilerplate(text: str) -> bool:
    if not text:
        return True
    t = text.lower().strip()
    for m in _BOILERPLATE_MARKERS:
        if m in t:
            return True
    return False


def _is_clean_symptom(text: str) -> bool:
    return not _looks_like_question(text) and not _looks_like_boilerplate(text)


def _jaccard(a, b) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / len(union)


def _distinct_ratio(items: List[str]) -> float:
    if len(items) < 2:
        return 1.0
    return len(set(items)) / len(items)


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

class Vocab:
    def __init__(self):
        self.itos: List[str] = list(SPECIAL_TOKENS)
        self.stoi: Dict[str, int] = {t: i for i, t in enumerate(self.itos)}

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def encode(self, tokens: List[str],
               add_bos: bool = True, add_eos: bool = True) -> List[int]:
        out = [self.stoi[BOS]] if add_bos else []
        for t in tokens:
            out.append(self.stoi.get(t, self.stoi[UNK]))
        if add_eos:
            out.append(self.stoi[EOS])
        return out

    def decode(self, ids: List[int]) -> List[str]:
        out = []
        for i in ids:
            if 0 <= i < len(self.itos):
                t = self.itos[i]
                if t in (PAD, BOS, EOS):
                    continue
                out.append(t)
        return out

    def to_list(self) -> List[str]:
        return list(self.itos)

    @classmethod
    def from_list(cls, itos: List[str]) -> "Vocab":
        v = cls()
        v.itos = list(itos)
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v

    def __len__(self) -> int:
        return len(self.itos)


# ---------------------------------------------------------------------------
# CustomANN
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[:, None, None, :] == 0, -1e9)
        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mult, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class CustomANNBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_mult, dropout)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ff(self.ln2(x))
        return x


class CustomANN(nn.Module):
    def __init__(self, vocab_size, embed_dim=192, num_heads=6,
                 num_layers=4, ff_mult=4, max_seq_len=96, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, embed_dim=embed_dim,
                           num_heads=num_heads, num_layers=num_layers,
                           ff_mult=ff_mult, max_seq_len=max_seq_len,
                           dropout=dropout)
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            CustomANNBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, input_ids, attn_mask=None):
        B, T = input_ids.shape
        T = min(T, self.max_seq_len)
        input_ids = input_ids[:, :T]
        if attn_mask is not None:
            attn_mask = attn_mask[:, :T]
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask)
        return self.lm_head(self.ln_f(h))

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=48, temperature=0.8,
                 top_k=40, top_p=0.95, repetition_penalty=1.15,
                 eos_id=None, pad_id=0):
        self.eval()
        out = prompt.clone()
        generated: List[int] = []
        for _ in range(max_new_tokens):
            if out.shape[1] >= self.max_seq_len:
                break
            attn = (out != pad_id).long() if pad_id is not None else None
            logits = self.forward(out, attn_mask=attn)[:, -1, :]
            if repetition_penalty and generated:
                for tid in set(generated):
                    logits[0, tid] /= repetition_penalty
            logits = logits / max(temperature, 1e-6)
            if top_k and top_k < logits.shape[-1]:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            if top_p and 0.0 < top_p < 1.0:
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                rm = cum > top_p
                rm[..., 1:] = rm[..., :-1].clone()
                rm[..., 0] = False
                sl[rm] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            out = torch.cat([out, nxt], dim=-1)
            generated.append(int(nxt.item()))
            if eos_id is not None and nxt.item() == eos_id:
                break
        return out


# ---------------------------------------------------------------------------
# Trainer state
# ---------------------------------------------------------------------------

class TrainerState:
    def __init__(self):
        self.model: Optional[CustomANN] = None
        self.vocab: Optional[Vocab] = None
        self.ref_model: Optional[CustomANN] = None
        self.pretrained: bool = False
        self.ppo_trained: bool = False
        self.lock = threading.RLock()
        self.busy: bool = False
        self.last_pretrain_loss: float = 0.0
        self.last_ppo_reward: float = 0.0

        # Data-derived knowledge caches (learned once per process,
        # refreshed after pretraining).
        self.concept_map: Optional[Dict[str, str]] = None
        self.tfidf: Optional[Dict[str, Dict[str, float]]] = None


STATE = TrainerState()


def _canonical_disease_from_key(key: str) -> str:
    return " ".join(w.capitalize() for w in key.split())


# ---------------------------------------------------------------------------
# Learned concept clustering  (replaces a static synonym map)
# ---------------------------------------------------------------------------

def _learn_concept_clusters(
    min_support: int = 2,
    sim_threshold: float = 0.55,
) -> Dict[str, str]:
    """
    Learn synonym clusters from co-occurrence across diseases.

    Two symptom strings that consistently appear together under the same
    disease are likely synonyms. We build a co-occurrence graph and greedily
    merge clusters whose Jaccard similarity exceeds `sim_threshold`.

    Returns a mapping raw_normalized -> cluster_representative.
    """
    lib = _symptom_library()

    disease_symptoms: Dict[str, Set[str]] = {}
    for disease, symptoms in lib.items():
        items = (symptoms.get("symptoms") if isinstance(symptoms, dict)
                 else symptoms) or []
        clean = {_normalise_key(s) for s in items
                 if isinstance(s, str) and _is_clean_symptom(s)}
        if clean:
            disease_symptoms[_normalise_key(disease)] = clean

    symptom_diseases: Dict[str, Set[str]] = {}
    for disease, symptoms in disease_symptoms.items():
        for s in symptoms:
            symptom_diseases.setdefault(s, set()).add(disease)

    candidates = [s for s, ds in symptom_diseases.items()
                  if len(ds) >= min_support]

    clusters: List[Set[str]] = []
    for s in candidates:
        placed = False
        for cluster in clusters:
            rep = next(iter(cluster))
            j = _jaccard(symptom_diseases[s], symptom_diseases[rep])
            if j >= sim_threshold:
                cluster.add(s)
                placed = True
                break
        if not placed:
            clusters.append({s})

    mapping: Dict[str, str] = {}
    for cluster in clusters:
        # Prefer the shortest string (usually the clinical term).
        rep = min(cluster, key=len)
        for s in cluster:
            mapping[s] = rep

    for s in symptom_diseases:
        mapping.setdefault(s, s)

    logger.info(
        "Learned %d concept clusters from %d symptoms across %d diseases",
        len(clusters), len(symptom_diseases), len(disease_symptoms),
    )
    return mapping


def _get_concept_map() -> Dict[str, str]:
    if STATE.concept_map is None:
        STATE.concept_map = _learn_concept_clusters()
    return STATE.concept_map


def _canonical_term_learned(term: str) -> str:
    key = _normalise_key(term)
    if not key:
        return key
    return _get_concept_map().get(key, key)


def _refresh_concept_map() -> None:
    with STATE.lock:
        STATE.concept_map = _learn_concept_clusters()
        STATE.tfidf = None


# ---------------------------------------------------------------------------
# Learned characteristic symptoms  (replaces hand-labeled red flags)
# ---------------------------------------------------------------------------

def _compute_tfidf() -> Dict[str, Dict[str, float]]:
    """
    For each disease, score every concept by how characteristic it is of
    that disease relative to the whole library.

    Returns {disease_key: {concept: score in [0, 1]}}.
    """
    lib = _symptom_library()
    disease_docs: Dict[str, Counter] = {}

    for disease, symptoms in lib.items():
        items = (symptoms.get("symptoms") if isinstance(symptoms, dict)
                 else symptoms) or []
        d_key = _normalise_key(disease)
        c = Counter()
        for s in items:
            if not isinstance(s, str) or not _is_clean_symptom(s):
                continue
            c[_canonical_term_learned(s)] += 1
        if c:
            disease_docs[d_key] = c

    df = Counter()
    for c in disease_docs.values():
        for s in c:
            df[s] += 1
    n_docs = max(1, len(disease_docs))

    idf = {s: math.log((1 + n_docs) / (1 + d)) + 1.0
           for s, d in df.items()}

    out: Dict[str, Dict[str, float]] = {}
    for disease, c in disease_docs.items():
        total = sum(c.values())
        scores = {s: (tf / total) * idf[s] for s, tf in c.items()}
        max_s = max(scores.values()) if scores else 1.0
        out[disease] = {s: v / max_s for s, v in scores.items()}

    logger.info("TF-IDF computed for %d diseases", len(out))
    return out


def _get_tfidf() -> Dict[str, Dict[str, float]]:
    if STATE.tfidf is None:
        STATE.tfidf = _compute_tfidf()
    return STATE.tfidf


def _characteristic_symptoms(disease: str, top_k: int = 6) -> List[str]:
    """
    Return the top-k most characteristic concepts for a disease,
    scored by TF-IDF over the library itself.
    """
    d_key = _normalise_key(disease)
    scores = _get_tfidf().get(d_key, {})
    if not scores:
        # Substring fallback for slight name mismatches.
        for k, v in _get_tfidf().items():
            if d_key and (d_key in k or k in d_key):
                scores = v
                break
    return [s for s, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


# ---------------------------------------------------------------------------
# Model-derived concept refresh  (self-improving loop)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_concept_map_from_model(
    model: CustomANN,
    vocab: Vocab,
    sim_threshold: float = 0.75,
) -> Dict[str, str]:
    """
    Cluster vocabulary tokens by embedding cosine similarity. Yields a
    concept map that reflects what the model has actually learned.

    Only tokens that appear in the symptom library are considered, so
    function words don't distort the clustering.
    """
    emb = model.token_embed.weight
    emb = F.normalize(emb, dim=-1)
    sim = emb @ emb.t()

    lib_tokens: Set[str] = set()
    for symptoms in _symptom_library().values():
        items = (symptoms.get("symptoms") if isinstance(symptoms, dict)
                 else symptoms) or []
        for s in items:
            if not isinstance(s, str):
                continue
            for t in tokenize(s):
                lib_tokens.add(t)
    lib_token_ids = [vocab.stoi[t] for t in lib_tokens if t in vocab.stoi]
    if not lib_token_ids:
        return {}

    mapping: Dict[str, str] = {}
    assigned: Set[int] = set()
    for i in lib_token_ids:
        if i in assigned:
            continue
        cluster = {i}
        for j in lib_token_ids:
            if j in cluster or j in assigned:
                continue
            if sim[i, j].item() >= sim_threshold:
                cluster.add(j)
        rep = min(cluster, key=lambda k: len(vocab.itos[k]))
        for k in cluster:
            mapping[vocab.itos[k]] = vocab.itos[rep]
            assigned.add(k)
    logger.info("Extracted %d model-derived concept keys", len(mapping))
    return mapping


# ---------------------------------------------------------------------------
# Chat-log harvesting  (question/boilerplate filtered)
# ---------------------------------------------------------------------------

def _iter_all_chat_logs(limit_per_patient: int = 200) -> List[Dict[str, Any]]:
    try:
        db = config_db.get_db()
    except Exception as e:
        logger.warning("DB not reachable for chat-log harvest: %s", e)
        return []

    docs: List[Dict[str, Any]] = []
    try:
        for d in config_db.load_all_chat_logs():
            d.pop("_id", None)
            docs.append(d)
    except Exception as e:
        logger.warning("Chat-log harvest failed: %s", e)
        return []

    return docs


def _samples_from_chat_log(doc: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    """
    Turn one chat-log document into (disease, [clean symptom phrases]).
    Question-shaped and boilerplate endorsements are dropped.
    """
    try:
        from .chat_log_parser import parse_chat_log
        summary = parse_chat_log(doc)
    except Exception as e:
        logger.debug("Could not parse chat log: %s", e)
        return []

    diseases = summary.get("diseases") or []
    evidence = summary.get("symptom_evidence") or {}

    endorsed_by_branch: Dict[str, List[str]] = {}
    rejected_q = 0
    for text, meta in evidence.items():
        status = meta.get("status", "")
        if status not in ("yes_strong", "yes_but"):
            continue
        raw = (text or "").strip()
        if not raw:
            continue
        if not _is_clean_symptom(raw):
            rejected_q += 1
            continue
        branch = (meta.get("branch") or "").lower().strip()
        endorsed_by_branch.setdefault(branch, []).append(raw)

    if rejected_q:
        logger.debug("chat log %s: dropped %d question-shaped endorsements",
                     doc.get("session_id", "?"), rejected_q)

    out: List[Tuple[str, List[str]]] = []

    if diseases:
        for entry in diseases:
            canonical = entry.get("canonical") or entry.get("raw") or ""
            if not canonical:
                continue
            branch = (entry.get("source_branch") or "").lower().strip()
            items = list(endorsed_by_branch.get(branch, []))
            if not items:
                items = [t for lst in endorsed_by_branch.values() for t in lst]
            seen = set()
            clean: List[str] = []
            for t in items:
                k = t.lower()
                if k in seen:
                    continue
                seen.add(k)
                clean.append(t)
                if len(clean) >= 8:
                    break
            if clean:
                out.append((canonical, clean))
    else:
        for branch, items in endorsed_by_branch.items():
            if not branch or not items:
                continue
            disease = branch.capitalize()
            out.append((disease, items[:8]))

    return out


def _chat_log_corpus() -> List[Tuple[str, List[str]]]:
    """
    Aggregate (disease, [clean symptom phrases]) from every chat log.
    De-duplicated per disease.
    """
    docs = _iter_all_chat_logs()
    if not docs:
        return []

    per_disease: Dict[str, List[str]] = {}
    seen_by_disease: Dict[str, set] = {}

    for doc in docs:
        for disease, items in _samples_from_chat_log(doc):
            key = disease.strip()
            if not key:
                continue
            per_disease.setdefault(key, [])
            seen_by_disease.setdefault(key, set())
            for t in items:
                low = t.lower()
                if low in seen_by_disease[key]:
                    continue
                seen_by_disease[key].add(low)
                per_disease[key].append(t)
                if len(per_disease[key]) >= 12:
                    break

    return [(d, items) for d, items in per_disease.items() if items]


def _clinical_prose_corpus() -> List[Tuple[str, List[str]]]:
    """
    Extract clinician-style sentences from chat logs, aligned to concepts.

    A sentence qualifies only if it contains a concept that is
    characteristic of the disease (top-k TF-IDF). This filters out
    generic phrasing that would otherwise blur across diseases.
    """
    docs = _iter_all_chat_logs()
    if not docs:
        return []

    per_disease: Dict[str, List[str]] = {}
    per_disease_seen: Dict[str, Set[str]] = {}

    for doc in docs:
        try:
            from .chat_log_parser import parse_chat_log
            summary = parse_chat_log(doc)
        except Exception:
            continue

        evidence = summary.get("symptom_evidence") or {}
        for entry in summary.get("diseases") or []:
            disease = (entry.get("canonical") or entry.get("raw") or "").strip()
            if not disease:
                continue
            d_key = _normalise_key(disease)
            disease_concepts = set(_characteristic_symptoms(disease, top_k=8))
            if not disease_concepts:
                continue

            for text, meta in evidence.items():
                if meta.get("status") not in ("yes_strong", "yes_but"):
                    continue
                if not _is_clean_symptom(text):
                    continue
                sentence_concepts = {
                    _canonical_term_learned(t) for t in tokenize(text)
                }
                if not (sentence_concepts & disease_concepts):
                    continue
                k = text.lower()
                per_disease_seen.setdefault(d_key, set())
                if k in per_disease_seen[d_key]:
                    continue
                per_disease_seen[d_key].add(k)
                per_disease.setdefault(disease, []).append(text)
                if len(per_disease[disease]) >= 10:
                    break

    return [(d, items) for d, items in per_disease.items() if items]


# ---------------------------------------------------------------------------
# Corpus builder — library + chat logs + clinical prose
# ---------------------------------------------------------------------------

def _iter_library_symptoms() -> List[Tuple[str, List[str]]]:
    out: List[Tuple[str, List[str]]] = []
    for key, symptoms in _symptom_library().items():
        disease = _canonical_disease_from_key(key)
        items = (symptoms.get("symptoms") if isinstance(symptoms, dict)
                 else symptoms) or []
        clean = [s for s in items if isinstance(s, str) and _is_clean_symptom(s)]
        if clean:
            out.append((disease, clean))
    return out


def _build_corpus() -> Tuple[Vocab, List[List[str]], List[Tuple[str, List[str]]]]:
    """
    Build the training corpus from:
      1. Symptom library entries (clean).
      2. Clean chat-log symptom phrases.
      3. Clinical prose mined from chat logs, concept-aligned.

    Format per sample:
        <sep> disease <sep> symptom1 <stop> symptom2 <stop> ... <eos>
    """
    vocab = Vocab()
    samples: List[List[str]] = []
    structured: List[Tuple[str, List[str]]] = []

    lib_pairs = _iter_library_symptoms()
    chat_pairs = _chat_log_corpus()
    clinical_pairs = _clinical_prose_corpus()

    for disease, items in lib_pairs + chat_pairs + clinical_pairs:
        tokens: List[str] = [SEP] + tokenize(disease) + [SEP]
        for s in items:
            tokens += tokenize(s) + [STOP]
        samples.append(tokens)
        structured.append((disease, list(items)))

    for seq in samples:
        for t in seq:
            vocab.add(t)

    if not samples:
        raise RuntimeError("Training corpus is empty.")

    logger.info(
        "Corpus built: %d samples (%d library + %d chat-log + %d clinical)",
        len(samples), len(lib_pairs), len(chat_pairs), len(clinical_pairs),
    )
    return vocab, samples, structured


def _reward_reference_pool(disease: str) -> List[str]:
    lib = _symptom_library()
    key = _normalise_key(disease)
    ref = lib.get(key)
    if ref is None:
        for k, v in lib.items():
            if key and (key in k or k in key):
                ref = v
                break
    ref = [s for s in (ref or []) if _is_clean_symptom(s)]

    chat = dict(_chat_log_corpus())
    for c_key, c_items in chat.items():
        c_norm = _normalise_key(c_key)
        if c_norm == key or key in c_norm or c_norm in key:
            ref.extend(c_items)
            break

    seen = set()
    out: List[str] = []
    for s in ref:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def _run_pretraining(job_id: str, epochs: int = DEFAULT_PRETRAIN_EPOCHS) -> None:
    try:
        STORE.update(job_id, status="running", progress=0.01,
                     message="Enriching symptom library from APIs…")
        enrich_result = _enrich_symptom_library_from_api(
            job_id, progress_start=0.02, progress_end=0.10
        )
        logger.info("Pre-training enrichment: %s", enrich_result)

        # Refresh data-derived knowledge after enrichment.
        _refresh_concept_map()

        STORE.update(job_id, status="running", progress=0.11,
                     message="Building vocabulary and corpus…")
        vocab, samples, structured = _build_corpus()
        max_len = 96
        encoded = [vocab.encode(s)[:max_len] for s in samples]

        diseases = [d for d, _ in structured]
        corpus_distinct = _distinct_ratio(diseases)

        model = CustomANN(vocab_size=len(vocab), max_seq_len=max_len).to(DEVICE)
        with STATE.lock:
            STATE.vocab = vocab
            STATE.model = model
            STATE.pretrained = False
            STATE.ppo_trained = False
            STATE.ref_model = None

        optim = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  weight_decay=0.02, betas=(0.9, 0.95))
        total_steps = epochs * len(encoded)
        step = 0
        total_loss = 0.0

        for epoch in range(epochs):
            random.shuffle(encoded)
            for seq in encoded:
                if len(seq) < 4:
                    continue
                ids = torch.tensor(seq, dtype=torch.long,
                                   device=DEVICE).unsqueeze(0)
                inp, tgt = ids[:, :-1], ids[:, 1:]
                attn = (inp != vocab.stoi[PAD]).long()
                logits = model(inp, attn_mask=attn)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    tgt.reshape(-1),
                    ignore_index=vocab.stoi[PAD],
                    label_smoothing=0.05,
                )
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                total_loss += loss.item()
                step += 1
                if step % 5 == 0 or step == total_steps:
                    p = 0.11 + 0.83 * (step / max(total_steps, 1))
                    STORE.update(job_id, progress=p,
                                 message=f"Pretrain {epoch+1}/{epochs} — "
                                         f"step {step}/{total_steps} — "
                                         f"loss {loss.item():.3f}")

        torch.save({"state_dict": model.state_dict(),
                    "config": model.config,
                    "vocab": vocab.to_list(),
                    "epochs": epochs}, PRETRAIN_WEIGHTS)

        STATE.last_pretrain_loss = total_loss / max(step, 1)
        with STATE.lock:
            STATE.pretrained = True

        # Self-improving loop: refresh concept map from the trained model.
        try:
            model_map = _extract_concept_map_from_model(model, vocab)
            if model_map:
                # Merge with the co-occurrence map. Model-derived entries
                # take precedence only for tokens the co-occurrence map
                # didn't already cluster.
                existing = _get_concept_map()
                for k, v in model_map.items():
                    existing.setdefault(k, v)
                logger.info("Merged %d model-derived concept entries",
                            len(model_map))
        except Exception as e:
            logger.warning("Model-derived concept refresh skipped: %s", e)

        STORE.update(
            job_id, status="done", progress=1.0,
            message=(f"Pretraining complete — avg loss "
                     f"{STATE.last_pretrain_loss:.3f} "
                     f"(library appended: "
                     f"{enrich_result.get('symptoms_added', 0)} new symptoms; "
                     f"corpus: {len(samples)} samples; "
                     f"distinct_diseases: {corpus_distinct:.2f})"),
            output_path=str(PRETRAIN_WEIGHTS),
            result={"avg_loss": STATE.last_pretrain_loss,
                    "weights": str(PRETRAIN_WEIGHTS),
                    "enrichment": enrich_result,
                    "corpus_samples": len(samples),
                    "distinct_diseases": corpus_distinct,
                    "concept_clusters": len(set(
                        _get_concept_map().values()
                    ))},
        )
        logger.info("Pretraining done — %s", PRETRAIN_WEIGHTS)

    except Exception as e:
        logger.exception("Pretraining failed")
        STORE.update(job_id, status="error",
                     message="Pretraining failed",
                     error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Reward — data-derived, disease-discriminative, anti-collapse
# ---------------------------------------------------------------------------

_SYMPTOM_VOCAB_CACHE: Optional[set] = None
_SYMPTOM_VOCAB_CACHE_KEY: Optional[int] = None


def _known_symptom_vocab(library: Optional[dict] = None) -> set:
    global _SYMPTOM_VOCAB_CACHE, _SYMPTOM_VOCAB_CACHE_KEY
    lib = library if library is not None else _symptom_library()
    cache_key = len(lib)
    if _SYMPTOM_VOCAB_CACHE is not None and _SYMPTOM_VOCAB_CACHE_KEY == cache_key:
        return _SYMPTOM_VOCAB_CACHE

    vocab: set = set()
    for entry in lib.values():
        if isinstance(entry, dict):
            items = (entry.get("symptoms")
                     or entry.get("canonical_symptoms")
                     or entry.get("keys") or [])
        elif isinstance(entry, (list, tuple, set)):
            items = entry
        else:
            items = []
        for s in items:
            if not _is_clean_symptom(s):
                continue
            n = _normalise_key(s)
            if n:
                vocab.add(n)

    _SYMPTOM_VOCAB_CACHE = vocab
    _SYMPTOM_VOCAB_CACHE_KEY = cache_key
    return vocab


def _canonical_symptoms_for(disease: str, lib: dict) -> Set[str]:
    disease_key = _normalise_key(disease)
    entry = lib.get(disease) or lib.get(disease_key) or {}
    if isinstance(entry, dict):
        items = (entry.get("symptoms")
                 or entry.get("canonical_symptoms")
                 or entry.get("keys") or [])
    elif isinstance(entry, (list, tuple, set)):
        items = list(entry)
    else:
        items = []
    cleaned = [s for s in items if _is_clean_symptom(s)]
    return {_canonical_term_learned(s) for s in cleaned if s}


def _reward_for_generation(
    vocab,
    disease: str,
    generated_text: str,
    *,
    is_enriched: bool = False,
    enrichment_bonus: float = 0.10,
    library: Optional[dict] = None,
    negative_disease: Optional[str] = None,
    recent_generations: Optional[List[str]] = None,
    recent_diseases: Optional[List[str]] = None,
) -> float:
    """
    Data-derived reward.

    Hard rejections:
      - empty / question-shaped / boilerplate output

    Shaping terms:
      + concept_coverage    : fraction of THIS disease's concepts covered
      + concept_precision   : fraction of generated concepts that belong to it
      + characteristic      : bonus for the disease's top TF-IDF concepts
      + contrastive         : bonus for concepts that separate this disease
                              from the sampled negative disease
      + system_coverage     : bonus for covering multiple systems when the
                              disease is systemic; staying in-system when
                              organ-specific
      + disease_sim         : token overlap with disease name
      - repetition          : token duplication penalty
      - length_reg          : output length regularization
      - collapse_penalty    : near-duplicate of a DIFFERENT disease's output
    """
    if not generated_text or not generated_text.strip():
        return -0.5
    if _looks_like_question(generated_text):
        return -1.0
    if _looks_like_boilerplate(generated_text):
        return -1.0

    lib = library if library is not None else _symptom_library()

    gen_tokens = [t for t in tokenize(generated_text) if t.strip()]
    if not gen_tokens:
        return -0.5

    gen_norm_tokens = [_normalise_key(t) for t in gen_tokens]
    gen_concepts = {_canonical_term_learned(t) for t in gen_norm_tokens if t}
    gen_concepts.discard("")

    if not gen_concepts:
        return -0.5

    canonical_concepts = _canonical_symptoms_for(disease, lib)

    if canonical_concepts:
        hits = len(gen_concepts & canonical_concepts)
        coverage = hits / len(canonical_concepts)
        precision = (sum(1 for c in gen_concepts if c in canonical_concepts)
                     / len(gen_concepts))
    else:
        coverage = 0.0
        precision = 0.0

    # Characteristic bonus (replaces hand-labeled red flags).
    char_symptoms = set(_characteristic_symptoms(disease, top_k=6))
    char_hits = len(gen_concepts & char_symptoms) if char_symptoms else 0
    characteristic_bonus = (0.6 * (char_hits / len(char_symptoms))
                            if char_symptoms else 0.0)

    # Contrastive bonus (replaces hand-labeled discriminators).
    contrastive_bonus = 0.0
    if negative_disease:
        pos_chars = set(_characteristic_symptoms(disease, top_k=8))
        neg_chars = set(_characteristic_symptoms(negative_disease, top_k=8))
        pos_only = pos_chars - neg_chars
        if pos_only:
            contrastive_bonus = 0.5 * (len(gen_concepts & pos_only)
                                       / len(pos_only))

    # System coverage from the learned clusters themselves.
    # We infer "system" as the set of concepts shared with sibling diseases.
    disease_char = set(_characteristic_symptoms(disease, top_k=12))
    sibling_systems: Dict[str, Set[str]] = {}
    for other, scores in _get_tfidf().items():
        if other == _normalise_key(disease):
            continue
        for c, v in scores.items():
            if v >= 0.5:
                sibling_systems.setdefault(c, set()).add(other)
    systemic = len({tuple(sorted(s)) for s in sibling_systems.values()}) > 3
    if systemic:
        covered = sum(1 for c in gen_concepts if c in sibling_systems)
        system_coverage = min(1.0, covered / max(1, len(disease_char)))
    else:
        covered = sum(1 for c in gen_concepts if c in disease_char)
        system_coverage = min(1.0, covered / max(1, len(disease_char)))

    disease_tokens = {
        _normalise_key(t) for t in tokenize(disease) if _normalise_key(t)
    }
    disease_sim = (len(gen_concepts & disease_tokens) / len(disease_tokens)
                   if disease_tokens else 0.0)

    unique_ratio = (len(set(gen_norm_tokens)) / len(gen_norm_tokens)
                    if gen_norm_tokens else 0.0)
    repetition = 1.0 - unique_ratio

    n = len(gen_norm_tokens)
    if n < 3:
        length_reg = (3 - n) / 3.0
    elif n > 25:
        length_reg = min(1.0, (n - 25) / 25.0)
    else:
        length_reg = 0.0

    reward = (
        1.00 * coverage
        + 0.50 * precision
        + 0.60 * characteristic_bonus
        + 0.30 * contrastive_bonus
        + 0.25 * system_coverage
        + 0.20 * disease_sim
        - 0.40 * repetition
        - 0.20 * length_reg
    )

    collapse_penalty = 0.0
    if recent_generations and recent_diseases:
        cur_disease = _normalise_key(disease)
        for prev_text, prev_disease in zip(recent_generations, recent_diseases):
            if _normalise_key(prev_disease) == cur_disease:
                continue
            prev_tokens = [_normalise_key(t) for t in tokenize(prev_text)
                           if _normalise_key(t)]
            prev_concepts = {_canonical_term_learned(t)
                             for t in prev_tokens if t}
            if not prev_concepts:
                continue
            j = _jaccard(gen_concepts, prev_concepts)
            if j > 0.6:
                collapse_penalty = max(collapse_penalty, j - 0.6)
    reward -= 2.0 * collapse_penalty

    if is_enriched:
        reward += enrichment_bonus

    return float(max(-1.5, min(2.5, reward)))


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

_PPO_CHAT_WEIGHT = 0.30
_PPO_LIBRARY_WEIGHT = 1.00
_PPO_ENRICHED_WEIGHT = 2.00
_PPO_KL_COEFF = 0.20
_PPO_ENTROPY_COEFF = 0.02      # raised from 0.01 to encourage exploration
_PPO_ADV_CLAMP = 3.0
_PPO_COLLAPSE_ALARM = 0.30
_PPO_RECENT_WINDOW = 16


def _load_reference_model(fallback: CustomANN) -> CustomANN:
    """
    Prefer the PRETRAINED checkpoint as the KL reference. This prevents
    PPO from compounding an already-drifted policy.
    """
    try:
        if PRETRAIN_WEIGHTS.exists():
            data = torch.load(PRETRAIN_WEIGHTS, map_location=DEVICE)
            cfg = data["config"]
            ref = CustomANN(**cfg).to(DEVICE)
            ref.load_state_dict(data["state_dict"])
            ref.eval()
            for p in ref.parameters():
                p.requires_grad = False
            logger.info("KL reference: pretrained checkpoint")
            return ref
    except Exception as e:
        logger.warning("Could not load pretrained reference: %s", e)

    ref = copy.deepcopy(fallback)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    logger.warning("KL reference: deepcopy of current model (fallback)")
    return ref


def _run_ppo(job_id: str, epochs: int = DEFAULT_PPO_EPOCHS) -> None:
    try:
        STORE.update(job_id, status="running", progress=0.01,
                     message="Enriching symptom library from APIs…")
        enrich_result = _enrich_symptom_library_from_api(
            job_id, progress_start=0.02, progress_end=0.10
        )
        logger.info("PPO enrichment: %s", enrich_result)

        # Refresh data-derived knowledge after enrichment.
        _refresh_concept_map()

        with STATE.lock:
            if STATE.model is None or STATE.vocab is None:
                raise RuntimeError("Run pretraining first.")
            model = STATE.model
            vocab = STATE.vocab

        STORE.update(job_id, status="running", progress=0.11,
                     message="Freezing reference model for KL…")
        ref = _load_reference_model(model)
        with STATE.lock:
            STATE.ref_model = ref

        optim = torch.optim.AdamW(model.parameters(), lr=2e-5,
                                  weight_decay=0.01, betas=(0.9, 0.95))

        library_keys = list(_symptom_library().keys())
        chat_pairs = _chat_log_corpus()
        chat_keys = [d for d, _ in chat_pairs]

        enriched_keys_raw = (
            enrich_result.get("added_keys")
            or enrich_result.get("new_symptoms")
            or enrich_result.get("symptoms_added_keys")
            or []
        )
        if isinstance(enriched_keys_raw, int):
            enriched_keys_raw = []
        enriched_keys_raw = [k for k in enriched_keys_raw
                             if _is_clean_symptom(k)]

        try:
            enriched_key_set = {_normalise_key(k) for k in enriched_keys_raw
                                if _normalise_key(k)}
        except Exception:
            enriched_key_set = set()

        key_weights: Dict[str, float] = {}
        pool_index: Dict[str, str] = {}

        def _add_key(k: str, weight: float) -> None:
            kn = _normalise_key(k)
            if not kn:
                return
            if weight > key_weights.get(kn, 0.0):
                key_weights[kn] = weight
            pool_index.setdefault(kn, k)

        for k in library_keys:
            _add_key(k, _PPO_LIBRARY_WEIGHT)
        for k in chat_keys:
            _add_key(k, _PPO_CHAT_WEIGHT)
        for k in enriched_keys_raw:
            _add_key(k, _PPO_ENRICHED_WEIGHT)

        sampling_pool: List[str] = list(pool_index.values())
        sampling_weights: List[float] = [
            key_weights[_normalise_key(k)] for k in sampling_pool
        ]

        samples_per_epoch = max(4, len(sampling_pool))
        total_steps = epochs * samples_per_epoch
        step = 0
        total_reward = 0.0
        baseline = 0.0
        enriched_trained = 0
        covered_keys: set = set()

        pad_id = vocab.stoi[PAD]
        eos_id = vocab.stoi[EOS]
        stop_id = vocab.stoi[STOP]

        recent_generations: List[str] = []
        recent_diseases: List[str] = []

        for epoch in range(epochs):
            if sampling_weights and any(w > 0 for w in sampling_weights):
                epoch_keys = random.choices(
                    sampling_pool, weights=sampling_weights,
                    k=samples_per_epoch,
                )
            else:
                epoch_keys = list(sampling_pool)
                random.shuffle(epoch_keys)
                epoch_keys = epoch_keys[:samples_per_epoch]

            for key in epoch_keys:
                model.train()
                disease = _canonical_disease_from_key(key)
                key_norm = _normalise_key(key)
                covered_keys.add(key_norm)
                is_enriched = key_norm in enriched_key_set
                if is_enriched:
                    enriched_trained += 1

                prompt_ids = vocab.encode([SEP] + tokenize(disease) + [SEP],
                                          add_bos=True, add_eos=False)
                prompt = torch.tensor(prompt_ids, dtype=torch.long,
                                      device=DEVICE).unsqueeze(0)

                with torch.no_grad():
                    gen = model.generate(prompt, max_new_tokens=48,
                                         temperature=0.9, top_k=40, top_p=0.95,
                                         repetition_penalty=1.15,
                                         eos_id=eos_id, pad_id=pad_id)

                gen_ids = gen[0].tolist()
                gen_tokens = vocab.decode(gen_ids[len(prompt_ids):])
                if STOP in gen_tokens:
                    gen_tokens = gen_tokens[:gen_tokens.index(STOP)]
                generated_text = " ".join(gen_tokens)

                # Contrastive negative: a different disease from the pool.
                neg_candidates = [k for k in sampling_pool
                                  if _normalise_key(k) != key_norm]
                neg_key = (random.choice(neg_candidates)
                           if neg_candidates else None)
                negative_disease = (_canonical_disease_from_key(neg_key)
                                    if neg_key else None)

                r = _reward_for_generation(
                    vocab, disease, generated_text,
                    is_enriched=is_enriched,
                    negative_disease=negative_disease,
                    recent_generations=recent_generations[-_PPO_RECENT_WINDOW:],
                    recent_diseases=recent_diseases[-_PPO_RECENT_WINDOW:],
                )
                recent_generations.append(generated_text)
                recent_diseases.append(disease)
                if len(recent_generations) > _PPO_RECENT_WINDOW * 4:
                    recent_generations = recent_generations[-_PPO_RECENT_WINDOW * 2:]
                    recent_diseases = recent_diseases[-_PPO_RECENT_WINDOW * 2:]

                total_reward += r
                adv = r - baseline
                baseline = 0.9 * baseline + 0.1 * r

                inp, tgt = gen[:, :-1], gen[:, 1:]
                attn = (inp != pad_id).long()

                logits = model(inp, attn_mask=attn)
                log_probs = F.log_softmax(logits, dim=-1)

                with torch.no_grad():
                    rl = ref(inp, attn_mask=attn)
                    rlp = F.log_softmax(rl, dim=-1)

                kl = (log_probs.exp() * (log_probs - rlp)).sum(-1).mean()
                nll = F.nll_loss(
                    log_probs.reshape(-1, log_probs.shape[-1]),
                    tgt.reshape(-1),
                    ignore_index=pad_id,
                )
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(-1).mean()
                adv_t = torch.tensor(adv, dtype=torch.float32, device=DEVICE)
                loss = (
                    -adv_t.clamp(-_PPO_ADV_CLAMP, _PPO_ADV_CLAMP) * nll
                    + _PPO_KL_COEFF * kl
                    - _PPO_ENTROPY_COEFF * entropy
                )

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                step += 1
                p = 0.11 + 0.83 * (step / max(total_steps, 1))
                tag = " [enriched]" if is_enriched else ""
                STORE.update(job_id, progress=p,
                             message=f"PPO {epoch+1}/{epochs} — "
                                     f"step {step}/{total_steps} — "
                                     f"reward {r:.3f}{tag}")

                if step % 32 == 0:
                    dr = _distinct_ratio(
                        recent_generations[-_PPO_RECENT_WINDOW:]
                    )
                    if dr < _PPO_COLLAPSE_ALARM:
                        raise RuntimeError(
                            f"Policy collapse detected: distinct_ratio="
                            f"{dr:.2f} < {_PPO_COLLAPSE_ALARM}. Aborting "
                            f"PPO. Check chat-corpus cleanliness."
                        )

        STATE.last_ppo_reward = total_reward / max(step, 1)

        torch.save({"state_dict": model.state_dict(),
                    "config": model.config,
                    "vocab": vocab.to_list(),
                    "epochs": epochs,
                    "avg_reward": STATE.last_ppo_reward}, PPO_WEIGHTS)

        with STATE.lock:
            STATE.ppo_trained = True

        final_distinct = _distinct_ratio(
            recent_generations[-_PPO_RECENT_WINDOW:]
        )

        STORE.update(
            job_id, status="done", progress=1.0,
            message=(f"PPO complete — avg reward "
                     f"{STATE.last_ppo_reward:.3f} "
                     f"(library appended: "
                     f"{enrich_result.get('symptoms_added', 0)} new symptoms; "
                     f"trained on {enriched_trained} enriched steps; "
                     f"sampling pool: {len(sampling_pool)}; "
                     f"distinct_ratio: {final_distinct:.2f})"),
            output_path=str(PPO_WEIGHTS),
            result={"avg_reward": STATE.last_ppo_reward,
                    "weights": str(PPO_WEIGHTS),
                    "enrichment": enrich_result,
                    "sampling_pool": len(sampling_pool),
                    "enriched_steps": enriched_trained,
                    "enriched_keys_covered": len(
                        covered_keys & enriched_key_set
                    ) if enriched_key_set else 0,
                    "distinct_ratio": final_distinct},
        )
        logger.info("PPO done — %s", PPO_WEIGHTS)

    except Exception as e:
        logger.exception("PPO failed")
        STORE.update(job_id, status="error", message="PPO failed",
                     error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_pretraining(epochs: int = DEFAULT_PRETRAIN_EPOCHS) -> str:
    with STATE.lock:
        if STATE.busy:
            raise RuntimeError("Another training job is running.")
        STATE.busy = True
    job = STORE.create()
    STORE.update(job.id, message="Pretraining queued")

    def _run():
        try:
            _run_pretraining(job.id, epochs=epochs)
        finally:
            with STATE.lock:
                STATE.busy = False

    threading.Thread(target=_run, name="pretrain", daemon=True).start()
    return job.id


def start_ppo(epochs: int = DEFAULT_PPO_EPOCHS) -> str:
    with STATE.lock:
        if STATE.busy:
            raise RuntimeError("Another training job is running.")
        if STATE.model is None:
            raise RuntimeError("Run pretraining first.")
        STATE.busy = True
    job = STORE.create()
    STORE.update(job.id, message="PPO queued")

    def _run():
        try:
            _run_ppo(job.id, epochs=epochs)
        finally:
            with STATE.lock:
                STATE.busy = False

    threading.Thread(target=_run, name="ppo", daemon=True).start()
    return job.id


def training_status() -> Dict[str, Any]:
    with STATE.lock:
        cm = STATE.concept_map or {}
        return {
            "pretrained": STATE.pretrained,
            "ppo_trained": STATE.ppo_trained,
            "busy": STATE.busy,
            "last_pretrain_loss": STATE.last_pretrain_loss,
            "last_ppo_reward": STATE.last_ppo_reward,
            "device": str(DEVICE),
            "model": "CustomANN",
            "pretrain_weights_exists": PRETRAIN_WEIGHTS.exists(),
            "ppo_weights_exists": PPO_WEIGHTS.exists(),
            "default_pretrain_epochs": DEFAULT_PRETRAIN_EPOCHS,
            "default_ppo_epochs": DEFAULT_PPO_EPOCHS,
            "concept_clusters": len(set(cm.values())) if cm else 0,
            "concept_terms_mapped": len(cm),
        }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _clean_symptom_fragment(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment:
        return ""
    fragment = re.sub(r"\s+", " ", fragment)
    fragment = fragment.strip(" ,;:.-")
    if not fragment:
        return ""
    return fragment[0].upper() + fragment[1:]


def generate_symptoms(disease: str,
                      max_tokens: int = 64) -> Optional[List[str]]:
    """
    Generate symptoms for a disease.

    Post-processing:
      - Drop question-shaped and boilerplate fragments.
      - Prefer the learned representative for each concept cluster.
      - De-duplicate by learned concept, not raw string.
    """
    with STATE.lock:
        model = STATE.model
        vocab = STATE.vocab
        ready = STATE.pretrained or STATE.ppo_trained
    if not ready or model is None or vocab is None:
        return None
    try:
        disease_clean = " ".join(w.capitalize() for w in tokenize(disease))
        prompt_ids = vocab.encode([SEP] + tokenize(disease_clean) + [SEP],
                                  add_bos=True, add_eos=False)
        prompt = torch.tensor(prompt_ids, dtype=torch.long,
                              device=DEVICE).unsqueeze(0)
        gen = model.generate(prompt, max_new_tokens=max_tokens,
                             temperature=0.7, top_k=30, top_p=0.9,
                             repetition_penalty=1.2,
                             eos_id=vocab.stoi[EOS], pad_id=vocab.stoi[PAD])
        ids = gen[0].tolist()
        new_ids = ids[len(prompt_ids):]
        decoded = vocab.decode(new_ids)

        fragments: List[List[str]] = [[]]
        for tok in decoded:
            if tok == STOP:
                fragments.append([])
            elif tok == EOS:
                break
            else:
                fragments[-1].append(tok)

        seen_concepts: Set[str] = set()
        clean: List[str] = []
        for ft in fragments:
            text = _clean_symptom_fragment(" ".join(ft))
            if not text or not _is_clean_symptom(text):
                continue
            concept = _canonical_term_learned(text)
            if concept in seen_concepts:
                continue
            seen_concepts.add(concept)
            clean.append(text)
            if len(clean) >= 8:
                break

        if len(clean) < 2:
            return None
        if all(len(f.split()) < 2 for f in clean):
            return None
        return clean
    except Exception as e:
        logger.debug("Generation failed for %s: %s", disease, e)
        return None


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_existing_weights() -> None:
    path = PPO_WEIGHTS if PPO_WEIGHTS.exists() else PRETRAIN_WEIGHTS
    if not path.exists():
        return
    try:
        data = torch.load(path, map_location=DEVICE)
        vocab = Vocab.from_list(data["vocab"])
        cfg = data["config"]
        model = CustomANN(**cfg).to(DEVICE)
        model.load_state_dict(data["state_dict"])
        model.eval()
        with STATE.lock:
            STATE.vocab = vocab
            STATE.model = model
            if path == PPO_WEIGHTS:
                STATE.pretrained = True
                STATE.ppo_trained = True
            else:
                STATE.pretrained = True
        # Refresh concept map from the loaded model so downstream
        # generation uses the same clustering the model learned.
        try:
            _refresh_concept_map()
            model_map = _extract_concept_map_from_model(model, vocab)
            if model_map:
                existing = _get_concept_map()
                for k, v in model_map.items():
                    existing.setdefault(k, v)
        except Exception as e:
            logger.warning("Concept refresh after weight load skipped: %s", e)
        logger.info("Loaded CustomANN weights from %s", path)
    except Exception as e:
        logger.warning("Failed to load weights %s: %s", path, e)


# ---------------------------------------------------------------------------
# Append-only API enrichment
# ---------------------------------------------------------------------------

def _enrich_symptom_library_from_api(
    job_id: str,
    progress_start: float = 0.02,
    progress_end: float = 0.10,
) -> Dict[str, Any]:
    """
    Append-only enrichment. Never overwrites, never deletes.
    API responses pass through question/boilerplate filters and are
    normalized to learned concept representatives before insertion.
    """
    library = dict(config_loader.SYMPTOM_LIBRARY)
    if not library:
        return {"status": "skipped", "reason": "empty library"}

    diseases = list(library.keys())
    total = len(diseases)

    delta: Dict[str, List[str]] = {}

    for i, disease in enumerate(diseases):
        try:
            api_symptoms = fetch_symptoms_for_disease(disease)
        except Exception as e:
            logger.debug("API fetch failed for %s: %s", disease, e)
            api_symptoms = []

        existing = library.get(disease, []) or []
        existing_lower = {s.lower() for s in existing if isinstance(s, str)}
        existing_concepts = {_canonical_term_learned(e).lower()
                             for e in existing if isinstance(e, str)}

        fresh: List[str] = []
        seen_fresh: set[str] = set()
        for s in api_symptoms:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            if not _is_clean_symptom(t):
                continue
            concept = _canonical_term_learned(t)
            key = concept.lower()
            if key in existing_lower or key in seen_fresh:
                continue
            if key in existing_concepts:
                continue
            seen_fresh.add(key)
            fresh.append(concept)

        if fresh:
            delta[disease] = fresh

        pct = (progress_start
               + (progress_end - progress_start) * (i + 1) / max(total, 1))
        try:
            STORE.update(
                job_id,
                progress=min(progress_end, pct),
                message=(f"Enriching symptom library from APIs — "
                         f"{i + 1}/{total}: {disease}"),
            )
        except Exception:
            pass

    if not delta:
        config_loader.refresh_symptom_library_after_write()
        return {
            "status": "ok",
            "diseases_processed": total,
            "diseases_inserted": 0,
            "diseases_appended": 0,
            "symptoms_added": 0,
            "diseases_skipped": total,
        }

    try:
        stats = config_db.append_symptoms_to_library(delta)
    except Exception as e:
        logger.warning("Could not persist enriched library: %s", e)
        return {"status": "error", "error": str(e),
                "diseases_processed": total}

    refreshed = config_loader.refresh_symptom_library_after_write()

    added_keys: List[str] = []
    for items in delta.values():
        added_keys.extend(items)

    return {
        "status": "ok",
        "diseases_processed": total,
        **stats,
        "added_keys": added_keys,
        "library_size": refreshed.get("symptom_library_size", len(library)),
    }