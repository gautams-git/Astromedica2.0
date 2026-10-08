"""
Background training — CustomANN pretraining + PPO fine-tuning.
Defaults: 20 epochs each. Weights are saved under weights/.

Training corpus is built from the symptom library only (curated +
API-enriched). Chat logs are not used here.
"""
from __future__ import annotations

import copy
import logging
import math
import random
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


STATE = TrainerState()


def _canonical_disease_from_key(key: str) -> str:
    return " ".join(w.capitalize() for w in key.split())


# ---------------------------------------------------------------------------
# Corpus builder — library only
# ---------------------------------------------------------------------------

def _build_corpus() -> Tuple[Vocab, List[List[str]], List[Tuple[str, List[str]]]]:
    """
    Build the training corpus from the (possibly enriched) symptom library.
    """
    vocab = Vocab()
    samples: List[List[str]] = []
    structured: List[Tuple[str, List[str]]] = []

    for key, symptoms in _symptom_library().items():
        disease = _canonical_disease_from_key(key)
        tokens: List[str] = [SEP] + tokenize(disease) + [SEP]
        for s in symptoms:
            tokens += tokenize(s) + [STOP]
        samples.append(tokens)
        structured.append((disease, list(symptoms)))

    for seq in samples:
        for t in seq:
            vocab.add(t)

    if not samples:
        raise RuntimeError("Symptom library is empty.")

    logger.info("Corpus built: %d samples from library", len(samples))
    return vocab, samples, structured


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def _run_pretraining(job_id: str, epochs: int = DEFAULT_PRETRAIN_EPOCHS) -> None:
    try:
        # ---- API enrichment (append-only) --------------------------------
        STORE.update(job_id, status="running", progress=0.01,
                     message="Enriching symptom library from APIs…")
        enrich_result = _enrich_symptom_library_from_api(
            job_id, progress_start=0.02, progress_end=0.10
        )
        logger.info("Pre-training enrichment: %s", enrich_result)

        # ---- Build corpus ------------------------------------------------
        STORE.update(job_id, status="running", progress=0.11,
                     message="Building vocabulary and corpus…")
        vocab, samples, _ = _build_corpus()
        max_len = 96
        encoded = [vocab.encode(s)[:max_len] for s in samples]

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

        STORE.update(
            job_id, status="done", progress=1.0,
            message=(f"Pretraining complete — avg loss "
                     f"{STATE.last_pretrain_loss:.3f} "
                     f"(library appended: "
                     f"{enrich_result.get('symptoms_added', 0)} new symptoms)"),
            output_path=str(PRETRAIN_WEIGHTS),
            result={"avg_loss": STATE.last_pretrain_loss,
                    "weights": str(PRETRAIN_WEIGHTS),
                    "enrichment": enrich_result},
        )
        logger.info("Pretraining done — %s", PRETRAIN_WEIGHTS)

    except Exception as e:
        logger.exception("Pretraining failed")
        STORE.update(job_id, status="error",
                     message="Pretraining failed",
                     error=f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def _reward_for_generation(vocab: Vocab, disease: str,
                           generated_text: str) -> float:
    tokens = tokenize(generated_text)
    token_set = set(tokens)
    key = _normalise_key(disease)
    ref = _symptom_library().get(key)
    if ref is None:
        for k, v in _symptom_library().items():
            if key and (key in k or k in key):
                ref = v
                break
    if ref is None:
        return 0.0
    ref_tokens: set[str] = set()
    for s in ref:
        ref_tokens.update(tokenize(s))
    if not ref_tokens:
        return 0.0
    recall = len(token_set & ref_tokens) / len(ref_tokens)
    n = len(token_set)
    if n == 0:
        return -0.5
    length_score = max(0.0, min(1.0, 1.0 - abs(n - 15) / 15.0))
    redundancy = 1.0 - (len(token_set) / max(len(tokens), 1))
    too_long = max(0, len(tokens) - 40) / 20.0
    return recall + 0.3 * length_score - 0.4 * redundancy - 0.1 * too_long


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------

def _run_ppo(job_id: str, epochs: int = DEFAULT_PPO_EPOCHS) -> None:
    try:
        # ---- API enrichment ----------------------------------------------
        STORE.update(job_id, status="running", progress=0.01,
                     message="Enriching symptom library from APIs…")
        enrich_result = _enrich_symptom_library_from_api(
            job_id, progress_start=0.02, progress_end=0.10
        )
        logger.info("PPO enrichment: %s", enrich_result)

        with STATE.lock:
            if STATE.model is None or STATE.vocab is None:
                raise RuntimeError("Run pretraining first.")
            model = STATE.model
            vocab = STATE.vocab

        STORE.update(job_id, status="running", progress=0.11,
                     message="Freezing reference model for KL…")
        ref = copy.deepcopy(model)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False
        with STATE.lock:
            STATE.ref_model = ref

        optim = torch.optim.AdamW(model.parameters(), lr=2e-5,
                                  weight_decay=0.01, betas=(0.9, 0.95))

        diseases = list(_symptom_library().keys())
        samples_per_epoch = max(4, len(diseases))
        total_steps = epochs * samples_per_epoch
        step = 0
        total_reward = 0.0
        baseline = 0.0

        pad_id = vocab.stoi[PAD]
        eos_id = vocab.stoi[EOS]
        stop_id = vocab.stoi[STOP]

        for epoch in range(epochs):
            random.shuffle(diseases)
            for key in diseases[:samples_per_epoch]:
                model.train()
                disease = _canonical_disease_from_key(key)
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

                r = _reward_for_generation(vocab, disease, generated_text)
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
                loss = (-adv_t.clamp(-3.0, 3.0) * nll
                        + 0.05 * kl - 0.01 * entropy)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                step += 1
                p = 0.11 + 0.83 * (step / max(total_steps, 1))
                STORE.update(job_id, progress=p,
                             message=f"PPO {epoch+1}/{epochs} — "
                                     f"step {step}/{total_steps} — "
                                     f"reward {r:.3f}")

        STATE.last_ppo_reward = total_reward / max(step, 1)

        torch.save({"state_dict": model.state_dict(),
                    "config": model.config,
                    "vocab": vocab.to_list(),
                    "epochs": epochs,
                    "avg_reward": STATE.last_ppo_reward}, PPO_WEIGHTS)

        with STATE.lock:
            STATE.ppo_trained = True

        STORE.update(
            job_id, status="done", progress=1.0,
            message=(f"PPO complete — avg reward "
                     f"{STATE.last_ppo_reward:.3f} "
                     f"(library appended: "
                     f"{enrich_result.get('symptoms_added', 0)} new symptoms)"),
            output_path=str(PPO_WEIGHTS),
            result={"avg_reward": STATE.last_ppo_reward,
                    "weights": str(PPO_WEIGHTS),
                    "enrichment": enrich_result},
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
        clean: List[str] = []
        seen = set()
        for ft in fragments:
            text = _clean_symptom_fragment(" ".join(ft))
            if not text:
                continue
            if text.lower() in seen:
                continue
            seen.add(text.lower())
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

        fresh: List[str] = []
        seen_fresh: set[str] = set()
        for s in api_symptoms:
            if not isinstance(s, str):
                continue
            t = s.strip()
            if not t:
                continue
            key = t.lower()
            if key in existing_lower or key in seen_fresh:
                continue
            seen_fresh.add(key)
            fresh.append(t)

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

    return {
        "status": "ok",
        "diseases_processed": total,
        **stats,
        "library_size": refreshed.get("symptom_library_size", len(library)),
    }