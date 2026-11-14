"""All switches and budgets. Defaults reproduce today's classic chat."""
import os


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# ── Master switch ───────────────────────────────────────
CHAT_MODE = os.getenv("CHAT_MODE", "classic")               # classic | agentic

# ── Model layer ─────────────────────────────────────────
CHAT_LLM        = os.getenv("CHAT_LLM", "gemini")           # gemini | lfm
CHAT_UNDERSTAND = os.getenv("CHAT_UNDERSTAND", "rules")     # rules | shadow | llm
CHAT_PHRASE_LLM = _flag("CHAT_PHRASE_LLM", "false")

# ── RAG layer ───────────────────────────────────────────
RAG_MODE = os.getenv("RAG_MODE", "off")                     # off | shadow | on

# ── RL layer ────────────────────────────────────────────
RL_HINT        = _flag("RL_HINT", "false")
RL_HINT_SHADOW = _flag("RL_HINT_SHADOW", "true")

# ── Budgets (seconds) ───────────────────────────────────
LLM_TIMEOUT  = float(os.getenv("CHAT_LLM_TIMEOUT", "6"))
RAG_TIMEOUT  = float(os.getenv("RAG_TIMEOUT", "1.5"))
RL_HINT_WAIT = float(os.getenv("RL_HINT_WAIT", "0.3"))

# ── Safety ──────────────────────────────────────────────
MAX_TURNS = int(os.getenv("CHAT_MAX_TURNS", "15"))
DISCLAIMER = (
    "\n\nThis is an astrological analysis for informational purposes only. "
    "It is NOT a medical diagnosis. Consult a qualified healthcare professional "
    "for any health concerns."
)