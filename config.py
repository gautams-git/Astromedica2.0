# Shared configuration for AstroMedica
#
# Keys and secrets come from the environment / the local .env file (gitignored)
# — never write a real key in this file. Names to set are in .env.example.

import os

from dotenv import load_dotenv

# server.py loads .env too; loading here as well covers scripts run on their own.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Prokerala API Credentials ──────────────────────
# Both server.py and jataka_fetch.py import these.
PROKERALA_CLIENT_ID     = os.environ.get("PROKERALA_CLIENT_ID", "")
PROKERALA_CLIENT_SECRET = os.environ.get("PROKERALA_CLIENT_SECRET", "")

# ── Gemini / LLM Config ────────────────────────────
# No API key: Gemini is reached through Vertex AI with the Google (ADC) login,
# see gemini_client.py. Region: GEMINI_LOCATION in .env.
GEMINI_MODEL   = "gemini-2.5-flash"
LLM_PROVIDER   = "gemini"

# Checked against the live models list (2026-10-02): text-embedding-004 is no
# longer served (404); gemini-embedding-001 is the stable embedding model.
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
# Must match the kb_chunks.embedding vector(768) column in the planned DB.
GEMINI_EMBEDDING_DIM = 768

# ── Session Signing ─────────────────────────────────
# The session cookie is signed with SECRET_KEY, so this must stay constant
# across restarts or every logged-in visitor gets silently signed out.
SECRET_KEY = os.environ.get("SECRET_KEY", "")

# ── De-identification Token Secret ─────────────────
# HMAC key used to derive the opaque patient token the AI/RAG/RL layers see.
# Deliberately separate from SECRET_KEY — a session-signing key and a
# de-identification key should never be the same secret, so a leak of one
# doesn't also compromise the other.
DEIDENTIFICATION_SECRET = os.environ.get("DEIDENTIFICATION_SECRET", "")

# ── Search API (LLM 1 / Step 3B web grounding) ─────
# Serper.dev key — used by llm1_search.py's real _web_search().
SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")

_missing = [name for name in ("PROKERALA_CLIENT_ID", "PROKERALA_CLIENT_SECRET",
                              "SECRET_KEY", "DEIDENTIFICATION_SECRET") if not globals()[name]]
if _missing:
    print(f"[Config] Missing in .env: {', '.join(_missing)} - see .env.example")
