"""
orchestrator.py — the single coordination layer everything routes through.

Nothing outside this class should call the astro engine, deidentification,
or chat_module directly. It owns session state (an in-memory dict today —
same pattern chat_module.py's conversation_store already used, just
generalized) and every response comes back through it before going
anywhere else, so state stays accurate for whatever reads it later
(eventually the Review Queue, RL Driver — neither exists yet).

Scope right now: chart generation + chat + full RAG (Steps 1-4, including
LLM 1/3B web search). Only the RL Driver has no routing here yet.
"""

from __future__ import annotations

import asyncio
import json

import chat_module
import llm1_search
import rag_pipeline
from combined_algo_clusters import compute_combined_algo_summary
from deidentification import deidentify
from rules import apply_rule_1

# The spec's exact constraint for LLM 2 (Chat Agent) — prepended onto
# chat_module's own system prompt, not a replacement for it. Kept as a
# fixed constant here rather than editing chat_module.py, so its existing
# clustering/screening-question logic stays untouched.
LLM2_SAFETY_CONSTRAINT = (
    "Never diagnose. Never prescribe. Explain astrological health signals "
    "in friendly, non-clinical language. No symptoms list. No medication."
)


class Orchestrator:
    def __init__(self):
        # session_id -> { patient_id, state, combined_scores, rag_context, ...chat_module's own keys }
        self._sessions: dict[str, dict] = {}
        # patient_id -> { combined_scores, priority_ladder } — chart-generation
        # results, keyed by patient_id because /generate-chart has no
        # session_id at all (only /start-consultation does). This is what
        # lets a chat session, started later under a different identifier,
        # find the chart data that was computed for this patient earlier.
        self._patient_chart_cache: dict[str, dict] = {}

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def remove_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _load_chart_from_disk(self, patient_id: str) -> dict:
        # The chart cache lives in memory, so after a server restart a patient
        # opened from the list (not re-generated) would chat with no RAG.
        # Rebuild the same two values from the files run_logic() saved.
        try:
            folder = f"patients/{patient_id}"
            with open(f"{folder}/processed_logic.json", encoding="utf-8") as f:
                rule1_result = json.load(f).get("rule1", {})
            cached = {
                "combined_scores": compute_combined_algo_summary(folder),
                "priority_ladder": rule1_result,
            }
        except (OSError, ValueError) as e:
            print(f"[Orchestrator] could not load chart data from disk for patient_id={patient_id}: {e}")
            return {}
        self._patient_chart_cache[patient_id] = cached
        print(f"[Orchestrator] chart data loaded from disk for patient_id={patient_id}")
        return cached

    def _get_or_create_session(self, session_id: str, patient_id: str | None = None) -> dict:
        if session_id not in self._sessions:
            cached = self._patient_chart_cache.get(patient_id, {}) if patient_id else {}
            if patient_id and not cached:
                cached = self._load_chart_from_disk(patient_id)
            self._sessions[session_id] = {
                "patient_id": patient_id,
                "state": "S0",  # placeholder until the RL Driver exists
                "combined_scores": cached.get("combined_scores"),
                "priority_ladder": cached.get("priority_ladder"),
                "rag_context": None,  # placeholder until the RAG pipeline exists
            }
        return self._sessions[session_id]

    # ── Chart generation ────────────────────────────────────────────

    def handle_chart_generation(
        self,
        patient_id: str,
        name: str,
        dob: str,
        birth_place: str,
        chart_data: dict,
        father_name: str | None = None,
        mother_name: str | None = None,
    ) -> dict:
        """
        Runs the priority ladder + reads the v1 Combined Algo consensus (from
        processed_logic.json, so it must run after run_logic()), caches the
        result under patient_id (see _patient_chart_cache above — there's no
        session_id yet at this point, /generate-chart doesn't have one), then
        de-identifies before returning.
        """
        print("-" * 50)
        print(f"[CHART GEN] patient_id={patient_id}")

        rule1_result = apply_rule_1(chart_data)
        # Same v1 Combined Algo the chat and the /results page use, so RAG
        # grounds on the organs the chat actually asks about. Needs
        # processed_logic.json, so server.py calls this after run_logic().
        combined_scores = compute_combined_algo_summary(f"patients/{patient_id}")
        self._patient_chart_cache[patient_id] = {
            "combined_scores": combined_scores,
            "priority_ladder": rule1_result,
        }
        print(f"[Orchestrator] chart cached under patient_id={patient_id} (no session yet)")

        result = deidentify(
            patient_id=patient_id,
            name=name,
            dob=dob,
            birth_place=birth_place,
            planets=chart_data.get("planets", []),
            houses=chart_data.get("houses", []),
            priority_ladder=rule1_result,
            combined_scores=combined_scores,
            father_name=father_name,
            mother_name=mother_name,
        )
        print("-" * 50)
        return result

    # ── Chat ─────────────────────────────────────────────────────────

    def start_chat_session(
        self,
        session_id: str,
        patient_id: str,
        call_gemini_fn,
        embed_fn=None,
        search_fn=None,
        parse_fn=None,
        use_rag: bool = True,
    ) -> dict:
        """
        Creates (or reuses) this session, pulling in any chart data already
        cached for patient_id (see _get_or_create_session). chat_module.
        start_chat() replaces conversation_store[session_id] wholesale (it
        doesn't know about Orchestrator's fields), so this preserves
        state/combined_scores/priority_ladder/rag_context across that and
        restores them afterward — otherwise the chart data just pulled in
        would immediately vanish again.

        Also prepends LLM2_SAFETY_CONSTRAINT onto the session's stored
        system_prompt — the spec's hard "never diagnose/prescribe" rule —
        then, if chart data is present and use_rag isn't disabled, enriches
        it further with RAG context on top of that. send_message() reads
        session["system_prompt"] fresh on every turn (confirmed by reading
        chat_module.py directly, not assumed), so both changes cover every
        future turn of the conversation automatically — nothing about
        chat_module.py itself changes.

        Known limitation, same shape as the RAG one: chat_module.start_chat()
        already calls Gemini once internally to produce the opening message,
        before this method gets a chance to prepend anything — so turn 1
        isn't covered by either the safety constraint or RAG grounding, only
        turn 2 onward is. Fixing that fully means editing chat_module.py's
        own prompt construction, which is out of scope here.

        Returns chat_module's own {"response": ..., "debug_context": ...}
        unchanged — pass it straight through, don't assume it's a string.
        """
        print("-" * 50)
        print(f"[CHAT START] session_id={session_id} patient_id={patient_id}")

        session = self._get_or_create_session(session_id, patient_id)
        had_cached_chart = bool(session.get("combined_scores"))
        if had_cached_chart:
            print(f"[Orchestrator] found cached chart data for patient_id={patient_id} -> linked to session")
        else:
            print(f"[Orchestrator] no cached chart data for patient_id={patient_id} - chat will run without RAG grounding")

        preserved = {
            "state": session.get("state", "S0"),
            "combined_scores": session.get("combined_scores"),
            "priority_ladder": session.get("priority_ladder"),
            "rag_context": session.get("rag_context"),
        }

        opening_message = chat_module.start_chat(
            session_id, patient_id, self._sessions, call_gemini_fn
        )

        self._sessions[session_id].update(preserved)

        session = self._sessions[session_id]
        print(f"[Orchestrator] state={session.get('state', 'S0')} -> safety constraint prepended to system_prompt")
        session["system_prompt"] = LLM2_SAFETY_CONSTRAINT + "\n\n" + session["system_prompt"]
        if use_rag and session.get("combined_scores"):
            # RAG only adds context; if any step fails (embedding API, DB...)
            # the chat still starts, just without it.
            try:
                session["system_prompt"] = self.build_rag_context(
                    session_id,
                    system_prompt=session["system_prompt"],
                    user_query=None,
                    embed_fn=embed_fn,
                    search_fn=search_fn,
                    parse_fn=parse_fn,
                )
            except Exception as e:
                print(f"[RAG] [FAIL] {type(e).__name__}: {str(e)[:200]} - chat continues without RAG")

        print("[LLM2] response received (turn 1)")
        print("-" * 50)
        return opening_message

    def handle_chat_message(self, session_id: str, user_message: str, call_gemini_fn) -> dict:
        """
        Wraps chat_module.send_message(), which mutates the session dict
        in place (appends to messages, saves chat_log.json) — so this is
        the "reply comes back through the Orchestrator before going to
        the patient" round trip the spec calls for, not a direct
        route -> chat_module -> patient path.
        """
        if session_id not in self._sessions:
            raise ValueError(
                f"no session '{session_id}' — call start_chat_session() first"
            )

        print("-" * 50)
        preview = user_message if len(user_message) <= 60 else user_message[:57] + "..."
        print(f"[User] \"{preview}\"")
        print(f"[Orchestrator] round trip: chat_module.send_message() -> Orchestrator -> patient")

        result = chat_module.send_message(
            session_id, user_message, self._sessions, call_gemini_fn
        )
        print(f"[CHAT MESSAGE] session_id={session_id} turn={result.get('turn')}")
        print(f"[LLM2] response received (turn {result.get('turn')})")
        print("-" * 50)
        return result

    # ── RAG (Steps 1/2/3A/3B/4 — Step 5 wiring is start_chat_session's job) ──

    def build_rag_context(
        self,
        session_id: str,
        system_prompt: str,
        user_query: str | None = None,
        embed_fn=None,
        search_fn=None,
        parse_fn=None,
        use_llm1_search: bool = True,
    ) -> str:
        """
        Runs Query Builder -> Query Embedding -> pgvector retrieval (3A) ->
        LLM 1 web search (3B) -> Augmenter, and stores the resulting prompt
        in session['rag_context'] before returning it — same "state updates
        before anything moves on" rule as the other handlers.

        3B is scoped to just the single top-ranked organ/disease pair, not
        all 3 from Query Builder's query — keeps this first integration
        pass to one search + one Gemini parse call rather than three of
        each. Easy to widen to all 3 pairs later if needed.

        Deliberately synchronous, matching chat_module.py's own convention:
        async network calls (embedding, search) are wrapped in asyncio.run()
        right here, not exposed as an outer async method — each asyncio.run()
        call is sequential, not nested inside another, so this is safe.
        chat_module.start_chat() already does the same thing internally for
        call_gemini_fn — if this method were async instead, calling it from
        start_chat_session() would nest a second asyncio.run() inside
        chat_module's, which Python does not allow (RuntimeError: cannot be
        called from a running event loop).

        embed_fn / search_fn default to the real Gemini/search calls but can
        be swapped for stubs in tests, same pattern chat_module already uses
        for call_gemini_fn.
        """
        if session_id not in self._sessions:
            raise ValueError(f"no session '{session_id}' — nothing to build RAG context from")

        session = self._sessions[session_id]
        combined_scores = session.get("combined_scores") or {}
        priority_ladder = session.get("priority_ladder") or {}

        recent_turns = [m.get("content", "") for m in session.get("messages", [])[-3:]]
        query_text = rag_pipeline.build_query(combined_scores, user_query, recent_turns)
        preview = query_text if len(query_text) <= 90 else query_text[:87] + "..."
        print(f'[RAG] Step 1 Query Builder -> "{preview}"')

        embed = embed_fn or rag_pipeline.embed_query
        embedding = asyncio.run(embed(query_text))
        print(f"[RAG] Step 2 Query Embedding -> Gemini embed call -> vector[{len(embedding)}]")

        chunks = rag_pipeline.retrieve_chunks(embedding)
        print(f"[RAG] Step 3A Retrieval (pgvector) -> db_layer.query_similar_chunks() -> {len(chunks)} chunks"
              + ("" if chunks else " (stub, no real DB yet)"))

        search_results = None
        if use_llm1_search:
            top_organs = combined_scores.get("top_organs") or []
            if top_organs:
                top_entry = top_organs[0]
                organ = top_entry.get("organ", "")
                diseases = top_entry.get("top_diseases") or []
                disease = diseases[0]["disease"] if diseases else organ
                try:
                    search_results = asyncio.run(
                        llm1_search.search_symptoms(
                            disease, organ, search_fn=search_fn, parse_fn=parse_fn
                        )
                    )
                    n_symptoms = len(search_results.get("symptoms", []))
                    print(f"[RAG] Step 3B Web Search (LLM1) -> {n_symptoms} symptoms found")
                except Exception as e:
                    print(f"[RAG] Step 3B Web Search (LLM1) -> [FAIL] {e.__class__.__name__}: {e} - search_results=None")
                    search_results = None

        augmented_prompt = rag_pipeline.build_augmented_prompt(
            system_prompt=system_prompt,
            priority_ladder=priority_ladder,
            combined_scores=combined_scores,
            kb_chunks=chunks,
            search_results=search_results,
            user_query=user_query,
        )

        used_tokens = rag_pipeline._estimate_tokens(augmented_prompt)
        print(f"[RAG] Step 4 Augmenter -> prompt assembled ({used_tokens} / {rag_pipeline.MAX_PROMPT_TOKENS} tokens used)")

        session["rag_context"] = augmented_prompt
        return augmented_prompt
