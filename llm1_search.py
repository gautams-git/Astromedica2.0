"""
llm1_search.py — Step 3B of the RAG pipeline: live web search grounding
("LLM 1 — Search AI" in the spec).

search_symptoms() is the entry point: takes a {disease, organ} pair,
searches the web (real Serper.dev call — see config.SERPER_API_KEY), and
asks Gemini (system prompt below, "clinical search") to parse the raw
results into {organs, diseases, symptoms[], web_sources[]} — the exact
shape rag_pipeline.build_augmented_prompt() expects for its Step 3B slot.

Both network-calling steps (_web_search, _parse_search_results) are
individually injectable via search_fn / parse_fn on search_symptoms(), so
tests can stub either — or both — without ever triggering a live API call
by accident. This was a real gap in an earlier version: stubbing only the
search step still let the parse step make a real Gemini call underneath.
"""

from __future__ import annotations

import json
import re

import httpx

import gemini_client
from config import GEMINI_MODEL, SERPER_API_KEY

LLM1_SYSTEM_PROMPT = (
    "You are a clinical search assistant. Given raw web search results about "
    "a disease affecting an organ, extract only the specific symptoms "
    "actually mentioned. Ignore unreliable or promotional sources (ads, "
    "forums, unverified blogs). Respond with strict JSON only, no other "
    'text: {"symptoms": ["..."], "web_sources": ["..."]}'
)

SERPER_SEARCH_URL = "https://google.serper.dev/search"


async def _web_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Real Serper.dev call. If SERPER_API_KEY turns out to be a different
    provider, this request format (POST + X-API-KEY header + {"q": ...}
    body, "organic" results array) needs updating to match that provider's
    actual API instead.
    """
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    payload = {"q": query, "num": top_k}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(SERPER_SEARCH_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "url": item.get("link", ""),
        }
        for item in data.get("organic", [])[:top_k]
    ]


def _safe_json_parse(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


async def _parse_search_results(disease: str, organ: str, raw_results: list[dict]) -> dict:
    """
    Calls Gemini to extract symptoms from raw search results. If there are
    no results, returns the honest empty shape without making a Gemini
    call at all — nothing to parse.
    """
    if not raw_results:
        return {"organs": [organ], "diseases": [disease], "symptoms": [], "web_sources": []}

    results_text = "\n\n".join(
        f"[{r.get('title', '')}]\n{r.get('snippet', '')}\n{r.get('url', '')}"
        for r in raw_results
    )
    prompt = (
        f"{LLM1_SYSTEM_PROMPT}\n\n"
        f"Disease: {disease}\nOrgan: {organ}\n\nSearch results:\n{results_text}"
    )

    # Vertex AI with the ADC login (gemini_client.py) - API keys are blocked.
    url = gemini_client.generate_url(GEMINI_MODEL)
    headers = await gemini_client.auth_headers()
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    parsed = _safe_json_parse(text)

    return {
        "organs": [organ],
        "diseases": [disease],
        "symptoms": parsed.get("symptoms", []),
        "web_sources": parsed.get("web_sources") or [r.get("url", "") for r in raw_results],
    }


async def search_symptoms(
    disease: str,
    organ: str,
    search_fn=None,
    parse_fn=None,
) -> dict:
    """
    Full Step 3B: search + parse. Both steps are independently injectable
    for testing — search_fn defaults to the real _web_search, parse_fn
    defaults to the real _parse_search_results — same dependency-injection
    pattern as embed_fn/call_gemini_fn elsewhere in this codebase.
    """
    search = search_fn or _web_search
    parse = parse_fn or _parse_search_results

    raw_results = await search(f"symptoms of {disease} {organ}")
    return await parse(disease, organ, raw_results)
