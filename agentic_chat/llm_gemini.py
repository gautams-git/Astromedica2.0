"""Gemini on Vertex AI — same auth path the app already uses."""
import httpx

import gemini_client
from config import GEMINI_MODEL
from agentic_chat import config as cfg
from agentic_chat.llm_provider import ChatLLM


class GeminiProvider(ChatLLM):
    name = "gemini"

    async def _call(self, system, user, schema, max_tokens):
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens},
        }
        if schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseSchema"] = schema

        headers = await gemini_client.auth_headers()
        async with httpx.AsyncClient(timeout=cfg.LLM_TIMEOUT) as client:
            r = await client.post(
                gemini_client.generate_url(GEMINI_MODEL),
                json=payload, headers=headers,
            )
        if r.status_code != 200:
            raise RuntimeError(
                gemini_client.auth_failure_message(r.status_code, r.text)
            )
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]