"""Gemini on Vertex AI — same auth path the app already uses."""
import httpx

import gemini_client
from config import GEMINI_MODEL
from agentic_chat import config as cfg
from agentic_chat.llm_provider import ChatLLM


def _vertex_schema(schema):
    """A copy of a JSON schema with the type names in upper case ("object" -> "OBJECT"), the
    spelling the Vertex AI REST documentation uses. The shared schema is not changed: the local
    LFM3 server wants the lower-case form. Property names and enum values are left alone."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, str):
            out[key] = value.upper()
        elif key == "properties" and isinstance(value, dict):
            out[key] = {name: _vertex_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _vertex_schema(value)
        else:
            out[key] = value
    return out


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
            payload["generationConfig"]["responseSchema"] = _vertex_schema(schema)

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