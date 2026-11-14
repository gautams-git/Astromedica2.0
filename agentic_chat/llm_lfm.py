"""LFM3 behind a local llama.cpp OpenAI-compatible server."""
import os
import httpx

from agentic_chat import config as cfg
from agentic_chat.llm_provider import ChatLLM


class LFMProvider(ChatLLM):
    name = "lfm3"
    base_url = os.getenv("LFM_BASE_URL", "http://127.0.0.1:8081")

    async def _call(self, system, user, schema, max_tokens):
        body = {
            "model": "lfm3",
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "out", "schema": schema},
            }
        async with httpx.AsyncClient(timeout=cfg.LLM_TIMEOUT) as client:
            r = await client.post(
                f"{self.base_url}/v1/chat/completions", json=body
            )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]