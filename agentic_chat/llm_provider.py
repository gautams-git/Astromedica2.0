"""Model-agnostic interface. Subclasses implement only `_call`."""
import json
from typing import Any, Dict, Optional

from agentic_chat import config as cfg
from agentic_chat.prompts import EXTRACT_SCHEMA, EXTRACT_SYSTEM, PHRASE_SYSTEM


class ChatLLM:
    name = "base"

    async def _call(self, system: str, user: str,
                    schema: Optional[dict], max_tokens: int) -> str:
        raise NotImplementedError

    async def extract(self, question: str, answer: str,
                      rag_examples: str = "") -> Dict[str, Any]:
        prompt = EXTRACT_SYSTEM.format(
            question=question,
            rag_examples=rag_examples or "(no examples)",
        )
        raw = await self._call(prompt, answer, EXTRACT_SCHEMA, 200)
        return json.loads(raw)

    async def phrase(self, bank: str, previous_answer: str,
                     symptoms: list) -> str:
        user = f"Question: {bank}\nPatient's last answer: {previous_answer}"
        text = await self._call(PHRASE_SYSTEM, user, None, 80)
        return text.strip().strip('"')


def get_provider() -> ChatLLM:
    if cfg.CHAT_LLM == "lfm":
        from agentic_chat.llm_lfm import LFMProvider
        return LFMProvider()
    from agentic_chat.llm_gemini import GeminiProvider
    return GeminiProvider()