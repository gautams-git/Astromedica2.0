"""Shared knowledge-base package."""

from app.knowledge.registry import KNOWLEDGE_SPECS, RL_SPECS, spec_by_id

__all__ = ["KNOWLEDGE_SPECS", "RL_SPECS", "spec_by_id"]
