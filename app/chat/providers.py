"""Model provider switch. Vertex stays the default and does not add an API key."""

from __future__ import annotations

from app.core.config import load_data_settings


class ModelProvider:
    name = "vertex"

    def describe(self) -> dict:
        return {"provider": self.name, "uses_api_key": False}


class LocalProvider(ModelProvider):
    name = "lfm3"

    def describe(self) -> dict:
        return {"provider": self.name, "uses_api_key": False, "status": "configured_not_default"}


def current_provider() -> ModelProvider:
    if load_data_settings().model_provider == "lfm3":
        return LocalProvider()
    return ModelProvider()
