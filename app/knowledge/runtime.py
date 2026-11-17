"""Activate one knowledge-base service for the process."""

from __future__ import annotations

import logging

from app.core.config import DataSettings, load_data_settings
from app.core.errors import MissingDatasetError
from app.knowledge.bindings import publish
from app.knowledge.service import KnowledgeBaseService

_LOG = logging.getLogger("astromedica.knowledge")
_service: KnowledgeBaseService | None = None


def activate_knowledge_base(plane=None, settings: DataSettings | None = None) -> KnowledgeBaseService:
    global _service
    settings = settings or load_data_settings()
    if plane is None:
        from app_core.runtime import get_services

        plane = getattr(get_services().repository, "data_plane", None)
    service = KnowledgeBaseService(plane, settings)
    if settings.kb_source == "firestore":
        if plane is None:
            raise MissingDatasetError("Firestore knowledge was requested but no data plane is configured")
        service.warm_required()
        publish(service)
        _LOG.info("knowledge base loaded from Firestore (%s datasets)", len(service.stats()["cached"]))
    elif settings.kb_strict:
        raise MissingDatasetError("KB_STRICT requires KB_SOURCE=firestore with a loaded dataset")
    else:
        _LOG.warning(
            "knowledge base source is %s. This is the explicit offline source, not a production fallback.",
            settings.kb_source,
        )
    _service = service
    return service


def get_knowledge_base() -> KnowledgeBaseService:
    if _service is None:
        raise MissingDatasetError("knowledge base has not been activated")
    return _service


def assert_calculation_source() -> None:
    """Refuse algorithm execution when strict mode is on and Firestore is not the source."""
    settings = load_data_settings()
    if not settings.kb_strict:
        return
    service = get_knowledge_base()
    if service.settings.kb_source != "firestore":
        raise MissingDatasetError("strict mode refuses embedded knowledge")
    service.warm_required()
