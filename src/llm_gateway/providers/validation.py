"""Provider request checks that must happen before SDK dispatch."""

from __future__ import annotations

from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError


def reject_file_attachments(request: LLMRequest, *, provider: str) -> None:
    """Do not silently drop remote files on a text-only adapter."""
    if request.attachments:
        raise ConfigurationError(f"{provider} does not support remote file attachments")
