"""Provider request checks that must happen before SDK dispatch."""

from __future__ import annotations

from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError


def reject_file_attachments(request: LLMRequest, *, provider: str) -> None:
    """Do not silently drop remote files on a text-only adapter."""
    if request.attachments:
        raise ConfigurationError(f"{provider} does not support remote file attachments")


def reject_tools(request: LLMRequest, *, provider: str) -> None:
    """Do not silently drop tools on an adapter that cannot speak them.

    Dropping them returns prose where the caller is waiting for a call, and the
    application then has nothing to dispatch and no error saying why.
    """
    if request.tools or request.tool_results:
        raise ConfigurationError(f"{provider} does not support tool calling")
