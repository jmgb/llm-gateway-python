"""Provider request checks that must happen before SDK dispatch."""

from __future__ import annotations

from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError
from llm_gateway.media import ImageRequest


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


def reject_unsendable_image_options(request: ImageRequest, *, provider: str) -> None:
    """Refuse the sizing options this provider has no field for.

    ``size`` and ``quality`` reach every image adapter, and only the providers
    that publish them can honour one. Dropping either returns a picture the
    caller did not ask for — the wrong shape, or a tier that costs four times
    what they budgeted — and nothing in the reply says so.
    """
    if request.size is not None:
        raise ConfigurationError(f"{provider} does not size images as WIDTHxHEIGHT")
    if request.quality is not None:
        raise ConfigurationError(f"{provider} publishes no image quality tiers")
