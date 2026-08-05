"""A declared capability must be one a caller can actually exercise.

`ProviderCapabilities` exists so an application can ask before requesting
something that would silently degrade. That only works if every declared
capability has a way into a request: a `True` a caller cannot reach is worse
than a `False`, because it reads as available and answers nothing.

This test is written against `LLMRequest`, not against a hard-coded list, so it
stops demanding `False` on the day the contract grows the field.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from typing import Any

import pytest

from llm_gateway import (
    ImageRequest,
    LLMRequest,
    ProviderCapabilities,
    TranscriptionRequest,
    VideoRequest,
)

# Capability -> the request field that would let a caller use it.
REQUEST_FIELD_FOR_CAPABILITY = {
    "function_calling": "tools",
    "inline_files": "attachments",
    "remote_files": "attachments",
    "audio_transcription": "audio",
    "image_generation": "prompt",
    "image_editing": "image",
    "video_generation": "prompt",
    "video_from_image": "image",
    "video_webhooks": "webhook_url",
    "structured_outputs": "response_schema",
    "json_mode": "response_format",
    "reasoning_effort": "reasoning_effort",
    "conversation_history": "messages",
    "verbosity": "verbosity",
    "upstream_routing": "routing",
}


def _declared_capabilities() -> dict[str, Any]:
    """Every provider module's declared capability set, by module name."""
    package = importlib.import_module("llm_gateway.providers")
    found: dict[str, Any] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"llm_gateway.providers.{info.name}")
        capabilities = getattr(module, "CAPABILITIES", None)
        if capabilities is not None:
            found[info.name] = capabilities
    return found


def _request_fields() -> set[str]:
    return (
        {field.name for field in dataclasses.fields(LLMRequest)}
        | {field.name for field in dataclasses.fields(TranscriptionRequest)}
        | {field.name for field in dataclasses.fields(ImageRequest)}
        | {field.name for field in dataclasses.fields(VideoRequest)}
    )


def test_every_capability_names_the_request_field_that_reaches_it() -> None:
    """A new capability is a new promise, so it needs a way to be kept."""
    declared = {field.name for field in dataclasses.fields(ProviderCapabilities)}
    declared -= {"reports_token_usage"}  # about the reply, not about the request

    assert declared == set(REQUEST_FIELD_FOR_CAPABILITY)


@pytest.mark.parametrize("provider", sorted(_declared_capabilities()))
def test_no_adapter_promises_what_the_request_cannot_express(provider: str) -> None:
    capabilities = _declared_capabilities()[provider]
    fields = _request_fields()

    unreachable = {
        capability
        for capability, request_field in REQUEST_FIELD_FOR_CAPABILITY.items()
        if request_field not in fields and capabilities.require(capability)
    }

    assert unreachable == set(), (
        f"{provider} declares {sorted(unreachable)}, which no request can ask for"
    )
