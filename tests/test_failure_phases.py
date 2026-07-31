"""Attempt phases distinguish failures before, during and after provider work."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import AllAttemptsFailed, LLMGateway, LLMRequest, ProviderRegistry, ResponseFormat
from llm_gateway.providers.openai import OpenAIAdapter


class FreeFormAnswer(BaseModel):
    metadata: dict[str, str]


async def test_a_request_rejected_before_dispatch_has_a_configuration_phase() -> None:
    provider_called = False

    async def create(**_kwargs: Any) -> Any:
        nonlocal provider_called
        provider_called = True
        raise AssertionError("configuration errors must fail before provider dispatch")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    registry = ProviderRegistry()
    registry.register(OpenAIAdapter(client), model_prefixes=("gpt-",))

    with pytest.raises(AllAttemptsFailed) as caught:
        await LLMGateway(registry=registry).generate(
            LLMRequest(
                model="gpt-5.6-luna",
                response_format=ResponseFormat.JSON_SCHEMA,
                response_schema=FreeFormAnswer,
            )
        )

    attempt = caught.value.attempts[0]
    assert provider_called is False
    assert attempt.billable is False
    assert attempt.failure_phase is not None
    assert attempt.failure_phase.value == "configuration"
