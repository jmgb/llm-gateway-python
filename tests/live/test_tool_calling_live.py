"""The tool round trip against real providers, which a fake client cannot prove.

A fake accepts any payload. It will happily take a tool declaration a provider
answers 400 to, and it will never reject a continuation whose `function_call`
and `function_call_output` drifted apart. Both of those are the failures this
contract exists to prevent, and both are only observable against the real API.

So each test does the whole round trip on the smallest prompt that forces a
call: declare one function, receive the call, answer it, and check the model
used the answer. Marked `live` and deselected by default:

    uv sync --extra openai --extra groq
    OPENAI_API_KEY=... GROQ_API_KEY=... uv run pytest -m live \
      tests/live/test_tool_calling_live.py -q

They skip rather than fail when the SDK or the key is missing: the canonical
development environment installs no extra, and a contributor need not hold an
account with every provider.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import Any

import pytest

from llm_gateway import (
    FunctionTool,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    RequiredTool,
    ToolResult,
)
from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.factories import create_groq_client, create_openai_client
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter

pytestmark = pytest.mark.live

OPENAI_MODEL = "gpt-5.6-luna"
GROQ_MODEL = "openai/gpt-oss-120b"

WEATHER = FunctionTool(
    name="get_weather",
    description="Current weather for a city. Call this instead of guessing.",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"],
    },
)
QUESTION = "What is the weather in Madrid right now? Use the tool."
# A temperature no city has, so an answer repeating it cannot have come from
# the model's own idea of Madrid in any season.
ANSWER = json.dumps({"celsius": 61, "sky": "clear"})


@pytest.fixture(params=["openai", "groq"])
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
async def gateway(provider: str) -> AsyncIterator[LLMGateway]:
    variable = "OPENAI_API_KEY" if provider == "openai" else "GROQ_API_KEY"
    key = os.environ.get(variable)
    if not key:
        pytest.skip(f"{variable} is not set")

    build = create_openai_client if provider == "openai" else create_groq_client
    try:
        client = build(api_key=key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))

    adapter = OpenAIAdapter(client) if provider == "openai" else GroqAdapter(client)
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=())
    try:
        yield LLMGateway(registry=registry)
    finally:
        await client.close()


def _request(provider: str, **kwargs: Any) -> LLMRequest:
    return LLMRequest(
        model=OPENAI_MODEL if provider == "openai" else GROQ_MODEL,
        messages=(Message("user", QUESTION),),
        tools=(WEATHER,),
        **kwargs,
    )


async def test_the_provider_accepts_the_declaration_and_answers_with_a_call(
    gateway: LLMGateway, provider: str
) -> None:
    """The 400 a fake cannot raise: a tool schema the API refuses."""
    result = await gateway.generate(_request(provider, tool_choice=RequiredTool("get_weather")))

    assert result.tool_calls, "the model was required to call and did not"
    call = result.tool_calls[0]
    assert call.name == "get_weather"
    assert call.id, "the provider's correlation id must survive"
    assert "madrid" in str(call.arguments.get("city", "")).lower()
    assert result.output is None


async def test_the_continuation_is_accepted_and_used(gateway: LLMGateway, provider: str) -> None:
    """A `tool` message or `function_call_output` the provider cannot correlate is a 400."""
    request = _request(provider, tool_choice=RequiredTool("get_weather"))
    first = await gateway.generate(request)
    call = first.tool_calls[0]

    second = await gateway.generate(
        replace(request, tool_choice=None, tool_results=(ToolResult(call, ANSWER),))
    )

    assert second.tool_calls == (), "the model already had its answer"
    assert "61" in second.text, "the model ignored the tool output it was given"
