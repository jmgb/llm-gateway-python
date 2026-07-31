"""Real calls to real providers, because some failures only they can produce.

Every other test in this suite drives an adapter with a fake client, which is
what keeps the suite free, fast and offline. That design has one blind spot,
and it is exactly where this package lost money: a fake client accepts any
payload and returns whatever was queued, so it cannot reject a request the way
a provider does, and it cannot invent field names the way a model does.

Two real failures were invisible to the fakes:

* Groq answers **HTTP 400** when `json_object` is requested and the word "json"
  appears nowhere in the messages. No fake will ever produce that status.
* A model given no schema answers valid JSON under keys of its own choosing,
  which is a *schema violation the gateway pays for* rather than an error.

So these tests spend real tokens on the smallest prompts that still prove the
point. They are marked `live` and deselected by default: `uv run pytest` stays
offline and free. Run them deliberately:

    uv sync --extra groq --extra openrouter
    GROQ_API_KEY=... uv run pytest -m live

Note the extras: the canonical environment installs none, because that is what
makes the "extra not installed" tests meaningful. So these tests skip when the
SDK is missing *and* when the key is absent — a contributor holding neither
should not see a red suite.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import (
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    ResponseFormat,
    RetryPolicy,
)
from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.factories import create_groq_client, create_openrouter_client
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter

pytestmark = pytest.mark.live

GROQ_MODEL = "openai/gpt-oss-120b"
OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"

# Enough budget to think *and* answer. A reasoning model spends this allowance
# on reasoning first, so a tight cap makes it hit the ceiling before emitting a
# single character of JSON — which Groq reports as HTTP 400
# `json_validate_failed` with an empty `failed_generation`, indistinguishable
# at a glance from the malformed-request 400 these tests exist to catch.
MAX_TOKENS = 1500
QUESTION = "Is a bicycle lane good for a city?"


class Verdict(BaseModel):
    """Field names a model would not invent on its own."""

    verdict: str
    rationale: str


def _model(provider: str) -> str:
    return GROQ_MODEL if provider == "groq" else OPENROUTER_MODEL


@pytest.fixture(params=["groq", "openrouter"])
def provider(request: pytest.FixtureRequest) -> Iterator[str]:
    yield request.param


@pytest.fixture
async def adapter(provider: str) -> AsyncIterator[Any]:
    """A real adapter, whose SDK client is closed before the test ends.

    Two reasons to skip rather than fail, and both describe the canonical
    development environment: it installs no provider extra, and a contributor
    need not hold an account with every provider. Either gap is a test that
    cannot run, not a test that failed.

    Closing the client matters because leaving it to be garbage-collected emits
    `ResourceWarning` for the open socket, and this suite turns warnings into
    errors — so an unclosed client would fail a test that had already passed.
    """
    variable = "GROQ_API_KEY" if provider == "groq" else "OPENROUTER_API_KEY"
    key = os.environ.get(variable)
    if not key:
        pytest.skip(f"{variable} is not set")

    build = create_groq_client if provider == "groq" else create_openrouter_client
    try:
        client = build(api_key=key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))

    adapter_class = GroqAdapter if provider == "groq" else OpenRouterAdapter
    try:
        yield adapter_class(client)
    finally:
        await client.close()


async def test_a_plain_json_request_is_accepted_without_the_caller_saying_json(
    adapter: Any, provider: str
) -> None:
    """The HTTP 400 no fake client can raise.

    The system prompt here deliberately never mentions JSON. Before the
    adapter took responsibility for the word, this exact call was a 400.
    """
    model = _model(provider)

    response = await adapter.generate(
        LLMRequest(
            model=model,
            system_prompt="You are a helpful assistant.",
            messages=(Message("user", QUESTION),),
            response_format=ResponseFormat.JSON_OBJECT,
            max_output_tokens=MAX_TOKENS,
        ),
        model=model,
    )

    assert response.output_text is not None
    assert response.output_text.strip().startswith("{")


async def test_a_schema_request_comes_back_with_the_requested_field_names(
    adapter: Any, provider: str
) -> None:
    """The silent failure: a model given no schema invents its own keys.

    No system prompt at all, so nothing but the adapter can be telling the
    model which names to use.
    """
    model = _model(provider)

    response = await adapter.generate(
        LLMRequest(
            model=model,
            messages=(Message("user", QUESTION),),
            response_format=ResponseFormat.JSON_SCHEMA,
            response_schema=Verdict,
            max_output_tokens=MAX_TOKENS,
        ),
        model=model,
    )

    assert response.output_text is not None
    Verdict.model_validate_json(response.output_text)


async def test_a_structured_call_is_answered_by_the_model_that_was_asked(
    adapter: Any, provider: str
) -> None:
    """The end that shows up on the invoice.

    Driven through the gateway rather than the adapter, with a fallback armed
    and no system prompt. A schema the model never saw would fail validation
    here and the second model would answer — which is precisely what used to
    happen on every structured call, at the second model's price.
    """
    model = _model(provider)
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=(model,))
    gateway = LLMGateway(registry=registry)

    result = await gateway.generate(
        LLMRequest(
            model=model,
            messages=(Message("user", QUESTION),),
            response_format=ResponseFormat.JSON_SCHEMA,
            response_schema=Verdict,
            max_output_tokens=MAX_TOKENS,
            retry_policy=RetryPolicy.disabled(),
            fallback_policy=FallbackPolicy.disabled(),
        )
    )

    assert isinstance(result.output, Verdict)
    assert result.execution.model_used.endswith(model.split("/")[-1])
    assert result.execution.fallback_used is False
    assert result.execution.attempt_count == 1


async def test_a_caller_who_already_asked_for_json_still_gets_json(
    adapter: Any, provider: str
) -> None:
    """Nothing is appended to a prompt that already met the requirement."""
    model = _model(provider)

    response = await adapter.generate(
        LLMRequest(
            model=model,
            system_prompt="Answer as a JSON object with a single key.",
            messages=(Message("user", QUESTION),),
            response_format=ResponseFormat.JSON_OBJECT,
            max_output_tokens=MAX_TOKENS,
        ),
        model=model,
    )

    assert response.output_text is not None
    assert response.output_text.strip().startswith("{")
