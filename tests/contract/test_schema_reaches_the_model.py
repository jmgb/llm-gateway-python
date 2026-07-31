"""A requested schema has to reach the model, one way or another.

`ResponseFormat.JSON_SCHEMA` is a caller asking for a specific shape. An
adapter that declares `structured_outputs=True` hands the schema to the
provider, which enforces it. An adapter that declares `False` cannot do that —
but "cannot enforce" is not the same as "may discard". Today those adapters
send `{"type": "json_object"}` and drop the schema entirely, so the model is
asked for *some* JSON and never told which.

The failure that follows is the expensive kind, because nothing looks broken:

* the model answers valid JSON with keys it invented;
* the gateway validates afterwards, correctly rejects it, and bills it;
* the next model in the plan answers instead.

So every structured call through such a provider is served by the fallback, at
the fallback's price, plus the discarded attempt. The result is right, which is
why no behavioural test catches it: the cost is not in the logic, it is on the
invoice. Two consumers of this package hit exactly this against Groq.

These tests use a client that behaves like a model given no schema — it echoes
back whichever keys it was shown, and invents one when it was shown none.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import LLMRequest, Message, ResponseFormat
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter

ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT: dict[str, tuple[Callable[[Any], Any], str]] = {
    "groq": (GroqAdapter, "openai/gpt-oss-120b"),
    "openrouter": (OpenRouterAdapter, "deepseek/deepseek-v4-flash"),
}


class Answer(BaseModel):
    verdict: str
    rationale: str


def _echoing_client(seen: dict[str, Any]) -> Any:
    """A model with no schema of its own: it answers with what it was told.

    Given field names anywhere in the request it returns exactly those, which
    is the best case for a provider that does not enforce anything. Given none
    it invents its own key, which is what really happens in production.
    """

    async def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        blob = json.dumps(kwargs, default=str)
        fields = [name for name in Answer.model_fields if name in blob]
        payload = {name: "..." for name in fields} if fields else {"summary": "..."}
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload)),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model=kwargs["model"],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _request(model: str) -> LLMRequest:
    return LLMRequest(
        model=model,
        system_prompt="You are a careful assistant.",
        messages=(Message("user", "a question"),),
        response_format=ResponseFormat.JSON_SCHEMA,
        response_schema=Answer,
    )


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_the_field_names_reach_a_provider_that_cannot_enforce_them(provider: str) -> None:
    """Without the field names in the payload, the model cannot guess them."""
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))

    await adapter.generate(_request(model), model=model)

    payload = json.dumps(seen, default=str)
    for name in Answer.model_fields:
        assert name in payload, f"{provider} never told the model about {name!r}"


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_the_answer_satisfies_the_requested_schema(provider: str) -> None:
    """The end the caller cares about: a model shown the shape returns it."""
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    adapter = adapter_class(_echoing_client({}))

    response = await adapter.generate(_request(model), model=model)

    assert response.output_text is not None
    Answer.model_validate_json(response.output_text)


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_the_word_json_reaches_a_provider_that_requires_it(provider: str) -> None:
    """Groq rejects `json_object` outright unless the messages mention JSON.

    OpenAI documents the same requirement, and `openai.py` already carries the
    system prompt as a message because of it. Here the adapter asks for JSON
    mode itself, so it owns the requirement rather than hoping the caller's
    prompt happens to satisfy it.
    """
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))

    await adapter.generate(_request(model), model=model)

    assert "json" in json.dumps(seen["messages"], default=str).casefold()


def _plain_json_request(model: str, *, system_prompt: str | None = None) -> LLMRequest:
    return LLMRequest(
        model=model,
        system_prompt=system_prompt,
        messages=(Message("user", "a question"),),
        response_format=ResponseFormat.JSON_OBJECT,
    )


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_a_plain_json_request_carries_no_schema(provider: str) -> None:
    """`JSON_OBJECT` asks for no particular shape, so none is described."""
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))

    await adapter.generate(_plain_json_request(model), model=model)

    assert "verdict" not in json.dumps(seen, default=str)


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_a_plain_json_request_still_says_json(provider: str) -> None:
    """Groq answers HTTP 400 when `json_object` is asked for without the word.

    `JSON_OBJECT` describes no schema, so the schema instruction does not apply
    — but the requirement is about the mode, not about the shape. Whoever sets
    `response_format` owns satisfying it; leaving it to the caller's prompt
    turns a plain JSON request into a 400 whenever they did not think to
    mention the word.
    """
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))

    await adapter.generate(_plain_json_request(model), model=model)

    assert "json" in json.dumps(seen["messages"], default=str).casefold()


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_a_caller_who_already_said_json_is_left_alone(provider: str) -> None:
    """The requirement is already met, so nothing is added to their prompt."""
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))
    spoken_for = "Answer as JSON, with one key per finding."

    await adapter.generate(_plain_json_request(model, system_prompt=spoken_for), model=model)

    assert seen["messages"][0]["content"] == spoken_for


@pytest.mark.parametrize("provider", sorted(ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT))
async def test_the_caller_system_prompt_survives(provider: str) -> None:
    """Describing the schema must not cost the caller their own instructions."""
    adapter_class, model = ADAPTERS_WITHOUT_SCHEMA_ENFORCEMENT[provider]
    seen: dict[str, Any] = {}
    adapter = adapter_class(_echoing_client(seen))

    await adapter.generate(_request(model), model=model)

    assert "You are a careful assistant." in json.dumps(seen["messages"], default=str)
