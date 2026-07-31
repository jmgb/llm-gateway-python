"""Structured output fails the same way on every provider.

An answer that does not parse, or does not satisfy the schema, is a failed
attempt that was nevertheless billed — and the next model in the plan gets a
turn. That rule is only worth anything if it holds for every adapter, so this
matrix drives each real adapter through the gateway with a fake SDK client, and
checks the same four things: the fallback ran, the rejected attempt is on the
invoice, the totals add up exactly, and an exhausted call raises.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from llm_gateway import (
    AllAttemptsFailed,
    FailurePhase,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ProviderRegistry,
    ResponseFormat,
    StaticPriceCatalog,
)
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter
from llm_gateway.providers.openrouter import OpenRouterAdapter

INPUT_TOKENS = 10
OUTPUT_TOKENS = 5
# One microUSD per input token and two per output token: 20 per attempt.
COST_PER_ATTEMPT = INPUT_TOKENS * 1 + OUTPUT_TOKENS * 2

UNPARSEABLE = "I am afraid I cannot answer that."
WRONG_SHAPE = '{"otra_cosa": 1}'
VALID = '{"veredicto": "ok"}'


class Answer(BaseModel):
    veredicto: str


def _openai_client(texts: list[str]) -> Any:
    async def create(**kwargs: Any) -> Any:
        return SimpleNamespace(
            output_text=texts.pop(0),
            usage=SimpleNamespace(input_tokens=INPUT_TOKENS, output_tokens=OUTPUT_TOKENS),
            status="completed",
            model=kwargs["model"],
        )

    return SimpleNamespace(responses=SimpleNamespace(create=create))


def _gemini_client(texts: list[str]) -> Any:
    async def generate_content(**kwargs: Any) -> Any:
        return SimpleNamespace(
            text=texts.pop(0),
            usage_metadata=SimpleNamespace(
                prompt_token_count=INPUT_TOKENS,
                candidates_token_count=OUTPUT_TOKENS,
            ),
            candidates=(),
            model_version=kwargs["model"],
        )

    return SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
    )


def _chat_completions_client(texts: list[str]) -> Any:
    async def create(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=texts.pop(0)), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=INPUT_TOKENS, completion_tokens=OUTPUT_TOKENS),
            model=kwargs["model"],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


@dataclass(frozen=True)
class ProviderCase:
    """One provider, its fake transport, and two models it really serves."""

    adapter: Callable[[Any], Any]
    client: Callable[[list[str]], Any]
    primary: str
    fallback: str


CASES: dict[str, ProviderCase] = {
    "openai": ProviderCase(
        adapter=OpenAIAdapter,
        client=_openai_client,
        primary="gpt-5.2-2025-12-11",
        fallback="gpt-5.1-2025-11-13",
    ),
    "gemini": ProviderCase(
        adapter=GeminiAdapter,
        client=_gemini_client,
        primary="gemini-3.5-flash",
        fallback="gemini-3.5-flash-lite",
    ),
    "groq": ProviderCase(
        adapter=GroqAdapter,
        client=_chat_completions_client,
        primary="openai/gpt-oss-120b",
        fallback="openai/gpt-oss-20b",
    ),
    "openrouter": ProviderCase(
        adapter=OpenRouterAdapter,
        client=_chat_completions_client,
        primary="deepseek/deepseek-v4-flash",
        fallback="deepseek/deepseek-v4-pro",
    ),
}


def _run(provider: str, *texts: str) -> Any:
    """Drive the real adapter through the gateway with the queued answers."""
    case = CASES[provider]
    adapter = case.adapter(case.client(list(texts)))
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=(case.primary, case.fallback))
    prices = StaticPriceCatalog(
        version="matrix-1",
        rates={
            model: ModelRate(Decimal("1"), Decimal("2")) for model in (case.primary, case.fallback)
        },
    )
    gateway = LLMGateway(registry=registry, price_catalog=prices)
    return gateway.generate(
        LLMRequest(
            model=case.primary,
            messages=(Message("user", "a question"),),
            response_format=ResponseFormat.JSON_SCHEMA,
            response_schema=Answer,
            fallback_policy=FallbackPolicy.models_in_order(case.fallback),
        )
    )


def test_every_adapter_is_covered_by_this_matrix() -> None:
    """A new adapter has to state how it behaves when the output is unusable."""
    package = importlib.import_module("llm_gateway.providers")
    adapters = {
        info.name
        for info in pkgutil.iter_modules(package.__path__)
        if hasattr(importlib.import_module(f"llm_gateway.providers.{info.name}"), "CAPABILITIES")
    }

    assert adapters == set(CASES)


@pytest.mark.parametrize("provider", sorted(CASES))
async def test_an_unparseable_answer_falls_back_and_is_still_billed(provider: str) -> None:
    result = await _run(provider, UNPARSEABLE, VALID)

    assert isinstance(result.output, Answer)
    assert result.execution.fallback_used is True
    assert result.execution.model_used == CASES[provider].fallback
    rejected = result.execution.attempts[0]
    assert rejected.billable is True
    assert rejected.failure_phase is FailurePhase.OUTPUT_PARSING


@pytest.mark.parametrize("provider", sorted(CASES))
async def test_a_schema_violation_falls_back_and_is_still_billed(provider: str) -> None:
    result = await _run(provider, WRONG_SHAPE, VALID)

    assert result.execution.attempts[0].failure_phase is FailurePhase.SCHEMA_VALIDATION
    assert result.execution.attempts[0].cost.microusd == COST_PER_ATTEMPT


@pytest.mark.parametrize("provider", sorted(CASES))
async def test_the_total_is_the_exact_sum_of_both_attempts(provider: str) -> None:
    result = await _run(provider, WRONG_SHAPE, VALID)

    assert result.usage.input_tokens == 2 * INPUT_TOKENS
    assert result.usage.output_tokens == 2 * OUTPUT_TOKENS
    assert result.cost.microusd == 2 * COST_PER_ATTEMPT


@pytest.mark.parametrize("provider", sorted(CASES))
async def test_two_unusable_answers_exhaust_the_call_without_losing_the_money(
    provider: str,
) -> None:
    with pytest.raises(AllAttemptsFailed) as caught:
        await _run(provider, UNPARSEABLE, WRONG_SHAPE)

    attempts = caught.value.attempts
    assert len(attempts) == 2
    assert sum(attempt.cost.microusd or 0 for attempt in attempts) == 2 * COST_PER_ATTEMPT
    assert caught.value.last_error == "SchemaValidationError"
