"""Reasoning tokens mean the same thing in every adapter.

Providers disagree about where thinking is counted, so each adapter normalises
it at its own boundary. That agreement is invisible from inside any single
adapter: it exists only if *all* of them uphold it, and it is exactly what a
fifth adapter reintroduces by copying a fourth. Cost is computed from
``output_tokens`` alone, so an adapter that leaves reasoning outside it
under-bills, and one that folds it in twice bills it twice.

The private ``_usage`` mapper is the boundary in question, so it is what these
tests call.
"""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# One call, reported by each provider in its own shape: ten tokens of prompt,
# thirty-four of output, of which twenty were spent thinking and fourteen came
# back as text. Every adapter must arrive at the same three numbers.
INPUT_TOKENS = 10
OUTPUT_TOKENS = 34
REASONING_TOKENS = 20
VISIBLE_TOKENS = 14

NATIVE_PAYLOADS: dict[str, Any] = {
    "openai": SimpleNamespace(
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        output_tokens_details=SimpleNamespace(reasoning_tokens=REASONING_TOKENS),
    ),
    "gemini": SimpleNamespace(
        prompt_token_count=INPUT_TOKENS,
        # Gemini is the odd one out: thoughts are *not* inside the candidates.
        candidates_token_count=VISIBLE_TOKENS,
        thoughts_token_count=REASONING_TOKENS,
    ),
    "groq": SimpleNamespace(
        prompt_tokens=INPUT_TOKENS,
        completion_tokens=OUTPUT_TOKENS,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=REASONING_TOKENS),
    ),
    "openrouter": SimpleNamespace(
        prompt_tokens=INPUT_TOKENS,
        completion_tokens=OUTPUT_TOKENS,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=REASONING_TOKENS),
    ),
}


def _usage_mappers() -> dict[str, ModuleType]:
    """Every provider module that maps a usage payload."""
    package = importlib.import_module("llm_gateway.providers")
    found: dict[str, ModuleType] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"llm_gateway.providers.{info.name}")
        if hasattr(module, "_usage"):
            found[info.name] = module
    return found


def test_every_adapter_is_covered_by_this_contract() -> None:
    """A new adapter has to declare how its provider reports thinking."""
    assert set(_usage_mappers()) == set(NATIVE_PAYLOADS)


@pytest.mark.parametrize("provider", sorted(NATIVE_PAYLOADS))
def test_reasoning_is_normalised_into_the_output_count(provider: str) -> None:
    usage = _usage_mappers()[provider]._usage(NATIVE_PAYLOADS[provider])

    assert usage.output_tokens == OUTPUT_TOKENS
    assert usage.reasoning_tokens == REASONING_TOKENS
    assert usage.visible_output_tokens == VISIBLE_TOKENS


@pytest.mark.parametrize("provider", sorted(NATIVE_PAYLOADS))
def test_reasoning_is_never_billed_on_top_of_the_output(provider: str) -> None:
    usage = _usage_mappers()[provider]._usage(NATIVE_PAYLOADS[provider])

    assert usage.billable_output_tokens == usage.output_tokens
    assert usage.reasoning_tokens <= usage.output_tokens


@pytest.mark.parametrize("provider", sorted(NATIVE_PAYLOADS))
def test_an_unreported_usage_payload_is_unknown_not_zero(provider: str) -> None:
    usage = _usage_mappers()[provider]._usage(None)

    assert usage.complete is False
    assert usage.output_tokens is None


# The same call, reported inconsistently: more thinking than output, or a
# negative count, neither of which TokenUsage will represent. The reply has
# already been billed by the time it is read, so refusing it loses the attempt.
INCONSISTENT_PAYLOADS: dict[str, Any] = {
    "openai": SimpleNamespace(
        input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS,
        output_tokens_details=SimpleNamespace(reasoning_tokens=OUTPUT_TOKENS + 1),
    ),
    "gemini": SimpleNamespace(
        prompt_token_count=INPUT_TOKENS,
        candidates_token_count=VISIBLE_TOKENS,
        thoughts_token_count=-REASONING_TOKENS,
    ),
    "groq": SimpleNamespace(
        prompt_tokens=INPUT_TOKENS,
        completion_tokens=OUTPUT_TOKENS,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=OUTPUT_TOKENS + 1),
    ),
    "openrouter": SimpleNamespace(
        prompt_tokens=INPUT_TOKENS,
        completion_tokens=OUTPUT_TOKENS,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=OUTPUT_TOKENS + 1),
    ),
}


def test_every_adapter_is_given_an_inconsistent_payload() -> None:
    assert set(INCONSISTENT_PAYLOADS) == set(NATIVE_PAYLOADS)


@pytest.mark.parametrize("provider", sorted(INCONSISTENT_PAYLOADS))
def test_an_inconsistent_breakdown_is_dropped_rather_than_raised(provider: str) -> None:
    """A ValueError here would escape the adapter as an untyped failure, and
    the attempt it belongs to — already billed — would never be counted."""
    usage = _usage_mappers()[provider]._usage(INCONSISTENT_PAYLOADS[provider])

    assert usage.input_tokens == INPUT_TOKENS
    assert usage.reasoning_tokens is None


@pytest.mark.parametrize("provider", sorted(NATIVE_PAYLOADS))
def test_a_negative_count_is_unreported_rather_than_raised(provider: str) -> None:
    payload = SimpleNamespace(**vars(NATIVE_PAYLOADS[provider]))
    for field in ("input_tokens", "prompt_token_count", "prompt_tokens"):
        if hasattr(payload, field):
            setattr(payload, field, -1)

    usage = _usage_mappers()[provider]._usage(payload)

    assert usage.input_tokens is None
    assert usage.complete is False


def test_groq_reports_cached_prompt_tokens_like_openrouter() -> None:
    """Both speak Chat Completions; a cache hit Groq reports was being lost."""
    payload = SimpleNamespace(
        **vars(NATIVE_PAYLOADS["groq"]), prompt_tokens_details=SimpleNamespace(cached_tokens=4)
    )

    assert _usage_mappers()["groq"]._usage(payload).cached_input_tokens == 4
