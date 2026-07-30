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
