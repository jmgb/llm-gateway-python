"""A reasoning effort the catalogue accepts is one the adapter actually sends.

The two gate it independently. The gateway keeps an effort only when the
target model's catalogue entry declares it, and each adapter then decides by
model id whether to put it on the wire at all. When the two disagree nothing
raises: the call succeeds without thinking, at a quality and a price nobody
asked for. This walks every catalogued model that declares efforts through the
adapter that serves it, so a new id — a floating alias above all — cannot be
catalogued with efforts its adapter silently discards.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import LLMRequest, Message
from llm_gateway.models import MODEL_CATALOG, ModelInfo
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.groq import GroqAdapter
from llm_gateway.providers.openai import OpenAIAdapter


class Recorder:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.kwargs: dict[str, Any] = {}

    async def __call__(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self.response


def _openai(recorder: Recorder) -> Any:
    recorder.response = SimpleNamespace(output_text="x", usage=None, status="completed")
    return OpenAIAdapter(SimpleNamespace(responses=SimpleNamespace(create=recorder)))


def _gemini(recorder: Recorder) -> Any:
    recorder.response = SimpleNamespace(text="x", usage_metadata=None)
    return GeminiAdapter(
        SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=recorder)))
    )


def _groq(recorder: Recorder) -> Any:
    recorder.response = SimpleNamespace(choices=[], usage=None)
    return GroqAdapter(
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))
    )


# Per provider: how to build the adapter around a recorder, and where on the
# wire the effort ends up. A provider missing here cannot forward an effort.
WIRES: dict[str, tuple[Callable[[Recorder], Any], Callable[[dict[str, Any]], Any]]] = {
    "openai": (_openai, lambda sent: sent.get("reasoning", {}).get("effort")),
    "gemini": (
        _gemini,
        lambda sent: sent["config"].get("thinking_config", {}).get("thinking_level"),
    ),
    "groq": (_groq, lambda sent: sent.get("reasoning_effort")),
}

THINKING_MODELS: tuple[ModelInfo, ...] = tuple(
    info for info in MODEL_CATALOG.values() if info.reasoning_efforts
)


def test_there_are_thinking_models_to_check() -> None:
    assert THINKING_MODELS


@pytest.mark.parametrize("info", THINKING_MODELS, ids=lambda info: info.id)
async def test_every_declared_effort_is_forwarded_by_the_adapter_that_serves_it(
    info: ModelInfo,
) -> None:
    assert info.provider in WIRES, (
        f"{info.id} declares reasoning efforts, but no {info.provider} adapter forwards one"
    )
    build, sent_effort = WIRES[info.provider]

    for effort in info.reasoning_efforts:
        recorder = Recorder(None)
        await build(recorder).generate(
            LLMRequest(
                model=info.id, messages=(Message("user", "a question"),), reasoning_effort=effort
            ),
            model=info.id,
        )

        assert sent_effort(recorder.kwargs) == effort, f"{info.id} dropped {effort!r}"
