"""What every Chat Completions adapter reads and writes the same way.

Groq and OpenRouter are separate providers with separate capabilities, but the
wire shape they share is identical, and keeping two copies of it is how one of
them stopped reading the cached-token count the other already reported. Only
the shape lives here; what each provider sends on top of it stays in its own
adapter.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.contracts import LLMRequest
from llm_gateway.providers.base import reported_usage
from llm_gateway.providers.schema_prompt import system_prompt_for
from llm_gateway.usage import TokenUsage


def base_messages(request: LLMRequest) -> list[dict[str, Any]]:
    """The system prompt, with whatever the format needs said, then the turns."""
    messages: list[dict[str, Any]] = []
    system_prompt = system_prompt_for(request)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend({"role": m.role, "content": m.content} for m in request.messages)
    return messages


def first_choice(raw: Any) -> Any:
    choices = getattr(raw, "choices", None) or ()
    return choices[0] if choices else None


def token_usage(raw: Any) -> TokenUsage:
    if raw is None:
        return TokenUsage.unknown()
    return reported_usage(
        input_tokens=getattr(raw, "prompt_tokens", None),
        # Chat Completions counts reasoning inside completion_tokens, so the
        # breakdown changes no amount. It is read anyway: without it a thinking
        # model looks like it returned every token it was billed for. Models
        # that do not think report no details, and the breakdown stays unknown.
        output_tokens=getattr(raw, "completion_tokens", None),
        reasoning_tokens=getattr(
            getattr(raw, "completion_tokens_details", None), "reasoning_tokens", None
        ),
        cached_input_tokens=getattr(
            getattr(raw, "prompt_tokens_details", None), "cached_tokens", None
        ),
    )
