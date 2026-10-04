"""OpenRouter adapter.

Chat Completions shape. The client is injected, so this module imports no SDK;
an ``AsyncOpenAI`` pointed at OpenRouter's ``base_url`` is the usual one.

OpenRouter is an aggregator, and that is the whole reason it is a provider of
its own rather than the OpenAI adapter wearing a different ``base_url``:

* Its stable surface is Chat Completions, not the Responses API.
* What a call supports depends on the model underneath, so the capabilities
  declared here are the floor every route honours, not the ceiling the best
  ones reach.
* The model that answers is not always the model requested — ``openrouter/auto``
  chooses one — so the reply's own model id is reported back.
"""

from __future__ import annotations

from typing import Any

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest, ResponseFormat
from llm_gateway.policies import RoutingPreference
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.chat_completions import base_messages, first_choice
from llm_gateway.providers.chat_completions import token_usage as _usage
from llm_gateway.providers.error_mapping import classify_provider_error
from llm_gateway.providers.validation import reject_file_attachments, reject_tools

CAPABILITIES = ProviderCapabilities(
    # Declared as the floor, not the best case: an aggregator cannot promise on
    # behalf of every model it routes to, and a capability that silently does
    # not apply is worse than one that was never claimed.
    structured_outputs=False,
    json_mode=True,
    # Tool calling is model-dependent across routes, so the adapter rejects it
    # instead of promising support that an upstream may silently ignore.
    function_calling=False,
    inline_files=False,
    remote_files=False,
    audio_transcription=False,
    reasoning_effort=False,
    verbosity=False,
    # The one capability an aggregator has and a single-provider adapter does
    # not: the same model id is served by several upstreams, and which one
    # answers changes throughput, price and availability.
    upstream_routing=True,
    conversation_history=True,
    reports_token_usage=True,
)


class OpenRouterAdapter:
    """Translates the neutral contract to Chat Completions."""

    name = "openrouter"
    capabilities = CAPABILITIES

    def __init__(self, client: Any) -> None:
        self._client = client

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        reject_file_attachments(request, provider=self.name)
        reject_tools(request, provider=self.name)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._build_messages(request),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            kwargs["max_tokens"] = request.max_output_tokens
        if request.routing.stated:
            # `provider` belongs to OpenRouter's request body, not to the
            # OpenAI SDK signature used as transport. Provider-only fields
            # must travel through the SDK's explicit passthrough.
            kwargs["extra_body"] = {"provider": _provider_routing(request.routing)}
        if request.response_format in (ResponseFormat.JSON_OBJECT, ResponseFormat.JSON_SCHEMA):
            # Schema enforcement depends on the model behind the route, so the
            # most this provider can honestly ask for is JSON. The gateway
            # validates the payload afterwards.
            kwargs["response_format"] = {"type": "json_object"}

        try:
            raw = await self._client.chat.completions.create(**kwargs)
        except Exception as error:
            raise classify_provider_error(error) from error

        choice = first_choice(raw)
        return ProviderResponse(
            output_text=getattr(getattr(choice, "message", None), "content", None),
            usage=_usage(getattr(raw, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
            model_used=getattr(raw, "model", None),
        )

    def _build_messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Carry whatever the requested format needs the model to be told.

        Whether the route behind the request enforces a schema, or requires the
        word "json" before it accepts JSON mode, is unknowable from here. So
        both are stated rather than assumed: a model that needs neither reads a
        redundant sentence, and one that needs them reads its only
        specification.
        """
        return base_messages(request)


def _provider_routing(preference: RoutingPreference) -> dict[str, Any]:
    """Only the halves the caller stated.

    An empty ``order`` or a null ``sort`` is not the same instruction as an
    absent one: OpenRouter reads what it is sent, so a blank half would narrow
    the routing the caller left open.
    """
    routing: dict[str, Any] = {}
    if preference.order:
        routing["order"] = list(preference.order)
    if preference.optimise_for is not None:
        routing["sort"] = preference.optimise_for
    return routing
