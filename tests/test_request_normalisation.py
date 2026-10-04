"""Each model attempt receives only the request options it accepts."""

from __future__ import annotations

from llm_gateway import (
    FallbackPolicy,
    FileAttachment,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    ProviderResponse,
    RateLimitedError,
    TokenUsage,
)


def test_a_remote_attachment_only_keeps_the_provider_file_id() -> None:
    attachment = FileAttachment("file-1")

    assert not hasattr(attachment, "mime_type")


class RecordingAdapter:
    def __init__(self, name: str, *responses: ProviderResponse | Exception) -> None:
        self.name = name
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.requests.append(request)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(text: str) -> ProviderResponse:
    return ProviderResponse(
        output_text=text,
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        finish_reason="stop",
    )


def _gateway(*adapters: RecordingAdapter) -> LLMGateway:
    registry = ProviderRegistry()
    for adapter in adapters:
        # Catalogued models route by provider name; `gpt-` reaches the
        # uncatalogued id the last test sends.
        registry.register(adapter, model_prefixes=("gpt-",) if adapter.name == "openai" else ())
    return LLMGateway(registry=registry)


def _request(model: str, *, temperature: float, fallback: str | None = None) -> LLMRequest:
    return LLMRequest(
        model=model,
        messages=(Message("user", "hello"),),
        temperature=temperature,
        fallback_policy=(
            FallbackPolicy.models_in_order(fallback)
            if fallback is not None
            else FallbackPolicy.disabled()
        ),
    )


async def test_temperature_is_dropped_for_a_model_that_rejects_it() -> None:
    adapter = RecordingAdapter("openai", _response("x"))

    await _gateway(adapter).generate(_request("gpt-6-luna", temperature=0.2))

    assert adapter.requests[0].temperature is None


async def test_temperature_survives_for_a_model_that_accepts_it() -> None:
    adapter = RecordingAdapter("openrouter", _response("x"))

    await _gateway(adapter).generate(_request("x-ai/grok-4.5", temperature=0.2))

    assert adapter.requests[0].temperature == 0.2


async def test_a_fallback_does_not_inherit_a_temperature_its_model_rejects() -> None:
    primary = RecordingAdapter("openrouter", RateLimitedError("429"))
    fallback = RecordingAdapter("openai", _response("from the fallback"))

    result = await _gateway(primary, fallback).generate(
        _request(
            "x-ai/grok-4.5",
            temperature=0.2,
            fallback="gpt-5.6-terra",
        )
    )

    assert result.output == "from the fallback"
    assert primary.requests[0].temperature == 0.2
    assert fallback.requests[0].temperature is None


async def test_an_uncatalogued_model_keeps_the_temperature_it_was_given() -> None:
    """Silence in the catalogue is not evidence that an option is rejected."""
    adapter = RecordingAdapter("openai", _response("x"))

    await _gateway(adapter).generate(_request("gpt-brand-new", temperature=0.2))

    assert adapter.requests[0].temperature == 0.2
