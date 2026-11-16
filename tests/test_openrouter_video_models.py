"""Current video-understanding routes preserve inputs and account for long context."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from llm_gateway import (
    CostMeasurement,
    ImageInput,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    StaticPriceCatalog,
    TokenUsage,
    VideoInput,
    builtin_price_catalog,
    lookup_model,
)
from llm_gateway.factories import build_registry

MODELS = (
    ("bytedance-seed/seed-2-1-turbo", "0.50", "2.50", ("text", "image", "video")),
    ("bytedance-seed/seed-2.0-lite", "0.25", "2.00", ("text", "image", "video")),
    ("bytedance-seed/seed-2.0-mini", "0.10", "0.40", ("text", "image", "video")),
    ("qwen/qwen3.8-omni-flash", "0.15", "0.47", ("text", "image", "audio", "video")),
    ("google/gemini-4-hydrogen", "1.50", "7.50", ("text", "image", "audio", "video", "file")),
    ("google/gemini-4-carbon", "2.00", "10.00", ("text", "image", "audio", "video", "file")),
)


@pytest.mark.parametrize(("model", "input_price", "output_price", "modalities"), MODELS)
async def test_current_video_models_route_with_visual_inputs_and_their_published_prices(
    model: str, input_price: str, output_price: str, modalities: tuple[str, ...]
) -> None:
    info = lookup_model(model)
    assert info is not None
    assert info.input_modalities == modalities
    assert info.input_usd_per_mtok == Decimal(input_price)
    assert info.output_usd_per_mtok == Decimal(output_price)
    calls: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="a scene"), finish_reason="stop")
            ],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=100),
        )

    registry = build_registry(
        groq_client=SimpleNamespace(),
        openrouter_client=SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    result = await LLMGateway(registry=registry).generate(
        LLMRequest(
            model=model,
            messages=(
                Message(
                    "user",
                    "Describe.",
                    images=(ImageInput(data=b"image"),),
                    videos=(VideoInput(data=b"video", mime_type="video/mp4"),),
                ),
            ),
        )
    )
    assert registry.resolve(model).name == "openrouter"
    assert calls[0]["model"] == model
    assert calls[0]["messages"][0]["content"][1]["type"] == "image_url"
    assert calls[0]["messages"][0]["content"][2]["type"] == "video_url"
    assert result.cost.microusd == int(1000 * Decimal(input_price) + 100 * Decimal(output_price))


@pytest.mark.parametrize("model", ["bytedance-seed/seed-2.0-mini", "bytedance-seed/seed-2.0-lite"])
@pytest.mark.parametrize("input_tokens", [127999, 128000, 128001, 200000])
def test_seed_long_context_pricing_switches_both_rates_at_the_published_threshold(
    model: str, input_tokens: int
) -> None:
    info = lookup_model(model)
    assert info is not None
    multiplier = 2 if input_tokens >= 128000 else 1
    cost = builtin_price_catalog().estimate(
        model, TokenUsage(input_tokens=input_tokens, output_tokens=1000)
    )
    expected = (
        input_tokens * info.input_usd_per_mtok + 1000 * info.output_usd_per_mtok
    ) * multiplier
    assert cost.microusd == int(expected.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    assert cost.measurement is CostMeasurement.ACTUAL


def test_context_pricing_uses_total_input_including_cached_tokens() -> None:
    catalog = StaticPriceCatalog(
        version="context-test",
        rates={
            "m": ModelRate(
                Decimal("1"),
                Decimal("2"),
                context_tiers=(
                    (100, Decimal("3"), Decimal("4")),
                    (200, Decimal("5"), Decimal("6")),
                ),
            )
        },
    )
    cost = catalog.estimate(
        "m", TokenUsage(input_tokens=200, cached_input_tokens=150, output_tokens=10)
    )
    assert cost.microusd == 1060
    assert cost.measurement is CostMeasurement.ACTUAL


def test_unknown_input_keeps_the_base_output_rate_as_an_explicit_lower_bound() -> None:
    catalog = StaticPriceCatalog(
        version="context-test",
        rates={
            "m": ModelRate(
                Decimal("1"),
                Decimal("2"),
                context_tiers=((100, Decimal("3"), Decimal("4")),),
            )
        },
    )
    cost = catalog.estimate("m", TokenUsage(output_tokens=10))
    assert cost.microusd == 20
    assert cost.measurement is CostMeasurement.ESTIMATED


@pytest.mark.parametrize(
    "tiers",
    [
        ((0, Decimal("1"), Decimal("2")),),
        ((100, Decimal("-1"), Decimal("2")),),
        ((100, Decimal("1"), Decimal("NaN")),),
        ((100, Decimal("1"), Decimal("2")), (100, Decimal("3"), Decimal("4"))),
        ((200, Decimal("1"), Decimal("2")), (100, Decimal("3"), Decimal("4"))),
    ],
)
def test_context_price_tiers_require_ordered_unique_positive_thresholds_and_valid_rates(
    tiers: Any,
) -> None:
    with pytest.raises(ValueError):
        ModelRate(Decimal("1"), Decimal("2"), context_tiers=tiers)


def test_retrieved_documents_count_towards_the_context_threshold() -> None:
    catalog = StaticPriceCatalog(
        version="context-test",
        rates={
            "m": ModelRate(
                Decimal("1"),
                Decimal("2"),
                context_tiers=((100, Decimal("3"), Decimal("4")),),
            )
        },
    )
    cost = catalog.estimate(
        "m", TokenUsage(input_tokens=90, retrieved_document_tokens=20, output_tokens=10)
    )
    assert cost.microusd == 370


def test_unknown_input_uses_the_cheapest_possible_output_band_as_its_lower_bound() -> None:
    catalog = StaticPriceCatalog(
        version="context-test",
        rates={
            "m": ModelRate(
                Decimal("1"),
                Decimal("2"),
                context_tiers=((100, Decimal("0.5"), Decimal("1")),),
            )
        },
    )
    cost = catalog.estimate("m", TokenUsage(output_tokens=10))
    assert cost.microusd == 10
    assert cost.measurement is CostMeasurement.ESTIMATED
