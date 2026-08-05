"""Image generation is its own operation, with its own usage and its own cost.

The catalogue is what decides: a model declared ``modality="image"`` never
enters the token path, exactly as an audio-priced model never does.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from llm_gateway import (
    ConfigurationError,
    CostMeasurement,
    GeneratedImage,
    ImageInput,
    ImageRate,
    ImageRequest,
    ImageUsage,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ProviderRegistry,
    ProviderResponse,
    StaticImagePriceCatalog,
    TokenUsage,
    builtin_image_price_catalog,
    lookup_model,
)


class UnusedAdapter:
    name = "gemini"

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("an image model must never reach the token path")


def test_an_image_model_is_declared_in_the_catalogue_not_inferred_from_its_name() -> None:
    info = lookup_model("gemini-3.1-flash-image")

    assert info is not None
    assert info.modality == "image"
    assert lookup_model("gemini-3.1-flash-lite-preview") is not None
    assert lookup_model("gemini-3.1-flash-lite-preview").modality == "text"  # type: ignore[union-attr]


def test_generate_refuses_an_image_model_instead_of_returning_empty_text() -> None:
    """The failure this prevents: Gemini returning parts the text path drops."""
    registry = ProviderRegistry()
    registry.register(UnusedAdapter(), model_prefixes=("gemini-3",))
    gateway = LLMGateway(registry=registry)

    with pytest.raises(ConfigurationError, match="generate_image"):
        import asyncio

        asyncio.run(
            gateway.generate(
                LLMRequest(
                    model="gemini-3.1-flash-image",
                    messages=(Message(role="user", content="a cat"),),
                )
            )
        )


def test_an_image_request_requires_a_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        ImageRequest(model="gemini-3.1-flash-image", prompt="   ")


def test_an_edit_source_must_carry_bytes_or_a_url() -> None:
    with pytest.raises(ValueError, match="data or url"):
        ImageInput()

    assert ImageInput(url="https://example.test/cat.png").url is not None


def test_a_generated_image_must_carry_bytes_or_a_url() -> None:
    """Providers disagree: Gemini returns bytes, Replicate returns a URL."""
    with pytest.raises(ValueError, match="data or url"):
        GeneratedImage()

    assert GeneratedImage(data=b"\x89PNG").data == b"\x89PNG"

    with pytest.raises(ValueError, match="non-empty"):
        GeneratedImage(data=b"")
    with pytest.raises(ValueError, match="non-empty"):
        GeneratedImage(url="   ")


def test_unknown_image_usage_is_never_zero_images() -> None:
    unknown = ImageUsage.unknown()

    assert unknown.images is None
    assert unknown.complete is False
    assert ImageUsage(images=1).complete is True


def test_image_usage_cannot_turn_a_provider_bug_into_negative_cost() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ImageUsage(images=-1)


def test_merging_an_unknown_attempt_taints_the_total() -> None:
    merged = ImageUsage(images=1).merge(ImageUsage.unknown())

    assert merged.images == 1
    assert merged.complete is False


def test_a_per_image_model_is_priced_by_the_images_it_returned() -> None:
    catalog = StaticImagePriceCatalog(
        version="image-test",
        rates={"black-forest-labs/flux-kontext-pro": ImageRate(usd_per_image=Decimal("0.04"))},
    )

    cost = catalog.estimate("black-forest-labs/flux-kontext-pro", ImageUsage(images=2))

    assert cost.amount_usd == Decimal("0.080000")
    assert cost.measurement is CostMeasurement.ACTUAL
    assert cost.pricing_version == "image-test"


def test_a_token_priced_image_model_is_priced_by_its_tokens() -> None:
    """Gemini bills image generation as tokens; the unit is the model's, not the operation's."""
    catalog = StaticImagePriceCatalog(
        version="image-test",
        rates={
            "gemini-3.1-flash-image": ImageRate(
                token_rate=ModelRate(
                    input_microusd_per_token=Decimal("0.50"),
                    output_microusd_per_token=Decimal("3.00"),
                )
            )
        },
    )

    cost = catalog.estimate(
        "gemini-3.1-flash-image",
        ImageUsage(images=1, tokens=TokenUsage(input_tokens=10, output_tokens=1000)),
    )

    assert cost.amount_usd == Decimal("0.003005")
    assert cost.measurement is CostMeasurement.ACTUAL


def test_gemini_image_output_uses_the_published_image_token_rate() -> None:
    catalog = builtin_image_price_catalog()

    cost = catalog.estimate(
        "gemini-3.1-flash-image",
        ImageUsage(images=1, tokens=TokenUsage(input_tokens=10, output_tokens=1120)),
    )

    assert cost.amount_usd == Decimal("0.067205")


def test_an_unpriced_image_model_costs_unavailable_never_zero() -> None:
    catalog = StaticImagePriceCatalog(version="image-test", rates={})

    cost = catalog.estimate("prunaai/p-image", ImageUsage(images=1))

    assert cost.microusd is None
    assert cost.measurement is CostMeasurement.UNAVAILABLE


def test_a_rate_that_prices_nothing_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="per-image price or token rate"):
        ImageRate()


def test_an_image_rate_cannot_declare_two_billing_units() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ImageRate(
            usd_per_image=Decimal("0.04"),
            token_rate=ModelRate(
                input_microusd_per_token=Decimal("0.50"),
                output_microusd_per_token=Decimal("60.00"),
            ),
        )
