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


def test_openai_bills_a_reference_photo_above_its_prompt_text() -> None:
    """The two input rates differ by 60%, and a photo is twice a prompt in tokens.

    Measured against gpt-image-2.5-flare at 768x1376, quality high: 507 text
    tokens, 992 for one reference photo, 991 for the picture it returned.
    Pricing the whole input at the text rate would report 0.037 for a call
    that cost 0.040.
    """
    catalog = builtin_image_price_catalog()

    cost = catalog.estimate(
        "gpt-image-2.5-flare",
        ImageUsage(
            images=1,
            tokens=TokenUsage(input_tokens=1499, output_tokens=991),
            input_image_tokens=992,
        ),
    )

    assert cost.amount_usd == Decimal("0.040201")
    assert cost.measurement is CostMeasurement.ACTUAL


def test_a_prompt_only_generation_is_priced_entirely_at_the_text_rate() -> None:
    catalog = builtin_image_price_catalog()

    cost = catalog.estimate(
        "gpt-image-2.5-flare",
        ImageUsage(
            images=1,
            tokens=TokenUsage(input_tokens=507, output_tokens=239),
            input_image_tokens=0,
        ),
    )

    assert cost.amount_usd == Decimal("0.009705")


def test_two_input_rates_with_no_split_reported_cost_unavailable_not_the_cheaper_one() -> None:
    """Absence is not zero: an unsplit input could be either rate, so it is neither."""
    catalog = builtin_image_price_catalog()

    cost = catalog.estimate(
        "gpt-image-2.5-flare",
        ImageUsage(images=1, tokens=TokenUsage(input_tokens=1499, output_tokens=991)),
    )

    assert cost.microusd is None
    assert cost.measurement is CostMeasurement.UNAVAILABLE


def test_a_single_input_rate_ignores_a_split_no_provider_reported() -> None:
    """Gemini charges one rate for text and image alike, so nothing changes."""
    catalog = builtin_image_price_catalog()

    cost = catalog.estimate(
        "gemini-3.1-flash-image",
        ImageUsage(images=1, tokens=TokenUsage(input_tokens=10, output_tokens=1120)),
    )

    assert cost.amount_usd == Decimal("0.067205")


def test_an_image_input_rate_needs_the_token_rate_it_completes() -> None:
    with pytest.raises(ValueError, match="token rate to complete"):
        ImageRate(
            usd_per_image=Decimal("0.04"),
            input_image_microusd_per_token=Decimal("8.00"),
        )


def test_the_gpt_image_models_are_openai_image_models_in_the_catalogue() -> None:
    for model_id in ("gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
        info = lookup_model(model_id)
        assert info is not None
        assert info.provider == "openai"
        assert info.modality == "image"


def test_a_blank_size_is_refused_rather_than_sent() -> None:
    with pytest.raises(ValueError, match="size"):
        ImageRequest(model="gpt-image-2.5-flare", prompt="a cat", size="  ")


def test_a_blank_quality_is_refused_rather_than_sent() -> None:
    with pytest.raises(ValueError, match="quality"):
        ImageRequest(model="gpt-image-2.5-flare", prompt="a cat", quality="  ")
