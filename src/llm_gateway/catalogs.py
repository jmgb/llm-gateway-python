"""Price tables built from the shared catalogue.

## Units

Prices are declared in **USD per million tokens**, as every provider publishes
them. Rates are consumed as **microUSD per token**. Those are the same number:

    1 USD / 1,000,000 tokens = 1e-6 USD / token = 1 microUSD / token

so the conversion is the identity, and there is no factor to get wrong. There
is a test asserting exactly that.

Models billed by audio duration or per image keep their own rate and their own
table. They are routable through the catalogue but are intentionally excluded
from the token price catalog, so a minute or a picture can never be priced as
if it were a token.

## Updating

Change the price, bump ``CATALOG_VERSION``, tag a release. Consumers pin a tag,
so nobody's cost figures move without an explicit upgrade — and every recorded
amount carries the version that produced it, so old numbers stay reconcilable.
"""

from __future__ import annotations

from decimal import Decimal

from llm_gateway.models import CATALOG_VERSION, MODEL_CATALOG
from llm_gateway.pricing import (
    ImageRate,
    ModelRate,
    StaticAudioPriceCatalog,
    StaticImagePriceCatalog,
    StaticPriceCatalog,
    StaticVideoPriceCatalog,
    VideoRate,
)


def builtin_price_catalog(
    *,
    overrides: dict[str, tuple[Decimal, Decimal]] | None = None,
    version: str | None = None,
) -> StaticPriceCatalog:
    """The shared price table, optionally overridden by a consumer.

    ``overrides`` maps a model id to ``(input, output)`` in USD per million
    tokens — for negotiated rates, or for a model this catalogue does not know
    yet. Supplying one without a ``version`` would make an amount
    unreconcilable, so pass a version that identifies *your* table.
    """
    rates = {
        model_id: info.rate
        for model_id, info in MODEL_CATALOG.items()
        if info.pricing_unit == "tokens" and info.modality == "text"
    }
    resolved_version = version or CATALOG_VERSION

    if overrides:
        if version is None:
            raise ValueError(
                "overriding prices requires a version identifying your own table, "
                f"so an amount is never attributed to {CATALOG_VERSION!r}"
            )
        for model_id, (input_price, output_price) in overrides.items():
            rates[model_id] = ModelRate(
                input_microusd_per_token=input_price,
                output_microusd_per_token=output_price,
            )

    return StaticPriceCatalog(version=resolved_version, rates=rates)


def builtin_image_price_catalog(*, version: str | None = None) -> StaticImagePriceCatalog:
    """Build the image table: token-billed and per-image models together.

    A model whose provider publishes no per-image rate is left out rather than
    priced at zero, so its cost reports UNAVAILABLE and the application can
    inject the rate it negotiated.
    """
    rates: dict[str, ImageRate] = {}
    for model_id, info in MODEL_CATALOG.items():
        if info.modality != "image":
            continue
        rate = info.image_rate
        if rate is not None:
            rates[model_id] = rate
    return StaticImagePriceCatalog(version=version or CATALOG_VERSION, rates=rates)


def builtin_video_price_catalog(*, version: str | None = None) -> StaticVideoPriceCatalog:
    """Build the video table, keyed by resolution where the provider prices it so."""
    rates: dict[str, VideoRate] = {}
    for model_id, info in MODEL_CATALOG.items():
        if info.modality != "video":
            continue
        rate = info.video_rate
        if rate is not None:
            rates[model_id] = rate
    return StaticVideoPriceCatalog(version=version or CATALOG_VERSION, rates=rates)


def builtin_audio_price_catalog(*, version: str | None = None) -> StaticAudioPriceCatalog:
    """Build the duration-based table, excluding every token model."""
    rates = {
        model_id: info.audio_rate
        for model_id, info in MODEL_CATALOG.items()
        if info.pricing_unit == "audio_minutes" and info.audio_usd_per_minute is not None
    }
    return StaticAudioPriceCatalog(version=version or CATALOG_VERSION, rates=rates)


__all__ = [
    "builtin_audio_price_catalog",
    "builtin_image_price_catalog",
    "builtin_price_catalog",
    "builtin_video_price_catalog",
]
