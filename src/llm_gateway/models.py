"""The shared model catalogue: identity, provider and price.

This is the single place where a model's identity and pricing metadata are
updated. One versioned table beats several copies drifting apart, and a
model's price is a fact about the *provider*, not about any product — which is
exactly the test for what belongs in this package.

What stays out: **which model a feature should use**. That is a product
decision, and putting it here would make one repository's choice everybody's.

Prices are declared in USD per million tokens; :mod:`llm_gateway.catalogs`
turns this table into the price catalogues the gateways use.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Literal

from llm_gateway.contracts import ReasoningEffort
from llm_gateway.pricing import AudioRate, ImageRate, ModelRate, VideoRate

CATALOG_VERSION = "2026-08-15.1"
"""Bump on every price change. Recorded alongside every amount."""

Provider = str
PricingUnit = Literal["tokens", "audio_minutes", "images", "video_seconds"]
Modality = Literal["text", "audio", "image", "video"]
"""What a model does, which is not how it is billed: Gemini's image models are
billed in tokens and still cannot answer a text call. Routing reads this,
pricing reads ``pricing_unit``."""
OPENAI_56_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "none",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
GEMINI_3_FLASH_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
)
GEMINI_3_PRO_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
)
GROQ_GPT_OSS_REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "low",
    "medium",
    "high",
)


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """One model: provider, pricing metadata, and supported request options."""

    id: str
    provider: Provider
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    deprecated: bool = False
    notes: str = ""
    reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    supports_temperature: bool = True
    """False for models whose API rejects the option outright.

    Declared rather than inferred, and defaulting to the permissive answer:
    an entry that says nothing keeps sending what the caller asked for.
    """
    pricing_unit: PricingUnit = "tokens"
    audio_usd_per_minute: Decimal | None = None
    """Audio rate for models billed by duration instead of tokens."""
    audio_minimum_seconds: int = 0
    modality: Modality = "text"
    image_usd_per_image: Decimal | None = None
    """Per-image rate, or ``None`` where the provider publishes none — Replicate
    bills community models by GPU time, so a fixed number would be a guess."""
    image_output_usd_per_mtok: Decimal | None = None
    """Image-output token rate when it differs from the model's text output rate."""
    video_usd_per_second: Decimal | None = None
    video_usd_per_second_by_resolution: Mapping[str, Decimal] = field(
        default_factory=lambda: MappingProxyType({})
    )
    """Per-second video rates. Resolution changes the rate on every video
    provider seen so far, so the table is keyed by it when the provider
    publishes more than one."""

    @property
    def rate(self) -> ModelRate:
        """USD per million tokens read as microUSD per token — same number."""
        if self.pricing_unit != "tokens":
            raise ValueError(f"{self.id!r} is priced in {self.pricing_unit}, not tokens")
        return ModelRate(
            input_microusd_per_token=self.input_usd_per_mtok,
            output_microusd_per_token=self.output_usd_per_mtok,
        )

    @property
    def audio_rate(self) -> AudioRate:
        if self.pricing_unit != "audio_minutes" or self.audio_usd_per_minute is None:
            raise ValueError(f"{self.id!r} does not have an audio-minute rate")
        return AudioRate(
            usd_per_minute=self.audio_usd_per_minute,
            minimum_billable_seconds=self.audio_minimum_seconds,
        )

    @property
    def video_rate(self) -> VideoRate | None:
        """How a second of video is priced, or ``None`` when nothing prices it."""
        if self.modality != "video":
            raise ValueError(f"{self.id!r} does not generate video")
        if self.video_usd_per_second is None and not self.video_usd_per_second_by_resolution:
            return None
        return VideoRate(
            usd_per_second=self.video_usd_per_second,
            usd_per_second_by_resolution=self.video_usd_per_second_by_resolution,
        )

    @property
    def image_rate(self) -> ImageRate | None:
        """How one generated image is priced, or ``None`` when nothing prices it."""
        if self.modality != "image":
            raise ValueError(f"{self.id!r} does not generate images")
        if self.pricing_unit == "tokens":
            output_rate = self.image_output_usd_per_mtok or self.output_usd_per_mtok
            return ImageRate(
                token_rate=ModelRate(
                    input_microusd_per_token=self.input_usd_per_mtok,
                    output_microusd_per_token=output_rate,
                )
            )
        if self.image_usd_per_image is None:
            return None
        return ImageRate(usd_per_image=self.image_usd_per_image)


def _m(
    model_id: str,
    provider: Provider,
    input_price: str,
    output_price: str,
    *,
    deprecated: bool = False,
    notes: str = "",
    reasoning_efforts: tuple[ReasoningEffort, ...] = (),
    supports_temperature: bool = True,
    pricing_unit: PricingUnit = "tokens",
    audio_price_per_minute: str | None = None,
    audio_minimum_seconds: int = 0,
    modality: Modality = "text",
    image_price: str | None = None,
    image_output_price_per_mtok: str | None = None,
    video_price_per_second: str | None = None,
    video_prices_by_resolution: dict[str, str] | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        input_usd_per_mtok=Decimal(input_price),
        output_usd_per_mtok=Decimal(output_price),
        deprecated=deprecated,
        notes=notes,
        reasoning_efforts=reasoning_efforts,
        supports_temperature=supports_temperature,
        pricing_unit=pricing_unit,
        audio_usd_per_minute=(
            Decimal(audio_price_per_minute) if audio_price_per_minute is not None else None
        ),
        audio_minimum_seconds=audio_minimum_seconds,
        modality=modality,
        image_usd_per_image=(Decimal(image_price) if image_price is not None else None),
        image_output_usd_per_mtok=(
            Decimal(image_output_price_per_mtok)
            if image_output_price_per_mtok is not None
            else None
        ),
        video_usd_per_second=(
            Decimal(video_price_per_second) if video_price_per_second is not None else None
        ),
        video_usd_per_second_by_resolution=MappingProxyType(
            {
                resolution: Decimal(price)
                for resolution, price in (video_prices_by_resolution or {}).items()
            }
        ),
    )


_ENTRIES: tuple[ModelInfo, ...] = (
    # ---- OpenAI ---------------------------------------------------------
    _m("gpt-5.1-2025-11-13", "openai", "1.25", "10.00", deprecated=True),
    _m("gpt-5.2-2025-12-11", "openai", "1.75", "14.00", deprecated=True),
    # The 5.6 family rejects `temperature`: reasoning replaces it, and sending
    # it fails the whole call. A fallback onto one of these must not inherit it.
    _m(
        "gpt-5.6-sol",
        "openai",
        "5.00",
        "30.00",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
        supports_temperature=False,
    ),
    _m(
        "gpt-5.6-terra",
        "openai",
        "2.00",
        "12.00",
        notes="max output 128K tokens",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
        supports_temperature=False,
    ),
    _m(
        "gpt-5.6-luna",
        "openai",
        "0.20",
        "1.20",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
        supports_temperature=False,
    ),
    _m("gpt-realtime-2.1", "openai", "32.00", "64.00", notes="realtime audio"),
    _m("gpt-realtime-2.1-mini", "openai", "10.00", "20.00", notes="realtime audio"),
    _m("gpt-realtime-2025-08-28", "openai", "32.00", "64.00", deprecated=True),
    _m("gpt-realtime-mini-2025-10-06", "openai", "10.00", "20.00", deprecated=True),
    _m("gpt-realtime-mini-2025-12-15", "openai", "10.00", "20.00", deprecated=True),
    _m("gpt-realtime-1.5-2026-02-25", "openai", "32.00", "64.00", deprecated=True),
    _m(
        "gpt-transcribe",
        "openai",
        "0",
        "0",
        notes="speech-to-text; billed at USD 0.0045 per audio minute",
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.0045",
    ),
    # ---- Groq (OpenAI-compatible ids, served by Groq) -------------------
    _m(
        "whisper-large-v3-turbo",
        "groq",
        "0",
        "0",
        notes="Whisper speech-to-text; billed at USD 0.04 per audio hour",
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.0006666666666666666666666667",
        audio_minimum_seconds=10,
    ),
    _m(
        "whisper-large-v3",
        "groq",
        "0",
        "0",
        notes="Whisper speech-to-text; billed at USD 0.111 per audio hour",
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.00185",
        audio_minimum_seconds=10,
    ),
    _m(
        "distil-whisper-large-v3-en",
        "groq",
        "0",
        "0",
        notes="English-only Whisper speech-to-text; billed at USD 0.02 per audio hour",
        deprecated=True,
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.0003333333333333333333333333",
        audio_minimum_seconds=10,
    ),
    _m(
        "openai/gpt-oss-120b",
        "groq",
        "0.15",
        "0.60",
        notes="the openai/ prefix is not OpenAI",
        reasoning_efforts=GROQ_GPT_OSS_REASONING_EFFORTS,
    ),
    _m(
        "openai/gpt-oss-20b",
        "groq",
        "0.075",
        "0.30",
        notes="the openai/ prefix is not OpenAI",
        reasoning_efforts=GROQ_GPT_OSS_REASONING_EFFORTS,
    ),
    _m("meta-llama/llama-4-scout-17b-16e-instruct", "groq", "0.11", "0.34", deprecated=True),
    _m("meta-llama/llama-4-maverick-17b-128e-instruct", "groq", "0.50", "0.77", deprecated=True),
    # ---- AssemblyAI ------------------------------------------------------
    _m(
        "assemblyai-universal-3-pro",
        "assemblyai",
        "0",
        "0",
        notes="speech-to-text; billed at USD 0.21 per audio hour",
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.0035",
    ),
    _m(
        "assemblyai-universal-2",
        "assemblyai",
        "0",
        "0",
        notes="speech-to-text; billed at USD 0.15 per audio hour",
        supports_temperature=False,
        pricing_unit="audio_minutes",
        audio_price_per_minute="0.0025",
    ),
    # ---- Google Gemini --------------------------------------------------
    _m(
        "gemini-3-flash-preview",
        "gemini",
        "0.50",
        "3.00",
        deprecated=True,
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3-pro-preview",
        "gemini",
        "2.00",
        "12.00",
        deprecated=True,
        reasoning_efforts=GEMINI_3_PRO_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3-pro-image",
        "gemini",
        "2.00",
        "12.00",
        modality="image",
        image_output_price_per_mtok="120.00",
    ),
    _m(
        "gemini-3.1-flash-lite-preview",
        "gemini",
        "0.25",
        "1.50",
        notes="text/image/video share",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3.1-flash-lite-image",
        "gemini",
        "0.25",
        "1.50",
        modality="image",
        image_output_price_per_mtok="30.00",
    ),
    _m(
        "gemini-3.1-flash-image",
        "gemini",
        "0.50",
        "3.00",
        modality="image",
        image_output_price_per_mtok="60.00",
    ),
    _m(
        "gemini-3.1-flash-image-preview",
        "gemini",
        "0.50",
        "3.00",
        deprecated=True,
        modality="image",
        image_output_price_per_mtok="60.00",
    ),
    _m(
        "gemini-3.1-pro-preview",
        "gemini",
        "2.00",
        "12.00",
        reasoning_efforts=GEMINI_3_PRO_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3.5-flash",
        "gemini",
        "1.50",
        "9.00",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3.5-flash-lite",
        "gemini",
        "0.30",
        "2.50",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3.6-flash",
        "gemini",
        "1.50",
        "7.50",
        deprecated=True,
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3.7-flash",
        "gemini",
        "0.75",
        "3.75",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-pro-latest",
        "gemini",
        "2.00",
        "12.00",
        notes="floating alias; prefer a pinned id",
        reasoning_efforts=GEMINI_3_PRO_REASONING_EFFORTS,
    ),
    _m(
        "gemini-flash-latest",
        "gemini",
        "1.50",
        "9.00",
        notes="floating alias; prefer a pinned id",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-flash-lite-latest",
        "gemini",
        "0.25",
        "1.50",
        notes="floating alias; prefer a pinned id",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    # ---- OpenRouter -----------------------------------------------------
    _m("google/gemini-3-flash-preview", "openrouter", "0.50", "3.00", deprecated=True),
    _m("google/gemini-3-pro-preview", "openrouter", "2.00", "12.00", deprecated=True),
    _m("google/gemini-3.1-flash-lite-preview", "openrouter", "0.25", "1.50"),
    # OpenRouter resells Google's own rates, image tokens included: its
    # `image_output` is 60.00/Mtok, while the 3.00 above is the text one.
    _m(
        "google/gemini-3.1-flash-image",
        "openrouter",
        "0.50",
        "3.00",
        modality="image",
        image_output_price_per_mtok="60.00",
    ),
    _m("google/gemini-3.1-pro-preview", "openrouter", "2.00", "12.00"),
    _m("google/gemini-3.5-flash", "openrouter", "1.50", "9.00"),
    _m("google/gemini-3.5-flash-lite", "openrouter", "0.30", "2.50"),
    _m("google/gemini-3.6-flash", "openrouter", "1.50", "7.50", deprecated=True),
    _m("google/gemini-3.7-flash", "openrouter", "0.75", "3.75"),
    _m("anthropic/claude-sonnet-4.6", "openrouter", "3.00", "15.00"),
    _m("x-ai/grok-4.5", "openrouter", "2.00", "6.00"),
    _m(
        "~anthropic/claude-sonnet-latest",
        "openrouter",
        "2.00",
        "10.00",
        notes="floating alias; prefer a pinned id",
        supports_temperature=False,
    ),
    _m(
        "~anthropic/claude-opus-latest",
        "openrouter",
        "5.00",
        "25.00",
        notes="floating alias; prefer a pinned id",
    ),
    _m(
        "~deepseek/deepseek-v4-flash-latest",
        "openrouter",
        "0.09",
        "0.18",
        notes="floating alias; prefer a pinned id",
    ),
    _m(
        "~moonshotai/kimi-latest",
        "openrouter",
        "2.90",
        "14.00",
        notes="floating alias; prefer a pinned id",
    ),
    _m("qwen/qwen3.8-max", "openrouter", "2.00", "6.00"),
    _m("deepseek/deepseek-chat-v3.1", "openrouter", "0.28", "0.42", deprecated=True),
    _m(
        "deepseek/deepseek-r1-distill-qwen-7b",
        "openrouter",
        "0.55",
        "2.19",
        deprecated=True,
    ),
    _m("deepseek/deepseek-v4-flash", "openrouter", "0.14", "0.28"),
    _m("deepseek/deepseek-v4-pro", "openrouter", "0.435", "0.87"),
    _m("moonshotai/kimi-k2-thinking", "openrouter", "0.50", "1.50", deprecated=True),
    _m("moonshotai/kimi-k2.6", "openrouter", "0.74", "3.49", deprecated=True),
    # ---- Replicate (image generation, billed per image or per GPU second) --
    _m(
        "black-forest-labs/flux-kontext-pro",
        "replicate",
        "0",
        "0",
        notes="image editing; USD 0.04 per output image",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
        image_price="0.04",
    ),
    _m(
        "prunaai/p-image",
        "replicate",
        "0",
        "0",
        notes="image generation; USD 5 per thousand output images, so 0.005 each. "
        "Replicate used to bill this one by GPU time and published no rate at all",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
        image_price="0.005",
    ),
    _m(
        "prunaai/z-image-turbo",
        "replicate",
        "0",
        "0",
        notes="image generation, no safety checker of its own. Billed per output "
        "megapixel and not per image — 0.005 per megapixel at the 1MP tier, "
        "rising to 0.02 at 4MP — and ImageUsage counts images, not pixels, so no "
        "rate is catalogued and cost reports UNAVAILABLE. The adapter sends no "
        "dimensions, so Replicate's 1024x1024 default applies: about USD 0.0052 "
        "an image, which an ImagePriceCatalog can state as fact once measured",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
    ),
    _m(
        "bytedance/seedream-4",
        "replicate",
        "0",
        "0",
        notes="no per-image rate verified; supply an ImagePriceCatalog to price it",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
    ),
    _m(
        "wan-video/wan-2.2-5b-fast",
        "replicate",
        "0",
        "0",
        notes="text-to-video and image-to-video, billed by GPU time, so no "
        "per-second rate exists to publish; supply a VideoPriceCatalog to price it",
        supports_temperature=False,
        pricing_unit="video_seconds",
        modality="video",
    ),
    _m(
        "kwaivgi/kling-v3-video",
        "replicate",
        "0",
        "0",
        notes="3-15s clips; 'mode' selects 720p/1080p/4K. No per-second rate "
        "verified against the API, so cost reports UNAVAILABLE rather than a guess",
        supports_temperature=False,
        pricing_unit="video_seconds",
        modality="video",
    ),
    _m(
        "bytedance/seedance-2.5",
        "replicate",
        "0",
        "0",
        notes="clips up to 30s at 480p or 720p; USD 0.1028 per second at 480p, "
        "0.2312 at 720p. Those are the rates for the variant this package can "
        "reach: sending a reference video costs about four times as much, and "
        "VideoRequest has no field that would send one",
        supports_temperature=False,
        pricing_unit="video_seconds",
        modality="video",
        video_prices_by_resolution={"480p": "0.1028", "720p": "0.2312"},
    ),
    # ---- WaveSpeed ------------------------------------------------------
    _m(
        "wavespeed-ai/minimax-h3/image-to-video",
        "wavespeed",
        "0",
        "0",
        notes="MiniMax H3 open weights; USD 0.04 per second at 480p, 0.08 at 768p",
        supports_temperature=False,
        pricing_unit="video_seconds",
        modality="video",
        video_prices_by_resolution={"480p": "0.04", "768p": "0.08"},
    ),
    _m(
        "wavespeed-ai/hidream-i1-dev",
        "wavespeed",
        "0",
        "0",
        notes="image generation; USD 0.012 per image, rising with output size",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
        image_price="0.012",
    ),
    _m(
        "wavespeed-ai/chroma",
        "wavespeed",
        "0",
        "0",
        notes="image generation; USD 0.015 per image, flat. Unfiltered by "
        "design, which is a property of the weights and not a licence: the "
        "provider's own safety checker and terms still apply",
        supports_temperature=False,
        pricing_unit="images",
        modality="image",
        image_price="0.015",
    ),
)

MODEL_CATALOG: dict[str, ModelInfo] = {entry.id: entry for entry in _ENTRIES}


def lookup_model(model_id: str) -> ModelInfo | None:
    """Catalogue entry for a model, or ``None`` when it is not catalogued."""
    return MODEL_CATALOG.get(model_id)


def models_by_provider(provider: Provider) -> tuple[ModelInfo, ...]:
    return tuple(m for m in MODEL_CATALOG.values() if m.provider == provider)


# Only this provider-owned namespace is routable without a catalogue entry.
# Model families and vendor namespaces are deliberately not inferred: Groq's
# official list includes namespaced ids that also exist on OpenRouter.
_EXPLICIT_PROVIDER_PREFIXES: tuple[tuple[str, Provider], ...] = (("openrouter/", "openrouter"),)


def resolve_provider(model_id: str) -> Provider | None:
    """Which provider serves this model.

    The catalogue is authoritative. An id that is not catalogued is only
    accepted for the explicit ``openrouter/`` router namespace; every other
    unknown id returns ``None`` rather than being guessed into the wrong
    provider.
    """
    if _is_non_three_gemini_model(model_id):
        return None

    catalogued = MODEL_CATALOG.get(model_id)
    if catalogued is not None:
        return catalogued.provider

    for prefix, provider in _EXPLICIT_PROVIDER_PREFIXES:
        if model_id.startswith(prefix):
            return provider
    return None


def _is_non_three_gemini_model(model_id: str) -> bool:
    """Keep removed Gemini generations out of both direct and namespaced routes."""
    for component in model_id.lower().split("/"):
        if not component.startswith("gemini-"):
            continue
        major = component.removeprefix("gemini-").split(".", 1)[0]
        if major.isdigit() and major != "3":
            return True
    return False
