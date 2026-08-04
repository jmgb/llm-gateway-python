"""The shared model catalogue: identity, provider and price.

This is the single place where a model's identity and pricing metadata are
updated. One versioned table beats several copies drifting apart, and a
model's price is a fact about the *provider*, not about any product — which is
exactly the test for what belongs in this package.

What stays out: **which model a feature should use**. That is a product
decision, and putting it here would make one repository's choice everybody's.

## Units

Prices are declared in **USD per million tokens**, as every provider publishes
them. Rates are consumed as **microUSD per token**. Those are the same number:

    1 USD / 1,000,000 tokens = 1e-6 USD / token = 1 microUSD / token

so the conversion is the identity, and there is no factor to get wrong. There
is a test asserting exactly that.

Models billed by audio duration use ``pricing_unit="audio_minutes"`` and keep
their per-minute rate separately. They are routable through the catalogue but
are intentionally excluded from the token price catalog.

## Updating

Change the price, bump ``CATALOG_VERSION``, tag a release. Consumers pin a tag,
so nobody's cost figures move without an explicit upgrade — and every recorded
amount carries the version that produced it, so old numbers stay reconcilable.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from llm_gateway.contracts import ReasoningEffort
from llm_gateway.pricing import AudioRate, ModelRate, StaticAudioPriceCatalog, StaticPriceCatalog

CATALOG_VERSION = "2026-08-04.4"
"""Bump on every price change. Recorded alongside every amount."""

Provider = str
PricingUnit = Literal["tokens", "audio_minutes"]
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
    )


_ENTRIES: tuple[ModelInfo, ...] = (
    # ---- OpenAI ---------------------------------------------------------
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
    # ---- AssemblyAI ------------------------------------------------------
    _m(
        "assemblyai-universal-3-5-pro",
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
        "gemini-3.1-flash-lite-preview",
        "gemini",
        "0.25",
        "1.50",
        notes="text/image/video share",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m("gemini-3.1-flash-lite-image", "gemini", "0.25", "1.50"),
    _m("gemini-3.1-flash-image", "gemini", "0.50", "3.00"),
    _m("gemini-3.1-flash-image-preview", "gemini", "0.50", "3.00"),
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
    _m("google/gemini-3.1-flash-lite-preview", "openrouter", "0.25", "1.50"),
    _m("google/gemini-3.1-flash-image", "openrouter", "0.50", "3.00"),
    _m("google/gemini-3.1-pro-preview", "openrouter", "2.00", "12.00"),
    _m("google/gemini-3.5-flash", "openrouter", "1.50", "9.00"),
    _m("google/gemini-3.5-flash-lite", "openrouter", "0.30", "2.50"),
    _m("google/gemini-3.6-flash", "openrouter", "1.50", "7.50"),
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
    _m("deepseek/deepseek-v4-flash", "openrouter", "0.14", "0.28"),
    _m("deepseek/deepseek-v4-pro", "openrouter", "0.435", "0.87"),
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
        if info.pricing_unit == "tokens"
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


def builtin_audio_price_catalog(*, version: str | None = None) -> StaticAudioPriceCatalog:
    """Build the duration-based catalogue, excluding every token model."""
    rates = {
        model_id: info.audio_rate
        for model_id, info in MODEL_CATALOG.items()
        if info.pricing_unit == "audio_minutes" and info.audio_usd_per_minute is not None
    }
    return StaticAudioPriceCatalog(version=version or CATALOG_VERSION, rates=rates)
