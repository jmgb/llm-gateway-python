"""The shared model catalogue: identity, provider and price.

This is the single place where a price is updated. One versioned table beats
several copies drifting apart, and a model's price is a fact about the
*provider*, not about any product — which is exactly the test for what belongs
in this package.

What stays out: **which model a feature should use**. That is a product
decision, and putting it here would make one repository's choice everybody's.

## Units

Prices are declared in **USD per million tokens**, as every provider publishes
them. Rates are consumed as **microUSD per token**. Those are the same number:

    1 USD / 1,000,000 tokens = 1e-6 USD / token = 1 microUSD / token

so the conversion is the identity, and there is no factor to get wrong. There
is a test asserting exactly that.

## Updating

Change the price, bump ``CATALOG_VERSION``, tag a release. Consumers pin a tag,
so nobody's cost figures move without an explicit upgrade — and every recorded
amount carries the version that produced it, so old numbers stay reconcilable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from llm_gateway.contracts import ReasoningEffort
from llm_gateway.pricing import ModelRate, StaticPriceCatalog

CATALOG_VERSION = "2026-07-31"
"""Bump on every price change. Recorded alongside every amount."""

Provider = str
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
    """One model: provider, price, and explicitly supported request options."""

    id: str
    provider: Provider
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal
    deprecated: bool = False
    notes: str = ""
    aliases: tuple[str, ...] = field(default=())
    reasoning_efforts: tuple[ReasoningEffort, ...] = field(default=())

    @property
    def rate(self) -> ModelRate:
        """USD per million tokens read as microUSD per token — same number."""
        return ModelRate(
            input_microusd_per_token=self.input_usd_per_mtok,
            output_microusd_per_token=self.output_usd_per_mtok,
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
) -> ModelInfo:
    return ModelInfo(
        id=model_id,
        provider=provider,
        input_usd_per_mtok=Decimal(input_price),
        output_usd_per_mtok=Decimal(output_price),
        deprecated=deprecated,
        notes=notes,
        reasoning_efforts=reasoning_efforts,
    )


_ENTRIES: tuple[ModelInfo, ...] = (
    # ---- OpenAI ---------------------------------------------------------
    _m("gpt-5.1-2025-11-13", "openai", "1.25", "10.00"),
    _m("gpt-5.2-2025-12-11", "openai", "1.75", "14.00"),
    _m(
        "gpt-5.6-sol",
        "openai",
        "5.00",
        "30.00",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
    ),
    _m(
        "gpt-5.6-terra",
        "openai",
        "2.00",
        "12.00",
        notes="max output 128K tokens",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
    ),
    _m(
        "gpt-5.6-luna",
        "openai",
        "0.20",
        "1.20",
        reasoning_efforts=OPENAI_56_REASONING_EFFORTS,
    ),
    _m("gpt-realtime-2025-08-28", "openai", "32.00", "64.00", notes="realtime audio"),
    _m("gpt-realtime-mini-2025-10-06", "openai", "10.00", "20.00", notes="realtime audio"),
    _m("gpt-realtime-mini-2025-12-15", "openai", "10.00", "20.00", notes="realtime audio"),
    _m("gpt-realtime-1.5-2026-02-25", "openai", "32.00", "64.00", notes="realtime audio"),
    # ---- Groq (OpenAI-compatible ids, served by Groq) -------------------
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
    _m("meta-llama/llama-4-scout-17b-16e-instruct", "groq", "0.11", "0.34"),
    _m("meta-llama/llama-4-maverick-17b-128e-instruct", "groq", "0.50", "0.77"),
    # ---- Google Gemini --------------------------------------------------
    _m(
        "gemini-3-flash-preview",
        "gemini",
        "0.50",
        "3.00",
        reasoning_efforts=GEMINI_3_FLASH_REASONING_EFFORTS,
    ),
    _m(
        "gemini-3-pro-preview",
        "gemini",
        "2.00",
        "12.00",
        reasoning_efforts=GEMINI_3_PRO_REASONING_EFFORTS,
    ),
    _m("gemini-3-pro-image", "gemini", "2.00", "12.00"),
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
    _m("google/gemini-3-flash-preview", "openrouter", "0.50", "3.00"),
    _m("google/gemini-3-pro-preview", "openrouter", "2.00", "12.00"),
    _m("google/gemini-3.1-flash-lite-preview", "openrouter", "0.25", "1.50"),
    _m("google/gemini-3.1-flash-image", "openrouter", "0.50", "3.00"),
    _m("google/gemini-3.1-pro-preview", "openrouter", "2.00", "12.00"),
    _m("google/gemini-3.5-flash", "openrouter", "1.50", "9.00"),
    _m("google/gemini-3.5-flash-lite", "openrouter", "0.30", "2.50"),
    _m("google/gemini-3.6-flash", "openrouter", "1.50", "7.50"),
    _m("deepseek/deepseek-chat-v3.1", "openrouter", "0.28", "0.42"),
    _m("deepseek/deepseek-r1-distill-qwen-7b", "openrouter", "0.55", "2.19"),
    _m("deepseek/deepseek-v4-flash", "openrouter", "0.14", "0.28"),
    _m("deepseek/deepseek-v4-pro", "openrouter", "0.435", "0.87"),
    _m("moonshotai/kimi-k2-thinking", "openrouter", "0.50", "1.50"),
    _m("moonshotai/kimi-k2.6", "openrouter", "0.74", "3.49"),
)

MODEL_CATALOG: dict[str, ModelInfo] = {entry.id: entry for entry in _ENTRIES}


def lookup_model(model_id: str) -> ModelInfo | None:
    """Catalogue entry for a model, or ``None`` when it is not catalogued."""
    return MODEL_CATALOG.get(model_id)


def models_by_provider(provider: Provider) -> tuple[ModelInfo, ...]:
    return tuple(m for m in MODEL_CATALOG.values() if m.provider == provider)


# Prefix rules for models that are not (yet) catalogued. Order matters: the
# `openai/` prefix belongs to Groq, so it must be tested before any generic
# namespace rule sends it to OpenRouter.
_PREFIX_RULES: tuple[tuple[str, Provider], ...] = (
    ("openai/gpt-oss", "groq"),
    ("models/gemini-3", "gemini"),
    ("meta-llama/", "groq"),
    ("gemini-3", "gemini"),
    ("gpt-", "openai"),
    ("chatgpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("llama", "groq"),
    ("mixtral", "groq"),
    ("gemma", "groq"),
    ("qwen", "groq"),
    ("kimi", "groq"),
    ("groq/", "groq"),
)


def resolve_provider(model_id: str) -> Provider | None:
    """Which provider serves this model.

    The catalogue is authoritative. Prefix rules only cover models released
    after the catalogue was last updated; an id that matches nothing returns
    ``None`` rather than being guessed into the wrong provider.
    """
    if _is_non_three_gemini_model(model_id):
        return None

    catalogued = MODEL_CATALOG.get(model_id)
    if catalogued is not None:
        return catalogued.provider

    lowered = model_id.lower()
    for prefix, provider in _PREFIX_RULES:
        if lowered.startswith(prefix):
            return provider

    # A namespaced id that matched no rule is almost certainly an aggregator.
    if "/" in lowered:
        return "openrouter"
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
    rates = {model_id: info.rate for model_id, info in MODEL_CATALOG.items()}
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
