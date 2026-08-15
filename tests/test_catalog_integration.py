"""The catalogue wired in: routing, default prices and catalogue-aware fallback."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from llm_gateway import (
    CostMeasurement,
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    ProviderResponse,
    TokenUsage,
)
from llm_gateway.factories import build_registry
from llm_gateway.models import lookup_model


class StubAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[str] = []

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.calls.append(model)
        return ProviderResponse(
            output_text="ok",
            usage=TokenUsage(input_tokens=1_000_000, output_tokens=0),
            finish_reason="stop",
        )


class TestRoutingUsesTheCatalogue:
    def test_a_catalogued_groq_model_is_not_routed_by_its_openai_prefix(self) -> None:
        registry = build_registry(openai_client=SimpleNamespace(), groq_client=SimpleNamespace())

        assert registry.resolve("openai/gpt-oss-120b").name == "groq"

    def test_catalogued_models_route_to_their_declared_provider(self) -> None:
        registry = build_registry(
            openai_client=SimpleNamespace(),
            gemini_client=SimpleNamespace(),
            groq_client=SimpleNamespace(),
        )

        assert registry.resolve("gemini-3.7-flash").name == "gemini"
        assert registry.resolve("gpt-5.6-sol").name == "openai"
        assert registry.resolve("openai/gpt-oss-120b").name == "groq"


class TestDefaultPricing:
    async def test_the_gateway_prices_with_the_shared_catalogue_by_default(self) -> None:
        adapter = StubAdapter("gemini")
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gemini-3.5-flash-lite",))
        gateway = LLMGateway(registry=registry)

        result = await gateway.generate(
            LLMRequest(model="gemini-3.5-flash-lite", messages=(Message("user", "x"),))
        )

        assert result.cost.amount_usd == Decimal("0.300000")
        assert result.cost.measurement is CostMeasurement.ACTUAL
        assert result.cost.pricing_version

    async def test_an_explicit_catalog_still_wins(self) -> None:
        from llm_gateway import ModelRate, StaticPriceCatalog

        adapter = StubAdapter("gemini")
        registry = ProviderRegistry()
        registry.register(adapter, model_prefixes=("gemini-3.5-flash-lite",))
        gateway = LLMGateway(
            registry=registry,
            price_catalog=StaticPriceCatalog(
                version="mine",
                rates={"gemini-3.5-flash-lite": ModelRate(Decimal("2"), Decimal("2"))},
            ),
        )

        result = await gateway.generate(
            LLMRequest(model="gemini-3.5-flash-lite", messages=(Message("user", "x"),))
        )

        assert result.cost.amount_usd == Decimal("2.000000")
        assert result.cost.pricing_version == "mine"


class TestCatalogueAwareFallback:
    def test_cheaper_alternatives_are_ordered_by_price(self) -> None:
        policy = FallbackPolicy.cheaper_than("gemini-3.7-flash", limit=5)

        prices = []
        for model in policy.models:
            info = lookup_model(model)
            assert info is not None
            prices.append(info.input_usd_per_mtok + info.output_usd_per_mtok)

        assert prices == sorted(prices)

    def test_alternatives_stay_within_the_same_provider(self) -> None:
        policy = FallbackPolicy.cheaper_than("gemini-3.7-flash", limit=3)

        for model in policy.models:
            info = lookup_model(model)
            assert info is not None
            assert info.provider == "gemini"

    def test_the_cheapest_model_has_no_cheaper_alternative(self) -> None:
        policy = FallbackPolicy.cheaper_than("gemini-3.1-flash-lite-preview")

        assert policy.models == ()

    def test_an_uncatalogued_model_cannot_derive_a_fallback(self) -> None:
        with pytest.raises(ValueError, match="not in the catalogue"):
            FallbackPolicy.cheaper_than("model-that-does-not-exist")

    def test_deprecated_models_are_never_proposed_as_fallback(self) -> None:
        policy = FallbackPolicy.cheaper_than("gpt-5.6-sol", limit=5)

        for model in policy.models:
            info = lookup_model(model)
            assert info is not None
            assert info.deprecated is False
