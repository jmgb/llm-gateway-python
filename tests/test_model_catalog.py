"""The shared model catalogue: identity, provider and price, in one place.

This is the answer to "where do I update a price?". One repository, one
version string, every consumer.
"""

from decimal import Decimal

import pytest

from llm_gateway import CostMeasurement, TokenUsage
from llm_gateway.models import (
    CATALOG_VERSION,
    MODEL_CATALOG,
    ModelInfo,
    builtin_price_catalog,
    lookup_model,
    models_by_provider,
    resolve_provider,
)


class TestIdentity:
    def test_a_known_model_reports_its_provider(self) -> None:
        gemini = lookup_model("gemini-3.5-flash-lite")
        openai = lookup_model("gpt-5.6-luna")

        assert gemini is not None and gemini.provider == "gemini"
        assert openai is not None and openai.provider == "openai"

    def test_an_unknown_model_is_none_not_a_guess(self) -> None:
        assert lookup_model("some-model-that-does-not-exist") is None

    def test_every_entry_declares_a_provider_the_package_can_serve(self) -> None:
        servable = {"openai", "gemini", "groq", "openrouter"}

        for model_id, info in MODEL_CATALOG.items():
            assert info.provider in servable, f"{model_id} has provider {info.provider}"

    def test_every_entry_is_keyed_by_its_own_id(self) -> None:
        for model_id, info in MODEL_CATALOG.items():
            assert info.id == model_id


class TestProviderRouting:
    def test_a_catalogued_model_routes_by_its_declared_provider(self) -> None:
        assert resolve_provider("gemini-3.6-flash") == "gemini"

    def test_an_openai_prefixed_groq_model_is_not_mistaken_for_openai(self) -> None:
        """`openai/gpt-oss-120b` is served by Groq, despite the prefix."""
        assert resolve_provider("openai/gpt-oss-120b") == "groq"

    def test_a_namespaced_model_defaults_to_openrouter(self) -> None:
        assert resolve_provider("somevendor/some-new-model") == "openrouter"

    def test_a_models_prefixed_gemini_id_routes_to_gemini(self) -> None:
        assert resolve_provider("models/gemini-3.5-flash-lite") == "gemini"

    def test_an_uncatalogued_gpt_routes_to_openai(self) -> None:
        assert resolve_provider("gpt-6-unreleased") == "openai"

    def test_an_unroutable_id_returns_none_rather_than_guessing_wrong(self) -> None:
        assert resolve_provider("totally-unknown") is None


class TestBuiltinPrices:
    def test_the_builtin_catalog_prices_a_known_model(self) -> None:
        catalog = builtin_price_catalog()

        cost = catalog.estimate(
            "gemini-3.5-flash-lite", TokenUsage(input_tokens=1_000_000, output_tokens=0)
        )

        assert cost.amount_usd == Decimal("0.300000")
        assert cost.measurement is CostMeasurement.ACTUAL

    def test_the_builtin_catalog_is_versioned(self) -> None:
        assert builtin_price_catalog().version == CATALOG_VERSION
        assert CATALOG_VERSION

    def test_an_unknown_model_is_unavailable_not_free(self) -> None:
        cost = builtin_price_catalog().estimate(
            "not-a-model", TokenUsage(input_tokens=10, output_tokens=10)
        )

        assert cost.measurement is CostMeasurement.UNAVAILABLE

    def test_output_is_priced_at_the_output_rate(self) -> None:
        cost = builtin_price_catalog().estimate(
            "gemini-3.5-flash-lite", TokenUsage(input_tokens=0, output_tokens=1_000_000)
        )

        assert cost.amount_usd == Decimal("2.500000")

    def test_a_consumer_can_override_a_price_without_forking(self) -> None:
        catalog = builtin_price_catalog(
            overrides={"gemini-3.5-flash-lite": (Decimal("9"), Decimal("9"))},
            version="my-negotiated-rates-2026-07",
        )

        cost = catalog.estimate(
            "gemini-3.5-flash-lite", TokenUsage(input_tokens=1_000_000, output_tokens=0)
        )

        assert cost.amount_usd == Decimal("9.000000")
        assert cost.pricing_version == "my-negotiated-rates-2026-07"


class TestPricesMatchTheDeclaredCatalogue:
    """USD per million tokens and microUSD per token are the same number."""

    def test_the_conversion_is_the_identity(self) -> None:
        info = lookup_model("gemini-3.5-flash-lite")
        assert info is not None

        cost = builtin_price_catalog().estimate(
            info.id, TokenUsage(input_tokens=1_000_000, output_tokens=0)
        )

        assert cost.amount_usd == info.input_usd_per_mtok.quantize(Decimal("0.000001"))


class TestCatalogueHygiene:
    def test_no_price_is_negative(self) -> None:
        for info in MODEL_CATALOG.values():
            assert info.input_usd_per_mtok >= 0
            assert info.output_usd_per_mtok >= 0

    def test_deprecated_models_are_marked_not_deleted(self) -> None:
        """Removing a model breaks a consumer that still calls it."""
        for info in MODEL_CATALOG.values():
            assert isinstance(info.deprecated, bool)

    def test_models_can_be_listed_per_provider(self) -> None:
        gemini = models_by_provider("gemini")

        assert "gemini-3.5-flash-lite" in {m.id for m in gemini}
        assert all(m.provider == "gemini" for m in gemini)

    def test_no_model_id_is_declared_twice(self) -> None:
        """A duplicate key silently discards one of the two prices.

        A real case: two constants pointing at the same model id, one of the
        declared prices never applying, and nobody noticing because a dict just
        keeps the last one.
        """
        from llm_gateway.models import _ENTRIES

        ids = [entry.id for entry in _ENTRIES]
        duplicates = {model_id for model_id in ids if ids.count(model_id) > 1}

        assert duplicates == set(), f"duplicated model ids: {duplicates}"

    def test_a_model_entry_is_immutable(self) -> None:
        info = lookup_model("gemini-3.5-flash-lite")
        assert info is not None

        with pytest.raises((AttributeError, TypeError)):
            info.input_usd_per_mtok = Decimal("0")  # type: ignore[misc]


class TestModelInfo:
    def test_a_rate_is_derived_from_the_declared_price(self) -> None:
        info = ModelInfo(
            id="x",
            provider="openai",
            input_usd_per_mtok=Decimal("1.25"),
            output_usd_per_mtok=Decimal("10"),
        )

        assert info.rate.input_microusd_per_token == Decimal("1.25")
        assert info.rate.output_microusd_per_token == Decimal("10")
