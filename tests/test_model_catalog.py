"""The shared model catalogue: identity, provider and price, in one place.

This is the answer to "where do I update a price?". One repository, one
version string, every consumer.
"""

import hashlib
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

    def test_reasoning_efforts_are_declared_per_model(self) -> None:
        openai_expected = ("none", "low", "medium", "high", "xhigh", "max")

        for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            info = lookup_model(model_id)
            assert info is not None
            assert info.provider == "openai"
            assert info.reasoning_efforts == openai_expected

        for model_id in (
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ):
            info = lookup_model(model_id)
            assert info is not None
            assert info.reasoning_efforts == ("minimal", "low", "medium", "high")

        for model_id in ("gemini-3-pro-preview", "gemini-3.1-pro-preview"):
            info = lookup_model(model_id)
            assert info is not None
            assert info.reasoning_efforts == ("low", "medium", "high")

        assert lookup_model("gemini-2.5-flash") is None
        assert lookup_model("gemini-2.5-flash-lite") is None

        oss = lookup_model("openai/gpt-oss-120b")
        assert oss is not None
        assert oss.provider == "groq"
        assert oss.reasoning_efforts == ("low", "medium", "high")


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

    def test_removed_gemini_25_models_are_not_routable(self) -> None:
        assert resolve_provider("gemini-2.5-flash") is None
        assert resolve_provider("google/gemini-2.5-flash") is None

    def test_an_uncatalogued_gpt_routes_to_openai(self) -> None:
        assert resolve_provider("gpt-6-unreleased") == "openai"

    def test_an_unroutable_id_returns_none_rather_than_guessing_wrong(self) -> None:
        assert resolve_provider("totally-unknown") is None


class TestBuiltinPrices:
    def test_luna_uses_the_current_published_rates(self) -> None:
        luna = lookup_model("gpt-5.6-luna")

        assert luna is not None
        assert luna.input_usd_per_mtok == Decimal("0.20")
        assert luna.output_usd_per_mtok == Decimal("1.20")

    def test_terra_uses_the_current_published_rates(self) -> None:
        terra = lookup_model("gpt-5.6-terra")

        assert terra is not None
        assert terra.input_usd_per_mtok == Decimal("2")
        assert terra.output_usd_per_mtok == Decimal("12")

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


class TestPricesAndVersionMoveTogether:
    """A price change that keeps the old version makes past amounts unauditable.

    ``CATALOG_VERSION`` is recorded next to every amount so an old figure can be
    recomputed from the table that produced it. Editing a rate without bumping
    it points two different tables at the same name, and the invoice that no
    longer reconciles is discovered months later, by someone else.

    The fingerprint below is the guard. Changing a price fails this test, and
    the honest fix is to bump the version in ``models.py`` and both constants
    here in the same commit.
    """

    PRICED_AT_VERSION = "2026-07-31"
    PRICE_FINGERPRINT = "90d7b71c6d5ddfddd0a728608d6a16ac0313ff0859615c40365da678b995daf9"

    @staticmethod
    def _fingerprint() -> str:
        """Identity and rates only, at a fixed precision.

        Quantising is what keeps ``2`` and ``2.00`` the same price, so the test
        fires on a rate that moved, never on a literal that was retyped.
        """
        micro = Decimal("0.000001")
        priced = sorted(
            f"{info.id}\t{info.input_usd_per_mtok.quantize(micro)}"
            f"\t{info.output_usd_per_mtok.quantize(micro)}"
            for info in MODEL_CATALOG.values()
        )
        return hashlib.sha256("\n".join(priced).encode()).hexdigest()

    def test_no_price_changes_without_a_new_catalog_version(self) -> None:
        assert self._fingerprint() == self.PRICE_FINGERPRINT, (
            f"the priced catalogue changed while CATALOG_VERSION stayed "
            f"{CATALOG_VERSION!r}: bump it, then update this fingerprint"
        )

    def test_the_pinned_fingerprint_belongs_to_the_current_version(self) -> None:
        """Bumping the version without repinning would leave the guard checking a ghost."""
        assert CATALOG_VERSION == self.PRICED_AT_VERSION


class TestDeclaredRequestOptions:
    """What a model accepts is declared, never inferred from its id."""

    def test_the_openai_56_family_declares_that_it_rejects_temperature(self) -> None:
        for model_id in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            info = lookup_model(model_id)
            assert info is not None
            assert info.supports_temperature is False, model_id

    def test_a_model_that_says_nothing_keeps_accepting_temperature(self) -> None:
        """Silence is not evidence of a refusal; the permissive answer is the default."""
        info = lookup_model("gpt-5.2-2025-12-11")

        assert info is not None
        assert info.supports_temperature is True


class TestModelInfo:
    def test_it_has_no_unused_alias_surface(self) -> None:
        info = lookup_model("gpt-5.6-luna")

        assert info is not None
        assert not hasattr(info, "aliases")

    def test_a_rate_is_derived_from_the_declared_price(self) -> None:
        info = ModelInfo(
            id="x",
            provider="openai",
            input_usd_per_mtok=Decimal("1.25"),
            output_usd_per_mtok=Decimal("10"),
        )

        assert info.rate.input_microusd_per_token == Decimal("1.25")
        assert info.rate.output_microusd_per_token == Decimal("10")
