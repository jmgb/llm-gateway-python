"""The shared model catalogue: identity, provider and price, in one place.

This is the answer to "where do I update a price?". One repository, one
version string, every consumer.
"""

import hashlib
from decimal import Decimal

import pytest

from llm_gateway import AudioUsage, CostMeasurement, TokenUsage, builtin_audio_price_catalog
from llm_gateway.catalogs import builtin_price_catalog
from llm_gateway.models import (
    CATALOG_VERSION,
    MODEL_CATALOG,
    ModelInfo,
    lookup_model,
    models_by_provider,
    resolve_provider,
)


class TestIdentity:
    def test_current_openrouter_models_are_catalogued_with_their_published_rates(self) -> None:
        expected = {
            "anthropic/claude-sonnet-4.6": ("3", "15", True),
            "x-ai/grok-4.5": ("2", "6", True),
            "~anthropic/claude-sonnet-latest": ("2", "10", False),
            "~anthropic/claude-opus-latest": ("5", "25", True),
            "~deepseek/deepseek-v4-flash-latest": ("0.09", "0.18", True),
            "~moonshotai/kimi-latest": ("2.9", "14", True),
            "qwen/qwen3.8-max": ("2", "6", True),
        }

        for model_id, (input_price, output_price, supports_temperature) in expected.items():
            info = lookup_model(model_id)

            assert info is not None
            assert info.provider == "openrouter"
            assert resolve_provider(model_id) == "openrouter"
            assert info.input_usd_per_mtok == Decimal(input_price)
            assert info.output_usd_per_mtok == Decimal(output_price)
            assert info.supports_temperature is supports_temperature

    def test_legacy_models_remain_routable_but_are_marked_deprecated(self) -> None:
        legacy = {
            "gemini-3-flash-preview",
            "gemini-3-pro-preview",
            "gemini-3.1-flash-image-preview",
            "deepseek/deepseek-chat-v3.1",
            "deepseek/deepseek-r1-distill-qwen-7b",
            "google/gemini-3-flash-preview",
            "google/gemini-3-pro-preview",
            "moonshotai/kimi-k2-thinking",
            "moonshotai/kimi-k2.6",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "gpt-5.1-2025-11-13",
            "gpt-5.2-2025-12-11",
            "gpt-realtime-2025-08-28",
            "gpt-realtime-mini-2025-10-06",
            "gpt-realtime-mini-2025-12-15",
            "gpt-realtime-1.5-2026-02-25",
        }

        assert legacy <= MODEL_CATALOG.keys()
        assert all(MODEL_CATALOG[model_id].deprecated for model_id in legacy)

    def test_current_gemini_image_models_are_not_marked_legacy(self) -> None:
        for model_id in (
            "gemini-3-pro-image",
            "gemini-3.1-flash-image",
            "gemini-3.1-flash-lite-image",
        ):
            info = lookup_model(model_id)
            assert info is not None
            assert info.deprecated is False

    def test_wavespeed_models_use_their_complete_provider_ids(self) -> None:
        info = lookup_model("wavespeed-ai/hidream-i1-dev")

        assert info is not None
        assert info.provider == "wavespeed"
        assert info.image_usd_per_image == Decimal("0.012")

    def test_p_image_now_carries_the_rate_replicate_publishes(self) -> None:
        """It was unpriced while Replicate billed it by GPU time; now it is not."""
        info = lookup_model("prunaai/p-image")

        assert info is not None
        assert info.image_usd_per_image == Decimal("0.005")

    def test_z_image_turbo_is_catalogued_without_a_rate_it_cannot_apply(self) -> None:
        """Replicate bills it per output megapixel, and ImageUsage counts images.

        A per-image figure here would mean assuming the output size, and the
        model spans 0.005 per megapixel to 0.02 — a guess that could be four
        times wrong in either direction. Routing does not need a price, so the
        model is reachable and its cost says UNAVAILABLE until a consumer
        states its own.
        """
        info = lookup_model("prunaai/z-image-turbo")

        assert info is not None
        assert info.provider == "replicate"
        assert info.modality == "image"
        assert info.image_rate is None

    def test_chroma_is_priced_per_image_at_its_published_flat_rate(self) -> None:
        """Flat, so unlike HiDream the amount does not move with output size."""
        info = lookup_model("wavespeed-ai/chroma")

        assert info is not None
        assert info.provider == "wavespeed"
        assert info.modality == "image"
        assert info.pricing_unit == "images"
        assert info.image_usd_per_image == Decimal("0.015")

    def test_current_openai_audio_models_use_their_current_ids(self) -> None:
        expected = {
            "gpt-realtime-2.1",
            "gpt-realtime-2.1-mini",
            "gpt-transcribe",
        }

        assert expected <= {model.id for model in models_by_provider("openai")}

        transcribe = lookup_model("gpt-transcribe")
        assert transcribe is not None
        assert transcribe.pricing_unit == "audio_minutes"
        assert transcribe.audio_usd_per_minute == Decimal("0.0045")

    def test_assemblyai_and_groq_whisper_models_are_catalogued_as_audio(self) -> None:
        expected = {
            "assemblyai-universal-3-pro": ("assemblyai", "0.0035", False),
            "assemblyai-universal-2": ("assemblyai", "0.0025", False),
            "whisper-large-v3-turbo": (
                "groq",
                "0.0006666666666666666666666667",
                False,
            ),
            "whisper-large-v3": ("groq", "0.00185", False),
            "distil-whisper-large-v3-en": (
                "groq",
                "0.0003333333333333333333333333",
                True,
            ),
        }

        for model_id, (provider, rate, deprecated) in expected.items():
            info = lookup_model(model_id)
            assert info is not None
            assert info.provider == provider
            assert info.pricing_unit == "audio_minutes"
            assert info.audio_usd_per_minute == Decimal(rate)
            assert info.deprecated is deprecated

    def test_a_known_model_reports_its_provider(self) -> None:
        gemini = lookup_model("gemini-3.5-flash-lite")
        openai = lookup_model("gpt-5.6-luna")

        assert gemini is not None and gemini.provider == "gemini"
        assert openai is not None and openai.provider == "openai"

    def test_an_unknown_model_is_none_not_a_guess(self) -> None:
        assert lookup_model("some-model-that-does-not-exist") is None

    def test_every_entry_declares_a_provider_the_package_can_serve(self) -> None:
        servable = {
            "openai",
            "gemini",
            "groq",
            "openrouter",
            "assemblyai",
            "replicate",
            "wavespeed",
        }

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
            "gemini-3.1-flash-lite-preview",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ):
            info = lookup_model(model_id)
            assert info is not None
            assert info.reasoning_efforts == ("minimal", "low", "medium", "high")

        for model_id in ("gemini-3.1-pro-preview",):
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

    def test_an_uncatalogued_namespaced_model_is_not_guessed(self) -> None:
        assert resolve_provider("somevendor/some-new-model") is None

    def test_an_uncatalogued_models_namespace_is_not_guessed(self) -> None:
        assert resolve_provider("models/gemini-3.5-flash-lite") is None

    def test_removed_gemini_25_models_are_not_routable(self) -> None:
        assert resolve_provider("gemini-2.5-flash") is None
        assert resolve_provider("google/gemini-2.5-flash") is None

    def test_an_uncatalogued_direct_model_is_not_guessed(self) -> None:
        assert resolve_provider("gpt-6-unreleased") is None

    def test_uncatalogued_provider_namespaces_are_not_guessed(self) -> None:
        for model_id in (
            "qwen/future-model",
            "meta-llama/future-model",
            "moonshotai/future-model",
        ):
            assert resolve_provider(model_id) is None

    def test_uncatalogued_groq_family_ids_are_not_guessed(self) -> None:
        for model_id in (
            "llama-4-future",
            "mixtral-9-future",
            "gemma3-future",
        ):
            assert resolve_provider(model_id) is None

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

    def test_audio_pricing_is_not_mistaken_for_token_pricing(self) -> None:
        cost = builtin_price_catalog().estimate(
            "gpt-transcribe", TokenUsage(input_tokens=100, output_tokens=100)
        )

        assert cost.measurement is CostMeasurement.UNAVAILABLE

    def test_audio_catalog_prices_duration_and_applies_groq_minimum(self) -> None:
        catalog = builtin_audio_price_catalog()

        cost = catalog.estimate("whisper-large-v3-turbo", AudioUsage(duration_seconds=1.0))

        assert cost.amount_usd == Decimal("0.000111")
        assert cost.measurement is CostMeasurement.ACTUAL

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

    def test_every_token_billed_image_model_declares_its_image_token_rate(self) -> None:
        """The text output rate is not the image one, and it is twenty times off.

        Both numbers are published per model, they sit next to each other, and
        only one of them prices a generated image. A real case: the OpenRouter
        entry for `google/gemini-3.1-flash-image` carried `3.00` — Google's text
        rate — while the image tokens it actually bills cost `60.00`, so every
        image it priced was valued at a twentieth of what it cost. The Gemini
        entry for the very same model had the right figure all along.
        """
        missing = {
            info.id
            for info in MODEL_CATALOG.values()
            if info.modality == "image"
            and info.pricing_unit == "tokens"
            and info.image_output_usd_per_mtok is None
        }

        assert missing == set(), (
            f"image models billed by tokens with no image token rate, so their "
            f"images are priced at the text rate: {missing}"
        )

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

    PRICED_AT_VERSION = "2026-08-12.4"
    PRICE_FINGERPRINT = "b43a791ef76684e6d0705f4bf325a2cfdfbf856c1fe013c1e039dd81478516f1"

    @staticmethod
    def _fingerprint() -> str:
        """Identity and rates only, at a fixed precision.

        Quantising is what keeps ``2`` and ``2.00`` the same price, so the test
        fires on a rate that moved, never on a literal that was retyped.
        """
        micro = Decimal("0.000001")

        def by_resolution(info: ModelInfo) -> list[tuple[str, str]]:
            rates = info.video_usd_per_second_by_resolution
            return sorted((name, str(rate.quantize(micro))) for name, rate in rates.items())

        priced = sorted(
            f"{info.id}\t{info.pricing_unit}"
            f"\t{info.input_usd_per_mtok.quantize(micro)}"
            f"\t{info.output_usd_per_mtok.quantize(micro)}"
            f"\t{(info.audio_usd_per_minute or Decimal('0')).quantize(micro)}"
            f"\t{(info.image_usd_per_image or Decimal('0')).quantize(micro)}"
            f"\t{(info.image_output_usd_per_mtok or Decimal('0')).quantize(micro)}"
            f"\t{(info.video_usd_per_second or Decimal('0')).quantize(micro)}"
            f"\t{by_resolution(info)}"
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
        info = lookup_model("gpt-realtime-2.1-mini")

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
