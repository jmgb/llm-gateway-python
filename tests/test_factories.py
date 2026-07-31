"""Optional convenience for building a registry.

The extras are not installed in this environment, which is exactly what makes
these tests meaningful: they prove the failure is a readable instruction and
not an ImportError traceback.
"""

import pytest

from llm_gateway import LLMRequest, ProviderNotInstalled, ProviderRegistry, ProviderResponse
from llm_gateway.factories import (
    build_registry,
    create_gemini_client,
    create_groq_client,
    create_openai_client,
    create_openrouter_client,
)


class FakeAdapter:
    name = "fake"

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise AssertionError("registration never calls the adapter")


def test_a_missing_extra_names_the_install_command() -> None:
    with pytest.raises(ProviderNotInstalled, match=r"neutral-llm-gateway\[openai\]"):
        create_openai_client(api_key="unused")


def test_each_provider_names_its_own_extra() -> None:
    with pytest.raises(ProviderNotInstalled, match=r"\[gemini\]"):
        create_gemini_client(api_key="unused")
    with pytest.raises(ProviderNotInstalled, match=r"\[groq\]"):
        create_groq_client(api_key="unused")


def test_openrouter_names_its_own_extra_even_though_it_ships_no_sdk() -> None:
    """It installs `openai`, but the extra a consumer asked for is its own."""
    with pytest.raises(ProviderNotInstalled, match=r"\[openrouter\]"):
        create_openrouter_client(api_key="unused")


def test_an_empty_api_key_is_rejected_before_importing_anything() -> None:
    with pytest.raises(ValueError, match="api_key"):
        create_openai_client(api_key="")


def test_build_registry_accepts_clients_the_application_already_owns() -> None:
    from types import SimpleNamespace

    registry = build_registry(openai_client=SimpleNamespace(), groq_client=SimpleNamespace())

    assert isinstance(registry, ProviderRegistry)
    assert registry.provider_names == ("groq", "openai")


def test_build_registry_routes_known_model_families() -> None:
    from types import SimpleNamespace

    registry = build_registry(
        openai_client=SimpleNamespace(),
        gemini_client=SimpleNamespace(),
        groq_client=SimpleNamespace(),
    )

    assert registry.resolve("gpt-5.6-luna").name == "openai"
    assert registry.resolve("o4-mini").name == "openai"
    assert registry.resolve("gemini-3.5-flash-lite").name == "gemini"
    assert registry.resolve("llama-3.3-70b").name == "groq"


def test_an_empty_registry_is_rejected_rather_than_silently_useless() -> None:
    with pytest.raises(ValueError, match="at least one"):
        build_registry()


def test_registering_the_same_provider_twice_is_rejected_before_it_is_overwritten() -> None:
    registry = ProviderRegistry()
    registry.register(FakeAdapter(), model_prefixes=("first-",))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeAdapter(), model_prefixes=("second-",))

    assert registry.resolve("first-model").name == "fake"


class TestOpenRouter:
    """An aggregator is a provider, not a `base_url` on somebody else's."""

    def _registry(self) -> ProviderRegistry:
        from types import SimpleNamespace

        return build_registry(
            openai_client=SimpleNamespace(),
            gemini_client=SimpleNamespace(),
            groq_client=SimpleNamespace(),
            openrouter_client=SimpleNamespace(),
        )

    def test_it_registers_under_its_own_name(self) -> None:
        assert self._registry().provider_names == ("gemini", "groq", "openai", "openrouter")

    def test_catalogued_models_route_without_any_extra_configuration(self) -> None:
        """The bug this replaced: these needed an undocumented prefix argument."""
        registry = self._registry()

        assert registry.resolve("deepseek/deepseek-chat-v3.1").name == "openrouter"
        assert registry.resolve("moonshotai/kimi-k2.6").name == "openrouter"

    def test_an_uncatalogued_namespaced_model_reaches_the_aggregator(self) -> None:
        assert self._registry().resolve("somevendor/released-yesterday").name == "openrouter"

    def test_openrouters_own_ids_route_to_it(self) -> None:
        assert self._registry().resolve("openrouter/auto").name == "openrouter"

    def test_registering_it_does_not_hijack_the_direct_providers(self) -> None:
        """Two OpenAI-shaped clients used to collide on the name `openai`."""
        registry = self._registry()

        assert registry.resolve("gpt-5.6-luna").name == "openai"
        assert registry.resolve("gemini-3.5-flash").name == "gemini"
        assert registry.resolve("openai/gpt-oss-120b").name == "groq"

    def test_a_gemini_model_served_by_openrouter_is_not_confused_with_googles(self) -> None:
        """Same model, two routes, two prices. The namespace is what decides."""
        registry = self._registry()

        assert registry.resolve("google/gemini-3-pro-preview").name == "openrouter"
        assert registry.resolve("gemini-3-pro-preview").name == "gemini"

    def test_without_the_client_the_error_names_the_missing_provider(self) -> None:
        from types import SimpleNamespace

        from llm_gateway import UnknownModelError

        registry = build_registry(openai_client=SimpleNamespace())

        with pytest.raises(UnknownModelError, match="openrouter"):
            registry.resolve("deepseek/deepseek-chat-v3.1")
