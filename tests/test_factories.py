"""Optional convenience for building a registry.

The extras are not installed in this environment, which is exactly what makes
these tests meaningful: they prove the failure is a readable instruction and
not an ImportError traceback.
"""

import pytest

from llm_gateway import ProviderNotInstalled, ProviderRegistry
from llm_gateway.factories import (
    build_registry,
    create_gemini_client,
    create_groq_client,
    create_openai_client,
)


class FakeAdapter:
    name = "fake"


def test_a_missing_extra_names_the_install_command() -> None:
    with pytest.raises(ProviderNotInstalled, match=r"internal-llm-gateway\[openai\]"):
        create_openai_client(api_key="unused")


def test_each_provider_names_its_own_extra() -> None:
    with pytest.raises(ProviderNotInstalled, match=r"\[gemini\]"):
        create_gemini_client(api_key="unused")
    with pytest.raises(ProviderNotInstalled, match=r"\[groq\]"):
        create_groq_client(api_key="unused")


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
