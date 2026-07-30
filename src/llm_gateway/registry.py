"""Model identifier to provider resolution.

Explicit and inspectable: routing is a lookup table the application can read
and extend, not a chain of substring guesses buried in a 900-line function.
"""

from __future__ import annotations

from llm_gateway.errors import UnknownModelError
from llm_gateway.providers.base import ProviderAdapter


class ProviderRegistry:
    """Maps model identifiers to the adapter that serves them."""

    def __init__(self) -> None:
        self._by_prefix: list[tuple[str, ProviderAdapter]] = []
        self._by_name: dict[str, ProviderAdapter] = {}

    def register(self, adapter: ProviderAdapter, *, model_prefixes: tuple[str, ...]) -> None:
        """Register an adapter for every model whose id starts with a prefix."""
        self._by_name[adapter.name] = adapter
        for prefix in model_prefixes:
            self._by_prefix.append((prefix, adapter))
        # Longest prefix wins, so "gemini-embedding" can outrank "gemini".
        self._by_prefix.sort(key=lambda pair: len(pair[0]), reverse=True)

    def resolve(self, model: str) -> ProviderAdapter:
        for prefix, adapter in self._by_prefix:
            if model.startswith(prefix):
                return adapter
        raise UnknownModelError(
            f"no provider is registered for model {model!r}; "
            f"known prefixes: {sorted({p for p, _ in self._by_prefix})}"
        )

    def by_name(self, provider: str) -> ProviderAdapter:
        try:
            return self._by_name[provider]
        except KeyError as error:
            raise UnknownModelError(f"no provider named {provider!r}") from error

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))
