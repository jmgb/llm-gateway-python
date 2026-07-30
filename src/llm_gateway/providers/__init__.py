"""Provider adapters.

Each adapter imports its SDK lazily, inside the adapter module, so that
importing ``llm_gateway`` works with no provider extra installed at all.
"""

from llm_gateway.providers.base import ProviderAdapter, ProviderResponse

__all__ = ["ProviderAdapter", "ProviderResponse"]
