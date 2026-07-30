"""Neutral LLM gateway.

This package knows about providers, not about products. It never imports a
consuming application, never reads credentials on import and never logs prompt
or response content by default.
"""

from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import (
    Attempt,
    AttemptOutcome,
    Execution,
    LLMRequest,
    LLMResult,
    Message,
    ReasoningEffort,
    ResponseFormat,
    Role,
)
from llm_gateway.errors import (
    AllAttemptsFailed,
    AuthenticationError,
    ConfigurationError,
    InvalidRequestError,
    LLMGatewayError,
    OutputError,
    OutputParsingError,
    ProviderError,
    ProviderNotInstalled,
    ProviderTimeoutError,
    RateLimitedError,
    SchemaValidationError,
    ServiceUnavailableError,
    UnknownModelError,
)
from llm_gateway.gateway import LLMGateway
from llm_gateway.models import (
    CATALOG_VERSION,
    MODEL_CATALOG,
    ModelInfo,
    builtin_price_catalog,
    lookup_model,
    models_by_provider,
    resolve_provider,
)
from llm_gateway.policies import FallbackPolicy, RetryPolicy, TimeoutPolicy
from llm_gateway.ports import (
    AlertSink,
    EventSink,
    NullAlertSink,
    NullEventSink,
    NullUsageSink,
    UsageRecord,
    UsageSink,
)
from llm_gateway.pricing import (
    Cost,
    CostMeasurement,
    ModelRate,
    NullPriceCatalog,
    PriceCatalog,
    StaticPriceCatalog,
)
from llm_gateway.providers.base import ProviderAdapter, ProviderResponse
from llm_gateway.registry import ProviderRegistry
from llm_gateway.usage import TokenUsage

__all__ = [
    "CATALOG_VERSION",
    "MODEL_CATALOG",
    "AlertSink",
    "AllAttemptsFailed",
    "Attempt",
    "AttemptOutcome",
    "AuthenticationError",
    "ConfigurationError",
    "Cost",
    "CostMeasurement",
    "EventSink",
    "Execution",
    "FallbackPolicy",
    "InvalidRequestError",
    "LLMGateway",
    "LLMGatewayError",
    "LLMRequest",
    "LLMResult",
    "Message",
    "ModelInfo",
    "ModelRate",
    "NullAlertSink",
    "NullEventSink",
    "NullPriceCatalog",
    "NullUsageSink",
    "OutputError",
    "OutputParsingError",
    "PriceCatalog",
    "ProviderAdapter",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderNotInstalled",
    "ProviderRegistry",
    "ProviderResponse",
    "ProviderTimeoutError",
    "RateLimitedError",
    "ReasoningEffort",
    "ResponseFormat",
    "RetryPolicy",
    "Role",
    "SchemaValidationError",
    "ServiceUnavailableError",
    "StaticPriceCatalog",
    "TimeoutPolicy",
    "TokenUsage",
    "UnknownModelError",
    "UsageRecord",
    "UsageSink",
    "builtin_price_catalog",
    "lookup_model",
    "models_by_provider",
    "resolve_provider",
]
