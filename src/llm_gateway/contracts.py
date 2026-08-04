"""Neutral request and result contracts.

The result never mixes the model's output with technical metadata in one
dictionary: ``output``, ``usage``, ``execution`` and ``cost`` are separate, so
a caller can never mistake a token count for a business field. Legacy facades
may flatten it during migration; the package does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel

from llm_gateway.policies import FallbackPolicy, RetryPolicy, TimeoutPolicy
from llm_gateway.pricing import Cost
from llm_gateway.usage import TokenUsage

Role = Literal["system", "user", "assistant"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class FileAttachment:
    """An already-uploaded remote file referenced by provider id."""

    file_id: str
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if not self.file_id.strip():
            raise ValueError("file_id must be non-empty")


class ResponseFormat(Enum):
    """What shape the caller expects back."""

    TEXT = "text"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Everything needed to make one call, and nothing about any product."""

    model: str
    messages: tuple[Message, ...] = ()
    attachments: tuple[FileAttachment, ...] = ()
    system_prompt: str | None = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    response_schema: type[BaseModel] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.disabled)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy.disabled)
    request_id: str | None = None
    """Correlation id. Supplied by the application; the package never invents one."""
    source: str | None = None
    """Which feature made the call. Used for observability, never for routing."""

    def __post_init__(self) -> None:
        if (
            self.response_schema is not None
            and self.response_format is not ResponseFormat.JSON_SCHEMA
        ):
            raise ValueError("response_schema requires response_format=ResponseFormat.JSON_SCHEMA")
        if self.response_format is ResponseFormat.JSON_SCHEMA and self.response_schema is None:
            raise ValueError("response_format=JSON_SCHEMA requires a response_schema")
        if not self.model.strip():
            raise ValueError("a model identifier is required")
        if self.attachments and not any(message.role == "user" for message in self.messages):
            raise ValueError("file attachments require at least one user message")


class AttemptOutcome(Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class FailurePhase(Enum):
    """Where an attempt broke.

    The exception class alone forces a reader to rebuild the context: a
    rate limit and a malformed payload are both "an error", but only one of
    them means the provider answered and was paid for it. Dashboards and
    alerting want that distinction without importing the error hierarchy.
    """

    CONFIGURATION = "configuration"
    """The request was rejected before provider dispatch."""

    PROVIDER = "provider"
    """The provider rejected the call or never answered."""

    TIMEOUT = "timeout"
    """The attempt ran out of its own time budget."""

    OUTPUT_PARSING = "output_parsing"
    """An answer arrived, but JSON could not be recovered from it."""

    SCHEMA_VALIDATION = "schema_validation"
    """JSON parsed, and did not satisfy the requested schema."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """One call to one model. Recorded whether it succeeded or not."""

    index: int
    model: str
    provider: str
    outcome: AttemptOutcome
    usage: TokenUsage
    cost: Cost
    latency_ms: int
    error_type: str | None = None
    billable: bool = True
    """False only when the call never reached the provider."""
    failure_phase: FailurePhase | None = None
    """``None`` on a successful attempt, and only then."""


@dataclass(frozen=True, slots=True)
class Execution:
    """What actually happened, as opposed to what was asked for."""

    requested_model: str
    model_used: str
    provider: str
    finish_reason: str | None
    attempts: tuple[Attempt, ...]
    latency_ms: int

    @property
    def fallback_used(self) -> bool:
        """True when the gateway reached a model from its fallback plan."""
        return bool(self.attempts and self.attempts[-1].model != self.requested_model)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True, slots=True)
class LLMResult:
    """Output and metadata, deliberately kept apart."""

    output: Any
    usage: TokenUsage
    execution: Execution
    cost: Cost

    @property
    def text(self) -> str:
        """The output as text. Raises when the output was structured."""
        if not isinstance(self.output, str):
            raise TypeError(
                f"output is {type(self.output).__name__}, not text; use .output instead"
            )
        return self.output
