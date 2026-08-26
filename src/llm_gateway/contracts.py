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

from llm_gateway.policies import FallbackPolicy, RetryPolicy, RoutingPreference, TimeoutPolicy
from llm_gateway.pricing import Cost
from llm_gateway.tools import FunctionTool, RequiredTool, ToolCall, ToolChoice, ToolResult
from llm_gateway.usage import TokenUsage

Role = Literal["system", "user", "assistant"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
Verbosity = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of conversation."""

    role: Role
    content: str


@dataclass(frozen=True, slots=True)
class FileAttachment:
    """An already-uploaded remote file referenced by provider id."""

    file_id: str

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
    tools: tuple[FunctionTool, ...] = ()
    """Functions the model may ask to have run. The package runs none of them."""
    tool_choice: ToolChoice | RequiredTool | None = None
    """Unset means the provider's own default, which is ``AUTO`` once tools exist."""
    tool_results: tuple[ToolResult, ...] = ()
    """Answers to the calls of a previous turn, replayed with the calls they answer."""
    system_prompt: str | None = None
    response_format: ResponseFormat = ResponseFormat.TEXT
    response_schema: type[BaseModel] | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: ReasoningEffort | None = None
    verbosity: Verbosity | None = None
    """How much prose to spend on the answer, where the provider offers the dial.

    Separate from ``max_output_tokens``: that truncates an answer already being
    written, and pays for every token up to the cut. This asks for a shorter one.
    """
    routing: RoutingPreference = field(default_factory=RoutingPreference)
    """Which upstream should serve the call, honoured only by an aggregator."""
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
        self._validate_tools()

    def _validate_tools(self) -> None:
        """Reject here what a provider would reject after being paid for it."""
        names: set[str] = set()
        for tool in self.tools:
            if tool.name in names:
                raise ValueError(f"two tools share the name {tool.name!r}")
            names.add(tool.name)

        if self.tool_choice is not None and not self.tools:
            raise ValueError("tool_choice needs at least one tool to choose from")
        if isinstance(self.tool_choice, RequiredTool) and self.tool_choice.name not in names:
            raise ValueError(f"tool_choice names {self.tool_choice.name!r}, which is not declared")

        seen: set[str] = set()
        for result in self.tool_results:
            if result.call.id in seen:
                raise ValueError(f"two results answer the call {result.call.id!r}")
            seen.add(result.call.id)


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
    error_message: str | None = None
    """What the failure actually said, truncated to stay loggable.

    ``error_type`` alone names the class of problem and stops there: a rate
    limit, a refused schema and a wrong API key are all ``ProviderError``. The
    message is the only part that says which one, so an operator reading an
    alert does not have to reproduce the call to find out.
    """
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
    def fallback_cause(self) -> Attempt | None:
        """The failure that made the gateway leave the requested model.

        A fallback is only ever reported as a pair of model names, which says
        that something went wrong and not what. This returns the last failed
        attempt on the requested model, so the reason travels with the alert
        instead of having to be dug out of a log that may already be gone.
        """
        for attempt in reversed(self.attempts):
            if attempt.model == self.requested_model and attempt.outcome is AttemptOutcome.FAILED:
                return attempt
        return None

    @property
    def failures(self) -> tuple[Attempt, ...]:
        """Every attempt that failed, in the order they were made.

        ``fallback_cause`` names the one failure that ended the requested
        model's turn, which is the headline. It is not the whole story: a plan
        with a retry and two models can fail three times for three different
        reasons, and a reader who sees only the first cannot tell a provider
        having a bad minute from a request no model will accept.
        """
        return tuple(a for a in self.attempts if a.outcome is AttemptOutcome.FAILED)

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
    tool_calls: tuple[ToolCall, ...] = ()
    """What the model asked to have run instead of answering.

    Never flattened into ``output``: a caller that treats a call as text sends
    the model's own request for a function straight to a user. When this is
    non-empty ``output`` is ``None``, and the two are checked separately.
    """

    @property
    def text(self) -> str:
        """The output as text. Raises when the output was structured."""
        if not isinstance(self.output, str):
            raise TypeError(
                f"output is {type(self.output).__name__}, not text; use .output instead"
            )
        return self.output
