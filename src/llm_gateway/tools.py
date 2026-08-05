"""Provider-neutral function tool calling.

The package declares tools, receives the calls a model makes and puts the
results back on the wire in the dialect each provider expects. It stops there:
it never runs an application function, never decides whether one is allowed and
never owns the loop that would call it again. Execution is where authorisation,
side effects and business schemas live, and none of those belong to a package
that knows about providers rather than about products.

So the contract is deliberately one round trip. An application receives typed
calls, does its own work, and supplies typed results in the *next* request.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

NO_PARAMETERS: Mapping[str, Any] = MappingProxyType({"type": "object", "properties": {}})
"""What a function taking no arguments declares. Providers reject a bare ``{}``."""


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """One function the model may ask to have run.

    ``parameters`` is JSON Schema, passed through as written. The package does
    not rewrite a caller's schema beyond what a provider refuses to accept.
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=lambda: NO_PARAMETERS)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a function tool needs a name")


class ToolChoice(Enum):
    """How free the model is to decide."""

    AUTO = "auto"
    """It may call a tool or answer. The default when tools are declared."""

    NONE = "none"
    """It must answer. The tools stay declared but unreachable this turn."""

    REQUIRED = "required"
    """It must call one of them, and may not answer instead."""


@dataclass(frozen=True, slots=True)
class RequiredTool:
    """Force one named function, rather than any of them."""

    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a forced tool needs a name")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the model made, with its arguments already parsed.

    ``id`` is the provider's own correlation id, kept verbatim: it is what the
    continuation has to quote, and a regenerated one answers the wrong call.
    """

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a tool call needs the provider's correlation id")
        if not self.name.strip():
            raise ValueError("a tool call needs a function name")


@dataclass(frozen=True, slots=True)
class ToolResult:
    """What the application's function returned, and which call it answers.

    The call is held rather than only its id, so a result cannot be built
    without the call it belongs to. Providers need both halves back on the wire
    — the call the model made *and* its output — and a pair that drifted apart
    is a 400 at best and an answer to the wrong question at worst.

    ``output`` is already serialised by the application: what a function
    returns is a business shape, and this package does not model those.
    """

    call: ToolCall
    output: str


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    """One call as the provider worded it, arguments still unparsed.

    Adapters normalise to this and stop. Parsing happens in the gateway,
    because a reply whose arguments are unusable was still answered and still
    billed, and only the gateway records attempts.
    """

    id: str
    name: str
    arguments: str
