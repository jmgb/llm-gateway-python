"""Typed errors.

Exhausted calls raise; they never return a dictionary that a caller could
mistake for a successful response. Errors carry the attempts already made, so
that a failure still accounts for the money it spent.

No error message may contain credentials, prompts or response content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_gateway.audio import AudioAttempt
    from llm_gateway.contracts import Attempt
    from llm_gateway.media import ImageAttempt, VideoAttempt


#: A provider message can carry its whole response body. This text ends up in
#: alerts and logs, so it is cut: what identifies the failure is at the front,
#: and the rest only fills a screen.
MAX_ERROR_MESSAGE_CHARS = 500


def message_of(failure: BaseException) -> str:
    """The failure as text, truncated and never empty.

    The same rule as every other message in this module applies: what a
    provider says about *why* a call was rejected is safe to record, its echo
    of the prompt is not, so adapters keep response content out of the errors
    they raise and this only ever passes along what they built.

    An exception with no message would record ``""``, which reads in a log as
    "there was no error" rather than "the provider did not explain itself".
    """
    text = str(failure).strip() or type(failure).__name__
    if len(text) <= MAX_ERROR_MESSAGE_CHARS:
        return text
    return text[:MAX_ERROR_MESSAGE_CHARS] + "…"


class LLMGatewayError(Exception):
    """Root of every error raised by this package."""

    transient: bool = False


class ConfigurationError(LLMGatewayError):
    """The request could not even be dispatched."""


class UnknownModelError(ConfigurationError):
    """No provider claims this model identifier."""


class ProviderNotInstalled(ConfigurationError):
    """The provider's optional extra is not installed."""

    @classmethod
    def for_provider(cls, provider: str) -> ProviderNotInstalled:
        """Name the extra without assuming how the caller installs things.

        Telling a uv-managed project to run ``pip install`` would install
        outside its lockfile, so both commands are offered rather than one.
        """
        target = f"neutral-llm-gateway[{provider}]"
        return cls(
            f"provider {provider!r} is not installed; add the {provider!r} extra: "
            f'uv add "{target}"  (or: pip install "{target}")'
        )


class ProviderError(LLMGatewayError):
    """The provider was reached, or reaching it failed."""


class AuthenticationError(ProviderError):
    """Credentials rejected. Retrying cannot help."""


class InvalidRequestError(ProviderError):
    """The provider rejected the request as malformed. Retrying cannot help."""


class RateLimitedError(ProviderError):
    """Quota exhausted for now."""

    transient = True


class ServiceUnavailableError(ProviderError):
    """Provider outage or overload."""

    transient = True


class ProviderTimeoutError(ProviderError):
    """The attempt exceeded its timeout."""

    transient = True


class OutputError(LLMGatewayError):
    """The call returned, but the payload was unusable."""


class OutputParsingError(OutputError):
    """The model did not return parseable JSON."""


class SchemaValidationError(OutputError):
    """The payload parsed, but did not satisfy the requested schema."""


class AllAttemptsFailed(LLMGatewayError):
    """Every model and every retry failed.

    ``attempts`` preserves what was spent, so a failed call is still auditable.
    """

    def __init__(self, message: str, *, attempts: tuple[Attempt, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts

    @property
    def last_error(self) -> str | None:
        return self.attempts[-1].error_type if self.attempts else None

    @property
    def last_error_message(self) -> str | None:
        """What the final failure said, not just which class it belonged to."""
        return self.attempts[-1].error_message if self.attempts else None


class AllTranscriptionsFailed(LLMGatewayError):
    """Every audio model and retry failed, with duration accounting preserved."""

    def __init__(self, message: str, *, attempts: tuple[AudioAttempt, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts

    @property
    def last_error(self) -> str | None:
        return self.attempts[-1].error_type if self.attempts else None

    @property
    def last_error_message(self) -> str | None:
        """What the final failure said, not just which class it belonged to."""
        return self.attempts[-1].error_message if self.attempts else None


class AllImagesFailed(LLMGatewayError):
    """Every image model and retry failed, with image accounting preserved."""

    def __init__(self, message: str, *, attempts: tuple[ImageAttempt, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts

    @property
    def last_error(self) -> str | None:
        return self.attempts[-1].error_type if self.attempts else None

    @property
    def last_error_message(self) -> str | None:
        """What the final failure said, not just which class it belonged to."""
        return self.attempts[-1].error_message if self.attempts else None


class AllVideosFailed(LLMGatewayError):
    """Every video model and retry failed, with video accounting preserved."""

    def __init__(self, message: str, *, attempts: tuple[VideoAttempt, ...]) -> None:
        super().__init__(message)
        self.attempts = attempts

    @property
    def last_error(self) -> str | None:
        return self.attempts[-1].error_type if self.attempts else None

    @property
    def last_error_message(self) -> str | None:
        """What the final failure said, not just which class it belonged to."""
        return self.attempts[-1].error_message if self.attempts else None
