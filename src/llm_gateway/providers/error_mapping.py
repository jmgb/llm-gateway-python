"""Classify provider SDK exceptions into the package's typed errors.

Deliberately structural. Importing ``openai`` just to write
``except openai.RateLimitError`` would make error handling depend on an extra
being installed, and would break the moment an SDK reorganises its exception
tree. HTTP status is the stable signal; the class name is the fallback.

The original exception is preserved as ``__cause__``, but its message is not
copied into the new one: provider messages routinely echo request payloads and
sometimes credentials. What is copied is the provider's short error code
(``json_validate_failed``, ``invalid_request_error``, ``INVALID_ARGUMENT``):
one status covers failures with nothing in common — Groq answers 400 both for
a bad parameter and for output that failed its JSON check — and the code is
the only part of the body that says which, while being an identifier that can
carry no payload.
"""

from __future__ import annotations

import re

from llm_gateway.errors import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    ServiceUnavailableError,
)

_BY_STATUS: dict[int, type[ProviderError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: InvalidRequestError,
    408: ProviderTimeoutError,
    413: InvalidRequestError,
    422: InvalidRequestError,
    429: RateLimitedError,
}

_BY_NAME_FRAGMENT: tuple[tuple[str, type[ProviderError]], ...] = (
    ("ratelimit", RateLimitedError),
    ("resourceexhausted", RateLimitedError),
    ("timeout", ProviderTimeoutError),
    ("deadline", ProviderTimeoutError),
    ("unavailable", ServiceUnavailableError),
    ("serviceunavailable", ServiceUnavailableError),
    ("internalserver", ServiceUnavailableError),
    ("authentication", AuthenticationError),
    ("permissiondenied", AuthenticationError),
    ("unauthorized", AuthenticationError),
    ("badrequest", InvalidRequestError),
    ("invalidargument", InvalidRequestError),
    ("notfound", InvalidRequestError),
)


def classify_provider_error(error: BaseException) -> ProviderError:
    """Map any exception raised by a provider SDK to a typed gateway error."""
    typed = _classify(error)
    typed.__cause__ = error
    return typed


def _classify(error: BaseException) -> ProviderError:
    status = _status_code(error)
    if status is not None:
        message = f"provider returned HTTP {status}"
        code = _error_code(error)
        if code is not None:
            message = f"{message} ({code})"
        if status in _BY_STATUS:
            return _BY_STATUS[status](message)
        if 500 <= status < 600:
            return ServiceUnavailableError(message)
        return ProviderError(message)

    if isinstance(error, TimeoutError):
        return ProviderTimeoutError("the provider call timed out")

    name = type(error).__name__.lower()
    for fragment, error_type in _BY_NAME_FRAGMENT:
        if fragment in name:
            return error_type(f"provider raised {type(error).__name__}")

    return ProviderError(f"provider raised {type(error).__name__}")


def _status_code(error: BaseException) -> int | None:
    for attribute in ("status_code", "http_status", "code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and 100 <= value < 600:
            return value
    response = getattr(error, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


#: A provider code is a bare identifier. Anything else — spaces, quotes,
#: brackets — is free text, and free text is where request payloads end up.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _error_code(error: BaseException) -> str | None:
    """The provider's short reason, if the SDK exposes one as an identifier.

    ``code`` is what OpenAI-compatible SDKs (OpenAI, Groq, OpenRouter) set from
    the body; ``type`` is their coarser family when there is no code; ``status``
    is the Google SDK's ``INVALID_ARGUMENT``-style name. An ``int`` in any of
    them is an HTTP status, already reported, not a reason.
    """
    for attribute in ("code", "type", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, str) and _IDENTIFIER.match(value):
            return value
    return None
