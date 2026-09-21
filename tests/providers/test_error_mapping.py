"""SDK exceptions become typed errors without importing any SDK.

Classification is structural — HTTP status first, class name second — so the
package can map a provider failure it has never imported, and so installing an
extra is never required just to interpret an error.
"""

from llm_gateway import (
    AuthenticationError,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitedError,
    ServiceUnavailableError,
)
from llm_gateway.providers.error_mapping import classify_provider_error


class FakeSDKError(Exception):
    def __init__(
        self,
        message: str = "boom",
        status_code: int | None = None,
        *,
        code: object = None,
        type: str | None = None,  # the SDK attribute is called `type`
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        self.code = code
        self.type = type


def test_429_is_rate_limited() -> None:
    assert isinstance(classify_provider_error(FakeSDKError(status_code=429)), RateLimitedError)


def test_503_is_service_unavailable() -> None:
    error = classify_provider_error(FakeSDKError(status_code=503))

    assert isinstance(error, ServiceUnavailableError)
    assert error.transient is True


def test_401_is_authentication_and_is_not_transient() -> None:
    error = classify_provider_error(FakeSDKError(status_code=401))

    assert isinstance(error, AuthenticationError)
    assert error.transient is False


def test_400_is_an_invalid_request() -> None:
    assert isinstance(classify_provider_error(FakeSDKError(status_code=400)), InvalidRequestError)


def test_class_name_is_used_when_there_is_no_status_code() -> None:
    class RateLimitError(Exception):
        pass

    assert isinstance(classify_provider_error(RateLimitError()), RateLimitedError)


def test_timeouts_are_recognised_by_class_name() -> None:
    class APITimeoutError(Exception):
        pass

    assert isinstance(classify_provider_error(APITimeoutError()), ProviderTimeoutError)


def test_builtin_timeout_is_recognised() -> None:
    assert isinstance(classify_provider_error(TimeoutError()), ProviderTimeoutError)


def test_an_unrecognised_error_stays_a_generic_provider_error() -> None:
    error = classify_provider_error(ValueError("something odd"))

    assert isinstance(error, ProviderError)
    assert error.transient is False


def test_the_original_exception_is_preserved_as_the_cause() -> None:
    original = FakeSDKError(status_code=429)

    assert classify_provider_error(original).__cause__ is original


def test_the_message_never_leaks_the_original_payload() -> None:
    original = FakeSDKError("api key sk-secret-value rejected", status_code=401)

    assert "sk-secret-value" not in str(classify_provider_error(original))


def test_a_provider_error_code_is_named_in_the_message() -> None:
    """Groq answers 400 for a bad parameter and for output that failed its
    JSON check alike; only the short ``code`` tells them apart."""
    original = FakeSDKError(
        "Failed to generate JSON. <the whole body>",
        status_code=400,
        code="json_validate_failed",
    )

    assert str(classify_provider_error(original)) == (
        "provider returned HTTP 400 (json_validate_failed)"
    )


def test_the_error_type_names_the_failure_when_there_is_no_code() -> None:
    original = FakeSDKError(status_code=400, type="invalid_request_error")

    assert str(classify_provider_error(original)) == (
        "provider returned HTTP 400 (invalid_request_error)"
    )


def test_a_code_that_is_not_an_identifier_is_not_copied() -> None:
    """Only a bare identifier is safe: anything with spaces could be the
    provider's free text echoing the request."""
    original = FakeSDKError(status_code=400, code="api key sk-secret-value rejected")

    assert str(classify_provider_error(original)) == "provider returned HTTP 400"


def test_a_numeric_code_is_a_status_not_a_reason() -> None:
    original = FakeSDKError(code=429)

    assert str(classify_provider_error(original)) == "provider returned HTTP 429"
