"""Typed errors, and the policies that decide what to do about them."""

import pytest

from llm_gateway import (
    AllAttemptsFailed,
    AuthenticationError,
    FallbackPolicy,
    InvalidRequestError,
    LLMGatewayError,
    OutputParsingError,
    ProviderNotInstalled,
    RateLimitedError,
    RetryPolicy,
    ServiceUnavailableError,
    TimeoutPolicy,
    UnknownModelError,
)


class TestErrorTaxonomy:
    def test_every_error_shares_one_root(self) -> None:
        for error in (
            RateLimitedError("x"),
            AuthenticationError("x"),
            OutputParsingError("x"),
            ProviderNotInstalled("x"),
            UnknownModelError("x"),
        ):
            assert isinstance(error, LLMGatewayError)

    def test_rate_limiting_and_unavailability_are_transient(self) -> None:
        assert RateLimitedError("x").transient is True
        assert ServiceUnavailableError("x").transient is True

    def test_authentication_and_bad_requests_are_permanent(self) -> None:
        assert AuthenticationError("x").transient is False
        assert InvalidRequestError("x").transient is False

    def test_missing_extra_names_the_install_target(self) -> None:
        error = ProviderNotInstalled.for_provider("groq")

        assert "neutral-llm-gateway[groq]" in str(error)


class TestRetryPolicy:
    def test_disabled_allows_a_single_attempt(self) -> None:
        assert RetryPolicy.disabled().max_attempts == 1

    def test_transient_policy_retries_transient_errors(self) -> None:
        policy = RetryPolicy.transient(max_attempts=3)

        assert policy.should_retry(RateLimitedError("x"), attempt_number=1) is True

    def test_transient_policy_never_retries_permanent_errors(self) -> None:
        policy = RetryPolicy.transient(max_attempts=3)

        assert policy.should_retry(AuthenticationError("x"), attempt_number=1) is False

    def test_retry_stops_at_the_attempt_ceiling(self) -> None:
        policy = RetryPolicy.transient(max_attempts=2)

        assert policy.should_retry(RateLimitedError("x"), attempt_number=2) is False

    def test_backoff_grows_with_each_attempt(self) -> None:
        policy = RetryPolicy.transient(max_attempts=4, base_delay_seconds=1.0)

        assert policy.delay_before(attempt_number=1) == 1.0
        assert policy.delay_before(attempt_number=2) == 2.0

    def test_max_attempts_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one attempt"):
            RetryPolicy.transient(max_attempts=0)


class TestFallbackPolicy:
    def test_disabled_offers_no_alternative_models(self) -> None:
        assert FallbackPolicy.disabled().models == ()

    def test_disabled_is_the_default_so_silence_is_never_the_default(self) -> None:
        assert FallbackPolicy().models == ()

    def test_models_are_tried_in_declared_order(self) -> None:
        policy = FallbackPolicy.models_in_order("b", "c")

        assert policy.models == ("b", "c")


class TestTimeoutPolicy:
    def test_per_attempt_timeout_defaults_to_the_total(self) -> None:
        policy = TimeoutPolicy(total_seconds=30)

        assert policy.per_attempt_seconds == 30

    def test_non_positive_timeout_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TimeoutPolicy(total_seconds=0)


class TestAllAttemptsFailed:
    def test_it_carries_the_cost_already_incurred(self) -> None:
        error = AllAttemptsFailed("every attempt failed", attempts=())

        assert isinstance(error, LLMGatewayError)
        assert error.attempts == ()
