"""`total_seconds` must be a total, not a per-attempt allowance.

Reported in review: with retries enabled, a 200s "total" budget allowed roughly
400s plus backoff, because the value was only ever applied per attempt. A
number that names itself a total and is not one is precisely the class of bug
this package exists to prevent.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from llm_gateway import (
    AllAttemptsFailed,
    LLMGateway,
    LLMRequest,
    Message,
    ProviderRegistry,
    ProviderResponse,
    RetryPolicy,
    TimeoutPolicy,
)


class SlowAdapter:
    """Never returns. Only a timeout can end a call to it."""

    name = "slow"

    def __init__(self, delay: float = 30.0) -> None:
        self._delay = delay
        self.calls = 0

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        self.calls += 1
        await asyncio.sleep(self._delay)
        raise AssertionError("unreachable: the delay always outlasts the budget")


def _gateway(adapter: SlowAdapter) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(adapter, model_prefixes=("slow",))
    return LLMGateway(registry=registry)


def _request(**kwargs: object) -> LLMRequest:
    return LLMRequest(model="slow-x", messages=(Message("user", "x"),), **kwargs)  # type: ignore[arg-type]


class TestTotalBudget:
    async def test_retries_cannot_exceed_the_total_budget(self) -> None:
        """Six attempts of 0.15s each would be 0.90s against a 0.30s budget."""
        adapter = SlowAdapter()
        started = time.perf_counter()

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(
                _request(
                    timeout_policy=TimeoutPolicy(
                        total_seconds=0.30, per_attempt_seconds_override=0.15
                    ),
                    retry_policy=RetryPolicy.transient(max_attempts=6, base_delay_seconds=0),
                )
            )

        elapsed = time.perf_counter() - started
        assert elapsed < 0.60, f"total budget not enforced: {elapsed:.2f}s against a 0.30s budget"

    async def test_a_single_attempt_may_use_the_whole_budget(self) -> None:
        adapter = SlowAdapter()
        started = time.perf_counter()

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(
                _request(timeout_policy=TimeoutPolicy(total_seconds=0.30))
            )

        assert adapter.calls == 1
        assert time.perf_counter() - started >= 0.25, "the attempt was cut short of its budget"

    async def test_exhausting_the_budget_still_reports_its_attempts(self) -> None:
        adapter = SlowAdapter()

        with pytest.raises(AllAttemptsFailed) as caught:
            await _gateway(adapter).generate(
                _request(
                    timeout_policy=TimeoutPolicy(
                        total_seconds=0.30, per_attempt_seconds_override=0.10
                    ),
                    retry_policy=RetryPolicy.transient(max_attempts=6, base_delay_seconds=0),
                )
            )

        assert caught.value.attempts, "a timed-out call must still account for what it spent"

    async def test_the_error_names_the_budget_it_exceeded(self) -> None:
        with pytest.raises(AllAttemptsFailed, match="total budget"):
            await _gateway(SlowAdapter()).generate(
                _request(timeout_policy=TimeoutPolicy(total_seconds=0.10))
            )

    async def test_a_per_attempt_override_bounds_each_try(self) -> None:
        adapter = SlowAdapter()

        with pytest.raises(AllAttemptsFailed):
            await _gateway(adapter).generate(
                _request(
                    timeout_policy=TimeoutPolicy(
                        total_seconds=5.0, per_attempt_seconds_override=0.05
                    ),
                    retry_policy=RetryPolicy.transient(max_attempts=3, base_delay_seconds=0),
                )
            )

        assert adapter.calls == 3, "each attempt should be cut individually, not the whole call"
