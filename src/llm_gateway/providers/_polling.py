"""Reading a running job's status without abandoning it over one bad answer.

A submit-and-poll adapter has already been paid when it starts polling. If one
status read that hit a 503 ended the attempt, the job would keep running and
billing with nobody waiting for it, and ``RetryPolicy.transient`` would then
submit — and pay for — a second one. A short run of transient failures on the
*read* is therefore ridden out; anything else, or a run that does not end, is
raised as before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from llm_gateway.errors import LLMGatewayError
from llm_gateway.providers.error_mapping import classify_provider_error

#: Consecutive transient failures one status read survives. Small on purpose:
#: enough for a blip, not enough to hide a provider that is down.
TOLERATED_STATUS_ERRORS = 2


async def read_status(
    get: Callable[[str], Awaitable[dict[str, Any]]],
    path: str,
    *,
    interval_seconds: float,
) -> dict[str, Any]:
    """GET ``path``, retrying only transient failures, at most a few in a row."""
    failures = 0
    while True:
        try:
            return await get(path)
        except LLMGatewayError:
            raise
        except Exception as error:
            typed = classify_provider_error(error)
            if not typed.transient or failures >= TOLERATED_STATUS_ERRORS:
                raise typed from error
        failures += 1
        await asyncio.sleep(interval_seconds)
