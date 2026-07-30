# internal-llm-gateway

A neutral gateway for LLM calls: typed contracts, thin provider adapters, and
explicit accounting of retries, fallbacks, tokens and cost.

It knows about **providers**, never about products. It holds no credentials,
reads no environment variables, ships no prompts and stores no business
schemas. Everything product-shaped — ledgers, tenants, alerting, history,
prompts — stays in the application, wired in through optional ports.

```
application → local facade/adapter → llm_gateway → provider SDK
```

Never the reverse, and never application A → application B.

## Install

Provider SDKs are **optional extras**. Install only what you call:

```bash
pip install "internal-llm-gateway[gemini]"
pip install "internal-llm-gateway[openai,gemini,groq]"
```

Importing the package with no extra installed works by design; asking for a
provider you have not installed raises a typed error naming the exact install
command.

## Use

```python
from decimal import Decimal

from pydantic import BaseModel

from llm_gateway import (
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ModelRate,
    ResponseFormat,
    RetryPolicy,
    StaticPriceCatalog,
)
from llm_gateway.factories import build_registry, create_gemini_client


class Answer(BaseModel):
    verdict: str


gateway = LLMGateway(
    registry=build_registry(gemini_client=create_gemini_client(api_key=my_key)),
    price_catalog=StaticPriceCatalog(
        version="2026-07-30",
        rates={"gemini-3.5-flash-lite": ModelRate(Decimal("0.3"), Decimal("2.5"))},
    ),
)

result = await gateway.generate(
    LLMRequest(
        model="gemini-3.5-flash-lite",
        system_prompt="Answer strictly from the supplied evidence.",
        messages=(Message("user", question),),
        response_format=ResponseFormat.JSON_SCHEMA,
        response_schema=Answer,
        temperature=0,
        retry_policy=RetryPolicy.transient(max_attempts=2),
        fallback_policy=FallbackPolicy.disabled(),
        request_id=request_id,
        source="my-feature",
    )
)

result.output                    # Answer instance — no metadata mixed in
result.usage.input_tokens        # None means "not reported", not zero
result.cost.amount_usd           # None when unavailable, never a fake 0
result.cost.measurement          # ACTUAL | ESTIMATED | UNAVAILABLE
result.execution.model_used      # what actually answered
result.execution.fallback_used   # a fallback is never silent
result.execution.attempts        # every attempt, including the failed ones
```

## The guarantees

These are enforced by tests, not by convention:

| Guarantee | Why it matters |
|---|---|
| Unreported usage is `None`, not `0` | A zero token count silently under-bills |
| Unknown cost is `UNAVAILABLE`, not `USD 0` | "Free" and "unknown" are different facts |
| Cost aggregates **every billable attempt** | A retry that failed may still be invoiced |
| Fallback is off by default and always visible | A silent model switch corrupts A/B comparisons and cost attribution |
| Exhausted calls **raise** | They never return something that looks like a success |
| Errors carry the attempts already made | A failure still accounts for the money it spent |
| Output, usage, execution and cost are separate | A token count can never be mistaken for a business field |
| Sinks never receive prompts or responses | Observability without storing content |
| No module reads the environment | The application owns its credentials |
| Importing needs no provider extra | Eight consumers, eight dependency sets |

## Extending

Ports are optional and default to no-op: `UsageSink`, `EventSink`, `AlertSink`,
`PriceCatalog`. Implement what you need; the package will not reach into your
application to find them.

Adding to the public API follows the **two-consumer rule**: nothing is promoted
into the core until two distinct applications need it. Until then it belongs in
that application's local adapter.

## Providers

| Provider | Extra | Notes |
|---|---|---|
| OpenAI | `[openai]` | Responses API. Also serves **OpenRouter** via `base_url` |
| Google Gemini | `[gemini]` | `google-genai` async surface, not the retired `google-generativeai` |
| Groq | `[groq]` | Chat Completions. Declares no schema enforcement; the gateway validates after |

Capabilities are declared per provider and never faked as identical — query
`adapter.capabilities` before relying on one.

## Not in this version

Tools/function calling, file attachments and Gemini File Search are
deliberately absent from `0.1.0`. Each is a real capability with its own cost
and failure model, and none is needed by the first consumers. See `CHANGELOG.md`.

## Development

```bash
uv sync
uv run pytest        # no network, no cost, no extras required
uv run ruff check .
uv run mypy
uv build
```

Python 3.11+ (the floor is set by the oldest consumer), Pydantic v2.
