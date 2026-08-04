# Pricing

## The problem this solves

Every application that calls an LLM ends up with a table of token prices. Keep
several of those tables and they drift: one gets updated when a provider
changes a rate, the others quietly keep billing yesterday's number. The figures
then stop reconciling against the invoice, and nobody notices until someone
asks why the dashboard disagrees with the bill.

So the table lives here, once, in `src/llm_gateway/models.py`.

A model's price is a fact about the **provider**, not about your product —
which is exactly the test for what belongs in this package.

## Units

Prices are declared in **USD per million tokens**, the way providers publish
them. Rates are consumed as **microUSD per token**. Those are the same number:

```
1 USD / 1,000,000 tokens  =  1e-6 USD / token  =  1 microUSD / token
```

The conversion is the identity, so there is no factor to get wrong. A test
asserts it.

Arithmetic is done in **whole microdollars** with `ROUND_HALF_UP`, so totals
add up exactly and there is no floating-point drift across many small calls.

The model identity catalogue also supports `pricing_unit="audio_minutes"` for
speech models. Those entries remain routable and carry their provider's
per-minute metadata, but `builtin_price_catalog()` excludes them because its
`TokenUsage` contract cannot represent audio duration. Use
`builtin_audio_price_catalog()` and `AudioUsage` for that path.

## The three measurements

`Cost.measurement` is as important as the amount:

| Value | Meaning |
|---|---|
| `ACTUAL` | Every billable dimension was reported by the provider and priced |
| `ESTIMATED` | A **lower bound**: something billable was missing or unpriced |
| `UNAVAILABLE` | No amount could be computed. **Not** the same as free |

An unknown cost is never rendered as `USD 0`. "Free" and "unknown" are
different facts, and conflating them under-declares spend precisely when you
can least afford it.

Similarly, a provider that reports no usage produces `TokenUsage.unknown()`,
whose fields are `None` — not zeroes.

## Reasoning tokens

`output_tokens` always means **everything billed at the output rate**,
reasoning included. `reasoning_tokens` is a breakdown of that number, never an
addition to it, and `visible_output_tokens` is the remainder the caller
actually received.

Providers disagree here, so each adapter normalises at its boundary:

| Provider | Reported as | Adapter |
|---|---|---|
| OpenAI (Responses) | `output_tokens_details.reasoning_tokens`, already inside `output_tokens` | passed through |
| OpenRouter, Groq (Chat Completions) | `completion_tokens_details.reasoning_tokens`, already inside `completion_tokens` | passed through |
| Gemini | `thoughts_token_count`, **outside** `candidates_token_count` | folded into `output_tokens` |
| AssemblyAI | no token usage; duration belongs to the audio contract | kept out of token pricing |

Adding the breakdown back on top of the total is a real bug this package had:
at a thinking effort where reasoning dominates the visible answer, it
overstated cost by roughly 1.5–2×. If you implement your own `PriceCatalog`,
price `usage.billable_output_tokens` and do not add `reasoning_tokens` yourself.

## Retries and fallbacks are billed

`result.cost` aggregates **every attempt that reached the provider**, including
the ones that failed. A call that timed out after the model had already
produced tokens is not free.

When a failed attempt's cost is unknown, the total degrades to `ESTIMATED`
rather than silently ignoring it.

An answer that arrived but could not be used — unparseable JSON, or a payload
that violates the schema — is billed like any other answer, with the exact
usage the provider reported. The model did the work and the invoice will say
so; the only thing that failed is the caller's ability to use the result. Those
attempts appear in `execution.attempts` with `billable=True` and a
`failure_phase` of `output_parsing` or `schema_validation`.

## Versioning

`CATALOG_VERSION` identifies the table that produced an amount, and travels
with every `Cost` and every `UsageRecord`. Store it alongside your figures:
it is what lets you recompute or audit an old number after prices have moved.

Consumers pin a package tag, so nobody's cost figures change without an
explicit upgrade.

## Updating a price

1. Edit the entry in `src/llm_gateway/models.py`.
2. Bump `CATALOG_VERSION`.
3. Note it in `CHANGELOG.md`.
4. Tag a release.

Never delete a model that consumers might still call — mark it
`deprecated=True`. Deleting it turns a priced call into an `UNAVAILABLE` one,
which is a silent loss of cost data.

Duplicate ids are rejected by a test: a repeated key silently discards one of
the two declared prices, and that failure is invisible until someone audits an
invoice.

## Using your own prices

Negotiated rates, or a model the catalogue does not know yet:

```python
from decimal import Decimal
from llm_gateway.models import builtin_price_catalog

catalog = builtin_price_catalog(
    overrides={"gemini-3.5-flash-lite": (Decimal("0.20"), Decimal("1.80"))},
    version="my-negotiated-rates-2026-07",  # required
)
```

The version is mandatory when overriding. Without it an amount would be
attributed to a catalogue version that did not produce it, which defeats the
whole point of recording the version.

For something more dynamic — prices from a database, per-customer rates —
implement the `PriceCatalog` protocol yourself and pass it to `LLMGateway`. It
needs a `version` property and an `estimate(model, usage)` method.

## Audio duration pricing

Audio and speech-to-text cost accounting is deliberately separate from the
token-based `PriceCatalog`. `LLMGateway.transcribe()` returns `AudioUsage` and
`AudioCost`, and records through `AudioUsageSink` when one is supplied.

The built-in catalogue currently includes:

| Model family | Rate | Minimum |
|---|---:|---:|
| OpenAI `gpt-transcribe` | `$0.0045` / minute | none |
| Groq Whisper | `$0.02`–`$0.111` / hour | 10 seconds |
| AssemblyAI Universal | `$0.15`–`$0.21` / hour | none |

Missing duration is `UNAVAILABLE`, never zero. Audio retries and fallbacks are
represented by `AudioAttempt`/`AudioExecution`; their cost never enters token
usage or token fallback pricing.
