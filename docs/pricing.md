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

The catalogue also supports `pricing_unit="audio_minutes"` for speech models,
`pricing_unit="images"` for image models and `pricing_unit="video_seconds"`
for video models. Those entries remain routable and carry their provider's own
rate, but `builtin_price_catalog()` excludes them: its `TokenUsage` contract
can represent neither a minute, a picture nor a second of footage. Use
`builtin_audio_price_catalog()` with `AudioUsage`,
`builtin_image_price_catalog()` with `ImageUsage` and
`builtin_video_price_catalog()` with `VideoUsage`, for those paths.

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
| Replicate | no token usage; images are billed per run | kept out of token pricing |
| WaveSpeed | no token usage; images are billed per image, video per second | kept out of token pricing |

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
from llm_gateway.catalogs import builtin_price_catalog

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

A provider-reported duration produces an `ACTUAL` amount. When the provider
omits duration but the caller supplied `AudioInput.duration_seconds`, the
amount is an `ESTIMATED` lower-confidence figure. If neither source has a
duration, cost is `UNAVAILABLE`, never zero. Audio retries and fallbacks are
represented by `AudioAttempt`/`AudioExecution`; their cost never enters token
usage or token fallback pricing.

## Image pricing

Image generation is billed in two incompatible ways, and the catalogue records
which one each model uses rather than picking a single fiction:

| Model | Provider | Unit | Rate |
|---|---|---|---:|
| `gemini-3.1-flash-image` | Gemini | tokens | `$0.50` input / `$60` image output per 1M tokens |
| `gemini-3.1-flash-lite-image` | Gemini | tokens | `$0.25` input / `$30` image output per 1M tokens |
| `gemini-3-pro-image` | Gemini | tokens | `$2` input / `$120` image output per 1M tokens |
| `black-forest-labs/flux-kontext-pro` | Replicate | per image | `$0.04` |
| `wavespeed-ai/hidream-i1-dev` | WaveSpeed | per image | `$0.012` |
| `prunaai/p-image`, `bytedance/seedream-4` | Replicate | per image | **not published** |

`StaticImagePriceCatalog` reads the model's unit, never the operation's: a
token-billed model is priced from `ImageUsage.tokens`, a per-image model from
`ImageUsage.images`. A model with no published rate is left out of the built-in
table, so its cost is `UNAVAILABLE` — Replicate bills community models by GPU
second, and a fixed per-image number for them would be a guess presented as a
fact. Applications that know their own figures inject an `ImagePriceCatalog`.

Gemini's image-output token rate is deliberately separate from its text-output
rate. Treating the two as one would understate an image invoice by up to 20×.
WaveSpeed charges `$0.012` per HiDream I1 Dev run at the documented default
shape.

Image retries and fallbacks are represented by `ImageAttempt`/`ImageExecution`,
recorded through `ImageUsageSink`, and never enter token usage or token
fallback pricing.

## Video pricing

Video is billed per second, and the rate depends on the resolution:

| Model | Provider | 480p | 768p |
|---|---|---:|---:|
| `wavespeed-ai/minimax-h3/image-to-video` | WaveSpeed | `$0.04` / second | `$0.08` / second |
| `wan-video/wan-2.2-5b-fast` | Replicate | *no published rate* | *no published rate* |
| `kwaivgi/kling-v3-video` | Replicate | *no verified rate* | *no verified rate* |
| `bytedance/seedance-2.0` | Replicate | *no verified rate* | *no verified rate* |

`StaticVideoPriceCatalog` prices `VideoUsage.seconds` at the rate for
`VideoUsage.resolution`. A resolution the table does not know yields
`UNAVAILABLE` rather than the cheaper rate: assuming 480p on a 768p clip would
halve a real invoice.

### The default resolution is always the model's cheapest tier

Resolution is the single biggest lever on a video bill — on MiniMax H3 it is
the difference between `$0.04` and `$0.08` per second, and Kling's tiers span
720p to 4K. So `VideoRequest.resolution` left unset does **not** fall through
to the provider's own default. The adapter sends the lowest tier its model
offers, and anything above it has to be named:

| Model | Provider's default | What this package sends |
|---|---|---|
| `wavespeed-ai/minimax-h3/image-to-video` | unset | `480p` |
| `wan-video/wan-2.2-5b-fast` | `720p` | `480p` |
| `kwaivgi/kling-v3-video` | `pro` (1080p) | `standard` (720p) |
| `bytedance/seedance-2.0` | `720p` | `480p` |

Every one of those providers defaults to something dearer than its floor, so
the request that says nothing is exactly the one that would quietly cost the
most.

It also keeps the amount computable. An adapter that sent no resolution would
get one back that it could not report, and `VideoUsage.resolution` of `None`
prices at `UNAVAILABLE` — so the silent request would be both the dearest and
the only unpriced one.

This holds in tests too, live ones included: the live suite animates its frame
at 480p on MiniMax H3 and 720p on Kling, the floor of each.

A model whose tiers this package has not read from a published schema gets no
default. Guessing a floor produces a rejected request, which is worse than
letting the provider choose.

WaveSpeed reports no clip length of its own and snaps output to the model's
frame grid, so its adapter reports the requested duration as an estimate. The
amount is therefore `ESTIMATED`, never `ACTUAL` — an honest lower-confidence
figure rather than a measurement nobody took.

Replicate bills Wan by GPU time, which no per-second table predicts and which
the prediction does not report. The catalogue therefore carries the model with
no rate at all and its cost is `UNAVAILABLE` — the same answer the image
catalogue gives for `prunaai/p-image`, and for the same reason: an invented
number is worse than an admitted gap. Applications with a measured
cost-per-clip supply a `VideoPriceCatalog` of their own.

Video retries and fallbacks are represented by `VideoAttempt`/`VideoExecution`
and recorded through `VideoUsageSink`.

### When a job is billed

`submit_video()` records nothing: at that point the clip does not exist, so any
amount would be invented. Neither does a poll that finds the job still running
— a job polled ten times is billed once. The poll that finds a terminal state
is the one that prices what was produced and writes the usage record.
