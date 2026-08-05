# Architecture

## The cut

The package exists because several applications had each grown their own
version of the same function — the largest of them close to two thousand lines,
mixing provider calls, retry policy, cost maths, a usage ledger and business
alerting in one place.

The split is **not** a common denominator of those seven signatures. A union of
every parameter would produce a bigger monster; an intersection would produce
something too thin to use. The question asked of each line was instead:

> Would this change if I switched **provider**, or if I switched **product**?

Provider-shaped code moved here. Product-shaped code stayed in the application.

| Concern | Where it lives | Why |
|---|---|---|
| Calling an SDK, mapping its response | `providers/` | Changes with the provider |
| Retry, fallback, attempt accounting | `gateway.py`, `audio_gateway.py` | Same policy shape, separate token/audio accounting |
| Token and audio cost arithmetic | `usage.py`, `pricing.py`, `models.py` | Separate usage/cost types prevent duration becoming tokens |
| JSON recovery | `json_payload.py` | A provider-shaped problem |
| Deciding whether an answer is usable | `gateway.py` | It decides whether the attempt failed, so it cannot sit after the attempt |
| Adapting a request to the target model | `gateway.py`, `models.py` | A fallback inherits a request written for another model |
| Satisfying what a response format needs said | `providers/schema_prompt.py` | Which providers enforce schemas, or demand the word "json", changes with the provider |
| Declaring tools, correlating calls and results | `tools.py`, `providers/` | Two dialects for one idea; the wire shape changes with the provider |
| Running a function the model asked for | The application | Authorisation and side effects change with the product |
| Ledger, tenant, alerting, history | The application | Changes with the product |
| Prompts, business schemas, model choice per feature | The application | Changes with the product |

Those last two rows sit closer together than they look. An adapter declaring
`structured_outputs=False` cannot bind the answer to a shape through any API
field, and Groq will not even accept `json_object` unless the messages say
"json". So the adapter appends to the system prompt — text this package sends
that the application did not write.

That is not a prompt, and the distinction is what keeps the boundary intact: it
carries no tone, no task framing and no examples, only the JSON Schema the
caller already declared, or one sentence naming the format they asked for. The
*shape* is the caller's; **whether it has to be spelled out is a fact about the
provider**, and a fact about the provider belongs here. The alternative is not
neutrality — dropping the schema makes every structured call fall back, and
omitting the word makes every plain JSON call a 400. Both are louder product
decisions than saying the sentence.

## Layers

```
LLMRequest
    │
    ▼
LLMGateway.generate()          ← retries, fallback, attempts, aggregation
    │
    ├── ProviderRegistry       ← model id → adapter
    ├── PriceCatalog (port)    ← injected by the application
    ├── UsageSink   (port)     ← injected, no-op by default
    ├── EventSink   (port)
    └── AlertSink   (port)
    │
    ▼
ProviderAdapter.generate()     ← one call, one translation, no policy
    │
    ▼
provider SDK

TranscriptionRequest
    │
    ▼
LLMGateway.transcribe()        ← audio retries, fallback and duration cost
    │
    └── AudioProviderAdapter.transcribe()

ImageRequest
    │
    ▼
LLMGateway.generate_image()    ← image retries, fallback and per-image cost
    │
    └── ImageProviderAdapter.generate_image()

VideoRequest
    │
    ▼
LLMGateway.generate_video()    ← video retries, fallback and per-second cost
    │
    └── VideoProviderAdapter.generate_video()
```

Four operations, four request types, four accounting seams — and the catalogue
decides which one a model belongs to. An audio, image or video model sent
through `generate()` raises instead of degrading: a transcription priced as
tokens and an image reply read as text are both silent failures, and both were
cheaper to make impossible than to detect.

Video is synchronous from the caller's side because the current WaveSpeed
integration can be polled: the adapter owns the loop, as the transcription
adapters do, and
`VideoRequest` defaults to a fifteen-minute total budget. A provider that only
answers through a webhook would need a two-phase submit-and-poll contract
instead, and that is deliberately left until one is actually adopted.

Provider-reported audio duration is actual usage. Caller-supplied duration is
kept as an estimate when a provider omits usage, and a missing duration remains
unknown. Provider adapters must reject unsupported transcription options rather
than silently dropping them.

Adapters are deliberately dumb. They do not retry, do not fall back, do not
price and do not aggregate. Every one of those, done once per provider, is how
the original file reached two thousand lines.

### Why the tool loop stays in the application

Every consumer that calls functions today wraps the provider call in a loop:
run what the model asked for, send the output back, repeat until it answers.
The loop looks identical in all of them, and it is not — each one carries its
own iteration cap, its own permission checks, its own idea of what a function
may touch and its own record of what was executed on whose behalf. Owning that
here would mean the package deciding, on the application's behalf, that a side
effect was allowed to happen.

So the contract is one round trip. The gateway declares tools, hands back typed
`ToolCall`s and puts typed `ToolResult`s back on the wire in each provider's
dialect; the application runs the function and asks again. A package-owned loop
becomes a separate question once two applications demonstrate the *same*
execution semantics, which they have not.

Correlation is structural rather than conventional: `ToolResult` holds the
`ToolCall` it answers instead of a loose id, because both halves have to go
back on the wire together — an assistant turn replaying the call, then its
output — and a pair assembled by zipping two lists answers the wrong question
the first time a provider returns them out of order.

## Why the result is four objects

`LLMResult` keeps `output`, `usage`, `execution` and `cost` apart. Flattening
them into one dictionary — as the legacy implementations did — means a caller
reading `result["tokens_in"]` cannot tell whether that key came from the model
or from the plumbing, and it means a model that happens to emit a field called
`cost` silently overwrites the real one.

Legacy facades may flatten during migration. The package does not.

## Failure is accounted for

Two things follow from "a failed call may still be billed":

* every attempt that reached the provider is recorded, whatever its outcome;
* `AllAttemptsFailed` carries those attempts, so an exception is still
  auditable.

A retry that timed out after the provider had already produced tokens is not
free, and reporting `USD 0` for it would under-declare spend precisely in the
worst case.

## Unusable output is a failed attempt, not a returned one

An answer that cannot be parsed as JSON, or that does not satisfy the requested
schema, is decided **inside** the attempt loop. Validating after the loop —
which is what this package did until 0.7.0 — has two consequences that are hard
to see and expensive to have:

* the fallback never runs, because the attempt was already accepted as the
  result;
* the tokens that produced the rejected answer are outside the totals, so the
  invoice is larger than the accounting.

The policy is deliberately simple and stated here because nothing in the type
system says it:

| Failure | What happens |
|---|---|
| `OutputParsingError`, `SchemaValidationError` | The attempt is recorded as `FAILED`, **billable**, with the usage and cost the provider reported. The next model in the plan is tried |
| Either of them on the last model | `AllAttemptsFailed`, with the output error as `__cause__` |

The same model is **not** retried after unusable output: the same prompt and
the same model reproduce the same malformed answer, so the retry buys a second
invoice for one failure. `RetryPolicy` still governs provider failures.

A tool call is judged by the same rule, from the other direction. A model that
called a function *answered*, so the attempt succeeds even though there is no
text and no JSON to validate — parsing a reply that contains no answer would
turn a correct call into a billed parsing failure. What does fail is a call the
application could not dispatch: arguments that do not parse, arguments that are
not a JSON object, or a name the request never declared. Those are
`OutputParsingError` and follow the table above exactly, arguments never
repeated in the message.

`Attempt.failure_phase` names which of the five phases ended an attempt —
`configuration`, `provider`, `timeout`, `output_parsing`,
`schema_validation` — so a dashboard does not have to rebuild that from an
exception class name. A configuration failure is non-billable because the
request never reached the provider.

## Requests are adapted per model, per attempt

`_request_for_model` runs before every attempt and removes options the target
model does not accept: a reasoning effort it does not declare, a `temperature`
its API rejects outright. A fallback inherits a request that was written for a
different model, and a provider refuses the whole call over one unknown option
— which would turn the fallback into a second, guaranteed failure.

Nothing raises: the fallback stays visible in `execution`, and only the
offending option is dropped. What the model accepts is declared in the
catalogue (`ModelInfo.reasoning_efforts`, `ModelInfo.supports_temperature`),
never inferred from the id.

An option only one provider can honour needs no catalogue entry, because it is
already unreachable elsewhere: `verbosity` and `routing` are read by the
adapters that declare the matching capability and by no other, so an adapter
without the field cannot forward what it never looked at. Adapters do not pass
unknown request fields through — that is the property this relies on, and it is
also why the request contract has no free-form provider dictionary: a
passthrough would carry whatever a caller put in it straight to an API, past
every capability declaration in the package.

## Error classification without SDK imports

`providers/error_mapping.py` classifies by HTTP status first and class name
second. Writing `except openai.RateLimitError` would make error handling
require the extra to be installed, and would break whenever an SDK reorganises
its exception tree.

Provider messages are **not** copied into the typed error: they routinely echo
request payloads and occasionally credentials. The original is preserved as
`__cause__` for local debugging.

## Extension without inversion

Ports are protocols with no-op defaults. The package never imports an
application to resolve one — the application constructs the gateway and hands
in what it wants.

This is what keeps the dependency rule intact:

```
application → facade → llm_gateway → SDK
```

## Growth control

The failure mode this package could reproduce is its own origin: an API that
grows by accumulation until it is the old giant function again, only now shared
by seven teams.

The rule: **nothing enters the public API until two distinct consumers need
it.** One consumer's requirement is that consumer's adapter. A second real case
is what makes a good general design possible.

Enforced structurally by `tests/contract/test_package_boundaries.py`, which
fails the build if any module exceeds 2000 lines.
