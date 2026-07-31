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
| Retry, fallback, attempt accounting | `gateway.py` | Same logic for every provider; divergence is the bug |
| Token and cost arithmetic | `usage.py`, `pricing.py` | Same maths everywhere; prices are injected |
| JSON recovery | `json_payload.py` | A provider-shaped problem |
| Deciding whether an answer is usable | `gateway.py` | It decides whether the attempt failed, so it cannot sit after the attempt |
| Adapting a request to the target model | `gateway.py`, `models.py` | A fallback inherits a request written for another model |
| Describing a schema a provider cannot enforce | `providers/schema_prompt.py` | Which providers enforce schemas changes with the provider |
| Ledger, tenant, alerting, history | The application | Changes with the product |
| Prompts, business schemas, model choice per feature | The application | Changes with the product |

Those last two rows sit closer together than they look. An adapter declaring
`structured_outputs=False` cannot bind the answer to a shape through any API
field, so it states the caller's schema in the messages — text this package
sends that the application did not write.

That is not a prompt, and the distinction is what keeps the boundary intact: it
carries no tone, no task framing and no examples, only the JSON Schema the
caller already declared and a sentence saying it is binding. The *shape* is the
caller's; **whether it has to be spelled out is a fact about the provider**, and
a fact about the provider belongs here. The alternative is not neutrality —
dropping the schema makes every structured call fail validation and fall back,
which is a louder product decision than describing it.

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
```

Adapters are deliberately dumb. They do not retry, do not fall back, do not
price and do not aggregate. Every one of those, done once per provider, is how
the original file reached two thousand lines.

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
fails the build if any module exceeds 500 lines.
