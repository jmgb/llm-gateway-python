# Architecture

## The cut

The package exists because seven applications had grown seven versions of the
same function — the largest of them 1,947 lines, mixing provider calls, retry
policy, cost maths, a usage ledger and business alerting in one place.

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
| Ledger, tenant, alerting, history | The application | Changes with the product |
| Prompts, business schemas, model choice per feature | The application | Changes with the product |

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
