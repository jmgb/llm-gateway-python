# Migrating a consumer

The rule that makes this safe: **do not migrate call sites and implementation in
the same change.** Keep the public name, replace what is behind it.

## 1. Keep the existing entry point

A consumer's `gpt_request` (or equivalent) keeps its name, its signature and its
return shape. It stops *performing* the call and starts *translating* it. The
16+ callers of a large consumer should not need to change at all.

```python
async def gpt_request(ai_model, system_prompt, user_message, ..., tenant_id=None):
    """Same signature as always. Now a facade."""
    result = await _gateway.generate(LLMRequest(
        model=ai_model,
        system_prompt=system_prompt,
        messages=_build_messages(user_message, ...),
        temperature=temperature,
        request_id=current_request_id(),
        source=source,
    ))

    _record_in_ledger(tenant_id, result.usage, result.cost)   # your effect
    if result.execution.fallback_used:
        _alert(...)                                            # your effect

    return _flatten_as_before(result)   # the keys your callers expect
```

## 2. Wire your effects through ports

| Application concern | Port |
|---|---|
| Usage ledger per company/user/org | `UsageSink` |
| Telegram / Sentry / business alerts | `AlertSink` |
| Structured logging, metrics | `EventSink` |
| Model prices | `PriceCatalog` |

Sinks receive metadata only. If you need to log prompts, do it in your
application, where your redaction policy lives.

## 3. Build the gateway once, at startup

Credentials belong to the application. Construct clients from your own settings
and hand them in:

```python
gateway = LLMGateway(
    registry=build_registry(
        openai_client=create_openai_client(api_key=settings.OPENAI_API_KEY),
        gemini_client=create_gemini_client(api_key=settings.GEMINI_API_KEY),
    ),
    price_catalog=MyPriceCatalog(),  # yours, versioned, reconcilable
    usage_sink=MyLedger(),
)
```

Do not construct it at import time, and do not read keys inside the gateway.

## 4. Prove parity before deleting anything

Before removing the old implementation, confirm on the same inputs:

- same model actually used;
- same token counts;
- same cost, or a documented explanation of the difference;
- same output;
- same fallback behaviour;
- ledger entries and tenant attribution unchanged.

Delete the local implementation **only** for the paths you have verified.

## 5. Migrate callers later, per feature

Once the facade is stable, individual features can move from the flattened dict
to the typed result:

```python
respuesta["tokens_in"]  # legacy
result.usage.input_tokens  # typed — and None means "not reported"
```

That migration is optional and incremental. It is not part of adopting the
package.

## Rollback

Each consumer pins an immutable tag, so rolling back is a lock change. During
`0.x`, keep the legacy implementation reachable behind a flag for at least one
release: pinning protects you from a bad upgrade, not from a design you have
not exercised in production yet.

## Upgrading from 0.6 to 0.7

Two changes need a look before the pin moves.

### Unusable output no longer raises its own error

A response that could not be parsed, or that violated the schema, used to reach
the caller as `OutputParsingError` or `SchemaValidationError`. It is now a
failed billable attempt: the fallback gets a turn, and a call that never
produces a usable answer raises `AllAttemptsFailed` like any other exhausted
call, carrying every attempt.

```python
except SchemaValidationError:      # 0.6 — no longer reached
except AllAttemptsFailed as error: # 0.7
    error.__cause__                # the OutputParsingError / SchemaValidationError
    error.attempts                 # what it cost, including the rejected answer
```

Catching `OutputError` still works for the attempt-level type through
`__cause__`, but not as the exception the call raises. A consumer that catches
`(AllAttemptsFailed, OutputError)` together already keeps working.

Expect the totals to move: the tokens that produced a rejected answer are now
inside `result.usage` and `result.cost`, where before they were absent. That is
the correction, not a regression — the provider had already invoiced them.

### A local override of the same behaviour must be deleted, not adapted

Some consumers worked around the 0.6 behaviour by subclassing `LLMGateway` and
reimplementing `_run`, importing `_interpret`, `_aggregate`,
`_request_for_model` or `_attempt_model` from `llm_gateway.gateway`.

Those are private, and 0.7 changed all of them: `_attempt_model` now returns a
completion or the failure that ended the model's turn, never `None`, because
validation moved inside it. **Delete the override.** Adapting it means running
the fallback plan twice — once in the subclass and once in the gateway — which
double-counts nothing but attempts every model twice as often as intended.

If you need behaviour the package does not offer, the supported seams are the
ports and the policies. An underscore-prefixed symbol is not one, and nothing
in `0.x` promises it will survive a minor.

## Upgrading to 0.14.1

Nothing to change. `0.14.1` only adds fields; every call site, port
implementation and stored payload from before keeps working unmodified.

Before this release, the `llm_fallback_used` alert told you *that* the gateway
degraded from one model to another, not *why*. An operator reading "degraded
from A to B" had to go reconstruct the reason from provider logs a deploy may
already have rotated away. The alert now carries `error_type`,
`error_message` and `failure_phase` for the failure that ended the requested
model's turn, plus a `failures` list with every attempt the call made. The
full payload shape is documented under "Reading a failure" in the README;
this section only covers what changes for an existing consumer.

If your `AlertSink.alert` implementation already forwards `fields` wholesale —
to Sentry, to a Telegram message, to a log line — the new keys arrive with no
code change. If it destructures specific keys instead, add the ones you want;
the old two (`requested_model`, `model_used`) are untouched.

The gap that matters is upstream of any of this: if your gateway was built
without an `AlertSink` (the default is a no-op sink that drops the alert),
none of this reaches you regardless of version. Registering one is the actual
migration step; everything else follows from it for free.

```python
class SentryFallbackAlerts:
    """Minimal AlertSink that reports why a fallback fired."""

    def alert(self, message: str, fields: dict[str, object]) -> None:
        if message != "llm_fallback_used":
            return
        sentry_sdk.capture_message(
            f"Fallback {fields['requested_model']} -> {fields['model_used']}: "
            f"{fields.get('error_type')}: {fields.get('error_message')}",
            level="warning",
        )
```

Reading the cause from typed code, instead of the alert payload, uses the same
three properties the README describes:

```python
result = await gateway.generate(request)
if result.execution.fallback_used:
    cause = result.execution.fallback_cause  # last failed attempt on the
    # requested model — not simply the last failure. With three models in
    # the plan, the last failure can belong to the second model, and
    # blaming it for why the first was abandoned would be wrong.
    log.warning(
        "fallback",
        cause=cause.error_message if cause else None,
        all_failures=[a.error_message for a in result.execution.failures],
    )
```

```python
try:
    result = await gateway.generate(request)
except AllAttemptsFailed as error:
    log.error("no model answered", reason=error.last_error_message)
```

## Notes for specific shapes

- **A second layer over the facade** (e.g. a runner module) must migrate too;
  migrating the service alone leaves the real callers uncovered.
- **A facade inside a very large module** should be extracted to its own module
  before its callers are touched, so the diff stays reviewable.
- **Comparison experiments** must pin the same model on both sides and fail
  loudly if the two diverge, or the comparison silently stops being one.

## Upgrading to 0.15.0

**Breaking for callers that name `gemini-3.7-flash` or
`google/gemini-3.7-flash`.** Both entries are renamed to `gemini-3.8-flash` and
`google/gemini-3.8-flash`. No `3.7` entry is left behind, so `lookup_model` on
the old id returns `None` — and a `None` here is quiet: the request still goes
out, but without the catalogue there is no `reasoning_efforts` to send and no
rate to price the usage with. Grep for the old ids in code, environment
variables and stored configuration before upgrading.

Rates are unchanged (`0.75` input / `3.75` output USD per MTok). What moved is
the identity that carries them, and `CATALOG_VERSION` moves with it to
`2026-09-03.1`, so an amount recorded under `2026-08-15.1` still points at the
table that produced it.
