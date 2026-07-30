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

## Notes for specific shapes

- **A second layer over the facade** (e.g. a runner module) must migrate too;
  migrating the service alone leaves the real callers uncovered.
- **A facade inside a very large module** should be extracted to its own module
  before its callers are touched, so the diff stays reviewable.
- **Comparison experiments** must pin the same model on both sides and fail
  loudly if the two diverge, or the comparison silently stops being one.
