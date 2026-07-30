# Contributing

Thanks for looking. This is a small, opinionated package, so it helps to know
what it is trying to be before proposing a change.

## What belongs here

One question decides everything:

> Would this change if you switched **provider**, or if you switched **product**?

Provider-shaped code belongs here. Product-shaped code belongs in your
application: prompts, business schemas, ledgers, tenant identity, alerting,
history loading, and which model a given feature should use.

## The two-consumer rule

**Nothing enters the public API until two distinct applications need it.**

One application's requirement is that application's adapter. A second real case
is what makes a good general design possible — and what stops this package from
slowly becoming the thousand-line function it was extracted from.

If you need something only you need, the ports (`UsageSink`, `EventSink`,
`AlertSink`, `PriceCatalog`) are there so you don't have to fork.

## Tooling

This project uses [uv](https://docs.astral.sh/uv/). Every command below assumes
it; `uv sync` creates the environment and installs from `uv.lock`, so there is
no virtualenv to activate by hand.

## Non-negotiables

Changes that break these will be declined, however convenient:

- **Absence is never zero.** Unreported usage is `None`; unknown cost is
  `UNAVAILABLE`. Neither may be quietly rendered as `0`.
- **Every billable attempt is counted**, including failed ones. A retry that
  failed may still appear on the invoice.
- **Fallback is opt-in and always visible** in the result.
- **Exhausted calls raise.** They never return something that reads as success.
- **No module reads the environment or constructs a credentialed client.**
  Applications own their credentials.
- **Sinks never receive prompt or response content.**
- **No provider SDK is imported at module import time.** Extras are optional;
  `import llm_gateway` must work with none of them installed.

`tests/contract/test_package_boundaries.py` enforces several of these.

## Development

```bash
uv sync
uv run pytest        # no network, no cost, no extras required
uv run ruff check .
uv run ruff format .
uv run mypy
uv build
```

Tests must not make network calls. If you cannot test something without a
network, that is usually a sign the seam is in the wrong place — reach for an
injected client, as the existing adapters do.

## Tests first

Write the failing test before the implementation, and watch it fail for the
reason you expect. A test that passes the moment you write it has not been
shown to test anything.

If you are fixing a bug, the pull request should contain a test that fails
without the fix.

## Adding a provider

1. A new module in `src/llm_gateway/providers/`, taking an **injected client**.
2. Declare its real `ProviderCapabilities` — do not claim parity it lacks.
3. Map errors through `classify_provider_error`; do not import the SDK to catch
   its exception types.
4. Add the SDK as an **optional extra** in `pyproject.toml`.
5. Tests with a fake client, covering usage mapping and the "usage not
   reported" case.

Before adding one, check whether the provider exposes an OpenAI-compatible API.
If it does, it is a `base_url` on the existing adapter, not a new provider.

## Adding or updating a model price

Edit `src/llm_gateway/models.py`, bump `CATALOG_VERSION`, and note it in the
changelog. Prices are declared in USD per million tokens. Never delete a model
that consumers may still call — mark it `deprecated=True`.

## Commit and PR

Explain *why*, not just what. If a change alters cost accounting or retry
behaviour in any way, say so explicitly in the description: those are the parts
people depend on being boring and predictable.
