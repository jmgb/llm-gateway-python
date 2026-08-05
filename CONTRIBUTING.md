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

If you need something only you need, the ports (`UsageSink`, `AudioUsageSink`,
`ImageUsageSink`, `EventSink`, `AlertSink`, `PriceCatalog`, `AudioPriceCatalog`,
`ImagePriceCatalog`) are there so you don't have to fork.

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

### The one exception: `-m live`

`tests/live/` calls real providers and spends real money. It is deselected by
default, so `uv run pytest` stays offline and free; run it deliberately, with
whichever keys you have:

```bash
GROQ_API_KEY=... OPENROUTER_API_KEY=... uv run pytest -m live
```

Image and video generation have their own live suite, and it is the more
expensive one — a five-second 480p clip costs USD 0.20, so it is run
deliberately and one file at a time:

```bash
uv sync --extra gemini --extra wavespeed
GEMINI_API_KEY=... WAVESPEED_API_KEY=... \
  uv run pytest -m live tests/live/test_media_live.py -q -s
```

It generates an image and then animates that exact image with a second
provider, which is the only way to prove the chain an application actually
runs. `LLM_GATEWAY_LIVE_MEDIA_DIR` decides where both land; the image test
writes the frame the video test reads, so running only the second one skips.

It exists because a fake client has one blind spot that has already cost money:
it accepts any payload, so it cannot reject a request the way a provider does,
and it cannot invent field names the way a model does. Groq's HTTP 400 for a
`json_object` request that never says "json" is not reproducible against a
fake, and neither is a model answering valid JSON under keys nobody asked for.

Keep it that way — a narrow suite for failures only a provider can produce, not
a second home for logic the fakes already cover. A provider whose key is absent
skips rather than fails.

## Tests first

Write the failing test before the implementation, and watch it fail for the
reason you expect. A test that passes the moment you write it has not been
shown to test anything.

If you are fixing a bug, the pull request should contain a test that fails
without the fix.

## Adding a provider

1. A new module in `src/llm_gateway/providers/`, taking an **injected client**.
2. Declare its real `ProviderCapabilities` — do not claim parity it lacks, and
   do not claim what `LLMRequest` cannot ask for. A capability no caller can
   reach reads as available and answers nothing; a contract test enforces it.
3. Map errors through `classify_provider_error`; do not import the SDK to catch
   its exception types.
4. Add the SDK as an **optional extra** in `pyproject.toml`.
5. Tests with a fake client, covering usage mapping and the "usage not
   reported" case.
6. Export whatever the adapter adds to the public API from
   `src/llm_gateway/__init__.py`'s `__all__`. A symbol reachable only through
   its module is not part of the API consumers may rely on.
7. Add the provider's row to the reasoning-token table in `docs/pricing.md`,
   stating where it reports thinking and what your adapter does about it. That
   table is the only place the agreement between adapters is written down; a
   test fails if a provider is missing from it.

A transcription provider follows the same boundary: injected client, typed
errors and honest capabilities. Its duration rate belongs in the audio table,
and unsupported fields such as prompts or speaker labels must raise instead of
being silently discarded.

An image provider follows it too, with one extra decision: mark its models
`modality="image"` so `generate()` refuses them, and state how they are billed.
`pricing_unit="images"` carries a per-image rate; a provider that bills the
picture as tokens keeps `pricing_unit="tokens"` and is priced from
`ImageUsage.tokens`. Where no rate is published, leave it out — an
`UNAVAILABLE` cost is the honest answer and a guessed one is not. The form of
the source image is the provider's: reject the one it cannot use rather than
downloading or hosting it, which is application work.

An OpenAI-compatible API is **not** on its own a reason to skip all this. It
decides the transport, not the provider. Ask instead:

- Does it speak the Responses API, or Chat Completions? The OpenAI adapter
  translates to the former only.
- Can it honestly declare the OpenAI adapter's capabilities? An aggregator
  cannot promise on behalf of every model it routes to.
- Does it need its own prices, or its own name in the catalogue?

Answer no to any of those and it is a provider, however familiar the wire
format. OpenRouter is the worked example: it reuses the `openai` SDK as a
transport and is a separate adapter in every other respect. What *is* just a
`base_url` on the existing adapter is the same API somewhere else — an Azure
deployment, vLLM, your own gateway — widened with
`build_registry(extra_openai_prefixes=...)`.

## Adding or updating a model price

Edit `src/llm_gateway/models.py`, bump `CATALOG_VERSION`, and note it in the
changelog. `src/llm_gateway/catalogs.py` turns that table into the token, audio
and image price catalogues; a new pricing unit is a change there too. Prices are declared in USD per million tokens. Never delete a model
that consumers may still call — mark it `deprecated=True`.

The version is not optional bookkeeping: it travels with every recorded amount
and is what allows an old figure to be recomputed. `TestPricesAndVersionMoveTogether`
in `tests/test_model_catalog.py` pins a fingerprint of the priced table, so a
rate that moves without a new version fails the suite. Repin both constants in
the same commit that changes the price.

## Commit and PR

Explain *why*, not just what. If a change alters cost accounting or retry
behaviour in any way, say so explicitly in the description: those are the parts
people depend on being boring and predictable.
