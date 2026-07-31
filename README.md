# neutral-llm-gateway

[![CI](https://github.com/jmgb/llm-gateway-python/actions/workflows/ci.yml/badge.svg)](https://github.com/jmgb/llm-gateway-python/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neutral-llm-gateway)](https://pypi.org/project/neutral-llm-gateway/)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://github.com/jmgb/llm-gateway-python/blob/main/pyproject.toml)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue)](https://github.com/jmgb/llm-gateway-python/blob/main/pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A small, honest gateway for LLM calls: typed contracts, thin provider adapters,
and accounting of retries, fallbacks, tokens and cost that refuses to lie to
you.

It knows about **providers**, never about products. It holds no credentials,
reads no environment variables, ships no prompts and stores no business
schemas. Everything product-shaped — ledgers, tenants, alerting, history,
prompts — stays in your application, wired in through optional ports.

```
your application → your facade → llm_gateway → provider SDK
```

Never the reverse.

## Why this exists

It was extracted from several applications that had each grown their own
version of the same "call an LLM" function — the largest close to two thousand
lines, mixing provider calls, retry policy, cost maths, a usage ledger and
business alerting in one place. Writing that from scratch a fourth time is how
subtle accounting bugs get copied around.

The bugs it is built to prevent are all the same shape: **a number that looks
like a fact but isn't.**

- A provider returns no usage, the code records `0` tokens, and the call bills
  as free.
- A model is missing from the price table, so its cost is `USD 0.00` —
  indistinguishable from a genuinely free call.
- A retry fails *after* the model produced tokens, and only the successful
  attempt gets counted.
- A fallback quietly answers with a different model, and the metrics attribute
  it to the one you asked for.

Here, unreported usage is `None`, unknown cost is `UNAVAILABLE`, every billable
attempt is counted, and a fallback is never silent.

### Is this for you?

**Probably yes** if you want a thin, auditable layer you can read in an
afternoon, you want to own your credentials, you care about per-call cost being
reconcilable against an invoice, and you prefer typed results to dictionaries.

**Probably not** if you want routing across dozens of providers, a proxy
server, streaming, or an agent framework.
[LiteLLM](https://github.com/BerriAI/litellm) and friends cover far more
surface than this does. This covers deliberately less, and is explicit about
what it does not know.

## Install

Provider SDKs are **optional extras**. Install only what you call.

```bash
# uv
uv add "neutral-llm-gateway[gemini]"

# pip
pip install "neutral-llm-gateway[gemini]"
```

Available extras: `openai`, `gemini`, `groq`, `openrouter`, `all`. Combine them
as `[openai,gemini]`. `openrouter` installs the `openai` SDK, so `[all]` already
covers it.

Importing the package with no extra installed works by design; asking for a
provider you have not installed raises a typed error naming the exact extra.

To try an unreleased commit, the PEP 508 git form works everywhere and pins a
tag or revision:

```bash
pip install "neutral-llm-gateway[gemini] @ git+https://github.com/jmgb/llm-gateway-python.git@v0.5.0"
```

## Use

```python
from pydantic import BaseModel

from llm_gateway import (
    FallbackPolicy,
    LLMGateway,
    LLMRequest,
    Message,
    ResponseFormat,
    RetryPolicy,
)
from llm_gateway.factories import build_registry, create_gemini_client


class Answer(BaseModel):
    verdict: str


# You build the client, so you keep the key. Prices come from the built-in
# versioned catalogue unless you pass your own.
gateway = LLMGateway(
    registry=build_registry(gemini_client=create_gemini_client(api_key=my_key)),
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

result.output  # Answer instance — no metadata mixed in
result.usage.input_tokens  # None means "not reported", not zero
result.cost.amount_usd  # None when unavailable, never a fake 0
result.cost.measurement  # ACTUAL | ESTIMATED | UNAVAILABLE
result.execution.model_used  # what actually answered
result.execution.fallback_used  # a fallback is never silent
result.execution.attempts  # every attempt, including the failed ones
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
| Importing needs no provider extra | Each application installs only the SDKs it calls |

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
| OpenAI | `[openai]` | Responses API |
| Google Gemini | `[gemini]` | `google-genai` async surface, not the retired `google-generativeai` |
| Groq | `[groq]` | Chat Completions. Declares no schema enforcement; the gateway validates after |
| OpenRouter | `[openrouter]` | Chat Completions. Aggregator: declares the floor every route honours, not the best case |

Capabilities are declared per provider and never faked as identical — query
`adapter.capabilities` before relying on one.

Reasoning effort is checked per model before each API attempt. OpenAI 5.6
models support `none`, `low`, `medium`, `high`, `xhigh`, and `max`; Gemini 3
Flash supports `minimal`, `low`, `medium`, and `high`; Gemini 3 Pro and Groq
GPT-OSS support `low`, `medium`, and `high`. If a fallback cannot honour the
requested effort, the gateway uses `medium` when available and otherwise omits
the reasoning option.

`[openrouter]` installs the `openai` SDK, because OpenRouter speaks the OpenAI
wire format and ships none of its own. That is a fact about the transport: the
adapter, the declared capabilities and the prices are OpenRouter's.

### Routing to OpenRouter

Models reach it by their namespace, so nothing needs configuring:

```python
from llm_gateway.factories import (
    build_registry,
    create_openai_client,
    create_openrouter_client,
)

registry = build_registry(
    openai_client=create_openai_client(api_key=...),
    openrouter_client=create_openrouter_client(api_key=...),
)

registry.resolve("gpt-5.6-luna")  # openai
registry.resolve("deepseek/deepseek-chat-v3.1")  # openrouter, from the catalogue
registry.resolve("somevendor/brand-new")  # openrouter, by the namespace rule
registry.resolve("openai/gpt-oss-120b")  # groq — the prefix is not OpenAI
```

`gemini-3-pro-preview` and `google/gemini-3-pro-preview` are the same model on
two routes, catalogued separately because they are billed separately.

For any *other* OpenAI-compatible endpoint — Azure, vLLM, your own gateway —
pass `base_url` to `create_openai_client` and widen the routing with
`build_registry(extra_openai_prefixes=...)`.

## Model catalogue and prices

The package ships a versioned table of models — provider, and price in USD per
million tokens — used by default, so a call is priced without you wiring
anything up. Override it for negotiated rates, or implement `PriceCatalog`
yourself. See [`docs/pricing.md`](docs/pricing.md).

## Not in this version

Tools/function calling, file attachments, streaming and Gemini File Search are
deliberately absent. Each is a real capability with its own cost and failure
model, and adding one badly is worse than not having it. See `CHANGELOG.md` for
the reasoning on each.

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the split was decided
- [`docs/pricing.md`](docs/pricing.md) — cost model and updating prices
- [`docs/migration.md`](docs/migration.md) — adopting it behind an existing function
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — what belongs here, and the non-negotiables

## Releasing without GitHub Actions

The local release runner keeps versioning and publication independent from
GitHub Actions minutes. Preview a release first:

```bash
uv run --offline python scripts/release.py --version 0.6.0 --dry-run
```

Prepare the release locally, including tests, the version in `pyproject.toml`
and `uv.lock`, the changelog, a release commit, and an annotated tag:

```bash
uv run --offline python scripts/release.py --version 0.6.0
```

Add `--push` to push `main` and the tag. Add `--publish` as well to publish
the matching wheel and sdist with `uv publish` and create the GitHub Release;
the latter requires GitHub CLI authentication and `UV_PUBLISH_TOKEN`.
`--publish` implies a real external release and therefore requires `--push`.

The local runner is the normal publisher when Actions minutes are unavailable.
The GitHub workflow is manual only; use one publisher per version to avoid
uploading the same PyPI files twice.

## Development

```bash
uv sync
uv run pytest        # no network, no cost, no extras required
uv run ruff check .
uv run mypy
uv build
```

Python 3.11+, Pydantic v2.

## Status

`0.x`: in production use, but the API may still change between minor versions.
Pin an exact version. Every release documents its changes, and cost-affecting
changes are called out explicitly.

## License

MIT.
