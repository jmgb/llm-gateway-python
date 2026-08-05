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

Available extras: `openai`, `gemini`, `groq`, `assemblyai`, `openrouter`,
`replicate`, `wavespeed`, `all`. Combine them as `[openai,assemblyai]`.
`openrouter` installs the `openai` SDK, and `assemblyai` and `wavespeed`
install the small HTTP transport used by their REST adapters.

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
result.execution.model_used  # provider-reported model that actually answered
result.execution.fallback_used  # whether the gateway used its fallback plan
result.execution.attempts  # every attempt, including the failed ones
```

If the model answers with something that is not valid JSON, or with JSON that
violates `Answer`, that attempt is recorded as failed **and billed** — the
tokens were spent — and the next model in `fallback_policy` is tried. When no
model produces a usable answer the call raises `AllAttemptsFailed`, carrying
every attempt, with the parsing or schema error as its `__cause__`.
Schema validation errors name each Pydantic `loc` and `type`, while dynamic
response keys and values stay out of the message.

### Tool calling

The gateway declares your functions, hands you the calls the model made, and
puts your results back on the wire. It never runs a function: authorisation,
side effects and business schemas are yours, and so is the loop.

```python
from dataclasses import replace

from llm_gateway import FunctionTool, LLMRequest, Message, ToolResult

request = LLMRequest(
    model="gpt-5.6",
    messages=(Message("user", "What is the weather in Madrid?"),),
    tools=(
        FunctionTool(
            name="get_weather",
            description="Current weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    ),
)
result = await gateway.generate(request)

if result.tool_calls:
    call = result.tool_calls[0]
    output = await my_app.run(call.name, call.arguments)  # yours, not the package's
    result = await gateway.generate(replace(request, tool_results=(ToolResult(call, output),)))

result.output  # the answer, once the model stopped asking for functions
```

`tool_choice` takes `ToolChoice.AUTO`, `NONE`, `REQUIRED` or
`RequiredTool("get_weather")` to force one. A `ToolResult` holds the `ToolCall`
it answers rather than a loose id, because both halves go back on the wire and
a pair that drifted apart answers the wrong question.

Tool calls never arrive as `output`: when the model calls a function `output`
is `None` and `tool_calls` is non-empty, and a requested JSON format is not
parsed out of a reply that contains no answer. Arguments the caller could not
dispatch — unparseable, not a JSON object, or naming a function the request
never declared — make that attempt a **billed failure**, so the fallback still
gets its turn and the money is still counted. The package checks what makes a
call dispatchable, not the whole JSON Schema: that needs a validator this
package does not depend on, and the code that owns the function validates
anyway. Definitions, arguments and results never reach a usage, event or alert
sink.

### Transcription

Speech-to-text is a separate operation with duration usage and audio pricing:

```python
from llm_gateway import AudioInput, LLMGateway, TranscriptionRequest
from llm_gateway.factories import build_registry, create_openai_client

gateway = LLMGateway(
    registry=build_registry(openai_client=create_openai_client(api_key=my_key)),
)
transcript = await gateway.transcribe(
    TranscriptionRequest(
        model="gpt-transcribe",
        audio=AudioInput(
            data=audio_bytes,
            filename="voice.webm",
            mime_type="audio/webm",
            duration_seconds=12.5,
        ),
        language="es",
        source="voice-note",
    )
)

transcript.text
transcript.usage.duration_seconds
transcript.cost.amount_usd  # audio minutes, never token pricing
```

`language` is optional and defaults to provider detection. Use
`assemblyai-universal-3-pro` or `assemblyai-universal-2` with a public
`AudioInput.url`, and `whisper-large-v3-turbo`/`whisper-large-v3` with Groq.
AssemblyAI Universal-3 Pro supports `prompt` and speaker labels; OpenAI and
Groq reject speaker labels rather than ignoring them. Fallbacks are explicit
through `FallbackPolicy.models_in_order(...)`.

### Image generation

Images are a separate operation too, for the same reason: the output is bytes
or a URL, and providers bill it either per image or with image-output tokens
whose rate can differ from text output.

```python
from llm_gateway import ImageInput, ImageRequest, LLMGateway
from llm_gateway.factories import build_registry, create_gemini_client

gateway = LLMGateway(
    registry=build_registry(gemini_client=create_gemini_client(api_key=my_key)),
)
result = await gateway.generate_image(
    ImageRequest(
        model="gemini-3.1-flash-image",
        prompt="a cat wearing a hat, studio lighting",
        image=ImageInput(data=original_bytes, mime_type="image/jpeg"),  # optional: an edit
        source="whatsapp-image",
    )
)

result.images[0].data  # bytes from Gemini; `.url` from Replicate and WaveSpeed
result.usage.images
result.cost.amount_usd  # per image, or per token where the model bills that way
```

Which form comes back is the provider's, not a choice: Gemini returns inline
bytes, Replicate and WaveSpeed return a URL. Downloading it, re-hosting it,
watermarking it and enforcing a per-user quota stay in the application, exactly
as decoding audio does.

Editing needs a source image in the form the provider accepts — bytes for
Gemini, a URL for Replicate — and an adapter that cannot use the form it was
given raises rather than dropping it. WaveSpeed's text-to-image models refuse
an edit outright. Sending an image model through `generate()` raises too: its
reply carries no text, and returning it as an empty success is the failure this
separation exists to prevent.

### Video generation

Video is a third seam, billed by the second:

```python
from llm_gateway import ImageInput, LLMGateway, VideoRequest
from llm_gateway.factories import build_registry, create_wavespeed_client

gateway = LLMGateway(
    registry=build_registry(wavespeed_client=create_wavespeed_client(api_key=my_key)),
)
result = await gateway.generate_video(
    VideoRequest(
        model="wavespeed-ai/minimax-h3/image-to-video",
        prompt="the lioness sprints and leaps at the gazelle",
        image=ImageInput(data=first_frame_bytes, mime_type="image/png"),
        resolution="480p",
        duration_seconds=5,
        source="wildlife-clip",
    )
)

result.videos[0].url
result.usage.seconds, result.usage.resolution
result.cost.amount_usd  # per second, at the rate for that resolution
```

The first frame may be bytes or a URL — WaveSpeed accepts an inline data URI,
which is what lets an image from one provider be animated by another without
hosting it anywhere first.

A clip takes minutes, so `VideoRequest` defaults to a fifteen-minute total
budget and the adapter owns the provider's polling loop. Resolution is part of
usage rather than a detail of the request because the rate depends on it: a
resolution the price table does not know costs `UNAVAILABLE`, never the
cheaper rate.

**Omit `resolution` and you get the cheapest tier the model offers**, never the
provider's own default — which is dearer on every catalogued video model. It is
the biggest single lever on a video bill (480p vs 768p is 2× on MiniMax H3), so
silence buys the floor and anything above it is asked for by name. See
[the table in `docs/pricing.md`](docs/pricing.md).

### Video jobs, for providers that answer minutes later

Replicate does not hand back a clip; it hands back a prediction id and finishes
later. Waiting inside one `await` would mean a four-minute request timeout and
no way to use the webhook, so the same `VideoRequest` takes a second route:

```python
from llm_gateway import LLMGateway, VideoJob, VideoJobStatus, VideoRequest
from llm_gateway.factories import build_registry, create_replicate_client

gateway = LLMGateway(registry=build_registry(replicate_client=create_replicate_client()))

job = await gateway.submit_video(
    VideoRequest(
        model="wan-video/wan-2.2-5b-fast",
        prompt="the lioness sprints and leaps at the gazelle",
        image=ImageInput(url="https://cdn.example/first-frame.png"),
        webhook_url="https://app.example/hooks/video",  # optional
    )
)

job.id, job.model, job.provider, job.status  # four strings: store them
```

`VideoJob` is plain data on purpose — the process that polls is usually not the
one that submitted. Save its fields, rebuild it later, and read it back from
anywhere:

```python
result = await gateway.poll_video(
    VideoJob(
        id=row.id,
        model=row.model,
        provider=row.provider,
        request_id=row.request_id,  # so the eventual cost stays attributable
    ),
    timeout_seconds=30.0,  # bounds the status call, not the job
)

if result.job.status is VideoJobStatus.SUCCEEDED:
    result.videos[0].url
    result.usage.seconds  # measured by the provider, where it reports one
    result.cost.amount_usd
elif result.job.status is VideoJobStatus.FAILED:
    result.error  # why the provider gave up, when it says
```

`request_id` and `source` are copied from the `VideoRequest` into the job, so
store them alongside the rest: the clip is billed minutes later from another
process, and they are what ties that amount back to the call that caused it.

A job the provider gave up on is a status, not an exception: the submission
worked, and an application storing the outcome wants the reason rather than a
traceback. `poll_video()` raises only when the *reading* failed — including
when the status call itself exceeds `timeout_seconds`, which stops a stalled
provider from blocking the worker that polled it.

Nothing is billed until the job is terminal. A submission records no usage —
the clip does not exist yet — and a job polled ten times is recorded once.

Three Replicate video models are catalogued, and the adapter translates each
one from its published input schema rather than a shared guess:

| Model | First frame | Duration | Resolution | Sent when unset |
|---|---|---|---|---|
| `wan-video/wan-2.2-5b-fast` | `image` | frames — `duration_seconds` refused | `480p`/`720p` | `480p` |
| `kwaivgi/kling-v3-video` | `start_image` | 3–15 s | `720p`/`1080p`/`4k`, sent as `mode` | `standard` (720p) |
| `bytedance/seedance-2.0` | `image` | 3–15 s | `480p`/`720p`/`1080p`/`4k` | `480p` |

That last column is the cheapest tier of each, and it is deliberately not what
Replicate would have picked — its own defaults are 720p, `pro` (1080p) and 720p
respectively.

A resolution a model does not offer raises rather than falling back to its
default: Replicate ignores keys it does not recognise, generates the default,
and bills for it, so a silent drop is a clip nobody asked for on the invoice.
None of the three has a per-second rate this package could verify, so their
cost is `UNAVAILABLE` — pass a `VideoPriceCatalog` with your measured rates.

## The guarantees

These are enforced by tests, not by convention:

| Guarantee | Why it matters |
|---|---|
| Unreported usage is `None`, not `0` | A zero token count silently under-bills |
| Unknown cost is `UNAVAILABLE`, not `USD 0` | "Free" and "unknown" are different facts |
| Cost aggregates **every billable attempt** | A retry that failed may still be invoiced |
| An unusable answer is a *billed, failed* attempt | Invalid JSON still cost money, and the fallback still gets a turn |
| Each attempt carries a typed `failure_phase` | `configuration`, `provider`, `timeout`, `output_parsing` or `schema_validation`, without parsing a message |
| Every attempt sends only options its model accepts | A fallback must not fail on a `temperature` the next model rejects |
| Fallback is off by default and always visible | A silent model switch corrupts A/B comparisons and cost attribution |
| Exhausted calls **raise** | They never return something that looks like a success |
| Errors carry the attempts already made | A failure still accounts for the money it spent |
| Output, usage, execution and cost are separate | A token count can never be mistaken for a business field |
| Sinks never receive prompts or responses | Observability without storing content |
| No module reads the environment | The application owns its credentials |
| Importing needs no provider extra | Each application installs only the SDKs it calls |

## Extending

Ports are optional and default to no-op: `UsageSink`, `EventSink`, `AlertSink`,
`AudioUsageSink`, `PriceCatalog` and `AudioPriceCatalog`. Implement what you
need; the package will not reach into your application to find them.

Adding to the public API follows the **two-consumer rule**: nothing is promoted
into the core until two distinct applications need it. Until then it belongs in
that application's local adapter.

## Providers

| Provider | Extra | Notes |
|---|---|---|
| OpenAI | `[openai]` | Responses API |
| Google Gemini | `[gemini]` | `google-genai` async surface, not the retired `google-generativeai` |
| Groq | `[groq]` | Chat Completions. Declares no schema enforcement; the schema is described in the messages and the gateway validates after |
| AssemblyAI | `[assemblyai]` | REST submit/poll transcription API |
| OpenRouter | `[openrouter]` | Chat Completions. Aggregator: declares the floor every route honours, not the best case |
| Replicate | `[replicate]` | Image generation and editing, plus video as a submitted job. Answers with a URL, and fetches the source image from one |
| WaveSpeed | `[wavespeed]` | REST submit/poll. Text-to-image, and image-to-video billed per second |

Capabilities are declared per provider and never faked as identical — query
`adapter.capabilities` before relying on one.

A provider that declares `structured_outputs=False` has no API field that binds
the answer to a shape, so its adapter states the requested schema in the system
prompt instead of dropping it. Otherwise the model answers valid JSON under keys
of its own choosing, validation rejects it, the attempt is billed anyway and the
fallback serves every structured call — a result that looks correct and shows up
only on the invoice.

The same adapters add one sentence asking for JSON when `JSON_OBJECT` is
requested, because Groq rejects that mode with HTTP 400 unless the word appears
in the messages. Setting `response_format` is what creates the obligation, so
the adapter meets it rather than the caller's prompt — and a prompt that already
says "json" is left as it is.

OpenAI and Groq declare `function_calling=True`, each in its own dialect: the
Responses API takes a flat tool and answers with `function_call` items carrying
a `call_id`, while Chat Completions nests the function and answers inside
`choices[0].message.tool_calls`. Groq sends tools *or* a JSON `response_format`
and never both, which costs nothing — it enforces no schema either way, the
system prompt still states the shape and the gateway still validates. Gemini
and OpenRouter declare `False` and reject a request carrying tools rather than
answering prose where a call was expected; their request, response and
continuation shapes get the capability once deterministic tests cover them.

`inline_files` remains unsupported. OpenAI declares
`remote_files=True`: `LLMRequest.attachments` accepts already-uploaded file IDs
and the Responses adapter appends them to the last user message as
`input_file` parts. Providers without that capability reject the request rather
than silently dropping the files. OpenAI, Groq and AssemblyAI expose
`audio_transcription=True` through the separate `TranscriptionRequest` API,
and Gemini, Replicate and WaveSpeed expose `image_generation=True` through
`ImageRequest`. WaveSpeed and Replicate both declare `video_generation=True`
and `video_from_image=True`, reached through `VideoRequest` — WaveSpeed by
awaiting `generate_video()`, Replicate through `submit_video()` and
`poll_video()`. Only Replicate declares `video_webhooks=True`, so an
application can ask whether it must poll before committing to a design.

Request options are adapted per model before each API attempt. A model that
rejects `temperature` — the OpenAI 5.6 family — never receives it, including
when it is reached through a fallback that inherited it from another model.

Reasoning effort is checked the same way. OpenAI 5.6
models support `none`, `low`, `medium`, `high`, `xhigh`, and `max`; Gemini 3
Flash supports `minimal`, `low`, `medium`, and `high`; Gemini 3 Pro and Groq
GPT-OSS support `low`, `medium`, and `high`. If a fallback cannot honour the
requested effort, the gateway uses `medium` when available and otherwise omits
the reasoning option.

`LLMRequest.verbosity` (`low`, `medium`, `high`) asks for a shorter or longer
answer, which is not what `max_output_tokens` does: that one truncates an answer
already being written and pays for every token up to the cut. OpenAI declares
`verbosity=True` and sends it for the `gpt-5` families; an adapter without the
capability has no field for it, so a fallback that inherits the option is
neither charged for it nor broken by it.

`LLMRequest.routing` states which upstream should serve the call. Only an
aggregator has that choice, so OpenRouter is the one adapter declaring
`upstream_routing=True`:

```python
from llm_gateway import LLMRequest, RoutingPreference

LLMRequest(
    model="google/gemini-3.6-flash",
    routing=RoutingPreference(order=("Groq", "SambaNova"), optimise_for="throughput"),
)
```

Stating nothing is the default, and it is not the same as stating an empty
preference: a blank instruction would override the aggregator's own routing, so
only the halves a caller filled in are sent.

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
registry.resolve("deepseek/deepseek-v4-pro")  # openrouter, from the catalogue
registry.resolve("somevendor/brand-new")  # raises: not in the catalogue
registry.resolve("openai/gpt-oss-120b")  # groq — the prefix is not OpenAI
```

`gemini-3.1-pro-preview` and `google/gemini-3.1-pro-preview` are the same model on
two routes, catalogued separately because they are billed separately.

For any *other* OpenAI-compatible endpoint — Azure, vLLM, your own gateway —
pass `base_url` to `create_openai_client` and widen the routing with
`build_registry(extra_openai_prefixes=...)`.

## Model catalogue and prices

The package ships a versioned table of models — provider, and price in USD per
million tokens — used by default, so a call is priced without you wiring
anything up. Audio models such as `gpt-transcribe`, Whisper and AssemblyAI are
catalogued with their duration unit and priced through a separate
`AudioPriceCatalog`; image models carry their per-image rate, or their token
rate where the provider bills images as tokens, and are priced through an
`ImagePriceCatalog`; video models carry a per-second rate that depends on the
resolution and are priced through a `VideoPriceCatalog`. Override prices for negotiated rates, or implement either
catalog protocol yourself. See
[`docs/pricing.md`](docs/pricing.md).

## Not in this version

Inline files, streaming and Gemini File Search remain absent, and so do
provider-hosted tools — web search, code execution, file search — and any
package-owned loop that would execute a function for you. Function tools,
remote file IDs for OpenAI, audio transcription, image generation and video
generation — both the polled shape and the submit/poll job shape — are
supported with their own capability and cost contracts.

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

Every built artifact is audited before it is uploaded, and the release is
refused if the archive contains an unexpected dotfile or a credential-shaped
name. What reaches a package index cannot be recalled — the file is mirrored
within minutes — so the check runs between the build and the upload, which is
the last moment it is still worth anything.

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
