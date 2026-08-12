# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[SemVer](https://semver.org/).

While the version is `0.x` the public API may still break between minors. Each
consumer pins an immutable tag and upgrades through its own pull request.

## [Unreleased]

### Fixed

- `json_object` no longer depends on the caller's prompt happening to say
  "json" when the provider is OpenAI. The Responses API answers HTTP 400 —
  *"Response input messages must contain the word 'json' in some form"* — and
  the OpenAI adapter was the only one that did not settle that debt, though it
  already carried the system prompt as a message so the check could see it.
  Groq and OpenRouter have called `system_prompt_for` since it existed; OpenAI
  now does too.

  The failure was expensive precisely because it did not look like one. Every
  such call was rejected, the fallback answered instead, and the caller got a
  correct result from the second model at the second model's price plus the
  latency of the rejected attempt. Nothing in the result said so.

### Changed

- `system_prompt_for` takes a keyword-only `structured_outputs` flag. An
  adapter that binds the shape through an API field passes `True` and no longer
  gets the schema repeated in the conversation; the `json_object` sentence is
  still added, because that precondition is about the word being present, not
  about enforcement. Default is `False`, so Groq and OpenRouter are unchanged.

## [0.12.0] — 2026-08-06

### Added

- Provider-neutral function tool calling for OpenAI and Groq through
  `FunctionTool`, `ToolChoice`, `RequiredTool`, `ToolCall`, `ToolResult` and
  `LLMResult.tool_calls`. Calls are translated and correlated but never
  executed, and malformed or undeclared calls remain billed failures.
- Submitted video jobs through `LLMGateway.submit_video()` and
  `LLMGateway.poll_video()`, including Replicate text/image-to-video support,
  webhook registration and catalogue entries for Wan 2.2, Kling v3 and
  Seedance 2.0.
- `LLMRequest.verbosity` for supported OpenAI GPT-5 models and typed
  `LLMRequest.routing` preferences for OpenRouter.

### Changed

- An unstated video resolution now selects the cheapest verified tier for the
  model instead of accepting a more expensive provider default. Unsupported
  resolutions are rejected rather than silently ignored.
- `CATALOG_VERSION` is now `2026-08-06.1` after adding the Replicate video
  models to the modality and price fingerprints.

### Fixed

- **Replicate reports the length of the clip it produced, and the new job path
  discarded it.** A finished prediction carries `metrics.video_output_duration_seconds`,
  measured on the video that was generated, plus `model_variant` — the tier
  that actually ran. The adapter now reads both, so `VideoUsage.seconds` is
  real usage rather than `None` and `VideoUsage.resolution` comes back in the
  package's own spelling. Anyone supplying a `VideoPriceCatalog` gets a
  computed amount where the answer used to be `UNAVAILABLE`. A prediction that
  reports no metrics still leaves the length unknown, never zero.
- **A video job's cost remains attributable.** `VideoJob` gained `request_id`
  and `source`, copied from the `VideoRequest` at submission, and the terminal
  poll now records them. Previously both reached the usage sink as `None`, so
  the one operation billed minutes later from another process was also the one
  whose spend could not be reconciled with the call that caused it. Stamped by
  the gateway rather than by each adapter, so no provider can omit it.
- **`poll_video()` is bounded.** It takes a `timeout_seconds` (default `30.0`)
  and raises `ProviderTimeoutError` when the status call exceeds it. It had no
  budget at all — a poll takes no `VideoRequest` and so inherited no
  `TimeoutPolicy` — which let a provider that stopped answering block the
  worker that polled it indefinitely. The bound is on the status call, never on
  the job, which is expected to run for minutes.
- `submit_video()` now enforces the total `TimeoutPolicy` budget across provider
  attempts and retry delays, not only each per-attempt timeout. Interrupted
  submissions are retained as potentially billable attempts because a remote
  provider may have accepted the job before the local timeout fired.
- Polling rejects non-positive and non-finite timeout values before dispatch,
  terminal job events retain the original `request_id` and `source`, and
  submission lifecycle events now use their job-specific name and report the
  successful attempt in their attempt count.
- Replicate treats non-finite duration metrics as unknown instead of allowing
  them to escape as invalid usage.
- OpenAI and Groq no longer invent tool-call correlation ids. A provider reply
  without its real id is a billed output-parsing failure, because a synthetic
  id cannot be used for a valid continuation. Manually constructed tool calls
  also reject blank ids and function names.
- OpenRouter provider-routing preferences now travel through the OpenAI SDK's
  `extra_body` passthrough instead of being supplied as an unsupported SDK
  keyword argument.

## [0.11.0] — 2026-08-06

### Added

- Provider-neutral **image generation and editing**: `ImageRequest`,
  `ImageInput`, `GeneratedImage`, `ImageResult` and `LLMGateway.generate_image()`,
  with the same retry, fallback and attempt accounting as the token and audio
  paths. Images are billed through a separate `ImagePriceCatalog`, `ImageUsage`
  and `ImageCost`, recorded through `ImageUsageSink`.
- Image adapters for **Gemini** (inline bytes, billed as tokens), **Replicate**
  (`[replicate]` extra, answers with a URL, edits from a URL) and **WaveSpeed**
  (`[wavespeed]` extra, REST submit/poll, text-to-image only). Capabilities are
  declared per adapter: `image_generation` and `image_editing`.
- Provider-neutral **video generation**: `VideoRequest`, `GeneratedVideo`,
  `VideoResult` and `LLMGateway.generate_video()`, with per-second usage and
  cost (`VideoUsage`, `VideoCost`, `VideoPriceCatalog`, `VideoUsageSink`). The
  rate depends on the resolution, and a resolution the table does not know
  reports `UNAVAILABLE` rather than the cheaper one. The WaveSpeed adapter
  implements it for `wavespeed-ai/minimax-h3/image-to-video` (USD 0.04 per
  second at 480p, 0.08 at 768p), accepting the first frame as a URL or as
  bytes it sends as a data URI — which is what lets an image generated by one
  provider be animated by another.
- Catalogue entries for `black-forest-labs/flux-kontext-pro` (USD 0.04 per
  image), `wavespeed-ai/hidream-i1-dev` (USD 0.012 per image), `prunaai/p-image` and
  `bytedance/seedream-4` (no published per-image rate, so their cost reports
  `UNAVAILABLE` rather than zero), plus a `modality` field on `ModelInfo`.
- Live tests for real image and video generation (`tests/live/test_media_live.py`),
  deselected by default like every other live test.

### Changed

- `CATALOG_VERSION` is now `2026-08-05.2`. The catalogue gained image models,
  and Gemini image output now uses its published image-token rates (`$30`,
  `$60` or `$120` per million) instead of the cheaper text-output rates. Both
  per-image and image-output-token rates are covered by the price fingerprint.
- The price-catalogue builders moved to `llm_gateway.catalogs`
  (`builtin_price_catalog`, `builtin_audio_price_catalog`,
  `builtin_image_price_catalog`). They are re-exported from `llm_gateway`, so
  only imports that reached into `llm_gateway.models` for them need updating.

### Fixed

- The per-module size budget is now 2000 lines, matching the repository's
  general rule instead of the stricter 500-line one.
- `generate()` refuses an image or video model instead of returning an empty
  success.
  Gemini's image reply carries inline image parts and no text, so the previous
  behaviour silently dropped the generated picture.
- `FallbackPolicy.cheaper_than()` and `builtin_price_catalog()` no longer treat
  a token-billed image model as a cheaper text model.
- WaveSpeed polling now uses the documented `/api/v3/predictions/...` endpoint,
  stops on every terminal task state, and rejects non-success API body codes.
- The current `gemini-3-pro-image` is no longer marked deprecated; the retired
  `gemini-3.1-flash-image-preview` is.
- Generated media, usage and resolution-price contracts reject empty, negative,
  non-finite or mutable values that could otherwise corrupt cost accounting.

## [0.10.1] — 2026-08-04

### Fixed

- CI now keeps each matrix job on its requested Python version and verifies the
  interpreter before running checks. The AssemblyAI factory test also supplies
  its own fake optional transport, so the no-extras job is deterministic.

## [0.10.0] — 2026-08-04

### Changed

- Updated the model catalogue and marked the requested legacy Gemini, DeepSeek,
  Groq Llama and GPT-5.1/5.2 identifiers as deprecated. They remain routable so
  existing consumers do not break during an upgrade.
- Added `gpt-realtime-2.1` and `gpt-realtime-2.1-mini`, and marked the older
  realtime identifiers as deprecated.
- Marked the obsolete OpenRouter entries for Gemini 3 preview and Kimi K2 as
  deprecated instead of deleting them.
- Removed provider-family and namespace guesses from fallback routing. Unknown
  models now require an explicit catalogue entry or consumer-supplied prefix.

### Added

- Added OpenAI's `gpt-transcribe` identity with its `$0.0045` per audio-minute
  rate. It is routed as an OpenAI model but remains outside token-based cost
  estimation.
- Added the independent transcription contract: `TranscriptionRequest`,
  `AudioInput`, `AudioUsage`, `AudioCost` and `LLMGateway.transcribe()`.
  OpenAI, Groq Whisper and AssemblyAI adapters support provider-specific audio
  transports, explicit fallback and duration-based accounting.
- Added OpenAI remote-file attachments for Responses analysis. Unsupported
  providers reject attachments instead of silently dropping them.
- Added `assemblyai` as an optional extra and catalogued AssemblyAI Universal,
  Groq Whisper and OpenAI transcription models outside token pricing.
- Added seven current OpenRouter entries: Claude Sonnet 4.6, Grok 4.5, the
  Sonnet and Opus latest aliases, the DeepSeek V4 Flash and MoonshotAI Kimi
  latest aliases, and Qwen3.8 Max.

### Fixed

- OpenAI `gpt-transcribe` now sends its language hint through the required
  plural `languages` field and reads provider duration from `usage.seconds`.
  Caller-supplied duration remains usable as an explicit estimate, never an
  actual provider measurement.
- AssemblyAI now uses the published `universal-3-pro` model identifier, sends
  its supported prompt, and maps HTTP, polling and malformed-response failures
  to typed gateway errors so retry and fallback remain available.
- OpenAI and Groq reject unsupported speaker labels instead of silently
  dropping them, and uploaded audio preserves its MIME type.
- A total timeout now records the provider attempt it interrupted, keeping a
  potentially billable call visible in failure accounting.
- Optional-extra tests now simulate missing SDKs explicitly and pass whether
  or not another development dependency installed the same transport.

## [0.9.0] — 2026-07-31

### Fixed

- **A requested schema now reaches providers that cannot enforce one.** The
  Groq and OpenRouter adapters declare `structured_outputs=False` and asked for
  `{"type": "json_object"}`, which discarded the caller's `response_schema`
  entirely. The model was told to answer JSON and never told which JSON, so it
  answered valid JSON under keys of its own invention, the gateway correctly
  rejected it, and the attempt was billed. Every structured call was therefore
  served by the fallback, at the fallback's price, plus the discarded attempt.

  Nothing about this looked like a failure — the caller received a correct
  result and a fallback notice — so it was visible on the invoice and nowhere
  else. Two applications hit it against Groq's `openai/gpt-oss-120b`.

  Both adapters now append what the requested format needs to the system
  prompt: the JSON Schema for `JSON_SCHEMA`, and one sentence asking for JSON
  for `JSON_OBJECT`.

  This changes cost accounting for anyone calling those providers with a
  schema: the fallback stops running on every call, so the model that answers,
  the amount billed and the attempt count all change — to what they should have
  been.

- **`ResponseFormat.JSON_OBJECT` no longer fails outright on Groq.** Groq
  rejects `json_object` with HTTP 400 unless the word "json" appears in the
  messages, and OpenAI documents the same rule. The adapter set that mode
  without guaranteeing the word, so a plain JSON request was a 400 whenever the
  caller's prompt did not happen to mention it — for a mode the caller asked
  for through the neutral contract, not through a provider quirk they could
  have known about.

  Whoever sets `response_format` now satisfies its precondition. A caller whose
  system prompt already says "json" is left untouched, so nothing is appended
  to a prompt that does not need it.

### Added

- **A `live` test suite that calls real providers.** Both fixes above are for
  failures a fake client cannot produce: it accepts any payload, so it never
  answers HTTP 400, and it returns queued text, so it never invents field
  names. `tests/live/` covers exactly that gap and is deselected by default —
  `uv run pytest` stays offline and free; `uv run pytest -m live` with the
  provider keys spends real tokens. A provider whose key is absent skips.

## [0.8.0] — 2026-07-31

### Fixed

- Release artifact selection now ignores signatures, partial downloads, files
  for similar version numbers, and stale distributions from earlier releases.
  The local uploader and the standalone audit now share the same selection;
  GitHub runs that audit immediately before its publishing action.
- Local publishing reads only publishing credentials from `.env`, rather than
  exposing unrelated values to the publishing subprocesses.
- `SchemaValidationError` now reports each Pydantic violation's location and
  type without copying dynamic keys or other response content into the error.
- `FallbackPolicy.cheaper_than()` now returns the cheapest candidates first,
  matching its documented contract.
- `Execution.model_used` now honours the model id reported by the provider.
  `fallback_used` continues to mean that the gateway used its fallback plan,
  so a provider alias or an `openrouter/auto` resolution does not trigger it.

### Changed

- Static type checking now includes the release and artifact-audit scripts.
- `ProviderRegistry.register()` rejects a second adapter with the same provider
  name instead of leaving name and prefix routing pointed at different clients.

### Removed

- Removed the unused `ModelInfo.aliases` field. It was never populated or read.

## [0.7.0] — 2026-07-31

### Fixed

- **The sdist declares what it ships.** With no `[tool.hatch.build.targets.sdist]`
  section, hatchling packages the whole project directory minus whatever the
  VCS ignored *at build time*, so files that were never part of the
  distribution — CI workflows, a local environment file — travelled to PyPI in
  the 0.6.0 sdist. Packaging is now an explicit allowlist, and the wheel is
  unaffected.

  The local release runner audits every built artifact before uploading it and
  refuses to publish one containing an unexpected dotfile or a credential-shaped
  name. An upload cannot be undone: the file is mirrored within minutes, so the
  only moment this can be stopped is between the build and the upload.

- **Output validation happens inside the attempt loop.** An answer that cannot
  be parsed as JSON, or that does not satisfy the requested schema, was
  previously validated *after* the attempt had already been accepted as the
  result. Two things followed, both invisible from the result: the fallback
  never ran, and the tokens that produced the rejected answer were left out of
  `usage`, `cost`, `execution.attempts` and every sink.

  Such an answer is now a failed **billable** attempt, recorded with the exact
  usage and cost the provider reported, and the next model in the plan is
  tried. The same model is not retried: the same prompt and the same model
  reproduce the same malformed answer, so a retry would buy a second invoice
  for one failure. `RetryPolicy` continues to govern provider failures.

- **OpenAI structured output is normalised to the strict subset.** Pydantic's
  schema was sent to the Responses API unchanged with `strict: true`, which
  rejects it: a field with a default is omitted from `required`, and no object
  declares `additionalProperties: false`. `providers/strict_schema.py` now
  rewrites the schema recursively — root, `$defs`, nested objects and union
  branches — removing nullable defaults and expanding a `$ref` only when
  sibling metadata has to be preserved. A construct strict mode cannot
  express, such as a free-form `dict[str, str]`, raises `ConfigurationError`
  naming the field instead of buying a provider 400 on every attempt.

- **Request options are adapted to the target model, not only reasoning
  effort.** A fallback onto a model of the OpenAI 5.6 family inherited the
  caller's `temperature`, which that API rejects outright — turning the
  fallback into a second, guaranteed failure. `ModelInfo.supports_temperature`
  declares the refusal and the option is dropped before the attempt.

### Changed

- **A call that never produces a usable answer raises `AllAttemptsFailed`**,
  carrying every attempt, with the `OutputParsingError` or
  `SchemaValidationError` as its `__cause__`. It previously raised the output
  error directly, which lost the attempts — and therefore the money spent.

- `function_calling`, `inline_files` and `remote_files` are declared `False` on
  every adapter. The providers support them; `LLMRequest` has no field that
  asks for any of them, so the declaration promised a capability no caller
  could exercise. `tests/contract/test_capability_honesty.py` is written
  against `LLMRequest`, so the declarations may return the day the contract
  grows the fields.

### Added

- `Attempt.failure_phase`, a typed `FailurePhase` — `configuration`,
  `provider`, `timeout`, `output_parsing` or `schema_validation`. `error_type`
  alone forced a consumer to rebuild the context from a class name to tell
  "the request never left the process" from "the provider answered, was paid,
  and the answer was unusable".

- `tests/contract/test_structured_fallback.py`: every adapter is driven through
  the gateway with a fake client and held to the same behaviour — invalid JSON,
  a schema violation, a successful fallback, total exhaustion, and the exact
  sum of tokens and cost across attempts.

## [0.6.0] — 2026-07-31

### Changed

- Updated `gpt-5.6-luna` pricing to USD 0.20 per input MTok and USD 1.20 per
  output MTok.
- Updated `gpt-5.6-terra` pricing to USD 2.00 per input MTok and USD 12.00 per
  output MTok; recorded its 128K-token maximum output.
- Made reasoning efforts model-specific: OpenAI 5.6 keeps `none`, `low`,
  `medium`, `high`, `xhigh`, and `max`; Gemini 3 Flash supports `minimal`,
  `low`, `medium`, and `high`; Gemini 3 Pro supports `low`, `medium`, and
  `high`; and Groq GPT-OSS supports `low`, `medium`, and `high`.
- Reasoning is adapted before every attempt. An unsupported effort becomes
  `medium` when the target model supports it; otherwise the reasoning option is
  omitted so a fallback cannot receive an invalid provider parameter.
- Removed legacy Gemini model entries from the direct and OpenRouter catalogues.

## [0.5.0] — 2026-07-30

### Added

- **OpenRouter is a provider, not a `base_url`.** `OpenRouterAdapter`,
  `create_openrouter_client`, an `[openrouter]` extra and an
  `openrouter_client` argument to `build_registry`.

  It was previously described as the OpenAI adapter pointed elsewhere, which
  was wrong in three ways. OpenRouter's stable surface is Chat Completions, not
  the Responses API. It is an aggregator, so it cannot promise the OpenAI
  adapter's capabilities: it declares the floor every route honours
  (`structured_outputs=False`, `json_mode=True`) and the gateway validates the
  payload afterwards, as it already did for Groq. And the model that answers is
  not always the model requested — `openrouter/auto` chooses one — so the
  reply's own model id is reported as `model_used`.

  The adapter maps `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens`, so cached input and reasoning
  are accounted for on this route too.

### Fixed

- **Reasoning tokens are no longer billed twice.** `billable_output_tokens`
  added `reasoning_tokens` on top of `output_tokens`, but the Responses API
  already counts reasoning inside `output_tokens` — input plus output
  reconciles to the reported total. With a thinking effort where reasoning
  dominates the visible answer, the estimate came out roughly 1.5–2× the real
  amount, which is exactly the kind of figure that stops reconciling against a
  provider invoice.

  `output_tokens` now means the same thing in every adapter — everything
  billed at the output rate — and `reasoning_tokens` is a breakdown of it,
  never an addition to it. Normalisation happens at the boundary, so the
  Gemini adapter folds `thoughts_token_count` into `output_tokens`, since
  `candidates_token_count` genuinely excludes it. `visible_output_tokens`
  reports the part the caller actually received.

  Cost figures produced by earlier versions for reasoning models are
  overstated and cannot be corrected in place: recompute them from the stored
  token counts.

  A contract test now holds every adapter to this: each one is handed its own
  provider's native shape for the same call and must arrive at the same three
  numbers. A new adapter fails it until it declares where its provider counts
  thinking.

- **The Groq adapter reports the reasoning breakdown.** It read only
  `prompt_tokens` and `completion_tokens`, so a thinking model looked like it
  had returned every token it was billed for. No amount changes — Chat
  Completions already counts reasoning inside `completion_tokens` — but
  `visible_output_tokens` was wrong, which is the number you compare against
  what the model actually said.

- **The OpenAI adapter sends the system prompt as a message.** It travelled as
  `instructions`, which is not part of the input — and `json_object` mode is
  rejected unless the word "json" appears in the input. A system prompt asking
  for JSON was invisible to that check, so a call whose user content did not
  happen to say "json" failed. The prompt is now the first input message, which
  is also the arrangement Chat Completions used.

- **The catalogue's OpenRouter models are reachable.** Sixteen models declared
  `provider="openrouter"`, but no adapter could register under that name, so
  `registry.resolve("deepseek/deepseek-chat-v3.1")` raised `UnknownModelError`
  unless the caller passed `build_registry(extra_openai_prefixes=...)` — an
  argument documented nowhere. Registering `openrouter_client` is now enough,
  and the namespace rule means uncatalogued `vendor/model` ids route too.

- **Two OpenAI-shaped clients no longer collide.** Registering one for OpenAI
  and one for OpenRouter left a single entry under the name `openai`, because
  both adapters reported that name. The second silently won: `gpt-5.6-luna`
  resolved to the OpenRouter client, sending OpenAI traffic through an
  aggregator with no error and no log line. The names are now distinct.

  `extra_openai_prefixes` remains, for what it is actually good at: widening
  the OpenAI adapter to an Azure deployment name or a self-hosted id.

## [0.4.1] — 2026-07-30

### Fixed

- **The wheel now ships a `py.typed` marker (PEP 561).** The code was always
  fully typed and checked under `mypy --strict`, but without the marker a
  consumer's type checker silently treated every import from `llm_gateway` as
  `Any` — annotations that look like a guarantee but aren't is exactly what
  this package is against. Types are now part of the public contract.

### Added

- Package metadata: classifiers (including `Typing :: Typed`), keywords and
  repository URLs in `pyproject.toml`.

## [0.4.0] — 2026-07-30

### Fixed

- **`TimeoutPolicy.total_seconds` is now a total.** It was only ever applied
  per attempt, so a 200s "total" with two retries allowed roughly 400s plus
  backoff. The whole call — every attempt, every retry, every backoff pause —
  is now bounded by it, and exceeding it raises `AllAttemptsFailed` carrying
  the attempts already made. `per_attempt_seconds_override` still bounds each
  individual try.

  This is a behaviour change: calls that previously ran past their declared
  budget will now be cut off at it. That is the point.

  Found in code review. A value that names itself a total and is not one is
  exactly the class of bug this package exists to prevent.

## [0.3.1] — 2026-07-30

### Fixed

- The "provider extra not installed" error no longer assumes pip. Telling a
  uv-managed project to run `pip install` would install outside its lockfile,
  so the hint now offers `uv add` and `pip install` and names the extra itself.

### Added

- Install instructions for uv, including the `[tool.uv.sources]` form and the
  portable PEP 508 alternative for projects that do not use uv.

## [0.3.0] — 2026-07-30

Prepared for public release.

### Changed

- **Renamed the distribution** to `neutral-llm-gateway`. The import stays
  `llm_gateway`. Update your dependency name; nothing else changes.

### Added

- `CONTRIBUTING.md` (what belongs here, the two-consumer rule, the
  non-negotiables), `SECURITY.md` (private reporting, and the properties a
  report can be measured against) and `docs/pricing.md`.
- A README that explains what this is for, and when you should use something
  else instead.

## [0.2.0] — 2026-07-30

The shared model catalogue. Prices now live here, so they are updated once
instead of once per consumer.

### Added

- `llm_gateway.models`: **46 models** with provider and price, merged from the
  two catalogues this package was extracted from. `CATALOG_VERSION` identifies it,
  and every recorded amount carries it.
- `builtin_price_catalog()` — used by `LLMGateway` **by default**, so a consumer
  that says nothing about prices gets real, versioned ones instead of
  `UNAVAILABLE`. An explicit catalogue still wins.
- `builtin_price_catalog(overrides=..., version=...)` for negotiated rates.
  Supplying overrides without a version is rejected: an amount must never be
  attributed to a catalogue version that did not produce it.
- `resolve_provider()` and catalogue-first routing in `ProviderRegistry`. Model
  ids can lie about their provider — `openai/gpt-oss-120b` is served by Groq —
  so the declared provider wins over any prefix rule.
- `FallbackPolicy.cheaper_than(model, limit=...)` — derives a same-provider,
  cheapest-first chain from the catalogue, skipping deprecated models. Whether
  degrading is acceptable remains the caller's decision.
- Test forbidding duplicate model ids. A duplicate key silently discards one of
  the two declared prices, which is a defect this extraction actually found.

### Notes

- Prices are declared in USD per million tokens and consumed as microUSD per
  token. Those are the same number; a test asserts the conversion is the
  identity.
- Audio/STT pricing (billed per hour, with a provider minimum) is **not**
  included: it is a different cost model, no gateway capability produces it,
  and only one consumer has it today. It moves here when that consumer migrates.

## [0.1.0] — 2026-07-30

First extraction. Nothing is migrated yet: this release exists so a first
consumer can be integrated behind a facade and compared for parity.

### Added

- Typed contracts: `LLMRequest`, `LLMResult`, `TokenUsage`, `Cost`,
  `Execution`, `Attempt`, `Message`, `ResponseFormat`.
- `LLMGateway.generate()` — orchestration of attempts, retries and fallback,
  with per-attempt usage and cost recorded whether the attempt succeeded or not.
- Policies as explicit values: `RetryPolicy`, `FallbackPolicy` (disabled by
  default), `TimeoutPolicy`.
- Typed error hierarchy rooted at `LLMGatewayError`, with transient/permanent
  classification, and `AllAttemptsFailed` carrying the attempts already billed.
- Provider adapters: OpenAI (Responses API, also serving OpenRouter via
  `base_url`), Google Gemini (`google-genai`), Groq (Chat Completions).
- Structural SDK error mapping that requires no SDK import, so an error can be
  classified without the corresponding extra installed.
- Cost accounting in whole microdollars with `ACTUAL` / `ESTIMATED` /
  `UNAVAILABLE`, injected through the `PriceCatalog` port.
- Optional no-op ports: `UsageSink`, `EventSink`, `AlertSink`.
- Declared, non-uniform `ProviderCapabilities` per provider.
- JSON payload recovery from fenced or prose-padded replies — location only,
  never semantic repair.
- Provider SDKs as optional extras: `[openai]`, `[gemini]`, `[groq]`, `[all]`.

### Deliberately not included

- **Tools / function calling.** Needs a bounded execution model and a story for
  tool-call accounting; no first consumer requires it.
- **File attachments.** Requires MIME, size and path validation before anything
  is sent, and capability differences between providers are large.
- **Gemini File Search / Interactions.** A separate capability with its own cost
  model: retrieved documents are billed as input context, and folding it into a
  plain `generate` call would hide that.
- **Streaming.** No first consumer needs it, and it changes the shape of both
  usage reporting and error handling.
- **A built-in price catalogue.** Prices age, and the authority on them is the
  application that reconciles the invoice.

### Notes

- `requires-python = ">=3.11"`, chosen to support still-common runtimes.
- `pydantic>=2.10,<3`; every target consumer is already on Pydantic v2.
- The test suite makes no network calls and costs nothing to run.
