# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[SemVer](https://semver.org/).

While the version is `0.x` the public API may still break between minors. Each
consumer pins an immutable tag and upgrades through its own pull request.

## [Unreleased]

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
