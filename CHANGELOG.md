# Changelog

All notable changes to this package are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[SemVer](https://semver.org/).

While the version is `0.x` the public API may still break between minors. Each
consumer pins an immutable tag and upgrades through its own pull request.

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

- `requires-python = ">=3.11"`, set by the oldest consumer in the fleet.
- `pydantic>=2.10,<3`; every target consumer is already on Pydantic v2.
- The test suite makes no network calls and costs nothing to run.
