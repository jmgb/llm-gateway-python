# Tasks

## Provider-neutral tool calling

- [x] **P1 — Add provider-neutral function tool calling as a v0.10 public contract.**

  Shipped as `FunctionTool`, `ToolChoice`, `RequiredTool`, `ToolCall` and
  `ToolResult`, reached through `LLMRequest.tools` / `tool_choice` /
  `tool_results` and answered through `LLMResult.tool_calls`, with OpenAI and
  Groq declaring the capability and Gemini and OpenRouter refusing it. See
  `[Unreleased]` in `CHANGELOG.md`, the tool sections of `README.md` and
  "Why the tool loop stays in the application" in `docs/architecture.md`.

  Two decisions worth carrying forward: a tool call is a *successful* attempt
  and skips output parsing entirely, while a call the application could not
  dispatch is a billed failure that lets the fallback run; and validation stops
  at what makes a call dispatchable — a declared name and a JSON object of
  arguments — rather than reaching for a JSON Schema validator this package
  does not depend on.

  ### Why this belongs in the package

  Versions 0.8.0 and 0.9.0 cannot represent tools: `LLMRequest` has no tool
  fields and every adapter honestly declares `function_calling=False`. Four
  independent applications now need the same provider-shaped behaviour, so the
  change satisfies the two-consumer rule:

  - Presupuestor uses tool calls together with its central `gpt_request`
    facade and provider fallback.
  - VirtualAssistant runs recursive tool continuations across OpenAI and Groq.
  - Sofia keeps tool requests on its provider-specific path while text/JSON
    requests use this package.
  - Apps has the same split between neutral text/JSON and a legacy dispatcher
    for tools, files and multimodal calls.

  The shared problem is transport and normalisation: declaring function tools,
  selecting one, receiving one or more calls, preserving provider call IDs and
  sending results back. Tool implementation, authorisation, execution,
  business schemas and side effects remain application concerns.

  ### First release scope

  Implement function tools only. Do not include provider-hosted tools such as
  web search, code execution or file search, and do not combine this task with
  attachments, multimodal input, streaming, remote files or agent runtimes.

  Add the smallest provider-neutral public types needed to express:

  - a function definition: stable name, optional description and JSON Schema
    parameters;
  - tool choice: `auto`, `none`, `required`, or one named function;
  - one or more returned tool calls with provider call ID, function name and
    validated JSON arguments;
  - a tool result correlated to its original call ID, suitable for a later
    continuation request.

  Extend `LLMRequest` and the response contract without flattening tool calls
  into text. Keep normal text/JSON/Pydantic output and technical metadata
  separate. Decide and document whether a tool-calling provider response is a
  successful gateway attempt and how malformed arguments or schema violations
  affect fallback; every provider request that may be billed must remain
  visible in `Execution.attempts`, usage and cost.

  The package must translate and normalise calls but must **not execute
  application functions** and must not own a recursive business loop. An
  application receives typed calls, performs its own permission checks and
  side effects, then supplies typed results in a subsequent request. Consider
  a package-owned orchestration loop only as a separate task after two
  applications demonstrate identical execution semantics.

  Start with the OpenAI and Groq adapters because they are the verified shared
  consumers. Set `function_calling=True` only after an adapter can exercise the
  complete neutral contract. Leave Gemini and OpenRouter false until their
  request, response and continuation shapes have deterministic adapter tests;
  wire-format similarity is not capability evidence.

  ### Non-negotiable behaviour

  - Requests without tools remain behaviourally unchanged.
  - Credentials and tool execution stay outside the package.
  - Tool definitions, arguments and results never reach usage, event or alert
    sinks; they may contain PII, credentials or business data.
  - Failed and successful billable provider attempts are still counted.
  - Fallback remains opt-in and visible.
  - Exhaustion raises `AllAttemptsFailed`; it never returns a tool call that
    looks successful.
  - Provider errors are classified structurally without importing SDK
    exception types at module import time.
  - `import llm_gateway` continues to work with no provider extras installed.

  ### Reference implementations to inspect

  These are sibling working copies used as design evidence, not dependencies.
  Extract provider-neutral behaviour and reproduce it with local fake-client
  fixtures; do not import application code or copy product policy into the
  package.

  1. **Presupuestor — provider translation and the package boundary**
     - `../presupuestor/backend/app/core/llm_gateway_compat.py`
     - `../presupuestor/backend/app/services/ai_client_service.py`
     - `../presupuestor/backend/tests/core/test_llm_gateway_integration.py`
     - `../presupuestor/docs/decisions/057-unified-llm-entrypoint.md`

     Study how tool definitions and results are translated for provider APIs,
     how the legacy facade is kept stable and how fallback/accounting surround
     the call. Do not copy `LegacyCallOptions`, `ContextVar`, file handling,
     prompts, product logging or the large compatibility adapters wholesale.

  2. **VirtualAssistant — correlation and recursive continuation semantics**
     - `../VirtualAssistant/ai_services/ai_request/universal_request.py`
     - `../VirtualAssistant/ai_services/ai_request/openai_helper.py`
     - `../VirtualAssistant/ai_services/ai_request/groq_helper.py`
     - `../VirtualAssistant/ai_services/ai_request/responses_utils.py`
     - `../VirtualAssistant/tests/unit/test_gpt_request_tools.py`

     Study preservation of `call_id`, multiple function calls, placement of
     `function_call_output`, and the boundary around `tool_executor`. The
     recursive execution loop, WhatsApp context, history loading and business
     dispatch stay in VirtualAssistant.

  3. **Sofia — safe partial migration and provider-specific carve-outs**
     - `../sofia-financial-reports/ai_services/ai_request/universal_request.py`
     - `../sofia-financial-reports/ai_services/ai_request/openai_helper.py`
     - `../sofia-financial-reports/ai_services/ai_request/groq_helper.py`
     - `../sofia-financial-reports/docs/LLM_GATEWAY.md`
     - `../sofia-financial-reports/sofia/tests/test_llm_gateway_chat.py`

     Use this repository to verify that tool requests continue through the
     legacy path until the new contract is complete. Keep OCR, files,
     multimodal input, conversations and prompt templates outside this task.

  4. **Apps — minimal dispatcher seam**
     - `../apps/backend/app/services/ai_client_service.py`
     - `../apps/backend/app/services/ai_text_gateway.py`
     - `../apps/backend/tests/core/test_ai_text_gateway.py`

     Study the early neutral/legacy split and preserve it until tool parity is
     proven. Do not bring Apps-specific pricing, notifications, files or video
     services into the package.

  ### Required tests and acceptance criteria

  Follow TDD and first demonstrate the missing contract with failing tests.
  Add provider-independent contract tests plus fake-client tests for each
  adapter that claims support. At minimum cover:

  - one function definition and every neutral tool-choice mode;
  - multiple returned calls in a single response;
  - stable call IDs, names and parsed arguments;
  - a continuation containing correctly correlated tool results;
  - malformed/non-object arguments and schema violations, with explicit
    fallback and accounting expectations;
  - provider failure before and after a tool-calling response;
  - usage and cost across every billable attempt;
  - capability honesty for supported and unsupported adapters;
  - requests without tools producing the same provider payload as v0.9.0;
  - sinks receiving metadata only;
  - optional SDK imports remaining lazy.

  Add live tests only for provider behaviour that fake clients cannot prove,
  keep them deselected by default, and never require credentials in CI.

  Update `README.md`, `docs/architecture.md`, provider capability docs,
  `CONTRIBUTING.md`, public exports and `[Unreleased]` in `CHANGELOG.md`. Keep
  every source file below the repository's 500-line boundary. Run the full
  offline test suite, Ruff, formatting and strict mypy before considering the
  task complete. Release and consumer upgrades are separate, explicitly
  authorised actions; do not tag, publish or push as part of implementation.

## Audio input and transcription

- [x] **P2 — Add provider-neutral audio analysis and transcription support.**

  `gpt-transcribe` is already present in the catalogue for identity and
  routing, but the gateway does not yet process audio. The feature must keep
  transcription billing by audio duration completely separate from
  token-based usage and cost.

  ### Reference behaviour

  `VirtualAssistant` has two relevant seams:

  - `ai_services/ai_request/gpt_request` accepts OpenAI `file_ids`; the OpenAI
    Responses adapter converts the last user message to content parts and
    appends `input_file` entries. This is the pattern needed to analyse an
    uploaded audio file with a normal multimodal model.
  - The WhatsApp audio pipeline decodes the received bytes, converts formats
    when needed, and calls `audio.transcriptions.create` with
    `gpt-transcribe`. Downloading, base64 handling and ffmpeg conversion are
    application concerns and must remain outside this package.

  Sofia also uses the `file_ids` path for document/OCR requests, so the
  attachment contract has a second real consumer and should not be shaped as
  a VirtualAssistant-specific API.

  ### First release scope

  Implement two explicit, related capabilities:

  1. A neutral remote-file attachment on `LLMRequest` for already-uploaded
     files, with an OpenAI Responses translation equivalent to
     `[{"type": "input_text", ...}, {"type": "input_file", "file_id": ...}]`.
     Do not accept local paths, base64 payloads or SDK file objects in the
     public contract.
  2. A separate transcription operation for audio bytes/file-like input,
     routed to `gpt-transcribe` and translated to OpenAI's audio transcription
     endpoint. It must not be sent through the normal token `generate` path.

  Keep attachment/transcription capabilities explicit on adapters. OpenAI
  may claim the capabilities only after deterministic fake-client tests prove
  both request shapes; Gemini, Groq and OpenRouter must reject unsupported
  requests without silently dropping the audio/file input.

  ### Cost and accounting contract

  - Preserve `TokenUsage` and token `Cost` for normal model calls.
  - Add a separate audio-duration usage/result/cost seam (or an equivalent
    typed port) for transcription; do not encode minutes as fake tokens.
  - Use the catalogue's `gpt-transcribe` per-minute metadata only from the
    audio path. A token `PriceCatalog` must never price that model as a zero-
    cost token call.
  - Audio duration must be reported when the provider reports it; missing
    duration is unknown, not zero. Retries/fallbacks and failed billable
    transcription attempts remain auditable.
  - Prompts, audio bytes, file IDs and transcription text must stay out of
    usage, event and alert sinks.

  ### Required tests and acceptance criteria

  Follow TDD and first demonstrate the missing contract with failing tests.
  Cover at minimum:

  - validation of remote file IDs and audio input metadata;
  - OpenAI Responses payload with text plus one or more `input_file` parts;
  - attachment requests rejected before dispatch by providers without the
    capability;
  - OpenAI transcription payload, response text and reported duration;
  - unsupported audio formats/empty input handled at the application seam;
  - `gpt-transcribe` never entering token pricing or token fallback chains;
  - separate audio cost arithmetic at `$0.0045` per minute, including unknown
    duration and retry accounting;
  - unchanged text-only payloads and token accounting;
  - lazy optional SDK imports, metadata-only sinks, Ruff and strict mypy.

  ### Reference files

  - `../VirtualAssistant/ai_services/ai_request/universal_request.py`
  - `../VirtualAssistant/ai_services/ai_request/openai_helper.py`
  - `../VirtualAssistant/ai_services/ai_request/responses_utils.py`
  - `../VirtualAssistant/projects/whatsapp/database/utils.py`
  - `../VirtualAssistant/tests/unit/test_gpt_request_tools.py`
  - `../VirtualAssistant/tests/integration/test_audio_pipeline_e2e.py`
  - `../sofia-financial-reports/sofia/handlers/document_ocr.py`

## Image and video generation

- [x] **P3 — Add provider-neutral image generation and editing.**

  Shipped as `ImageRequest`/`ImageResult`, `LLMGateway.generate_image()` and
  the Gemini, Replicate and WaveSpeed adapters, with image usage and cost kept
  out of token accounting. See `[Unreleased]` in `CHANGELOG.md` and the image
  sections of `README.md` and `docs/pricing.md`.

- [x] **P4 — Add provider-neutral video generation.** Both shapes are shipped:
  WaveSpeed's polled image-to-video (`VideoRequest`,
  `LLMGateway.generate_video()`, per-second cost by resolution) and Replicate's
  submitted job (`VideoJob`, `LLMGateway.submit_video()` / `poll_video()`,
  `webhook_url`, cost `UNAVAILABLE` because Replicate bills GPU time). The
  design record below is kept because it is the reasoning the contract rests
  on, not a to-do list.

  ### Why this belongs in the package

  Two applications already generate video against Replicate:

  - VirtualAssistant (`modules/replicate_client.py`) creates Replicate
    predictions for `wan-video/wan-2.2-5b-fast`, both text-to-video and
    image-to-video, and polls or receives a webhook.
  - Apps (`backend/app/services/replicate_client.py`,
    `video_generation_router.py`) runs the same Wan model behind a router that
    picks the client by model id — the product-side shape of what the
    catalogue would decide here.

  Both call `predictions.create`, keep the returned id, and later either call
  `predictions.get` or receive a webhook. That is the same code written twice.

  The shared problem is again transport and normalisation: submitting a job,
  correlating its id, reading its status, and pricing what it produced.

  ### What makes it a different contract from images

  A video is not returned by the call that requests it. Replicate answers with
  a prediction id and finishes minutes later, so the neutral contract is two
  phases — `submit_video()` returning a `VideoJob`, and `poll_video(job)`
  returning status and result — and the polling loop, the webhook endpoint and
  the storage stay with the application. Wrapping it in one `await` would put
  a four-minute timeout inside `TimeoutPolicy` and make a webhook impossible.

  A `VideoJob` outlives the process that submitted it: the application stores
  it and polls from a worker or a webhook handler, so it has to be plain
  serialisable data, reconstructible from what a database row can hold.

  ### Cost

  Replicate bills Wan by GPU time, which no request-shaped figure predicts. So
  the catalogue carries the model with no rate and the cost reports
  `UNAVAILABLE` rather than a guess — the same rule the image catalogue
  already follows for `prunaai/p-image`.

  ### Out of scope

  Prompt enrichment, moderation policy, per-user quotas, storage, thumbnails,
  re-hosting, cancellation, webhook signature verification and the webhook
  endpoint itself.
## More Replicate video models

- [x] **P5 — Catalogue and support `kwaivgi/kling-v3-video` and
  `bytedance/seedance-2.0`.**

  Both ride the `VideoJob` contract from P4. What each of them needed was the
  per-model input translation, read from the published schema rather than
  inferred from Wan — the three models agree on none of it:

  | | first frame | duration | resolution |
  |---|---|---|---|
  | `wan-video/wan-2.2-5b-fast` | `image` | frames, so seconds are refused | passed through |
  | `kwaivgi/kling-v3-video` | `start_image` | `duration`, 3-15 s | `mode`: standard/pro/4k |
  | `bytedance/seedance-2.0` | `image` | `duration`, 3-15 s | `resolution`: 480p-4k |

  A resolution a model does not offer raises rather than falling back to the
  model's default, and a Replicate model this package has not verified receives
  only prompt and first frame. Both rules exist because Replicate ignores keys
  it does not recognise: the option vanishes, the default is generated, and the
  caller pays for a clip they did not ask for.

  ### What was deliberately left out

  No per-second rate could be verified for either model — the API's model
  endpoint publishes none — so both carry no rate and report `UNAVAILABLE`. If
  a published rate is confirmed later it belongs in the catalogue with
  `pricing_unit="video_seconds"`, alongside a `CATALOG_VERSION` bump.

  Their remaining options have no second consumer and no neutral shape yet:
  Kling's `negative_prompt`, `multi_prompt`, `end_image` and `generate_audio`,
  and Seedance's `seed`, `last_frame_image` and the positional
  `reference_images`/`reference_videos`/`reference_audios`. Adding any of them
  to `VideoRequest` needs the two-consumer evidence first.
