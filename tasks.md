# Tasks

Open work only. What has shipped is recorded in `CHANGELOG.md`, and the
reasoning the shipped contracts rest on — the tool loop staying in the
application, the two shapes of video, unset options resolving to the cheapest
tier — lives in `docs/architecture.md`.

## Contract

- [ ] **P5 — Support inline files (images and PDFs as bytes) in the neutral contract.**

  Version 0.23.0 supports image bytes through `Message.images` on OpenRouter,
  alongside `Message.videos`. This item remains open for PDFs and inline-file
  handling on the other adapters. `LLMRequest.attachments` still carries only
  already-uploaded OpenAI file IDs, every provider declares `inline_files=False`,
  and OpenRouter rejects that attachment channel. Consumers still patch the
  missing paths themselves. Presupuestor does it in its compatibility layer and is the
  reference implementation to lift from (repo `presupuestor`,
  `backend/app/core/llm_gateway_compat.py`, commit `76804006c`, 2026-11-12):

  - **OpenRouter** (`PresupuestorOpenRouterAdapter`, `_openrouter_file_parts`):
    images as Chat Completions `image_url` data-URL parts, PDFs as a native
    `file` part (`{"type": "file", "file": {"filename", "file_data"}}`) appended
    to the last user message. Verified live on 2026-11-12: `google/gemini-3.5-flash-lite`
    described a JPEG and `google/gemini-3.8-flash` read a PDF, for under USD 0.003.
    OpenAI `file_ids` are refused, because OpenRouter cannot read them.
  - **Groq** (`PresupuestorGroqAdapter`): same `image_url` shape; PDFs are
    rasterised to PNG pages first, because Groq does not read PDFs.
  - **Gemini** (`PresupuestorGeminiAdapter`): inline byte parts per file.
  - **OpenAI** (`PresupuestorOpenAIAdapter`): uploads the bytes to the Files API
    and sends the resulting IDs as `input_file` parts.
  - Tests to port: `backend/tests/core/test_llm_gateway_integration.py`
    (`test_openrouter_adapter_sends_inline_images_as_image_url_parts`,
    `test_openrouter_adapter_sends_pdfs_natively_as_file_parts`,
    `test_openrouter_adapter_refuses_openai_file_ids_instead_of_answering_blind`).

  What the gateway needs:

  - an `InlineFile(data: bytes, mime_type: str, filename: str | None)` on
    `LLMRequest`, next to `attachments`;
  - `inline_files=True` declared per provider only once deterministic tests
    cover its request shape; providers without it reject the request rather
    than drop the files, as `remote_files` already does;
  - a decision on PDFs for providers that cannot read them (Groq): reject, or
    rasterise behind an optional extra so the core keeps no PDF dependency;
  - offline tests per adapter and one cheap `-m live` test per provider.

  Done when a release ships the contract for OpenAI, Gemini, Groq and
  OpenRouter, and Presupuestor bumps the pin and deletes its own file handling
  from `llm_gateway_compat.py` with its tests still green.

## Live coverage

- [ ] **P6 — Cover `bytedance/seedance-2.5` in the live suite.**

  Nothing has ever sent a Seedance request to Replicate. The offline tests
  assert what the adapter builds; they cannot assert that Replicate accepts it,
  and this model is exactly where that gap costs money — an unrecognised key is
  not a 422 there, it is a dropped option, a generated default and an invoice.
  Kling and MiniMax H3 each got a live test for that reason and Seedance did
  not.

  What it has to prove, in the shape `tests/live/test_media_live.py` already
  uses for Kling — `submit_video()`, poll, one terminal state:

  - `resolution="480p"` arrives as sent, and the returned clip is 480p rather
    than the 720p the model defaults to;
  - `duration` in seconds is honoured;
  - `VideoUsage.resolution` and `.seconds` come back populated, so the
    catalogued rate prices the call instead of yielding `UNAVAILABLE`;
  - the reported amount matches `0.1028 × seconds` — the assertion that would
    catch the day Replicate moves the rate or the request lands on the
    four-times-dearer `video_in` card.

  It is real money, so it stays deselected with the rest of `-m live`: about
  USD 0.51 for a five-second 480p clip, the model's own default length.

## Consumer rollout

- [ ] **Publish a release with AssemblyAI-STT-3.7.**

  Gateway support is on `origin/main` at `38e52f8`. Once authorised, follow
  the release workflow in `CLAUDE.md`. Verify that the version installed
  from PyPI can transcribe a public sample with `assemblyai-stt-3-7`.

- [ ] **Publish and deploy the 0.23.0 consumer upgrades.**

  The gateway is published on PyPI, tagged `v0.23.0`, and its GitHub CI and
  artifact audits passed. Dependency pins, lockfiles and local installations
  are updated in `../comunicador` (commit `ca5080e`, 35 targeted tests) and
  `../presupuestor` (commit `e7d6f9908`, 44 targeted tests). The selected video
  analysis model is `google/gemini-4-hydrogen`, recorded in consumer commits
  `7a3eb04` and `8f1525463` respectively; the gateway supplies no product default.

  These consumer commits have not been pushed or deployed by this session.
  Both worktrees now contain other agents' active changes: inspect their
  current state and pending history before publishing, run each project's
  required pre-push checks, and follow its deployment procedure with explicit
  authorisation. The user is coordinating model trials and video-flow
  integration in another session; do not duplicate those changes here.
