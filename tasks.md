# Tasks

Open work only. What has shipped is recorded in `CHANGELOG.md`, and the
reasoning the shipped contracts rest on — the tool loop staying in the
application, the two shapes of video, unset options resolving to the cheapest
tier — lives in `docs/architecture.md`.

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
