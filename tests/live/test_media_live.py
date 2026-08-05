"""Real image and video generation, end to end, against three providers.

Opt-in because it spends provider credits: roughly five cents for the Gemini
image, USD 0.20 for a five-second 480p clip on MiniMax H3, and an unpublished
amount for the Kling clip, which Replicate bills by GPU time.

    uv sync --extra gemini --extra wavespeed --extra replicate
    GEMINI_API_KEY=... WAVESPEED_API_KEY=... REPLICATE_API_TOKEN=... \
      uv run pytest -m live tests/live/test_media_live.py -q -s

The tests are a chain. The first generates a lioness running across the savanna
with `gemini-3.1-flash-lite-image`, the cheapest catalogued image model, and
writes the bytes where the others pick them up. The second and third animate
that same frame into the same hunt, from the same `VIDEO_PROMPT`, through the
two shapes video comes in:

- `wavespeed-ai/minimax-h3/image-to-video` is awaited — `generate_video()`
  polls inside the adapter and returns the clip;
- `kwaivgi/kling-v3-video` is submitted — `submit_video()` returns a job id and
  the poll loop lives in the test, which is where a real application keeps it.

Running both on one frame is the point: the clips are comparable because only
the model changed.

Every clip here is generated at the **lowest resolution its model offers** —
480p on MiniMax H3, 720p on Kling. That is the package's default and it is also
the rule for tests: a suite that a human reruns while working on it should
never be the expensive way to find out something broke. Both are stated
explicitly below rather than left to the default, so raising one is a visible
edit to a named constant.

Each writes its result to ``LLM_GATEWAY_LIVE_MEDIA_DIR`` (default: the system
temp directory) so a human can look at what was actually produced. The videos
are left as the URLs the providers returned; downloading and re-hosting them is
application work, and this package deliberately does not do it.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from llm_gateway import (
    CostMeasurement,
    ImageInput,
    ImageRequest,
    LLMGateway,
    ProviderRegistry,
    TimeoutPolicy,
    VideoJobStatus,
    VideoRequest,
)
from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.factories import (
    create_gemini_client,
    create_replicate_client,
    create_wavespeed_client,
)
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.replicate import ReplicateAdapter
from llm_gateway.providers.wavespeed import WaveSpeedAdapter

pytestmark = pytest.mark.live

IMAGE_MODEL = "gemini-3.1-flash-lite-image"
VIDEO_MODEL = "wavespeed-ai/minimax-h3/image-to-video"
VIDEO_RESOLUTION = "480p"  # MiniMax H3's cheapest tier; 768p costs twice as much
VIDEO_SECONDS = 5

IMAGE_PROMPT = (
    "A powerful adult lioness running at full speed across the African savanna, "
    "dust kicked up behind her paws, golden late-afternoon light, dry grass, "
    "acacia trees blurred in the background. Photorealistic wildlife photography, "
    "sharp focus on the lioness, shallow depth of field, 200mm telephoto lens."
)
VIDEO_PROMPT = (
    "The lioness sprints at full speed and attacks her prey, a gazelle. "
    "She leaps onto the gazelle, clamps her jaws hard on its throat and brings "
    "it down in a cloud of dust. Handheld wildlife documentary camera tracking "
    "the chase, natural savanna light."
)

_IMAGE_FILENAME = "live_lion.png"


def _output_dir() -> Path:
    return Path(os.environ.get("LLM_GATEWAY_LIVE_MEDIA_DIR", tempfile.gettempdir()))


def _gemini_gateway(key: str) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(GeminiAdapter(create_gemini_client(api_key=key)), model_prefixes=())
    return LLMGateway(registry=registry)


def _wavespeed_gateway(key: str) -> LLMGateway:
    registry = ProviderRegistry()
    registry.register(WaveSpeedAdapter(create_wavespeed_client(api_key=key)), model_prefixes=())
    return LLMGateway(registry=registry)


async def test_a_real_lion_image_is_generated_and_priced() -> None:
    """Gemini's cheapest image model returns bytes, usage and a token-based cost."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    try:
        gateway = _gemini_gateway(key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))

    result = await gateway.generate_image(
        ImageRequest(
            model=IMAGE_MODEL,
            prompt=IMAGE_PROMPT,
            aspect_ratio="16:9",
            source="live-media-integration",
            timeout_policy=TimeoutPolicy(total_seconds=300.0),
        )
    )

    image = result.images[0]
    assert image.data, "Gemini returned no image bytes"
    assert result.usage.images == 1
    assert result.usage.tokens is not None
    assert result.cost.amount_usd is not None
    assert result.cost.amount_usd > 0
    assert result.execution.provider == "gemini"

    destination = _output_dir() / _IMAGE_FILENAME
    destination.write_bytes(image.data)
    print(
        f"image: model={result.execution.model_used} bytes={len(image.data)} "
        f"cost=${result.cost.amount_usd} saved={destination}"
    )


async def test_the_real_lion_image_is_animated_into_a_hunt() -> None:
    """The generated frame is animated by a different provider, as an app would.

    It reuses the PNG the image test wrote, so run them in order. Falling back
    to generating the frame here would double the cost of a rerun.
    """
    wavespeed_key = os.environ.get("WAVESPEED_API_KEY")
    if not wavespeed_key:
        pytest.skip("WAVESPEED_API_KEY is not set")

    frame = _output_dir() / _IMAGE_FILENAME
    if not frame.is_file():
        pytest.skip(f"run the image test first: {frame} does not exist")

    try:
        gateway = _wavespeed_gateway(wavespeed_key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))

    result = await gateway.generate_video(
        VideoRequest(
            model=VIDEO_MODEL,
            prompt=VIDEO_PROMPT,
            image=ImageInput(data=frame.read_bytes(), mime_type="image/png"),
            resolution=VIDEO_RESOLUTION,
            duration_seconds=VIDEO_SECONDS,
            source="live-media-integration",
            timeout_policy=TimeoutPolicy(total_seconds=900.0),
        )
    )

    video = result.videos[0]
    assert video.url, "WaveSpeed returned no video URL"
    assert result.usage.seconds == float(VIDEO_SECONDS)
    assert result.usage.resolution == VIDEO_RESOLUTION
    # 5 seconds at USD 0.04/second, and ESTIMATED because WaveSpeed reports no
    # clip length of its own.
    assert result.cost.amount_usd is not None
    assert result.cost.amount_usd > 0
    assert result.execution.provider == "wavespeed"
    print(
        f"video: model={result.execution.model_used} "
        f"cost=${result.cost.amount_usd} ({result.cost.measurement.value}) url={video.url}"
    )


KLING_MODEL = "kwaivgi/kling-v3-video"
KLING_RESOLUTION = "720p"  # Kling's cheapest tier; sent as mode="standard", not "pro"
_KLING_POLL_SECONDS = 10.0
_KLING_MAX_WAIT_SECONDS = 900.0


@asynccontextmanager
async def _replicate_gateway(key: str) -> AsyncIterator[LLMGateway]:
    """Own the client for the duration of the test, then close it.

    The package never closes a client it did not create — the application
    builds it and keeps the key. Here the test *is* the application, and the
    SDK's `httpx.AsyncClient` left open emits a `ResourceWarning` that
    `filterwarnings = ["error"]` turns into an intermittent failure.
    """
    client = create_replicate_client(api_key=key)
    registry = ProviderRegistry()
    registry.register(ReplicateAdapter(client), model_prefixes=())
    try:
        yield LLMGateway(registry=registry)
    finally:
        await client._async_client.aclose()
        client._client.close()


async def test_the_same_lion_frame_is_animated_by_kling_as_a_submitted_job() -> None:
    """The job contract end to end, on the same frame and prompt as MiniMax H3.

    Deliberately the same PNG and the same `VIDEO_PROMPT` the WaveSpeed test
    uses, so the two clips are comparable: only the model changed.

    The polling loop lives here rather than in the package, which is the whole
    point of `submit_video()`/`poll_video()` — a real application runs this from
    a worker, or replaces it entirely with the webhook.
    """
    key = os.environ.get("REPLICATE_API_TOKEN") or os.environ.get("REPLICATE_API_KEY_TOKEN")
    if not key:
        pytest.skip("REPLICATE_API_TOKEN is not set")

    frame = _output_dir() / _IMAGE_FILENAME
    if not frame.is_file():
        pytest.skip(f"run the image test first: {frame} does not exist")

    try:
        manager = _replicate_gateway(key)
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))

    async with manager as gateway:
        await _run_kling_job(gateway, frame)


async def _run_kling_job(gateway: LLMGateway, frame: Path) -> None:
    job = await gateway.submit_video(
        VideoRequest(
            model=KLING_MODEL,
            prompt=VIDEO_PROMPT,
            image=ImageInput(data=frame.read_bytes(), mime_type="image/png"),
            resolution=KLING_RESOLUTION,
            duration_seconds=VIDEO_SECONDS,
            source="live-media-integration",
            timeout_policy=TimeoutPolicy(total_seconds=300.0),
        )
    )

    assert job.id, "Replicate returned no prediction id"
    assert job.provider == "replicate"
    assert job.model == KLING_MODEL
    assert job.status is not VideoJobStatus.FAILED
    print(f"kling job submitted: id={job.id} status={job.status.value}")

    # What an application's worker does: poll until the answer stops changing.
    deadline = time.monotonic() + _KLING_MAX_WAIT_SECONDS
    result = await gateway.poll_video(job)
    while not result.is_terminal:
        if time.monotonic() > deadline:
            pytest.fail(f"kling job {job.id} was still {result.job.status.value} after 15 minutes")
        await asyncio.sleep(_KLING_POLL_SECONDS)
        result = await gateway.poll_video(job)

    assert result.job.status is VideoJobStatus.SUCCEEDED, f"kling failed: {result.error}"
    assert result.job.id == job.id, "the job id must survive the round trip"
    video = result.videos[0]
    assert video.url, "Replicate returned no video URL"
    # Replicate bills Wan and Kling by GPU time and reports no clip length, so
    # this stays honest about not knowing rather than reporting a free clip.
    assert result.usage.seconds is None
    assert result.cost.measurement is CostMeasurement.UNAVAILABLE
    print(
        f"kling video: model={result.job.model} status={result.job.status.value} "
        f"cost={result.cost.measurement.value} url={video.url}"
    )
