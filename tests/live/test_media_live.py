"""Real image and video generation, end to end, against two providers.

Opt-in because it spends provider credits: roughly a fraction of a cent for the
Gemini image, and USD 0.20 for a five-second 480p clip on MiniMax H3.

    uv sync --extra gemini --extra wavespeed
    GEMINI_API_KEY=... WAVESPEED_API_KEY=... \
      uv run pytest -m live tests/live/test_media_live.py -q -s

The two tests are a chain: the first generates a lion running across the
savanna with `gemini-3.1-flash-lite-image`, the cheapest catalogued image
model, and writes the bytes where the second can pick them up. The second
animates that exact frame into a hunt with
`wavespeed-ai/minimax-h3/image-to-video` at 480p for five seconds — the shape a
consuming application uses, where one provider's output is another's input.

Both write their result to ``LLM_GATEWAY_LIVE_MEDIA_DIR`` (default: the
system temp directory) so a human can look at what was actually produced. The
video is left as the URL WaveSpeed returned; downloading and re-hosting it is
application work, and this package deliberately does not do it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from llm_gateway import (
    ImageInput,
    ImageRequest,
    LLMGateway,
    ProviderRegistry,
    TimeoutPolicy,
    VideoRequest,
)
from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.factories import (
    create_gemini_client,
    create_wavespeed_client,
)
from llm_gateway.providers.gemini import GeminiAdapter
from llm_gateway.providers.wavespeed import WaveSpeedAdapter

pytestmark = pytest.mark.live

IMAGE_MODEL = "gemini-3.1-flash-lite-image"
VIDEO_MODEL = "wavespeed-ai/minimax-h3/image-to-video"
VIDEO_RESOLUTION = "480p"
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
