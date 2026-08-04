"""Real speech-to-text calls for the three audio providers.

This test is deliberately opt-in because it spends provider credits. It uses
the same short Spanish construction-site recording for every provider:

    uv sync --extra openai --extra groq --extra assemblyai
    LLM_GATEWAY_LIVE_AUDIO_PATH=/path/to/acta_obra_1.m4a \
    LLM_GATEWAY_LIVE_AUDIO_DURATION_SECONDS=50.56 \
    OPENAI_API_KEY=... GROQ_API_KEY=... ASSEMBLYAI_API_KEY=... \
      uv run pytest -m live tests/live/test_audio_live.py -q -s

AssemblyAI requires a URL, so the test first uploads the local recording to
AssemblyAI's temporary upload endpoint and then exercises this package's
submit-and-poll adapter with the returned URL. OpenAI and Groq use the bytes
path directly, as the production callers do.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llm_gateway import AudioInput, LLMGateway, TranscriptionRequest
from llm_gateway.errors import ProviderNotInstalled
from llm_gateway.factories import (
    build_registry,
    create_assemblyai_client,
    create_groq_client,
    create_openai_client,
)

pytestmark = pytest.mark.live


@dataclass(frozen=True, slots=True)
class _ProviderCase:
    name: str
    key_variable: str
    model: str


_CASES = (
    _ProviderCase("openai", "OPENAI_API_KEY", "gpt-transcribe"),
    _ProviderCase("groq", "GROQ_API_KEY", "whisper-large-v3-turbo"),
    _ProviderCase("assemblyai", "ASSEMBLYAI_API_KEY", "assemblyai-universal-3-pro"),
)
_DEFAULT_AUDIO_DURATION_SECONDS = 50.56


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_real_audio_transcription_completes_end_to_end(case: _ProviderCase) -> None:
    """A real recording reaches each provider and returns priced audio usage."""
    key = os.environ.get(case.key_variable)
    if not key:
        pytest.skip(f"{case.key_variable} is not set")

    audio_path = _audio_path()
    if audio_path is None:
        pytest.skip("LLM_GATEWAY_LIVE_AUDIO_PATH is not set")
    audio_bytes = audio_path.read_bytes()
    assert audio_bytes, f"audio fixture is empty: {audio_path}"
    duration_seconds = float(
        os.environ.get(
            "LLM_GATEWAY_LIVE_AUDIO_DURATION_SECONDS",
            str(_DEFAULT_AUDIO_DURATION_SECONDS),
        )
    )

    try:
        audio, registry, provider_client = await _provider_client_and_registry(
            case, key, audio_bytes, audio_path, duration_seconds
        )
    except ProviderNotInstalled as absent:
        pytest.skip(str(absent))
    try:
        result = await LLMGateway(registry=registry).transcribe(
            TranscriptionRequest(
                model=case.model,
                audio=audio,
                language="es",
                source="live-audio-integration",
            )
        )
    finally:
        await _close_client(provider_client)

    assert result.text.strip(), f"{case.name} returned an empty transcript"
    assert result.usage.duration_seconds is not None
    assert result.usage.duration_seconds > 0
    assert result.cost.amount_usd is not None
    assert result.cost.amount_usd > 0
    assert result.execution.provider == case.name
    assert result.execution.model_used == case.model
    assert any(word in result.text.lower() for word in ("pasillo", "pladur", "pilar"))
    print(
        f"{case.name}: model={case.model} duration={result.usage.duration_seconds:.2f}s "
        f"cost=${result.cost.amount_usd} text={result.text[:160]!r}"
    )


def _audio_path() -> Path | None:
    raw_path = os.environ.get("LLM_GATEWAY_LIVE_AUDIO_PATH")
    if not raw_path:
        return None
    path = Path(raw_path)
    if not path.is_file():
        pytest.fail(f"LLM_GATEWAY_LIVE_AUDIO_PATH does not exist: {path}")
    return path


async def _provider_client_and_registry(
    case: _ProviderCase,
    key: str,
    audio_bytes: bytes,
    audio_path: Path,
    duration_seconds: float,
) -> tuple[AudioInput, Any, Any]:
    if case.name == "openai":
        client = create_openai_client(api_key=key)
        return (
            AudioInput(
                data=audio_bytes,
                filename=audio_path.name,
                mime_type="audio/mp4",
                duration_seconds=duration_seconds,
            ),
            build_registry(openai_client=client),
            client,
        )
    if case.name == "groq":
        client = create_groq_client(api_key=key)
        return (
            AudioInput(
                data=audio_bytes,
                filename=audio_path.name,
                mime_type="audio/mp4",
                duration_seconds=duration_seconds,
            ),
            build_registry(groq_client=client),
            client,
        )

    client = create_assemblyai_client(api_key=key)
    audio_url = await _upload_to_assemblyai(key, audio_bytes)
    return (
        AudioInput(
            url=audio_url,
            filename=audio_path.name,
            mime_type="audio/mp4",
            duration_seconds=duration_seconds,
        ),
        build_registry(assemblyai_client=client),
        client,
    )


async def _upload_to_assemblyai(api_key: str, audio_bytes: bytes) -> str:
    """Upload the fixture so the real AssemblyAI adapter can submit its URL."""
    try:
        import httpx
    except ImportError as error:
        raise ProviderNotInstalled.for_provider("assemblyai") from error

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.assemblyai.com/v2/upload",
            headers={"Authorization": api_key, "Content-Type": "application/octet-stream"},
            content=audio_bytes,
        )
        response.raise_for_status()
        payload = response.json()
    upload_url = payload.get("upload_url")
    if not isinstance(upload_url, str) or not upload_url:
        raise AssertionError("AssemblyAI upload did not return upload_url")
    return upload_url


async def _close_client(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
