"""AssemblyAI transcription adapter using the submit-and-poll REST API."""

from __future__ import annotations

import asyncio
from typing import Any, cast

from llm_gateway.audio import (
    ProviderTranscriptionResponse,
    TranscriptionRequest,
    normalize_provider_transcription,
)
from llm_gateway.capabilities import ProviderCapabilities
from llm_gateway.contracts import LLMRequest
from llm_gateway.errors import ConfigurationError, LLMGatewayError, ProviderError
from llm_gateway.providers.base import ProviderResponse
from llm_gateway.providers.error_mapping import classify_provider_error

CAPABILITIES = ProviderCapabilities(audio_transcription=True)

_MODEL_TO_SPEECH_MODEL = {
    "assemblyai-universal-3-pro": "universal-3-pro",
    "assemblyai-universal-2": "universal-2",
}


class AssemblyAIHttpClient:
    """Small injected REST client; it owns the key, never the gateway."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.assemblyai.com/v2") -> None:
        if not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": api_key}

    async def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        return await self._request("POST", path, json=json)

    async def get(self, path: str) -> dict[str, Any]:
        return await self._request("GET", path)

    async def _request(
        self, method: str, path: str, *, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.request(
                method,
                f"{self._base_url}/{path.lstrip('/')}",
                headers=self._headers,
                json=json,
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())


class AssemblyAIAdapter:
    """Translates a URL transcription into AssemblyAI's polling workflow."""

    name = "assemblyai"
    capabilities = CAPABILITIES

    def __init__(
        self,
        client: Any,
        *,
        max_poll_attempts: int = 120,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if max_poll_attempts < 1:
            raise ValueError("max_poll_attempts must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("poll interval cannot be negative")
        self._client = client
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval_seconds = poll_interval_seconds

    async def transcribe(
        self, request: TranscriptionRequest, *, model: str
    ) -> ProviderTranscriptionResponse:
        speech_model = _MODEL_TO_SPEECH_MODEL.get(model)
        if speech_model is None:
            raise ConfigurationError(f"unsupported AssemblyAI transcription model: {model}")
        if request.audio.url is None:
            raise ConfigurationError("AssemblyAI transcription requires a public audio URL")
        if request.prompt is not None and speech_model != "universal-3-pro":
            raise ConfigurationError("AssemblyAI Universal-2 does not support a prompt")

        payload: dict[str, Any] = {
            "audio_url": request.audio.url,
            "speaker_labels": request.speaker_labels,
            "speech_models": [speech_model],
        }
        if request.language is not None:
            payload["language_code"] = request.language
        if request.prompt is not None:
            payload["prompt"] = request.prompt

        try:
            submitted = await self._client.post("/transcript", json=payload)
            transcript_id = submitted.get("id")
            if not isinstance(transcript_id, str) or not transcript_id:
                raise ProviderError("AssemblyAI returned no transcript id")

            for attempt in range(self._max_poll_attempts):
                result = await self._client.get(f"/transcript/{transcript_id}")
                status = result.get("status")
                if status == "completed":
                    return normalize_provider_transcription(
                        result,
                        request=request,
                        model=model,
                        utterances_are_milliseconds=True,
                    )
                if status == "error":
                    raise ProviderError("AssemblyAI transcription failed")
                if attempt + 1 < self._max_poll_attempts:
                    await asyncio.sleep(self._poll_interval_seconds)

            raise TimeoutError("AssemblyAI transcription polling timed out")
        except LLMGatewayError:
            raise
        except Exception as error:
            raise classify_provider_error(error) from None

    async def generate(self, request: LLMRequest, *, model: str) -> ProviderResponse:
        raise ConfigurationError("AssemblyAI only supports transcription requests")


__all__ = ["CAPABILITIES", "AssemblyAIAdapter", "AssemblyAIHttpClient"]
