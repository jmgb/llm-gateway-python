"""Provider-neutral transcription contracts.

Audio is deliberately not represented as a token message. Providers receive a
common source and return duration usage, while the audio gateway owns retries,
fallbacks and duration-based accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from llm_gateway.contracts import AttemptOutcome, FailurePhase
from llm_gateway.policies import FallbackPolicy, RetryPolicy, TimeoutPolicy
from llm_gateway.pricing import AudioCost
from llm_gateway.usage import AudioUsage


@dataclass(frozen=True, slots=True)
class AudioInput:
    """Audio bytes and/or a provider-fetchable URL.

    URLs are used by AssemblyAI and Groq; bytes are used by OpenAI and Groq.
    Keeping both is useful when an application wants a cross-provider
    fallback. Downloading, decoding and transcoding remain application work.
    """

    data: bytes | None = None
    url: str | None = None
    filename: str = "audio.mp3"
    mime_type: str | None = None
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.url is None:
            raise ValueError("audio data or url is required")
        if self.data is not None and not self.data:
            raise ValueError("audio data must be non-empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("audio url must be non-empty")
        if not self.filename.strip():
            raise ValueError("audio filename must be non-empty")
        if self.duration_seconds is not None and (
            self.duration_seconds < 0 or not math.isfinite(self.duration_seconds)
        ):
            raise ValueError("audio duration must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
    """One speech-to-text operation, separate from ``LLMRequest``."""

    model: str
    audio: AudioInput
    language: str | None = None
    prompt: str | None = None
    speaker_labels: bool = False
    timeout_policy: TimeoutPolicy = field(default_factory=TimeoutPolicy)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy.disabled)
    fallback_policy: FallbackPolicy = field(default_factory=FallbackPolicy.disabled)
    request_id: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("a transcription model identifier is required")
        if self.language is not None and not self.language.strip():
            raise ValueError("language must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class AudioSegment:
    """A normalized transcript segment."""

    start_seconds: float
    end_seconds: float
    text: str
    speaker: str | None = None


@dataclass(frozen=True, slots=True)
class ProviderTranscriptionResponse:
    """One provider response before gateway accounting and orchestration."""

    text: str
    usage: AudioUsage
    segments: tuple[AudioSegment, ...] = ()
    language: str | None = None
    model_used: str | None = None


@dataclass(frozen=True, slots=True)
class AudioAttempt:
    """One billable or configuration-failed transcription attempt."""

    index: int
    model: str
    provider: str
    outcome: AttemptOutcome
    usage: AudioUsage
    cost: AudioCost
    latency_ms: int
    error_type: str | None = None
    billable: bool = True
    failure_phase: FailurePhase | None = None


@dataclass(frozen=True, slots=True)
class AudioExecution:
    """What happened during a transcription call."""

    requested_model: str
    model_used: str
    provider: str
    attempts: tuple[AudioAttempt, ...]
    latency_ms: int

    @property
    def fallback_used(self) -> bool:
        return bool(self.attempts and self.attempts[-1].model != self.requested_model)

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Text and audio accounting kept separate from token results."""

    text: str
    usage: AudioUsage
    execution: AudioExecution
    cost: AudioCost
    segments: tuple[AudioSegment, ...] = ()
    language: str | None = None

    @property
    def output(self) -> str:
        return self.text


def _value(raw: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw, dict):
        return raw.get(name, default)
    return getattr(raw, name, default)


def normalize_provider_transcription(
    raw: Any,
    *,
    request: TranscriptionRequest,
    model: str,
    utterances_are_milliseconds: bool = False,
) -> ProviderTranscriptionResponse:
    """Normalize OpenAI, Groq and AssemblyAI verbose transcript shapes."""

    text = raw if isinstance(raw, str) else (_value(raw, "text", "") or "")
    duration = _value(raw, "duration")
    if duration is None:
        duration = _value(raw, "audio_duration")
    if duration is None:
        duration = _value(_value(raw, "usage"), "seconds")
    estimated = duration is None and request.audio.duration_seconds is not None
    if estimated:
        duration = request.audio.duration_seconds
    raw_segments = _value(raw, "segments") or _value(raw, "utterances") or ()
    segments = tuple(
        AudioSegment(
            start_seconds=_seconds(_value(segment, "start", 0.0), utterances_are_milliseconds),
            end_seconds=_seconds(_value(segment, "end", 0.0), utterances_are_milliseconds),
            text=str(_value(segment, "text", "") or "").strip(),
            speaker=_value(segment, "speaker"),
        )
        for segment in raw_segments
    )
    return ProviderTranscriptionResponse(
        text=str(text),
        usage=AudioUsage(
            duration_seconds=_as_float(duration),
            partial_aggregate=estimated,
        ),
        segments=segments,
        language=_transcription_language(raw) or request.language,
        model_used=_value(raw, "model") or model,
    )


def _seconds(value: Any, milliseconds: bool) -> float:
    number = float(value or 0.0)
    return number / 1000.0 if milliseconds else number


def _as_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _transcription_language(raw: Any) -> str | None:
    direct = _value(raw, "language") or _value(raw, "language_code")
    if direct is not None:
        return str(direct)
    languages = _value(raw, "languages") or ()
    first = languages[0] if languages else None
    code = _value(first, "code")
    return str(code) if code is not None else None
