"""What a provider can actually do.

Providers sit behind one contract, but their capabilities are not pretended to
be identical. A caller can ask before requesting something that would silently
degrade.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderCapabilities:
    """Declared, not guessed. Defaults are deliberately conservative."""

    structured_outputs: bool = False
    json_mode: bool = False
    function_calling: bool = False
    inline_files: bool = False
    remote_files: bool = False
    audio_transcription: bool = False
    reasoning_effort: bool = False
    conversation_history: bool = True
    reports_token_usage: bool = True

    def require(self, capability: str) -> bool:
        """Read a capability by name, for generic checks."""
        if not hasattr(self, capability):
            raise AttributeError(f"unknown capability: {capability}")
        return bool(getattr(self, capability))
