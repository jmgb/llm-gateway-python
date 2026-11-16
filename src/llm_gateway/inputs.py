"""Visual sources supplied by the application, without network or filesystem work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImageInput:
    """An image to analyse or edit, as bytes or as a fetchable URL.

    Both exist because providers accept different ones and neither converts
    the other for free: Gemini takes inline bytes, Replicate takes a URL it
    fetches itself. An adapter rejects the form it cannot use rather than
    downloading or hosting on the caller's behalf.
    """

    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if self.data is None and self.url is None:
            raise ValueError("image data or url is required")
        if self.data is not None and not self.data:
            raise ValueError("image data must be non-empty")
        if self.url is not None and not self.url.strip():
            raise ValueError("image url must be non-empty")


@dataclass(frozen=True, slots=True)
class VideoInput:
    """A video to analyse, as bytes with a MIME type or a provider-fetchable URL.

    URLs may be base64 data URLs. The application owns reading files, trimming,
    encoding, and hosting; the adapter only encodes bytes for its wire format.
    """

    data: bytes | None = None
    url: str | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise ValueError("exactly one of video data or url is required")
        if self.data is not None:
            if not self.data:
                raise ValueError("video data must be non-empty")
            if not self.mime_type or not self.mime_type.startswith("video/"):
                raise ValueError("video bytes require a video MIME type")
        if self.url is not None and not self.url.strip():
            raise ValueError("video url must be non-empty")
