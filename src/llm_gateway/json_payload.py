"""Extracting JSON from a model that was asked for JSON.

Models wrap payloads in prose or fenced code blocks. Recovering from that is a
provider-shaped problem, so it lives here rather than in every application.

What this module will not do is repair the *semantics* of a payload: it never
invents a missing field or coerces a wrong type. A payload that does not match
its schema is an error, not something to be patched into looking correct.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_gateway.errors import OutputParsingError

_FENCE = re.compile(r"```(?:json)?\s*(?P<body>.+?)\s*```", re.DOTALL)


def parse_json_payload(text: str | None) -> Any:
    """Parse JSON that may be wrapped in a code fence or surrounding prose."""
    if text is None or not text.strip():
        raise OutputParsingError("the model returned an empty response where JSON was required")

    candidates = [text]
    fenced = _FENCE.search(text)
    if fenced:
        candidates.insert(0, fenced.group("body"))
    trimmed = _largest_braced_span(text)
    if trimmed is not None:
        candidates.append(trimmed)

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise OutputParsingError(
        f"the model did not return parseable JSON ({len(text)} characters received)"
    )


def _largest_braced_span(text: str) -> str | None:
    """The outermost {...} or [...] span, for replies padded with prose."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None
