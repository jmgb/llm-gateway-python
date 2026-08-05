"""The pricing document describes every adapter, not just the ones it was written for.

``docs/pricing.md`` holds the only statement of where each provider reports its
reasoning tokens and what its adapter does about it. Nothing in the code points
at that table, so a fifth adapter can be merged, be perfectly correct, and still
leave the document describing four providers. The damage is not the missing row:
it is that a table found to be incomplete once stops being trusted at all, and
the next person recomputes from the SDK docs the answer this package already
paid for.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from pathlib import Path
from typing import ClassVar

DOCUMENT = Path(__file__).resolve().parents[2] / "docs" / "pricing.md"
SECTION = "## Reasoning tokens"
HEADER_CELL = "provider"


def _adapter_names() -> set[str]:
    """Every name an adapter declares — the same one the registry routes by."""
    package = importlib.import_module("llm_gateway.providers")
    names: set[str] = set()
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"llm_gateway.providers.{info.name}")
        for member in vars(module).values():
            if not isinstance(member, type) or member.__module__ != module.__name__:
                continue
            declared = getattr(member, "name", None)
            if isinstance(declared, str) and hasattr(member, "capabilities"):
                names.add(declared)
    return names


def _documented_providers() -> set[str]:
    """The table's first column, read as adapter names.

    One row may cover several providers that report identically, so the cell is
    split on commas; the parenthetical names the wire format, not the provider.
    """
    parts = DOCUMENT.read_text(encoding="utf-8").split(SECTION, 1)
    assert len(parts) == 2, f"{DOCUMENT.name} has no {SECTION!r} section"
    section = parts[1].split("\n## ", 1)[0]

    documented: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        first_cell = line.strip("|").split("|")[0]
        for part in first_cell.split(","):
            provider = re.sub(r"\(.*?\)", "", part).strip().lower()
            if provider and provider != HEADER_CELL:
                documented.add(provider)
    return documented


def test_every_adapter_is_named_in_the_documented_normalisation_table() -> None:
    undocumented = _adapter_names() - _documented_providers()

    assert undocumented == set(), (
        f"adapters missing from the {SECTION!r} table in docs/pricing.md: {undocumented}"
    )


def test_the_documented_normalisation_table_names_no_adapter_that_is_gone() -> None:
    """A row for an adapter that no longer exists misleads exactly as badly."""
    stale = _documented_providers() - _adapter_names()

    assert stale == set(), (
        f"docs/pricing.md documents providers this package no longer adapts: {stale}"
    )


class TestTheDocumentedDefaultResolution:
    """`docs/pricing.md` promises the cheapest tier when the caller states none.

    The promise is per-adapter and easy to forget when adding a video model, so
    the table in the docs and the tables in the adapters are checked against
    each other. A model in one and not the other is the failure this catches.
    """

    DOCUMENTED: ClassVar[dict[str, str]] = {
        "wavespeed-ai/minimax-h3/image-to-video": "480p",
        "wan-video/wan-2.2-5b-fast": "480p",
        "kwaivgi/kling-v3-video": "720p",
        "bytedance/seedance-2.0": "480p",
    }

    def test_every_video_model_the_docs_promise_a_floor_for_has_one(self) -> None:
        from llm_gateway.providers.replicate import _VIDEO_SHAPES
        from llm_gateway.providers.wavespeed import _LOWEST_RESOLUTION

        declared = {model: shape.default_resolution for model, shape in _VIDEO_SHAPES.items()}
        declared.update(_LOWEST_RESOLUTION)

        assert declared == self.DOCUMENTED

    def test_each_floor_is_the_lowest_tier_that_model_actually_offers(self) -> None:
        """A "default" above the floor would quietly be the expensive answer."""
        from llm_gateway.providers.replicate import _VIDEO_SHAPES

        for model, shape in _VIDEO_SHAPES.items():
            assert shape.resolutions is not None, f"{model} declares no tiers"
            assert shape.default_resolution in shape.resolutions, model
            cheapest = min(shape.resolutions, key=_tier_order)
            assert shape.default_resolution == cheapest, (
                f"{model} defaults to {shape.default_resolution}, not its floor {cheapest}"
            )

    def test_the_documented_video_price_table_lists_every_floor(self) -> None:
        pricing = (Path(__file__).resolve().parents[2] / "docs" / "pricing.md").read_text()

        for model in self.DOCUMENTED:
            assert model in pricing, f"{model} is missing from docs/pricing.md"


def _tier_order(resolution: str) -> int:
    """Vertical pixels, so "4k" sorts above "1080p" rather than before it."""
    return {"480p": 480, "720p": 720, "768p": 768, "1080p": 1080, "4k": 2160}[resolution]
