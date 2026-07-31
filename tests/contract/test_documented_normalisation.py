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
