"""Pure release-file transformations used by the local release runner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_manifest import (
    bump_version,
    promote_unreleased,
    replace_lock_version,
    replace_project_version,
)

PROJECT = """[project]
name = "neutral-llm-gateway"
version = "0.5.0"
description = "example"

[build-system]
requires = ["hatchling"]
"""

LOCK = """[[package]]
name = "other"
version = "1.0.0"

[[package]]
name = "neutral-llm-gateway"
version = "0.5.0"
source = { editable = "." }
"""

CHANGELOG = """# Changelog

## [Unreleased]

### Changed

- Add a release change.

## [0.5.0] — 2026-07-30

### Added

- Existing release.
"""


def test_bump_version_uses_semver_parts() -> None:
    assert bump_version("0.5.0", "patch") == "0.5.1"
    assert bump_version("0.5.0", "minor") == "0.6.0"
    assert bump_version("0.5.0", "major") == "1.0.0"


def test_bump_version_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError, match="bump"):
        bump_version("0.5.0", "unsupported")


def test_project_version_replacement_preserves_other_metadata() -> None:
    updated = replace_project_version(PROJECT, "0.6.0")

    assert 'version = "0.6.0"' in updated
    assert 'version = "0.5.0"' not in updated
    assert 'description = "example"' in updated


def test_lock_version_replacement_targets_the_editable_project_only() -> None:
    updated = replace_lock_version(LOCK, "0.6.0")

    assert 'name = "other"\nversion = "1.0.0"' in updated
    assert 'name = "neutral-llm-gateway"\nversion = "0.6.0"' in updated


def test_promote_unreleased_creates_a_dated_release_section() -> None:
    updated = promote_unreleased(CHANGELOG, "0.6.0", "2026-07-31")

    assert "## [Unreleased]\n\n## [0.6.0] — 2026-07-31" in updated
    assert "- Add a release change." in updated
    assert "## [0.5.0] — 2026-07-30" in updated
