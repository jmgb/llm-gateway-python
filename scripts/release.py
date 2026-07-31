#!/usr/bin/env python3
"""Prepare and optionally publish a release without GitHub Actions."""

from __future__ import annotations

import argparse
import datetime as date
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_manifest import (
    bump_version,
    project_version,
    promote_unreleased,
    replace_lock_version,
    replace_project_version,
    unreleased_body,
    version_parts,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = ROOT / "pyproject.toml"
LOCK_FILE = ROOT / "uv.lock"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def _run(command: Sequence[str], *, capture: bool = False) -> str:
    print(f"$ {shlex.join(command)}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=capture,
        text=True,
    )
    return result.stdout.strip() if capture else ""


def _ensure_ready() -> None:
    if _run(("git", "status", "--porcelain"), capture=True):
        raise RuntimeError("working tree must be clean before a release")
    if _run(("git", "branch", "--show-current"), capture=True) != "main":
        raise RuntimeError("releases must be created from the main branch")


def _checks() -> None:
    commands = (
        ("uv", "run", "--offline", "pytest"),
        ("uv", "lock", "--check", "--offline"),
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "ruff", "format", "--check", "."),
        ("uv", "run", "mypy"),
        ("uv", "build"),
    )
    for command in commands:
        _run(command)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    version = parser.add_mutually_exclusive_group(required=True)
    version.add_argument("--version", help="release version in X.Y.Z format")
    version.add_argument("--bump", choices=("patch", "minor", "major"))
    parser.add_argument("--date", default=date.date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--push", action="store_true", help="push main and the tag")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish artifacts to PyPI and create the GitHub Release",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.publish and not args.push:
        raise SystemExit("--publish requires --push")

    try:
        current_version = project_version(PROJECT_FILE.read_text())
    except ValueError as error:
        raise SystemExit(str(error)) from error
    target = args.version or bump_version(current_version, args.bump)
    if version_parts(target) <= version_parts(current_version):
        raise SystemExit(f"target version {target} is not newer than {current_version}")

    changelog = CHANGELOG_FILE.read_text()
    release_notes = unreleased_body(changelog)
    if args.dry_run:
        print(f"Would release {target} from {current_version} on {args.date}.")
        return 0

    _ensure_ready()
    originals = {
        PROJECT_FILE: PROJECT_FILE.read_text(),
        LOCK_FILE: LOCK_FILE.read_text(),
        CHANGELOG_FILE: changelog,
    }
    updated_files = {
        PROJECT_FILE: replace_project_version(originals[PROJECT_FILE], target),
        LOCK_FILE: replace_lock_version(originals[LOCK_FILE], target),
        CHANGELOG_FILE: promote_unreleased(changelog, target, args.date),
    }
    for path, content in updated_files.items():
        path.write_text(content)
    try:
        _checks()
    except Exception:
        for path, content in originals.items():
            path.write_text(content)
        raise

    tag = f"v{target}"
    _run(("git", "add", "pyproject.toml", "uv.lock", "CHANGELOG.md"))
    _run(("git", "commit", "-m", f"chore: release {target}"))
    _run(("git", "tag", "--annotate", tag, "--message", f"Release {target}"))
    if args.push:
        _run(("git", "push", "origin", "HEAD", "--follow-tags"))
    if args.publish:
        artifacts = sorted(
            str(path.relative_to(ROOT)) for path in (ROOT / "dist").glob(f"*{target}*")
        )
        if not artifacts:
            raise RuntimeError(f"no build artifacts found for {target}")
        _run(("uv", "publish", *artifacts))
        _run(
            (
                "gh",
                "release",
                "create",
                tag,
                "--verify-tag",
                "--title",
                tag,
                "--notes",
                release_notes,
            )
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"release failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
