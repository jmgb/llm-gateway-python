#!/usr/bin/env python3
"""Prepare and optionally publish a release without GitHub Actions."""

from __future__ import annotations

import argparse
import datetime as date
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_dist import archive_members, artifacts_in
from scripts.release_manifest import (
    bump_version,
    project_version,
    promote_unreleased,
    publishing_env_values,
    replace_lock_version,
    replace_project_version,
    unpublishable_members,
    unreleased_body,
    version_parts,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = ROOT / "pyproject.toml"
LOCK_FILE = ROOT / "uv.lock"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
ENV_FILE = ROOT / ".env"


def _run(
    command: Sequence[str], *, capture: bool = False, env: Mapping[str, str] | None = None
) -> str:
    # The command line is printed, so no credential may ever travel in one:
    # `uv publish` and `gh` both read theirs from the environment.
    print(f"$ {shlex.join(command)}")
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=capture,
        text=True,
        env=None if env is None else dict(env),
    )
    return result.stdout.strip() if capture else ""


def _publishing_environment() -> dict[str, str]:
    """The environment for the upload, with a local ``.env`` as a fallback.

    ``.env`` is a reasonable place for a publishing token *because* it can no
    longer travel: it is absent from the sdist allowlist and the artifact audit
    refuses any archive containing one. Without reading it here the token has
    to be exported by hand before every release, and the shortcut people reach
    for instead is committing it. An already-exported value still wins.
    """
    environment = dict(os.environ)
    if ENV_FILE.exists():
        # Only the publishing keys: the file is read for one purpose, and the
        # rest of what a machine keeps in it has no business in the environment
        # of `uv publish` or `gh`.
        for key, value in publishing_env_values(ENV_FILE.read_text(encoding="utf-8")).items():
            environment.setdefault(key, value)
    return environment


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


def _artifacts_for(version: str) -> list[Path]:
    """This version's distributions, shared with the standalone auditor.

    Selecting them differently in the two publishers is how a file gets audited
    by one and uploaded by the other.
    """
    return artifacts_in(ROOT / "dist", version=version)


def _audit_artifacts(version: str) -> None:
    """Refuse to release an artifact carrying something that must stay local.

    An upload is irreversible in the only sense that matters: the file is
    mirrored and cached within minutes, so deleting the release does not
    unpublish what was inside it. The one moment this can still be stopped is
    here, between the build and the upload.
    """
    artifacts = _artifacts_for(version)
    if not artifacts:
        raise RuntimeError(f"no build artifacts found for {version}")
    for path in artifacts:
        members = archive_members(path)
        offenders = unpublishable_members(members)
        if offenders:
            raise RuntimeError(
                f"{path.name} would publish {offenders}; "
                f"remove them from the sdist include list before releasing"
            )
        print(f"  audited {path.name}: {len(members)} files, nothing local")


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
        _audit_artifacts(target)
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
        # Audited again against the files about to be uploaded: the earlier
        # pass ran before the release commit, and dist/ is not immutable.
        _audit_artifacts(target)
        artifacts = [str(path.relative_to(ROOT)) for path in _artifacts_for(target)]
        environment = _publishing_environment()
        _run(("uv", "publish", *artifacts), env=environment)
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
            ),
            env=environment,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"release failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
