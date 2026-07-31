#!/usr/bin/env python3
"""Refuse to publish an artifact carrying something that must stay local.

The same audit the local release runner performs, as a standalone command so
the GitHub workflow can run it too. Publishing happens from two places, and a
guard that only one of them honours is not a guard — the 0.6.0 sdist that
shipped a local `.env` would have travelled just as far through Actions.

Exit status is what matters: non-zero stops the job before the upload step.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.release_manifest import project_version, unpublishable_members

ROOT = Path(__file__).resolve().parents[1]


def archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def artifacts_in(directory: Path, *, version: str | None = None) -> list[Path]:
    """The distributions in a directory, optionally narrowed to one version."""
    found = (path for path in directory.glob("*") if path.name.endswith((".whl", ".tar.gz")))
    if version is None:
        return sorted(found)
    sdist_suffix = f"-{version}.tar.gz"
    wheel_marker = f"-{version}-"
    return sorted(
        path for path in found if path.name.endswith(sdist_suffix) or wheel_marker in path.name
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        # A directory named on the command line is audited whole: it is the
        # post-publish check, where everything present was just downloaded on
        # purpose and auditing less than all of it would miss the point.
        directory, version = Path(arguments[0]), None
    else:
        # `dist/` accumulates every artifact ever built here, and last year's
        # sdist cannot be fixed by this release. Auditing it anyway makes the
        # command fail permanently, and a guard that always fails is one whose
        # output stops being read — which is the failure this guard exists to
        # prevent, arriving by a different door.
        directory = ROOT / "dist"
        version = project_version((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    artifacts = artifacts_in(directory, version=version)
    if not artifacts:
        scope = directory if version is None else f"{directory} for version {version}"
        print(f"no artifacts to audit in {scope}", file=sys.stderr)
        return 1

    failed = False
    for path in artifacts:
        offenders = unpublishable_members(archive_members(path))
        if offenders:
            failed = True
            print(f"{path.name}: refusing to publish {offenders}", file=sys.stderr)
        else:
            print(f"{path.name}: nothing local")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
