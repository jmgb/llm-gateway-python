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

from scripts.release_manifest import unpublishable_members

ROOT = Path(__file__).resolve().parents[1]


def archive_members(path: Path) -> list[str]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    directory = Path(arguments[0]) if arguments else ROOT / "dist"
    artifacts = sorted(p for p in directory.glob("*") if p.suffix in {".whl", ".gz"})
    if not artifacts:
        print(f"no artifacts to audit in {directory}", file=sys.stderr)
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
