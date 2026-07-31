"""Pure version and changelog transformations for the local release runner."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import PurePosixPath

PROJECT_NAME = "neutral-llm-gateway"
VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")

# Names that carry a credential often enough that publishing one is never a
# deliberate act. `.env` is first for a reason: 0.6.0's sdist contained one.
SECRET_FILE_NAMES = frozenset(
    {".env", ".envrc", ".pypirc", ".netrc", ".npmrc", "id_rsa", "id_ed25519", "credentials"}
)
SECRET_FILE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
# The only dotfiles the distribution is meant to carry. `.gitignore` is not on
# the include list and lands in the sdist anyway: hatchling adds it so the
# unpacked tree rebuilds identically. Naming it here is the difference between
# a guard that passes and a guard that blocks every release until disabled.
PUBLISHABLE_DOTFILES = frozenset({".python-version", ".gitignore"})


def local_env_values(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines from a local ``.env``.

    Publishing credentials live in a file the distribution never carries: it is
    absent from the sdist allowlist, and ``unpublishable_members`` refuses any
    artifact containing one. Reading it here is what makes that arrangement
    usable — otherwise the token has to be exported by hand before every
    release, and the shortcut people take instead is to commit it.

    Deliberately small: no interpolation, no multi-line values, no shell
    semantics. A parser that evaluates its input is a second way to lose a
    credential.
    """
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def unpublishable_members(names: Iterable[str]) -> list[str]:
    """Paths in a built artifact that must never reach a package index.

    Applied to the archive before ``uv publish``, because an upload cannot be
    taken back: the file stays on mirrors and in caches long after the index
    entry is deleted, which is why a leaked credential has to be rotated rather
    than unpublished. The rule is deliberately blunt — every dotfile that is
    not explicitly expected is refused, so the *next* secret-bearing filename
    is caught without anyone having thought of it first.
    """
    offenders = set()
    for name in names:
        for part in PurePosixPath(name).parts:
            lowered = part.lower()
            if lowered in SECRET_FILE_NAMES or lowered.endswith(SECRET_FILE_SUFFIXES):
                offenders.add(name)
                break
            if part.startswith(".") and part not in PUBLISHABLE_DOTFILES:
                offenders.add(name)
                break
    return sorted(offenders)


def version_parts(version: str) -> tuple[int, int, int]:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"version must use X.Y.Z format: {version!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def bump_version(current: str, kind: str) -> str:
    """Return the next semantic version for a requested release kind."""
    major, minor, patch = version_parts(current)
    if kind == "patch":
        patch += 1
    elif kind == "minor":
        minor += 1
        patch = 0
    elif kind == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"unknown version bump: {kind!r}")
    return f"{major}.{minor}.{patch}"


def project_version(text: str) -> str:
    section = re.search(r"(?ms)^\[project\]\n(?P<body>.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ValueError("pyproject.toml has no [project] section")
    match = re.search(r'(?m)^version = "([^"]+)"$', section.group("body"))
    if match is None:
        raise ValueError("pyproject.toml has no project version")
    return match.group(1)


def replace_project_version(text: str, version: str) -> str:
    """Replace only the package version in a pyproject document."""
    version_parts(version)
    section = re.search(r"(?ms)^\[project\]\n(?P<body>.*?)(?=^\[|\Z)", text)
    if section is None:
        raise ValueError("pyproject.toml has no [project] section")
    match = re.search(r'(?m)^version = "([^"]+)"$', section.group("body"))
    if match is None:
        raise ValueError("pyproject.toml has no project version")
    start = section.start("body") + match.start(1)
    end = section.start("body") + match.end(1)
    return text[:start] + version + text[end:]


def replace_lock_version(text: str, version: str) -> str:
    """Keep the editable project entry in uv.lock aligned with pyproject.toml."""
    version_parts(version)
    blocks = re.finditer(r"(?ms)^\[\[package\]\]\n.*?(?=^\[\[package\]\]|\Z)", text)
    for block in blocks:
        content = block.group(0)
        if f'name = "{PROJECT_NAME}"' not in content:
            continue
        match = re.search(r'(?m)^version = "([^"]+)"$', content)
        if match is None:
            raise ValueError(f"uv.lock entry for {PROJECT_NAME!r} has no version")
        start = block.start() + match.start(1)
        end = block.start() + match.end(1)
        return text[:start] + version + text[end:]
    raise ValueError(f"uv.lock has no entry for {PROJECT_NAME!r}")


def unreleased_body(text: str) -> str:
    section = re.search(r"(?ms)^## \[Unreleased\]\n\n(?P<body>.*?)(?=^## \[|\Z)", text)
    if section is None or not section.group("body").strip():
        raise ValueError("CHANGELOG.md has no non-empty [Unreleased] section")
    return section.group("body").strip()


def promote_unreleased(text: str, version: str, release_date: str) -> str:
    """Move the unreleased notes under a dated version heading."""
    version_parts(version)
    body = unreleased_body(text)
    section = re.search(r"(?ms)^## \[Unreleased\]\n\n(?P<body>.*?)(?=^## \[|\Z)", text)
    assert section is not None
    replacement = f"## [Unreleased]\n\n## [{version}] — {release_date}\n\n{body}\n\n"
    return text[: section.start()] + replacement + text[section.end() :]
