"""Pure version and changelog transformations for the local release runner."""

from __future__ import annotations

import re

PROJECT_NAME = "neutral-llm-gateway"
VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


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
