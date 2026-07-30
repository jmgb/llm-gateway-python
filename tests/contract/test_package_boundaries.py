"""Architectural guarantees. These are the promises consumers rely on.

If one of these breaks, the package has stopped being safe to depend on from
seven repositories at once.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src" / "llm_gateway"

FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "openai",
    "groq",
    "google",
    "anthropic",
    "httpx",
    "requests",
}

CONSUMER_PACKAGES = {"app", "src", "backend", "ai_services"}


def _source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.rglob("*.py"))


def _module_level_imports(path: Path) -> set[str]:
    """Imports executed at import time, ignoring those inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def test_the_package_imports_with_no_provider_extra_installed() -> None:
    """The whole point of optional extras: importing must never need one."""
    result = subprocess.run(
        [sys.executable, "-c", "import llm_gateway; print(llm_gateway.__name__)"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "llm_gateway" in result.stdout


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_module_imports_a_provider_sdk_at_import_time(path: Path) -> None:
    offenders = _module_level_imports(path) & FORBIDDEN_TOP_LEVEL_IMPORTS

    assert not offenders, f"{path.name} imports {offenders} at module level"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_module_imports_a_consuming_application(path: Path) -> None:
    offenders = _module_level_imports(path) & CONSUMER_PACKAGES

    assert not offenders, f"{path.name} imports consumer package {offenders}"


def test_no_module_reads_the_environment() -> None:
    """Credentials belong to the application; the package never goes looking."""
    offenders = [
        path.name
        for path in _source_files()
        if "os.environ" in path.read_text(encoding="utf-8")
        or "getenv" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"modules reading the environment: {offenders}"


def test_the_public_api_is_explicit() -> None:
    """Ordering is ruff's job (RUF022); correctness is this test's job."""
    import llm_gateway

    exported = llm_gateway.__all__
    duplicates = {name for name in exported if exported.count(name) > 1}

    assert duplicates == set(), f"duplicated exports: {duplicates}"
    for name in exported:
        assert hasattr(llm_gateway, name), f"{name} is exported but missing"


def test_no_module_exceeds_the_size_budget() -> None:
    """The failure mode this package exists to prevent is a 2000-line function."""
    too_long = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in _source_files()
        if len(path.read_text(encoding="utf-8").splitlines()) > 500
    }

    assert too_long == {}, f"modules over 500 lines: {too_long}"
