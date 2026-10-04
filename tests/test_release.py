"""Pure release-file transformations used by the local release runner."""

from __future__ import annotations

import io
import subprocess
import sys
import tarfile
import tomllib
from collections.abc import Callable
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import release
from scripts.audit_dist import artifacts_in
from scripts.audit_dist import main as audit_dist_main
from scripts.release_manifest import (
    bump_version,
    local_env_values,
    promote_unreleased,
    publishing_env_values,
    replace_lock_version,
    replace_project_version,
    unpublishable_members,
    unreleased_body,
)

ROOT = Path(__file__).resolve().parents[1]

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


class TestTheChangelogIsReadAsWritten:
    """An exact `\\n\\n` after the heading made CRLF read as "no [Unreleased]".

    That message sends the reader looking for notes that are plainly there.
    """

    def test_crlf_line_endings_still_find_the_notes(self) -> None:
        text = CHANGELOG.replace("\n", "\r\n")

        assert unreleased_body(text).startswith("### Changed")

    def test_a_missing_blank_line_under_the_heading_still_finds_the_notes(self) -> None:
        text = CHANGELOG.replace("## [Unreleased]\n\n", "## [Unreleased]\n")

        assert unreleased_body(text).startswith("### Changed")

    def test_a_crlf_changelog_is_promoted_without_mixing_line_endings(self) -> None:
        updated = promote_unreleased(CHANGELOG.replace("\n", "\r\n"), "0.6.0", "2026-07-31")

        assert "## [Unreleased]\r\n\r\n## [0.6.0] — 2026-07-31\r\n\r\n### Changed" in updated
        assert "\n" not in updated.replace("\r\n", "")

    def test_an_empty_section_is_still_refused(self) -> None:
        text = CHANGELOG.replace("### Changed\n\n- Add a release change.\n\n", "")

        with pytest.raises(ValueError, match="Unreleased"):
            unreleased_body(text)


class TestNothingLocalIsPublishable:
    """An upload cannot be taken back, so the audit runs before it, not after.

    0.6.0 published an sdist containing the maintainer's `.env`, and with it a
    PyPI token, because hatchling packages the project directory minus whatever
    VCS ignored at build time — and the ignore rule was committed three minutes
    after the build. Deleting the release would not have unpublished the file:
    it was already mirrored. Only refusing to upload it would have.
    """

    def test_it_catches_the_file_that_actually_leaked(self) -> None:
        members = [
            "neutral_llm_gateway-0.6.0/.env",
            "neutral_llm_gateway-0.6.0/src/llm_gateway/gateway.py",
        ]

        assert unpublishable_members(members) == ["neutral_llm_gateway-0.6.0/.env"]

    def test_an_ordinary_sdist_has_nothing_to_report(self) -> None:
        members = [
            "neutral_llm_gateway-0.7.0/PKG-INFO",
            "neutral_llm_gateway-0.7.0/pyproject.toml",
            "neutral_llm_gateway-0.7.0/.python-version",
            "neutral_llm_gateway-0.7.0/.gitignore",
            "neutral_llm_gateway-0.7.0/src/llm_gateway/__init__.py",
            "neutral_llm_gateway-0.7.0/tests/test_release.py",
        ]

        assert unpublishable_members(members) == []

    def test_a_wheel_keeps_its_dist_info(self) -> None:
        """`.dist-info` is metadata the wheel format requires, not a dotfile."""
        members = [
            "llm_gateway/__init__.py",
            "neutral_llm_gateway-0.7.0.dist-info/METADATA",
            "neutral_llm_gateway-0.7.0.dist-info/RECORD",
        ]

        assert unpublishable_members(members) == []

    @pytest.mark.parametrize(
        "name",
        (
            "pkg-1.0/.envrc",
            "pkg-1.0/.pypirc",
            "pkg-1.0/deploy/id_rsa",
            "pkg-1.0/certs/server.pem",
            "pkg-1.0/.git/config",
            "pkg-1.0/.claude/settings.json",
        ),
    )
    def test_it_refuses_the_next_one_too(self, name: str) -> None:
        """Blunt on purpose: an unexpected dotfile is refused before anyone names it."""
        assert unpublishable_members([name]) == [name]

    @pytest.mark.parametrize(
        "name",
        (
            "pkg-1.0/credentials.json",
            "pkg-1.0/config/prod.env",
            "pkg-1.0/app.env.local",
            "pkg-1.0/.env.example",
            "pkg-1.0/keys/id_ecdsa",
            "pkg-1.0/keys/id_ed25519_sk",
            "pkg-1.0/putty.ppk",
            "pkg-1.0/release.jks",
            "pkg-1.0/android.keystore",
            "pkg-1.0/AuthKey_ABC123.p8",
            "pkg-1.0/client_secret_123.apps.googleusercontent.com.json",
            "pkg-1.0/service-account-prod.json",
            "pkg-1.0/service_account.json",
            "pkg-1.0/tests\\.env",
        ),
    )
    def test_a_credential_without_a_leading_dot_is_refused_by_its_name(self, name: str) -> None:
        assert unpublishable_members([name]) == [name]

    @pytest.mark.parametrize(
        "name",
        (
            "pkg-1.0/keys/id_ed25519.pub",
            "pkg-1.0/env.example",
            "pkg-1.0/src/pkg/environment.py",
            "pkg-1.0/src/pkg/credentials_test_data.py",
            "pkg-1.0/src/pkg/service_account_docs.md",
        ),
    )
    def test_a_name_that_only_resembles_a_credential_is_published(self, name: str) -> None:
        assert unpublishable_members([name]) == []

    def test_nothing_the_sdist_include_list_selects_is_refused(self) -> None:
        """The broadened rules must not block the release they exist to protect.

        Lists the tracked files the include list selects, so a new pattern that
        matches a real file in the project fails here rather than at release
        time. Tracked, not on disk: an ignored local `.env` is never packaged,
        and must not fail the suite of whoever happens to keep one.
        """
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        listed = subprocess.run(
            ["git", "ls-files", "--", *(entry[1:] for entry in include)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if listed.returncode != 0:
            pytest.skip("not a git checkout, so there is no tracked tree to compare")
        shipped = listed.stdout.splitlines()

        assert len(shipped) > len(include)
        assert unpublishable_members(shipped) == []


class TestTheLocalEnvIsReadNotEvaluated:
    """A `.env` may hold the publishing token, so parsing it must be boring."""

    def test_it_reads_a_plain_assignment(self) -> None:
        assert local_env_values("UV_PUBLISH_TOKEN=abc\n") == {"UV_PUBLISH_TOKEN": "abc"}

    def test_it_tolerates_comments_blanks_and_export(self) -> None:
        text = "# a comment\n\nexport UV_PUBLISH_TOKEN=abc\nGH_TOKEN='xyz'\n"

        assert local_env_values(text) == {"UV_PUBLISH_TOKEN": "abc", "GH_TOKEN": "xyz"}

    def test_a_value_containing_equals_is_kept_whole(self) -> None:
        """Base64 and JWT-shaped tokens end in `=`; splitting again truncates them."""
        assert local_env_values("T=a=b==")["T"] == "a=b=="

    def test_a_line_without_an_assignment_is_ignored_not_guessed(self) -> None:
        assert local_env_values("just some prose\n") == {}


class TestTheSdistDeclaresWhatItShips:
    def test_the_sdist_target_lists_its_contents_explicitly(self) -> None:
        """Without this list hatchling falls back to "everything VCS did not ignore".

        That default is what shipped a `.env`, and it fails silently: the
        artifact builds, uploads and installs perfectly well.
        """
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        include = config["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]

        assert include, "the sdist must declare an explicit include list"
        assert "/src" in include
        assert all(entry.startswith("/") for entry in include), (
            "unanchored patterns match at any depth, which is how a stray file gets in"
        )


class TestTheDefaultAuditScopeIsTheVersionBeingReleased:
    """`dist/` accumulates; the release does not.

    Auditing every artifact ever built means the command fails on last year's
    sdist and keeps failing, and an alarm that is always on is one nobody reads.
    A directory named explicitly is audited whole — that is the post-publish
    check, where everything present was just downloaded on purpose.
    """

    @staticmethod
    def _touch(directory: Path, *names: str) -> None:
        for name in names:
            (directory / name).write_bytes(b"")

    def test_a_version_selects_only_its_own_artifacts(self, tmp_path: Path) -> None:
        self._touch(
            tmp_path,
            "pkg-0.7.0.tar.gz",
            "pkg-0.7.0-py3-none-any.whl",
            "pkg-0.6.0.tar.gz",
            "pkg-10.7.0.tar.gz",
            "pkg-0.7.0rc1.tar.gz",
        )

        found = [path.name for path in artifacts_in(tmp_path, version="0.7.0")]

        assert found == ["pkg-0.7.0-py3-none-any.whl", "pkg-0.7.0.tar.gz"]

    def test_without_a_version_everything_present_is_audited(self, tmp_path: Path) -> None:
        self._touch(tmp_path, "pkg-0.7.0.tar.gz", "pkg-0.6.0.tar.gz")

        assert len(artifacts_in(tmp_path)) == 2

    def test_a_non_artifact_is_never_selected(self, tmp_path: Path) -> None:
        self._touch(
            tmp_path,
            "pkg-0.7.0.tar.gz",
            "pkg-0.7.0.tar.gz.asc",
            "pkg-0.7.0.tar.gz.part",
            "pkg-0.7.0.txt",
            "notes-0.7.0.md",
        )

        assert [path.name for path in artifacts_in(tmp_path)] == ["pkg-0.7.0.tar.gz"]


class TestTheStandaloneAuditorStopsTheJob:
    """The workflow reads nothing but the exit status, so it is the contract."""

    @staticmethod
    def _sdist(directory: Path, name: str, *members: str) -> None:
        with tarfile.open(directory / name, "w:gz") as archive:
            for member in members:
                info = tarfile.TarInfo(member)
                info.size = 0
                archive.addfile(info, io.BytesIO(b""))

    def test_an_archive_carrying_a_local_file_fails(self, tmp_path: Path) -> None:
        self._sdist(tmp_path, "pkg-0.7.0.tar.gz", "pkg-0.7.0/src/x.py", "pkg-0.7.0/.env")

        assert audit_dist_main([str(tmp_path)]) == 1

    def test_a_clean_archive_passes(self, tmp_path: Path) -> None:
        self._sdist(tmp_path, "pkg-0.7.0.tar.gz", "pkg-0.7.0/src/x.py", "pkg-0.7.0/README.md")

        assert audit_dist_main([str(tmp_path)]) == 0

    def test_an_empty_directory_fails_rather_than_reporting_success(self, tmp_path: Path) -> None:
        """Nothing audited is not the same as nothing wrong, and the difference is an upload."""
        assert audit_dist_main([str(tmp_path)]) == 1


class TestOnlyPublishingCredentialsAreExported:
    """A local `.env` is not a publishing config; it is whatever the machine holds.

    Handing all of it to `uv publish` and `gh` puts every unrelated key in the
    environment of two processes that have no use for them.
    """

    ENV = "UV_PUBLISH_TOKEN=publish-me\nGOOGLE_API_KEY=unrelated\nGH_TOKEN=gh\n"

    def test_the_publishing_token_is_kept(self) -> None:
        assert publishing_env_values(self.ENV)["UV_PUBLISH_TOKEN"] == "publish-me"

    def test_everything_the_upload_does_not_need_is_dropped(self) -> None:
        assert set(publishing_env_values(self.ENV)) == {"UV_PUBLISH_TOKEN", "GH_TOKEN"}


def _write_project(directory: Path, monkeypatch: pytest.MonkeyPatch) -> dict[Path, str]:
    """The three files a release rewrites, as copies the test can watch."""
    originals = {
        directory / "pyproject.toml": PROJECT,
        directory / "uv.lock": LOCK,
        directory / "CHANGELOG.md": CHANGELOG,
    }
    for path, content in originals.items():
        path.write_text(content)
    monkeypatch.setattr(release, "PROJECT_FILE", directory / "pyproject.toml")
    monkeypatch.setattr(release, "LOCK_FILE", directory / "uv.lock")
    monkeypatch.setattr(release, "CHANGELOG_FILE", directory / "CHANGELOG.md")
    return originals


class TestAnInterruptedReleaseLeavesNothingBehind:
    def test_ctrl_c_during_the_checks_restores_all_three_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The checks take minutes, which is when a release gets interrupted.

        Restoring only on `Exception` let a KeyboardInterrupt leave the version
        bumped and the changelog promoted, ready to be committed by accident.
        """
        originals = _write_project(tmp_path, monkeypatch)
        monkeypatch.setattr(release, "_ensure_ready", lambda target: None)

        def interrupted() -> None:
            raise KeyboardInterrupt

        monkeypatch.setattr(release, "_checks", interrupted)

        with pytest.raises(KeyboardInterrupt):
            release.main(["--version", "0.6.0", "--date", "2026-07-31"])

        assert {path: path.read_text() for path in originals} == originals


class TestTheReleaseDateIsAnIsoDate:
    @pytest.mark.parametrize("value", ("04/10/2026", "20261004", "2026-W40-1", "2026-02-30"))
    def test_anything_but_yyyy_mm_dd_is_refused_before_it_reaches_the_changelog(
        self, value: str
    ) -> None:
        with pytest.raises(SystemExit):
            release._parser().parse_args(["--version", "0.6.0", "--date", value])

    def test_a_real_date_is_kept_verbatim(self) -> None:
        args = release._parser().parse_args(["--version", "0.6.0", "--date", "2026-07-31"])

        assert args.date == "2026-07-31"


class TestTheRepositoryIsCheckedBeforeAnythingIsWritten:
    """Each refusal here used to arrive after the release commit existed.

    A taken tag failed at `git tag`, a stale `main` at `git push`, and both
    left a local commit and tag to unpick by hand.
    """

    HEAD = "a" * 40

    def _git(self, **overrides: str) -> Callable[..., str]:
        outputs = {
            "branch": "main",
            "status": "",
            "tag": "",
            "ls-remote": f"{self.HEAD}\trefs/heads/main",
            "rev-parse": self.HEAD,
        } | overrides

        def git(*arguments: str) -> str:
            return outputs[arguments[0]]

        return git

    def test_a_clean_main_in_step_with_origin_is_ready(self) -> None:
        release._ensure_ready("0.6.0", git=self._git())

    def test_a_branch_other_than_main_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="main branch"):
            release._ensure_ready("0.6.0", git=self._git(branch="feature"))

    def test_a_dirty_tree_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match="clean"):
            release._ensure_ready("0.6.0", git=self._git(status=" M CHANGELOG.md"))

    def test_a_version_already_tagged_locally_is_refused(self) -> None:
        with pytest.raises(RuntimeError, match=r"v0\.6\.0 already exists locally"):
            release._ensure_ready("0.6.0", git=self._git(tag="v0.6.0"))

    def test_a_version_already_tagged_on_origin_is_refused(self) -> None:
        remote = f"{self.HEAD}\trefs/heads/main\n{'b' * 40}\trefs/tags/v0.6.0"

        with pytest.raises(RuntimeError, match=r"v0\.6\.0 already exists on origin"):
            release._ensure_ready("0.6.0", git=self._git(**{"ls-remote": remote}))

    @pytest.mark.parametrize("remote_main", ("b" * 40, ""), ids=("moved", "missing"))
    def test_a_head_that_is_not_origins_main_is_refused(self, remote_main: str) -> None:
        """Ahead, behind or diverged: the release commit would not fast-forward cleanly."""
        remote = f"{remote_main}\trefs/heads/main" if remote_main else ""

        with pytest.raises(RuntimeError, match="not origin's main"):
            release._ensure_ready("0.6.0", git=self._git(**{"ls-remote": remote}))

    def test_the_dry_run_makes_the_same_checks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dry run that passes where the real run would refuse is a false all-clear."""
        _write_project(tmp_path, monkeypatch)
        checked: list[str] = []
        monkeypatch.setattr(release, "_ensure_ready", checked.append)

        assert release.main(["--version", "0.6.0", "--dry-run"]) == 0
        assert checked == ["0.6.0"]
