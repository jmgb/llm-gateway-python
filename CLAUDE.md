# Working in this repository

`CONTRIBUTING.md` is the design contract, not boilerplate: what belongs here,
the two-consumer rule, the non-negotiables, and the checklist for adding a
provider or a price. Read it before proposing a change. This file covers only
what it does not.

`CLAUDE.md` is the shared instruction source; `AGENTS.md` links to it.
Neither file belongs in the package distribution.

## Language

**Everything that lands in the repository is written in English** — comments,
docstrings, identifiers, test names, test data, commit messages, PR
descriptions, and all Markdown. This is a published package; its readers are
not assumed to share the maintainer's first language.

Conversation with the maintainer happens in Spanish. Nothing else does.

## Commands

```bash
uv run --offline pytest        # whole suite, no network, no extras needed
uv run ruff check .
uv run ruff format --check .   # what CI gates on; drop --check to rewrite
uv run mypy                    # strict, over src, tests *and* scripts
```

`uv` is at `~/.local/bin/uv`, not inside `.venv`. `.venv/bin/python -m pytest`
works too. No provider extra is installed here, which is deliberate: it is the
environment CI runs in, and it is what makes the "extra not installed" tests
meaningful.

Two pytest settings shape how tests are written: `asyncio_mode = "auto"`, so an
async test needs no marker, and `filterwarnings = ["error"]`, so any warning —
including a deprecation raised by a dependency — fails the suite rather than
scrolling past.

## What a change has to keep in sync

Behaviour that the tests cannot state is stated in prose, and the prose is part
of the deliverable:

- `README.md` — the public API surface and the quick start.
- `docs/architecture.md` — the provider/product cut and the layering.
- `docs/pricing.md` — cost semantics, plus the per-provider table of where
  reasoning tokens are reported. A new adapter adds a row to that table, and
  `tests/contract/test_documented_normalisation.py` fails in both directions
  until it does: an adapter missing from the table, or a row for an adapter
  that no longer exists.
- `CHANGELOG.md` — under `[Unreleased]`. It is the release input, not
  bookkeeping; see below.

A price change also repins `TestPricesAndVersionMoveTogether` in
`tests/test_model_catalog.py`, which fingerprints the priced table so a rate
cannot move while `CATALOG_VERSION` stays put. Bump the version and both
constants in the same commit.

A new public symbol also goes into `src/llm_gateway/__init__.py`'s `__all__`.
`test_the_public_api_is_explicit` checks every listed name resolves and none is
duplicated; ruff's RUF022 owns the ordering.

## Commits, versions and client updates

A commit records source changes; it does not notify clients, change an already
installed package, or publish a new artifact. Clients learn about a change
only when they explicitly consume the reference that contains it:

- A client pinned to PyPI or to a package version keeps using its installed
  artifact until it upgrades.
- A client installed from a Git branch or commit sees the change only after it
  reinstalls or updates that reference.
- A client pinned to an immutable Git tag or SHA never moves automatically.

Use this release flow for changes intended for clients: update
`[project].version` in `pyproject.toml`, add the release notes to
`CHANGELOG.md`, run the checks and build, create the matching `vX.Y.Z` tag, and
publish the artifact. Consumers then upgrade their version or tag explicitly.
Do not bump the package version for every local commit; bump it when publishing
the release that makes the committed changes available to clients.

### How this project publishes

**PyPI uploads go through GitHub Actions, by Trusted Publishing (OIDC).** The
`pypi` environment and the `release.yml` publisher are already registered on
both sides, so no PyPI token exists in the upload path — which is the point:
the credential that cannot be stored is the one that cannot leak. The local
runner prepares the release; it does **not** upload it.

The four steps, in order:

```bash
uv run --offline python scripts/release.py --version 0.7.0 --dry-run   # 1
uv run --offline python scripts/release.py --version 0.7.0 --push      # 2
gh workflow run "Manual Release" -f ref=v0.7.0                         # 3
gh release create v0.7.0 --verify-tag --title v0.7.0 --notes "<0.7.0 changelog body>"  # 4
```

1. **Dry run.** Also a real check of the changelog: an empty `[Unreleased]`
   aborts, so this is not just a print.
2. **Prepare, tag, push.** Version in `pyproject.toml` and `uv.lock`, the
   changelog promoted, the checks, the build, the artifact audit, the release
   commit, the annotated `vX.Y.Z` tag, and the push. `--push` is the first
   external action.
3. **Publish.** Watch it with `gh run watch <id> --exit-status`. The workflow
   checks out the tag, builds, runs `scripts/audit_dist.py`, and uploads by
   OIDC.
4. **GitHub Release.** The workflow does not create one, and `--publish` would
   upload to PyPI a second time, so it is created by hand from the same
   changelog body consumers read.

Then verify what was actually published rather than what was built — download
it back and audit it:

```bash
python3 -c "import json,urllib.request; d=json.load(urllib.request.urlopen('https://pypi.org/pypi/neutral-llm-gateway/0.7.0/json')); [urllib.request.urlretrieve(f['url'], f['filename']) for f in d['urls']]"
python3 scripts/audit_dist.py .
```

`--bump patch|minor|major` replaces `--version` when the next number follows
from the current one. The runner requires a clean `main` branch — an untracked
file is enough to stop it — equal to `origin/main` and a tag free locally and
on `origin`, checked with `git ls-remote` even in `--dry-run`, and if a check fails it restores `pyproject.toml`,
`uv.lock` and `CHANGELOG.md` before raising, so a failed run leaves nothing
half-applied.

`--publish` still exists and uploads from this machine with
`UV_PUBLISH_TOKEN`, read from the environment or from a local `.env`. It is the
fallback for when Actions is unavailable, not the normal path. **Never use both
publishers for one version**: PyPI refuses a re-upload of the same file, and
the run fails after the tag already exists.

Run this flow when a change is ready for consumers — when the maintainer asks
to publish, release, or make the version available to another repository. Do
not publish merely because a source commit was created. If GitHub
authentication is missing, stop before changing versions or creating a tag and
report the missing credential.

### What must never reach an index

An upload cannot be undone. The file is mirrored within minutes, so deleting a
release does not unpublish what was inside it — a leaked credential has to be
rotated, not withdrawn. This has already happened here: the 0.6.0 sdist carried
a local `.env`, because hatchling packaged the project directory minus whatever
VCS ignored *at build time*, and the ignore rule was committed three minutes
after the build.

Three layers now stand between a local file and PyPI, and all three are load-bearing:

- `.gitignore` keeps it out of the repository;
- the `[tool.hatch.build.targets.sdist]` allowlist in `pyproject.toml` decides
  what is packaged — files are included by declaration, never by default;
- `scripts/audit_dist.py` refuses any artifact carrying an unexpected dotfile
  or a credential-shaped name, and runs on **both** publishing paths, between
  the build and the upload.

When adding a file to the distribution, add it to the allowlist. When a build
is rejected by the audit, the answer is never to relax the audit.

## Traps that have cost time

- **Errors are classified structurally, never by message.**
  `classify_provider_error` reads HTTP status first and the exception's class
  name second, because provider messages routinely echo request payloads and
  sometimes credentials. A test that raises `RuntimeError("429 rate limited")`
  gets a plain `ProviderError`; set `status_code` on the exception instead.

- **An adapter's `name` must equal the provider string in `models.py`.**
  `ProviderRegistry.resolve` consults the catalogue first and looks the
  declared provider up *by name*; prefixes only decide models the catalogue
  does not know. An adapter registered under the wrong name makes every
  catalogued model of that provider unreachable — it raises rather than
  falling through to another provider's prefix — and a second adapter under a
  taken name is refused. `ProviderRegistry.resolve` also refuses any Gemini
  generation other than 3 outright, direct or namespaced, so such an id raises
  `UnknownModelError` however the prefixes are set.

- **`output_tokens` means everything billed at the output rate**, reasoning
  included. `reasoning_tokens` is a breakdown of it, never an addend;
  `visible_output_tokens` is the remainder. Providers disagree about where
  thinking is counted, so each adapter normalises at its own boundary and
  `tests/contract/test_usage_normalisation.py` holds all of them to the same
  three numbers. Getting this wrong overstates cost by 1.5–2× on thinking
  models — it has happened once already.

- **The catalogue is what silently strips a request option.**
  `_request_for_model` adapts every attempt before it is sent, and nothing
  raises: an effort the target model does not declare becomes the nearest one
  it does (ties go to the cheaper), and is dropped for a model that declares
  none or is not catalogued, and a model with
  `supports_temperature=False` loses its temperature. So an entry added without
  `reasoning_efforts=(...)` turns thinking off for that model with no error
  anywhere. Adapters gate reasoning a second time — `groq.py` only sends it for
  `openai/gpt-oss-`, `gemini.py` only for `gemini-3` and the `-latest` aliases — so catalogue and adapter
  have to agree, or one accepts what the other discards.

- **Keep every `.py` file under 2000 lines**, the same as the global rule, and
  enforced by `tests/contract/test_package_boundaries.py` over
  `src/llm_gateway/` — tests and `scripts/` are on the honour system, and
  `tests/providers/test_adapters.py` is already the closest to the line.
  The same file enforces the no-SDK-at-import-time and no-environment-reads
  guarantees.

## Prose

Docstrings and comments explain *why*, and name the failure mode being
prevented rather than restating the code. Test names are full sentences
stating the guarantee — `test_an_unknown_model_is_none_not_a_guess`, not
`test_lookup_model`. Match the surrounding density; this codebase comments
sparsely but substantially.

## Reglas transversales

- After completing a task and passing the relevant checks, commit its files locally without asking. Exclude unrelated changes; push only with explicit authorization.
- Puede haber varias sesiones de agentes trabajando a la vez en este repo: comprueba `git status` y que el HEAD es tuyo antes de stagear, commitear o reescribir historia.
- Aplica siempre KISS: la solución más pequeña y simple que resuelva el caso actual.

## Secretos

Leer o ver un secreto en una sesión no obliga a rotarlo. Se rotan cada 90 días para minimizar riesgos, salvo que se expongan en un sitio público o a terceros o queden invalidados: entonces, en el momento.
