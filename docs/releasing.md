# Releasing to PyPI

Releases are cut by pushing a version tag. A GitHub Actions workflow
([`.github/workflows/release.yml`](https://github.com/CodeHalwell/ADAPT-Agent/blob/main/.github/workflows/release.yml))
runs the full test matrix, builds and validates the distributions with
[uv](https://docs.astral.sh/uv/), publishes them to PyPI with `uv publish`, and
opens a GitHub Release.

```bash
git tag -a v0.3.0 -m "v0.3.0"
git push origin v0.3.0
```

Authentication uses **PyPI Trusted Publishing** (OIDC): GitHub proves the
workflow's identity to PyPI directly, so there is no API token stored in this
repository and nothing to rotate. That does require a one-time setup on PyPI —
see below.

The workflow installs uv with `astral-sh/setup-uv`, which tracks the latest
release. Attestation upload needs a recent uv; if you ever pin an older version
there, publishes will still succeed but without PEP 740 provenance.

## One-time setup

### 1. Create the trusted publisher on PyPI

`adapt-agent` has not been published yet, so the first release uses a **pending
publisher** (a trusted publisher for a project that does not exist on PyPI yet;
it becomes a normal one the moment the first upload lands).

Go to [PyPI → Your projects → Publishing](https://pypi.org/manage/account/publishing/)
and add a pending publisher with exactly these values:

| Field | Value |
| --- | --- |
| PyPI project name | `adapt-agent` |
| Owner | `CodeHalwell` |
| Repository name | `ADAPT-Agent` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

The environment name matters: the workflow's publish job runs in a GitHub
environment called `pypi`, and PyPI checks it. A mismatch here is the most
common cause of a first release failing with an authentication error.

### 2. Create the GitHub environment

In the repository, go to **Settings → Environments → New environment** and
create one named `pypi`. Nothing else is required, but this is a good place to
add a **required reviewer** so a human has to approve the publish step — the
tag then triggers the pipeline and waits for that approval before anything
reaches PyPI.

### 3. (Optional) The same again for TestPyPI

To rehearse releases, repeat step 1 on
[test.pypi.org](https://test.pypi.org/manage/account/publishing/) with
environment name `testpypi`, and create a matching `testpypi` GitHub
environment. Then use **Actions → Release → Run workflow** and pick
`testpypi` to run the entire pipeline against the test index without touching
the real one.

The publish target itself is already configured — `pyproject.toml` declares
TestPyPI under `[[tool.uv.index]]`, which is what `uv publish --index testpypi`
resolves. It is marked `explicit = true`, so it never participates in
dependency resolution.

## Cutting a release

1. **Update the version.** Edit `__version__` in `adapt_agent/__init__.py`.
   That is the single source of truth — `pyproject.toml` reads it via
   `[tool.setuptools.dynamic]`, and the workflow refuses to publish if the tag
   disagrees with it.
2. **Update `CHANGELOG.md`.** Rename the `[Unreleased]` heading to the new
   version with today's date, and open a fresh `[Unreleased]` section above it.
3. **Merge that to `main`** through the usual PR flow, so CI has run on the
   exact commit you are about to tag.
4. **Tag and push:**

   ```bash
   git checkout main && git pull
   git tag -a v0.3.0 -m "v0.3.0"
   git push origin v0.3.0
   ```

5. **Watch the run** under Actions → Release. If the `pypi` environment has a
   required reviewer, approve the publish step when it pauses.

Versions follow [Semantic Versioning](https://semver.org/). A PEP 440
pre-release suffix (`v0.3.0rc1`, `v0.3.0b1`, `v0.3.0.dev1`) is detected
automatically and marks the GitHub Release as a pre-release.

## What the workflow checks before publishing

Publishing is irreversible — a version number on PyPI can never be reused, even
after deletion — so the pipeline front-loads everything that could go wrong:

| Stage | Check |
| --- | --- |
| `version` | The tag matches `adapt_agent.__version__`, so `v0.3.0` cannot ship `0.2.0`. |
| `ci` | The same lint, type-check and 3.10–3.14 test matrix that guards `main`, re-run on the tagged commit. |
| `build` | Built with `uv build --no-sources`, then `twine check --strict` (via `uvx`) on both artifacts. |
| `build` | The built filenames carry the expected version. |
| `build` | The wheel really contains `SKILL.md` and its reference files, `py.typed`, and both console scripts; the sdist carries the skill too. |
| `build` | A clean `uv venv` installs the wheel, runs `adapt --version`, and runs `adapt install skill` end to end. |
| `publish-pypi` | `uv publish --trusted-publishing always` in the `pypi` environment. PEP 740 attestations are generated and uploaded by default, and `--check-url` makes a re-run skip files already on the index. |
| `github-release` | Creates the GitHub Release with generated notes and attaches the exact artifacts that went to PyPI. |

The skill and console-script assertions exist because those are *packaging
data*, not code: if a `package-data` glob stopped matching, the tests would
still pass while the published wheel silently lost `adapt install skill`. The
release refuses to publish such a wheel.

## Troubleshooting

**`Trusted publishing exchange failure` / 403 on upload.** The publisher on
PyPI does not match the run. Re-check owner, repository, workflow filename
(`release.yml`) and environment (`pypi`) — all four must agree exactly.

**`Tag vX.Y.Z does not match adapt_agent.__version__`.** Bump `__version__`
and re-tag, or delete the tag and push the right one:

```bash
git tag -d v0.3.0 && git push origin :refs/tags/v0.3.0
```

**`File already exists` on upload.** That version was already published; PyPI
never allows reuse. Bump to the next patch version and tag again.

**The run published nothing.** Check that the tag matched `v*` — the workflow
only triggers on tags with a leading `v`.

## Publishing manually

Should the workflow ever be unavailable, the same artifacts can be produced and
uploaded by hand — though this bypasses every check above, so prefer the tag:

```bash
uv build --no-sources
uvx twine check --strict dist/*
uv publish --dry-run               # verify the upload without sending anything
uv publish --token "$PYPI_TOKEN"   # needs a PyPI API token
```

`uv publish --dry-run` is also the quickest way to sanity-check a build locally
before tagging: it validates the files against the index without uploading and
needs no credentials. It needs a recent uv — the flag is absent from older
releases (0.7.x), where `uvx twine check --strict dist/*` alone is the
equivalent local check.
