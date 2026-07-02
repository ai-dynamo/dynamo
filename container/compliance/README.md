# Container Compliance Tooling

Inline pipeline that generates per-image license NOTICES at build time, gates the
build on a license policy, and ships a base-image SBOM corpus that drives both
baseline subtraction (so NOTICES attributes only what we redistribute on top of the
upstream base) and a CI drift check that fails fast when a base image moves.

There is no separate extraction job anymore. Every shipped image builds a `licenses`
stage (see `../templates/compliance.Dockerfile`) that runs the generators against its
own filesystem; CI extracts `/legal` and `/sboms` from that stage with a warm cache.

## Layout

| Path | Purpose |
|------|---------|
| `generators/` | Per-ecosystem NOTICES generators: `rust`, `python`, `dpkg`, `go`, `native`, plus `common.py` (shared `Component` + `render_notices`) and `__main__.py` (orchestrator). |
| `policy/` | `licenses.toml` (allow/deny SPDX lists + per-package `[[exceptions]]`) and `validate.py` (the build-failing policy gate). |
| `base_sboms/` | The baseline corpus: `manifest.json`, slim CycloneDX `*.cdx.json`, `capture_baseline_sbom.py`, and `check_drift.py`. |
| `osrb/` | Release-time OSRB submission packager (`package.py`) and its `distribution.yaml` / `linkage.yaml`. |
| `overrides.py`, `license_overrides.yaml` | Authoritative `(ecosystem, name) → SPDX` overrides consulted before automatic detection. |
| `native_packages.yaml` | From-source / binary components attributed via a hand-curated overlay (optional `license_text_path`). |
| `verify_sbom_diff.py` | Cross-checks generated NOTICES against the base SBOM corpus; fails on drift. |
| `resolve_diff_base.py` | Picks the baseline commit to diff a build's OSRB CSV against (PR→main / post-merge→main / release branch→prior release tag). |
| `diff_osrb_csv.py` | Diffs two OSRB CSVs into a change-typed `*.diff.csv` (additions, version bumps, license changes, removals). |

## How it works

The vllm/sglang/trtllm runtime images use the inline system below;
frontend/planner stay on the legacy `shared-compliance.yml` scan until they
migrate. For each runtime image, `templates/compliance.Dockerfile` adds a
`licenses` stage that `FROM`s the image's pre-compliance stage (`pre_runtime`)
and runs:

```bash
python3 -m compliance.generators \
    --ecosystem python,rust,dpkg[,native] \
    --venv ${VIRTUAL_ENV} \
    --output-dir /legal \
    ${BASELINE_SBOM_FILE:+--subtract-sbom /opt/compliance/base_sboms/${BASELINE_SBOM_FILE}}
```

Each generator emits a `NOTICES-<Ecosystem>.txt` (with the full upstream license text per
package where available) plus a `<ecosystem>-deps.csv` into `/legal`, then the policy gate
runs `compliance.policy.validate` against `licenses.toml` and the build fails on any
denied / `UNKNOWN` license not covered by an exception. The `sboms` and `legal` scratch
stages expose `/sboms` and `/legal` for CI extraction; the final runtime stage does
`COPY --from=licenses /legal /legal` so NOTICES ship inside the image.

`BASELINE_SBOM_FILE` is rendered from `container/context.yaml`'s per-(framework, device)
`baseline_sbom` key (see `render.py:_resolve_compliance_inputs`). When set, the generators
subtract the baseline's components so NOTICES attribute only what Dynamo adds on top of the
upstream base. When empty, NOTICES cover the full image (correct but unfiltered).

## Base SBOM corpus & drift

`base_sboms/manifest.json` maps each `(from_image, baseline_image)` pair to a slim
CycloneDX baseline. `capture_baseline_sbom.py` resolves digests, verifies the layer-prefix
invariant, syft-scans the baseline, and writes the slim SBOM. `check_drift.py` runs on every
PR and on a daily cron (`.github/workflows/compliance-base-drift.yml`); it fails if a recorded
digest moved or the layer-prefix invariant no longer holds, which means a vendor silently
switched a base image and the corpus must be re-captured.

## CI integration

- **Inline extraction** — `.github/actions/compliance-extract` extracts `/legal` + `/sboms`
  from the build's warm cache, runs `verify_sbom_diff.py`, and uploads the artifacts.
- **Drift check** — `.github/workflows/compliance-base-drift.yml` validates the corpus.
- **OSRB bundle** — `osrb/package.py` stitches per-image artifacts into a release submission.

Artifacts appear in the workflow run as `compliance-<prefix>-<suffix>-legal` /
`-sboms` / `-sources`.

### Per-build OSRB diff

Alongside each `osrb-<image>-<arch>-<sha8>.csv`, the compliance-extract action
writes `osrb-<image>-<arch>-<sha8>.diff.csv` comparing this build's dependency
set against a context-dependent baseline. The diff notes four change kinds —
additions, version bumps, license changes, removals — ordered by change bucket
(additions → version+license changes → version bumps → license-only changes →
removals), then by ecosystem (`rust, python, go, dpkg, native`), then name.

The baseline (resolved by `resolve_diff_base.py`) depends on where the build runs:

- **PR targeting `main`** → the latest commit on `main`.
- **Post-merge on `main`** → the previous commit on `main`.
- **PR targeting / post-merge on `release/*`** → the release tag `vX.Y.Z` that is
  the highest one strictly older than the current version (semver order first —
  `1.3.0` beats `1.2.5` for a `1.3.1` build regardless of publish date). Among
  tags sharing an `X.Y.Z` base (e.g. `v1.2.3-nemo-3` vs `v1.2.3-minimax`), the
  one published later wins.
- **Nightly** → the previous successful run of the nightly workflow (found via
  the Actions API); its commit names the baseline artifact.

The baseline CSV is fetched by downloading the baseline commit's
`compliance-<baseSHA>-<container>` artifact (needs `actions: read`). Baselines
live only as Actions artifacts, so an expired/absent one is not fatal: the diff
then contains a single `baseline_unavailable` row and the build stays green.

Each arch also gets a self-describing `baseline/` folder next to the diff so the
archive records exactly what was compared against:

```text
linux_<arch>/
  osrb-<container>-<arch>-<sha8>.diff.csv
  baseline/
    BASELINE.md                              # selection rules + baseline SHA, commit link, generating-run link, label
    osrb-<container>-<arch>-<baseSHA8>.csv   # a copy of the baseline CSV (absent when the baseline was unavailable)
```

## License detection

Detection is conservative: only unambiguous matches get an SPDX identifier. `UNKNOWN`
fails the policy gate, surfacing the package for an explicit override in
`license_overrides.yaml` or a signed-off `[[exceptions]]` entry in `policy/licenses.toml`.
