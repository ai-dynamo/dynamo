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

## License detection

Detection is conservative: only unambiguous matches get an SPDX identifier. `UNKNOWN`
fails the policy gate, surfacing the package for an explicit override in
`license_overrides.yaml` or a signed-off `[[exceptions]]` entry in `policy/licenses.toml`.
