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
upstream base. When empty, nothing is subtracted and the policy gate fails on any denied
license the base image carries, so an image is either baselined or skips these stages.

## Base SBOM corpus & drift

`base_sboms/manifest.json` maps each `(from_image, baseline_image)` pair to a slim
CycloneDX baseline. `capture_baseline_sbom.py` resolves digests, verifies the layer-prefix
invariant, syft-scans the baseline, and writes the slim SBOM. `check_drift.py` runs on every
PR and on a daily cron (`.github/workflows/compliance-base-drift.yml`); it fails if a recorded
digest moved or the layer-prefix invariant no longer holds, which means a vendor silently
switched a base image and the corpus must be re-captured.

A tag bump is the other way a baseline goes stale, and the upsert key includes `from_tag`, so
re-capturing on a new tag appends rather than replaces. Pass `--prune-superseded` to drop the
old tag's rows and the SBOM files nothing references any more; leaving them behind eventually
fails the drift check, which re-resolves every row against the registry.

For TRT-LLM this is automated: `.github/workflows/auto-dep-upgrade-trigger.yml` re-captures both
architectures against the new tag and commits the refreshed corpus alongside the version bump, so
`runtime_image_tag` and `baseline_sbom` never diverge. A capture that fails pushes nothing.

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

- **PR targeting `main`** → the PR's true merge-base (fork point), walking backward
  on first-parent history only when that commit lacks this container's
  `compliance-<sha>-<container>` artifact. Advancing `main` alone does not move
  this starting point; it moves when the PR is rebased or updated with `main`.
- **Post-merge on `main`** → the same walk, starting from the previous commit on `main`.
- **PR targeting / post-merge on `release/*`** → the release tag `vX.Y.Z` that is
  the highest one strictly older than the current version (semver order first —
  `1.3.0` beats `1.2.5` for a `1.3.1` build regardless of publish date). Among
  tags sharing an `X.Y.Z` base (e.g. `v1.2.3-nemo-3` vs `v1.2.3-minimax`), the
  one published later wins.
- **Nightly** → the previous successful scheduled run of the nightly workflow
  (found via the Actions API); manual runs are excluded and the selected run's
  commit names the baseline artifact.

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

## Shipped-file completeness

The frontend and vLLM / SGLang / TensorRT-LLM runtime builds supplement the
ecosystem generators with a pinned Syft scan of the pre-runtime filesystem. `shipped.py` adds components missing from the inventory
and feeds them through the same NOTICES writers, merged CSV, and license-policy
gate. The separate CI audit scans the final image again and compares it with the
build inventory. Discovery is not a license approval.

This covers files that an installed-wheel or dpkg inventory cannot fully describe:

| Shipped material | Attribution and evidence |
|---|---|
| Rust dependencies embedded in copied executables, such as `uv` | Discover the executable's auditable dependency graph. Fetch each exact published crate, verify its checksum against the static crates.io index (and Cargo.lock when present), then read that archive's manifest and upstream license/notice files. |
| Go dependencies in copied executables and shared libraries | Discover Go build information and resolve the exact module's upstream license files. Record each module separately from the executable's top-level native attribution. |
| Source Cargo.lock dependencies | Retain them in the inventory, including development and foreign-platform dependencies. Shipping source still requires attribution; a source declaration is not evidence that a dependency is linked into a binary. |
| Nested Python distributions | Walk nested `.dist-info` directories and use each distribution's own metadata and notices. Scan both the runtime venv and shipped system package directories. |
| Local source projects and synthetic Go main packages | Use a matching local Cargo manifest, including workspace-inherited licensing. When a local project has no independent declaration, require an exact wheel-file ownership record before using the owning project's license. This never covers its registry dependencies. |
| Bundled launchers, solvers, and test/template fixtures | Require an exact wheel ownership record and a reviewed resource-specific license rule. Keep separate inventory entries; never assume a parent's license covers every bundled file. Unrecognized resources remain `UNKNOWN`. |

For example, PuLP's MIT package license does not cover the bundled CBC solver's
EPL-1.0 license. The collector reads CBC's adjacent `coin-license.txt`; it does
not borrow the license of an unrelated Rust crate also named `cbc`. Likewise,
distlib's launcher uses the BSD-2-Clause notice from its own C source, rather
than the parent distribution's Python license. The upstream launcher notice is
retained in `resource_notices/`.

Inventory matching is scoped to the ecosystem. Python distribution names use
PEP 503 separator normalization (`ruamel.yaml` and `ruamel-yaml`); Go module paths
and other ecosystem names retain their own identity. The completeness audit
continues to compare names within an ecosystem, while supplemental generation
keeps distinct observed versions and validates their licenses individually.

### License evidence and source retention

`/legal/shipped-provenance.json` records the added component, discovery cataloger,
filesystem locations, license evidence, source URL, and downloaded-source checksum.
Unfiltered scanner evidence remains available even where license resolution uses
a more specific declaration. For example, a Go module's license-generation source
can contain GPL template strings without relicensing that module: resolution uses
actual LICENSE/COPYING/NOTICE files, rather than treating every `license_*.go`
string as a governing license. Distinct governing license detections are combined
with `AND`, not silently converted into an `OR` choice.

`/legal/license-review.json` flags reciprocal, restricted, and unknown licenses
across the merged above-baseline inventory, including licenses the policy allows.
An allowed MPL-2.0 dependency remains non-permissive; policy acceptance is not a
claim that source and notice obligations disappear. SPDX `OR` alternatives stay
intact for review. The existing policy gate still rejects denied or unknown
licenses. A component-specific exception permits the native CBC executable under
EPL-1.0 because it is distributed as a separate program. It does not permit EPL
for other components, the unrelated Rust `cbc` crate, or an unknown solver license.
The solver's own notice and provenance remain required, and its EPL license stays
visible in the license-review artifact. This exception records the distribution
rationale, not a claim of independent legal sign-off.

Crate sources whose expressions are not wholly recognized as permissive, and Go
module sources with such expressions, are retained under
`/sboms/supplemental-sources` in the licenses stage. The existing release
`sources_collect` stage includes these archives and their provenance in
`sources.zip`. Source-archival enablement and publication remain the existing
release pipeline's responsibility; a PR build alone does not publish sources.
The scanner and registry cache are build helpers, and the runtime image continues
to copy only `/legal` from this stage.

### Extending collection

Prefer upstream metadata and exact-version source notices over a growing list of
package names. Add a resource rule only when the file's owning distribution and
specific resource family have been verified, cite its upstream licensing source,
and test both the accepted ownership and an unrelated or unowned file. Do not add
blanket wheel-directory coverage, omit source/test/Windows files because they are
not executed on Linux, suppress scanner errors, or loosen the license policy to
make a build pass. A newly discovered denied license is a separate remediation
or approval decision.

Enable this path only when the licenses stage represents the filesystem being
shipped. The Planner currently inventories a builder and copies selected paths
into its distroless runtime, so scanning that builder would falsely attribute
build-only dependencies; it retains its existing collection path. A new target
must provide its actual runtime tree before enabling supplemental collection.

Both architectures execute collection against their own build filesystem and
baseline. The independent CI audit currently covers AMD64; it does not establish
ARM64 completeness. Collection adds one filesystem scan and cached registry reads
to the compliance stage, leaving inference code and dependency versions unchanged.
