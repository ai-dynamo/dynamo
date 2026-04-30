# Container Compliance Tooling

## Overview

This tooling extracts a Software Bill of Materials (SBOM) from a built container image and derives license-attribution artifacts from it. A syft scan produces a full SPDX 2.3 SBOM; a post-processor then emits per-ecosystem `ATTRIBUTIONS-*.md` files and a tabular `attribution.csv` suitable for legal review and third-party license compliance distribution. Lockfile inputs (Cargo, Go) and a hand-maintained `native_packages.yaml` overlay cover source-built components that syft cannot license on its own.

## Output files

Each extraction run produces the following under the output directory (e.g. `/tmp/compliance-target`):

| File | Description |
|------|-------------|
| `sbom.spdx.json` | Full SBOM in SPDX 2.3 JSON, emitted by syft |
| `dpkg.tsv` | 3-column TSV `package_name\tversion\tspdx_license` for Debian/Ubuntu packages (back-compat input to `process_results.py`) |
| `python.tsv` | 3-column TSV for Python packages (back-compat input to `process_results.py`) |
| `syft.err.txt` | syft stderr capture for debugging |
| `ATTRIBUTIONS-Apt.md` | Debian/Ubuntu package attributions |
| `ATTRIBUTIONS-Python.md` | Python package attributions (ai-dynamo\*/dynamo-\* first-party packages filtered out) |
| `ATTRIBUTIONS-Rust.md` | Rust crate attributions from `Cargo.lock` (only when `--cargo-lock` given) |
| `ATTRIBUTIONS-Go.md` | Go module attributions from `go.mod` (only when `--go-mod` given) |
| `ATTRIBUTIONS-Native.md` | From-source/native-build attributions sourced from `native_packages.yaml` overlay |
| `attribution.csv` | Tabular summary across all ecosystems |
| `attribution_diff.csv` | Delta vs. base image (only when `--base-dir` given) |

## Local usage

```sh
docker buildx build --platform linux/amd64 \
  --build-arg TARGET_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.2 \
  --output type=local,dest=/tmp/compliance-target \
  --pull --no-cache-filter extractor \
  -f container/compliance/Dockerfile.extract container/compliance/
python3 container/compliance/process_results.py \
  --target-dir /tmp/compliance-target \
  --cargo-lock Cargo.lock \
  --go-mod deploy/operator/go.mod \
  --output /tmp/compliance-target/attribution.csv
```

`--no-cache-filter extractor` forces the extractor stage to re-run so a cache hit cannot return stale output from a previous target image. Pass `--base-dir <dir>` to `process_results.py` (and run a second extraction against a base image) to additionally produce `attribution_diff.csv`.

### License lookups (optional)

`Cargo.lock` and `go.mod` carry version pins but no license metadata, so by default every Rust crate and Go module in the rendered output is `UNKNOWN`. Pass `--lookup-licenses` to resolve those against the upstream registries:

| Ecosystem | Source |
|-----------|--------|
| Rust | `https://crates.io/api/v1/crates/<name>/<version>` (SPDX expression in `version.license`) |
| Go | `https://api.deps.dev/v3alpha/systems/go/packages/<module>/versions/<version>` (SPDX list in `licenses[]`) |

Results are cached in a SQLite file (default `~/.cache/dynamo-compliance/license-lookup.sqlite`, override with `--license-cache <path>`). The cache is per-machine and intentionally not committed - it can always be rebuilt from the registries. After the first run, subsequent runs are instant and offline-safe; if the network is unreachable mid-run, unresolved entries fall back to `UNKNOWN` instead of failing the pipeline.

`--dedupe` collapses duplicate `(name, version)` rows that arise when syft sees the same package installed in two paths (e.g. system site-packages plus a venv). When two duplicates disagree on license, the one with a non-`UNKNOWN` SPDX value wins.

```sh
python3 container/compliance/process_results.py \
  --target-dir /tmp/compliance-target \
  --cargo-lock Cargo.lock --go-mod deploy/operator/go.mod \
  --lookup-licenses --dedupe \
  --output /tmp/compliance-target/attribution.csv
```

## Native packages overlay

`native_packages.yaml` is a manually maintained overlay describing components that land in the image via `RUN git clone …` or `RUN wget …tar.gz && make install` inside a Dockerfile. Syft sees the resulting binaries/libraries under `/usr/local` but cannot attribute them to an upstream package because they were not installed through a package manager, so they show up as unowned. The overlay fills that gap and feeds `ATTRIBUTIONS-Native.md`.

Schema:

```yaml
- repo: https://github.com/example/foo
  name: foo
  artifacts:
    - name: libfoo.so
      license: Apache-2.0
    - name: foo-cli
      license: Apache-2.0
```

When a new `RUN git clone …` or `RUN wget …tar…` is added to any Dockerfile under `container/`, add a corresponding entry listing the upstream `repo`, the logical `name`, and each binary/library dropped into the image with its SPDX `license`.

## CI integration

CI integration lives in [`.github/actions/compliance-scan/action.yml`](../../.github/actions/compliance-scan/action.yml). The action runs syft + `process_results.py` after every successful image build and uploads artifacts matching:

- `${inputs.artifact_name}*.csv`
- `sbom.spdx.json`
- `ATTRIBUTIONS-*.md`

The scan runs in parallel with tests and does not extend pipeline wall time.

## Architecture

Syft does the heavy lifting: it walks the target image filesystem and emits a full SPDX SBOM covering dpkg, Python, and anything else its catalogers recognize. Lockfile parsing (`Cargo.lock`, `go.mod`) covers statically-linked Rust and Go dependencies that syft cannot see because the compiler has already inlined them into stripped binaries. The `native_packages.yaml` overlay covers the inverse problem — from-source builds where syft *can* see the resulting files under `/usr/local` but has no package-manager metadata to license them against.

## Drift detection (future)

A planned CI lint will cross-check Dockerfile `RUN git clone` / `RUN wget …tar` lines against entries in `native_packages.yaml` and fail the build when a native component is introduced without a corresponding overlay entry. TODO: link to tracking issue (`<DEP-XXXX>`).
