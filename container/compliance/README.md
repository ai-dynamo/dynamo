# Container Compliance Tooling

Scripts for generating attribution CSVs from built container images, listing all installed dpkg and Python packages with their SPDX license identifiers where known.

## Output format

Each run produces up to two CSV files:

| Column | Description |
|--------|-------------|
| `package_name` | Package name as reported by dpkg or pip |
| `version` | Installed version |
| `type` | `dpkg` or `python` |
| `spdx_license` | SPDX identifier (e.g. `MIT`, `Apache-2.0`) or `UNKNOWN` |

Files are sorted by `(type, package_name)` for stable diffs.

When a base image is provided, a second `_diff.csv` file is written containing only packages that are new or version-changed relative to the base — i.e. what Dynamo's build layers added on top of the upstream image.

## Local usage

### Prerequisites

- Docker with [BuildKit](https://docs.docker.com/build/buildkit/) support (Docker 23+)
- Python 3.11+

### Step 1 — Create a local BuildKit builder (one-time)

```bash
docker buildx create --use --name compliance-builder
```

### Step 2 — Extract packages from an image

```bash
docker buildx build \
  --builder compliance-builder \
  --platform linux/amd64 \
  --build-arg TARGET_IMAGE=<image:tag> \
  --output type=local,dest=./output \
  --pull \
  --no-cache-filter extractor \
  --progress=plain \
  -f container/compliance/Dockerfile.extract \
  container/compliance/
```

This produces `./output/dpkg.tsv` and `./output/python.tsv` — tab-separated files
with `package_name\tversion\tspdx_license` per line.

> **Why `--no-cache-filter extractor`?** BuildKit's cache key for
> `RUN --mount=type=bind,from=<stage>` does not reliably include the mounted
> stage's content digest when the source is a stage name (vs. a direct image
> reference). Without this flag, a cache hit could return TSVs from a previous
> run against a different image even if `--pull` resolved a new digest.
> `--no-cache-filter extractor` forces only the extraction stage to re-run;
> the `python:3.12-slim` base layer and helper script COPYs are still cached.

### Step 3 — Convert to CSV

```bash
python container/compliance/process_results.py \
  --target-dir ./output \
  --output attribution.csv
```

### Full example with base image diff

Use `resolve_base_image.py` to look up the correct base image from `container/context.yaml`
rather than hardcoding the URI:

```bash
# Resolve base image from context.yaml (requires: pip install pyyaml)
BASE_IMAGE=$(python container/compliance/resolve_base_image.py \
  --framework vllm \
  --cuda-version 12.9 \
  --arch amd64)

# Extract target image
docker buildx build \
  --builder compliance-builder \
  --platform linux/amd64 \
  --build-arg TARGET_IMAGE=<image:tag> \
  --output type=local,dest=./output \
  --pull \
  --no-cache-filter extractor \
  -f container/compliance/Dockerfile.extract \
  container/compliance/

# Extract base image
docker buildx build \
  --builder compliance-builder \
  --platform linux/amd64 \
  --build-arg TARGET_IMAGE="${BASE_IMAGE}" \
  --output type=local,dest=./base-output \
  --pull \
  --no-cache-filter extractor \
  -f container/compliance/Dockerfile.extract \
  container/compliance/

# Generate CSV with diff
python container/compliance/process_results.py \
  --target-dir ./output \
  --base-dir ./base-output \
  --output attribution.csv
# Produces: attribution.csv (full) and attribution_diff.csv (delta from base)
```

### resolve_base_image.py flags

| Flag | Default | Description |
|------|---------|-------------|
| `--framework` | *(required)* | `vllm`, `sglang`, `trtllm`, or `dynamo` |
| `--target` | `runtime` | `runtime` or `frontend` |
| `--cuda-version` | — | Required for runtime targets (e.g. `12.9`, `13.0`, `13.1`) |
| `--arch` | `amd64` | `amd64`, `arm64`, `linux/amd64`, or `linux/arm64` |
| `--context-yaml` | `container/context.yaml` | Path to context.yaml |

### process_results.py flags

| Flag | Default | Description |
|------|---------|-------------|
| `--target-dir` | *(required)* | Directory containing `dpkg.tsv` and `python.tsv` from target extraction |
| `--base-dir` | — | Directory containing TSVs from base image extraction (enables `_diff.csv` output) |
| `--output`, `-o` | stdout | Output CSV path |

## Base image reference

| Framework | CUDA | Arch | Base image |
|-----------|------|------|------------|
| `vllm` | 12.9 | `amd64` | `vllm/vllm-openai:v0.18.0-x86_64` |
| `vllm` | 12.9 | `arm64` | `vllm/vllm-openai:v0.18.0-aarch64` |
| `vllm` | 13.0 | `amd64` | `vllm/vllm-openai:v0.18.0-x86_64-cu130` |
| `vllm` | 13.0 | `arm64` | `vllm/vllm-openai:v0.18.0-aarch64-cu130` |
| `sglang` | 12.9 | — | `lmsysorg/sglang:v0.5.9-runtime` |
| `sglang` | 13.0 | — | `lmsysorg/sglang:v0.5.9-cu130-runtime` |
| `trtllm` | 13.1 | — | `nvcr.io/nvidia/cuda-dl-base:25.12-cuda13.1-runtime-ubuntu24.04` |
| `dynamo` frontend | — | — | `nvcr.io/nvidia/base/ubuntu:noble-20250619` |

These values are sourced from `container/context.yaml`; the table above reflects the current defaults.

## How it works

Extraction uses BuildKit's bind-mount mechanism — the target image filesystem is
mounted read-only at `/target` inside a Python 3.12 builder container, and two
helper scripts read package metadata directly from disk without starting the target
container:

- **`helpers/dpkg_helper.py`** — parses `/target/var/lib/dpkg/status` for installed
  packages and reads `/target/usr/share/doc/<pkg>/copyright` (DEP-5 format) for license info.
- **`helpers/python_helper.py`** — enumerates site-packages directories under `/target`
  using `importlib.metadata`. License is read from `License-Expression` (PEP 639),
  then `License` metadata, then trove classifiers.

Both helpers are self-contained (stdlib only) and run inside the `python:3.12-slim`
extractor stage, not inside the target image.

## License detection

Detection is intentionally conservative: only unambiguous matches are assigned SPDX
identifiers. The `UNKNOWN` entries are expected; they can be resolved with additional
analysis against the raw copyright files.

## CI integration

Attribution CSVs are generated automatically as part of CI after every successful
image build. Artifacts are available in the GitHub Actions workflow run under:

- `compliance-{framework}-cuda{major}-{platform}` — runtime images
- `compliance-frontend-{arch}` — frontend image

The scan runs as a separate job in parallel with tests, so it does not extend
pipeline wall time.
