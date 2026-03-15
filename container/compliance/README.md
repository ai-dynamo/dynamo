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

## Usage

```bash
# Full scan, output to stdout
python container/compliance/generate_attributions.py <image:tag>

# Write to file
python container/compliance/generate_attributions.py <image:tag> -o attribution.csv

# With base image diff — auto-resolved from context.yaml
python container/compliance/generate_attributions.py <image:tag> \
    --framework vllm \
    --cuda-version 12.9 \
    -o attribution-vllm-cuda12-amd64.csv
# Produces: attribution-vllm-cuda12-amd64.csv  (full)
#           attribution-vllm-cuda12-amd64_diff.csv  (delta from base)

# With explicit base image override
python container/compliance/generate_attributions.py <image:tag> \
    --base-image nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04 \
    -o attribution.csv

# Frontend image
python container/compliance/generate_attributions.py <image:tag> \
    --framework dynamo \
    --target frontend \
    -o attribution-frontend-amd64.csv

# dpkg only
python container/compliance/generate_attributions.py <image:tag> \
    --types dpkg \
    -o attribution-dpkg.csv
```

### All flags

| Flag | Default | Description |
|------|---------|-------------|
| `image` | *(required)* | Container image to scan |
| `--output`, `-o` | stdout | Output CSV path |
| `--framework` | — | Auto-resolve base image from `context.yaml` (`vllm`, `sglang`, `trtllm`, `dynamo`) |
| `--target` | `runtime` | Build target for base resolution (`runtime` or `frontend`) |
| `--cuda-version` | — | CUDA version for base resolution (e.g. `12.9`, `13.0`, `13.1`) |
| `--base-image` | — | Explicit base image URI (overrides `--framework` auto-resolve) |
| `--context-yaml` | `container/context.yaml` | Path to context.yaml |
| `--types` | `dpkg,python` | Comma-separated list of types to extract |
| `--docker-cmd` | `docker` | Docker binary to use |
| `--verbose`, `-v` | — | Enable verbose logging to stderr |

## Base image reference

| Framework | CUDA | Base image |
|-----------|------|------------|
| `vllm` | 12.9 | `nvcr.io/nvidia/cuda:12.9.1-runtime-ubuntu24.04` |
| `vllm` | 13.0 | `nvcr.io/nvidia/cuda:13.0.2-runtime-ubuntu24.04` |
| `sglang` | 12.9 | `lmsysorg/sglang:v0.5.9-runtime` |
| `sglang` | 13.0 | `lmsysorg/sglang:v0.5.9-cu130-runtime` |
| `trtllm` | 13.1 | `nvcr.io/nvidia/cuda-dl-base:25.12-cuda13.1-runtime-ubuntu24.04` |
| `dynamo` frontend | — | `nvcr.io/nvidia/base/ubuntu:noble-20250619` |

These values are sourced from `container/context.yaml` at runtime; the table above reflects the current defaults.

## How it works

The script runs two lightweight helper scripts **inside the container** via `docker run --rm -v`:

- **dpkg extractor** — runs `dpkg-query` to list packages, then reads `/usr/share/doc/<pkg>/copyright` files for license info. Only DEP-5 machine-readable copyright files are parsed; ambiguous cases return `UNKNOWN`.
- **Python extractor** — uses `importlib.metadata.distributions()` to iterate installed packages. License is read from `License-Expression` (PEP 639), then `License` metadata, then trove classifiers. Ambiguous cases return `UNKNOWN`.

Both helpers are self-contained and have no external dependencies — they run with whatever Python is in the container.

## License detection

Detection is intentionally conservative: only unambiguous matches are assigned SPDX identifiers. The `UNKNOWN` entries are expected; they can be resolved with additional analysis against the raw copyright files.

## CI integration

Attribution CSVs are generated automatically as part of CI after every successful image build. Artifacts are available in the GitHub Actions workflow run under:
- `compliance-{framework}-cuda{major}-{platform}` — runtime images
- `compliance-frontend-{arch}` — frontend image

The scan runs as a separate lightweight job (`prod-default-small-v2`) in parallel with tests, so it does not extend pipeline wall time.

## Requirements

- Python 3.11+
- `docker` (or compatible CLI) with access to the target registry
- `pyyaml` — only required on the host when using `--framework`/`--cuda-version` base image auto-resolution (`pip install pyyaml`)
