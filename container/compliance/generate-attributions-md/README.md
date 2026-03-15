# dynamo-attributions

Generate `ATTRIBUTIONS-*.md` files for the Dynamo project from lock files and
package registry APIs (crates.io, PyPI, GitHub).

Complements `container/compliance/generate_attributions.py` (CSV from container
inspection) by producing the committed Markdown attribution files from lock
files and registry lookups.

## Install

```bash
pip install container/compliance/generate-attributions-md/
# or editable
pip install -e container/compliance/generate-attributions-md/
```

## Usage

Run from the dynamo repo root (or pass `--dynamo-path`):

```bash
# All ecosystems (Python via container image)
dynamo-attributions --branch main --image IMAGE --output-dir .

# Rust only (reads Cargo.lock via git-show, no container needed)
dynamo-attributions --ecosystem rust

# Go only (reads deploy/operator/go.mod via git-show)
dynamo-attributions --ecosystem go

# Python from container image (preferred -- uses importlib.metadata)
dynamo-attributions --ecosystem python --image myregistry/dynamo:vllm-runtime

# Python from pip freeze file (fallback)
docker run --rm IMAGE pip freeze > freeze.txt
dynamo-attributions --ecosystem python --pip-freeze-file freeze.txt
```

Also runnable as a module:

```bash
python3 -m dynamo_attributions --branch main --output-dir .
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--branch` | `main` | Git branch/ref to read lock files from |
| `--ecosystem` | `all` | `rust`, `python`, `go`, or `all` |
| `--image` | -- | Container image for Python extraction (preferred) |
| `--pip-freeze-file` | -- | Pip freeze output file (fallback for Python) |
| `--output-dir` | `.` | Directory for generated ATTRIBUTIONS-*.md files |
| `--dynamo-path` | `.` | Path to the dynamo git repo |
| `--license-cache` | `~/.dynamo_license_cache.json` | Persistent JSON cache for license lookups |
| `--github-token` | `$GITHUB_TOKEN` | GitHub token for Go module lookups (avoids 60 req/hr limit) |

## Python Extraction

Two modes for Python packages, in priority order:

1. **`--image IMAGE`** (preferred) -- Runs `python_helper.py` inside the container
   via `importlib.metadata`. Gets accurate package names, versions, and
   SPDX-normalized licenses in one step. No API calls needed. Handles `+cu129`
   variants correctly.

2. **`--pip-freeze-file FILE`** (fallback) -- Parses pip freeze output, then
   queries PyPI for each package's license. Slower and may miss packages with
   custom version suffixes.

## Caching

License lookups for Rust and Go are cached in `~/.dynamo_license_cache.json`.
The cache is keyed by `ecosystem:name:version` and reused across runs. First
run fetches from APIs (~13 min for Rust at 1 req/sec); subsequent runs with the
same lock files are near-instant.

Python packages extracted via `--image` do not use the cache (licenses come
directly from the container).

## Dependencies

Single runtime dependency: `requests` (for Rust/Go license API lookups).
