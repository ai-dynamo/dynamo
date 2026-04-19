# dynamo-attributions

Generate `ATTRIBUTIONS-*.md` files for the Dynamo project from lock files and
registry APIs (crates.io, GitHub) for Rust/Go, and container inspection for Python.

Complements `container/compliance/generate_attributions.py` (CSV from container
inspection) by producing the committed Markdown attribution files.

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

# Python from container image
dynamo-attributions --ecosystem python --image myregistry/dynamo:vllm-runtime
```

Also runnable as a module:

```bash
python3 -m dynamo_attributions --branch main --image IMAGE --output-dir .
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--branch` | `main` | Git branch/ref to read lock files from |
| `--ecosystem` | `all` | `rust`, `python`, `go`, or `all` |
| `--image` | -- | Container image for Python extraction (required for Python) |
| `--output-dir` | `.` | Directory for generated ATTRIBUTIONS-*.md files |
| `--dynamo-path` | `.` | Path to the dynamo git repo |
| `--license-cache` | `~/.dynamo_license_cache.json` | Persistent JSON cache for Rust/Go license lookups |
| `--github-token` | `$GITHUB_TOKEN` | GitHub token for Go module lookups (avoids 60 req/hr limit) |

## How It Works

- **Rust**: Parses `Cargo.lock` via `git show`, fetches license metadata from crates.io API
- **Go**: Parses `deploy/operator/go.mod` via `git show`, resolves vanity URLs, fetches from GitHub Licenses API
- **Python**: Runs `python_helper.py` inside the container via `docker run` using `importlib.metadata` to get package names, versions, and SPDX-normalized licenses in one step. Reuses the existing extractor at `container/compliance/extractors/python_pkgs.py`.

## Caching

License lookups for Rust and Go are cached in `~/.dynamo_license_cache.json`.
The cache is keyed by `ecosystem:name:version` and reused across runs. First
run fetches from APIs (~13 min for Rust at 1 req/sec); subsequent runs with the
same lock files are near-instant.

Python packages are extracted directly from the container — no cache needed.

## Dependencies

Single runtime dependency: `requests` (for Rust/Go license API lookups).
