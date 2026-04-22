# dynamo-attributions

Generate `ATTRIBUTIONS-*.md` files for the Dynamo project from lock files and
registry APIs (crates.io, GitHub) for Rust/Go, and container inspection for Python.

## Attribution System Overview

The Dynamo repo has a **three-PR** attribution system:

1. **Repo-root direct dependencies** (this tool):
   - Produces `ATTRIBUTIONS-Rust.md`, `ATTRIBUTIONS-Python.md`, `ATTRIBUTIONS-Go.md`
     at the repo root for direct dependencies visible in lock files
   - Uses `Cargo.lock`, `go.mod` files, and container `importlib.metadata` extraction
   - Generates Markdown suitable for repo-level compliance

2. **In-container comprehensive attributions** (`container/compliance/sbom/`):
   - Produces exhaustive container-image attributions from CycloneDX SBOMs
   - Covers all transitive dependencies including OS packages
   - Used for container compliance (NGC, OSRB submission)

3. **CSV extraction** (`container/compliance/generate_attributions.py`):
   - Extracts per-ecosystem CSV files from containers for OSRB import
   - Complements both Markdown generators

All three feed the OSRB (Open Source Review Board) process.

## Usage

Run from the dynamo repo root:

```bash
# All ecosystems (Python via container image)
python3 container/compliance/generate_root_attributions.py \
    --branch main --image IMAGE --output-dir .

# Rust only (reads Cargo.lock via git-show, no container needed)
python3 container/compliance/generate_root_attributions.py --ecosystem rust

# Go only (reads all go.mod files via git-show)
python3 container/compliance/generate_root_attributions.py --ecosystem go

# Python from container image
python3 container/compliance/generate_root_attributions.py \
    --ecosystem python --image myregistry/dynamo:vllm-runtime

# Show help
python3 container/compliance/generate_root_attributions.py --help
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
- **Go**: Parses all `go.mod` files in the repo via `git ls-tree` + `git show`, resolves vanity URLs, fetches from GitHub Licenses API
- **Python**: Runs `python_helper.py` inside the container via `docker run` using `importlib.metadata` to get package names, versions, and SPDX-normalized licenses in one step. Reuses the existing extractor at `container/compliance/extractors/python_pkgs.py`.

## Caching

License lookups for Rust and Go are cached in `~/.dynamo_license_cache.json`.
The cache is keyed by `ecosystem:name:version` and reused across runs. First
run fetches from APIs (~13 min for Rust at 1 req/sec); subsequent runs with the
same lock files are near-instant.

Python packages are extracted directly from the container — no cache needed.

## Dependencies

Single runtime dependency: `requests` (for Rust/Go license API lookups).
