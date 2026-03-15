# dynamo-attributions

Generate `ATTRIBUTIONS-*.md` files for the Dynamo project from lock files and
package registry APIs (crates.io, PyPI, GitHub).

## Install

```bash
pip install tools/attributions/
# or editable
pip install -e tools/attributions/
```

## Usage

Run from the dynamo repo root (or pass `--dynamo-path`):

```bash
# All ecosystems
dynamo-attributions --branch main --output-dir .

# Rust only (reads Cargo.lock via git-show)
dynamo-attributions --ecosystem rust

# Go only (reads deploy/operator/go.mod via git-show)
dynamo-attributions --ecosystem go

# Python (requires pip freeze from a container)
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
| `--pip-freeze-file` | — | Path to pip freeze output (required for Python) |
| `--output-dir` | `.` | Directory for generated ATTRIBUTIONS-*.md files |
| `--dynamo-path` | `.` | Path to the dynamo git repo |
| `--license-cache` | `~/.dynamo_license_cache.json` | Persistent JSON cache for license lookups |
| `--github-token` | `$GITHUB_TOKEN` | GitHub token for Go module lookups (avoids 60 req/hr rate limit) |

## Caching

License lookups are cached in `~/.dynamo_license_cache.json` by default. The cache
is keyed by `ecosystem:name:version` and reused across runs. First run fetches from
APIs (~13 min for Rust at 1 req/sec); subsequent runs with the same lock files are
near-instant.

## Dependencies

Single runtime dependency: `requests`.
