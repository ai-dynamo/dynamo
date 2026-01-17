---
orphan: true
---

# Building Documentation

This directory contains the documentation source files for NVIDIA Dynamo.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) package manager

## Build Instructions

### Option 1: Dedicated Docs Environment (Recommended)

This approach builds the docs without requiring the full project dependencies (including `ai-dynamo-runtime`):

```bash
# One-time setup: Create docs environment and install dependencies
uv venv .venv-docs
uv pip install --python .venv-docs --group docs

# Generate documentation
uv run --python .venv-docs --no-project docs/generate_docs.py
```

The generated HTML will be available in `docs/build/html/`.

### Option 2: Using Full Development Environment

If you already have the full project dependencies installed (i.e., you're actively developing the codebase), you can use `uv run` directly:

```bash
uv run --group docs docs/generate_docs.py
```

This will use your existing project environment and add the docs dependencies.

### Option 3: Using Docker

Build the docs in a Docker container with all dependencies isolated:

```bash
docker build -f container/Dockerfile.docs -t dynamo-docs .
```

The documentation will be built inside the container. To extract the built docs:

```bash
# Run the container and copy the output
docker run --rm -v $(pwd)/docs/build:/workspace/dynamo/docs/build dynamo-docs

# Or create a container to copy files from
docker create --name temp-docs dynamo-docs
docker cp temp-docs:/workspace/dynamo/docs/build ./docs/build
docker rm temp-docs
```

This approach is ideal for CI/CD pipelines or when you want complete isolation from your local environment.

## Directory Structure

- `docs/` - Documentation source files (Markdown and reStructuredText)
- `docs/conf.py` - Sphinx configuration
- `docs/_static/` - Static assets (CSS, JS, images)
- `docs/_extensions/` - Custom Sphinx extensions
- `docs/build/` - Generated documentation output (not tracked in git)

## Redirect Creation

When moving or renaming files a redirect must be created.

Redirect entries should be added to the `redirects` dictionary in `conf.py`. For detailed information on redirect syntax, see the [sphinx-reredirects usage documentation](https://documatt.com/sphinx-reredirects/usage/#introduction).

## Dependency Management

Documentation dependencies are defined in `pyproject.toml` under the `[dependency-groups]` section:

```toml
[dependency-groups]
docs = [
    "sphinx>=8.1",
    "nvidia-sphinx-theme>=0.0.8",
    # ... other doc dependencies
]
```

## Troubleshooting

### Build Warnings

The build process treats warnings as errors. Common issues:

- **Missing toctree entries**: Documents must be referenced in a table of contents
- **Non-consecutive headers**: Don't skip header levels (e.g., H1 → H3)
- **Broken links**: Ensure all internal and external links are valid

### Missing Dependencies

If you encounter import errors, ensure the docs dependencies are installed:

```bash
uv pip install --python .venv-docs --group docs
```

## Viewing the Documentation

After building, open `docs/build/html/index.html` in your, or use Python's built-in HTTP server:

```bash
cd docs/build/html
python -m http.server 8000
# Then visit http://localhost:8000 in your browser
```
