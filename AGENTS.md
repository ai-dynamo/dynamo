# AGENTS.md

## Cursor Cloud specific instructions

### Overview

NVIDIA Dynamo is a polyglot (Rust + Python + Go) distributed LLM inference orchestration framework. The Rust workspace (30 crates) provides the core runtime; Python (via PyO3/Maturin bindings) provides extensible components (frontend, router, planner, backends); Go modules provide Kubernetes tooling.

### Network restrictions (Cloud Agent VMs)

The `static.rust-lang.org`, `index.crates.io`, and `static.crates.io` domains are **blocked** by the Cloud Agent sandbox network. This means:

- `cargo build`, `cargo test`, and `cargo clippy` **cannot run** because Cargo cannot download crate dependencies.
- `rustup component add` (for `rustfmt`, `clippy`) also fails since component downloads go through `static.rust-lang.org`.
- The Rust 1.93.1 toolchain itself must be installed by extracting it from the Docker image `public.ecr.aws/docker/library/rust:1.93.1-slim`. The update script handles this.

**Working domains:** `github.com`, `api.github.com`, `pypi.org`, `files.pythonhosted.org`, `download.docker.com`, `archive.ubuntu.com`, `public.ecr.aws`.

### Python development workflow

Since Rust compilation is blocked, use the **pre-built `ai-dynamo-runtime` wheel from PyPI** instead of `maturin develop`:

```bash
source .venv/bin/activate
uv pip install --prerelease=allow "ai-dynamo-runtime==1.2.0"
uv pip install --no-deps -e .
```

The PyPI wheel contains the compiled Rust bindings (`dynamo._core`). This is sufficient for running and testing all Python components.

### Running services

- **Frontend (no GPU required):** `python3 -m dynamo.frontend --http-port 8000 --discovery-backend file`
  - `--discovery-backend file` avoids the etcd/NATS dependency for local dev.
  - Serves an OpenAI-compatible HTTP API at the specified port.
- **Backend workers require GPU.** Without GPUs, the frontend will start but `/v1/chat/completions` will return 404 (no model registered).

### Linting

- **Python:** `ruff check components/src/dynamo/` (ruff is configured in `pyproject.toml`)
- **Rust:** `cargo fmt --check` and `cargo clippy` — blocked in Cloud Agent VMs (see above).

### Testing

- **Python unit tests:** `pytest tests/ -m "unit"` — many tests are skipped without GPU or backend frameworks (vllm/sglang/trtllm).
- **Rust tests:** `cargo test` — blocked in Cloud Agent VMs.
- **Pre-merge marker:** Tests tagged `pre_merge` are the CI gate; run with `pytest -m pre_merge`.
- Parser parity tests in `tests/parity/` depend on the locally-built Rust parser; they may fail when using the PyPI wheel if the parser has changed since the last PyPI release.

### Key paths

| Component | Path |
|-----------|------|
| Python components | `components/src/dynamo/` |
| Rust workspace | `lib/` (30 crates) |
| Python bindings (Maturin) | `lib/bindings/python/` |
| Tests | `tests/` |
| Docker Compose (etcd + NATS) | `deploy/docker-compose.yml` |
| Container build templates | `container/` |
| Go K8s operator | `deploy/operator/` |
| Requirements files | `container/deps/requirements.*.txt` |
