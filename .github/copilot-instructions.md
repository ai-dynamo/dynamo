# Dynamo Project Instructions

## Project Structure

- `components/src/dynamo/` — Main Python package (hatchling build)
- `lib/` — Rust crates (maturin bindings): `llm/`, `runtime/`, `config/`, `kv-router/`, `kvbm-*/`, `velo-*/`, etc.
- `container/` — Docker build templates; `render.py` generates Dockerfiles per device/framework
- `deploy/` — Kubernetes operator, NATS/etcd setup, deployment configs
- `tests/` — Test suites organized by component: `basic/`, `router/`, `serve/`, `fault_tolerance/`, `deploy/`, `kvbm_integration/`, etc.
- `.github/` — CI workflows and composite actions
- `docs/` — Sphinx documentation
- `examples/`, `recipes/` — Usage examples and deployment recipes

Key config: `pyproject.toml` (Python), `Cargo.toml` + `rust-toolchain.toml` (Rust 1.93.1), `hatch_build.py` (custom build hook)

## CI/CD Conventions

### Pipeline Pattern

1. **changed-files** — Detect affected components: `core`, `planner`, `operator`, `deploy`, `vllm`, `sglang`, `trtllm`
2. **build** — `container/render.py` → `docker buildx build` per framework × device (cuda/xpu/cpu)
3. **test** — Pytest with markers, sequential for GPU/XPU, parallel for CPU
4. **copy** — Push to ECR, optionally copy to ACR

### Key Workflow Files

- `pr.yaml` — Main PR pipeline
- `build-test-distribute-flavor-matrix.yml` — Reusable matrix: framework → device → tests
- `build-test-distribute-flavor.yml` — Single flavor: build → test → compliance → copy
- `.github/actions/build-flavor/` — Tag calculation + image build
- `.github/actions/pytest/` — Docker test runner (GPU/XPU auto-detection)

### XPU vs CUDA

- XPU builds run on `xpu` runner with local docker builder (`use_runner_docker_builder: true`)
- CUDA builds use remote BuildKit on `prod-builder-v3`
- XPU test image suffix: `-xpu`; CUDA: `-cuda12`, `-cuda13`
- Job outputs containing secret-like values (ECR URIs) get silently skipped by GitHub — reconstruct tags locally in downstream jobs instead of passing full URIs as outputs

## Git Conventions

### Commit Messages

Use Conventional Commits with `-s` (DCO sign-off required):

```
feat(router): add KV cache eviction policy
fix(vllm): handle MultiModalUUIDDict import path change
ci: avoid secret-filtered image outputs in build-test workflow
test(cancellation): increase timeout to 660s for aggregated test
chore: update dependencies
docs: add XPU setup guide
```

### Branch Naming

`yourname/description` or `type/description`, e.g. `wenxinzh/fix-xpu-tag`, `test/add-router-e2e`

## Test Conventions

### Pytest Markers

**Lifecycle:** `pre_merge`, `post_merge`, `nightly`, `weekly`, `release`

**Hardware:**
- `gpu_0` (CPU-only), `gpu_1`, `gpu_2`, `gpu_4`, `gpu_8`
- `xpu_1`, `xpu_2` (Intel XPU)

**Framework:** `vllm`, `sglang`, `trtllm`, `lmcache`

**Component:** `router`, `planner`, `kvbm`, `fault_tolerance`, `deploy`, `e2e`

**Parallelism:** `parallel` (safe for xdist), `profiled_vram_gib(N)` (VRAM budget)

### Running Tests

```bash
# Pre-merge CPU tests (parallel)
pytest -m "pre_merge and gpu_0" -n auto --dist=loadscope -v --tb=short

# Single GPU tests (sequential)
pytest -m "pre_merge and vllm and gpu_1" -v --tb=short

# XPU tests
pytest -m "xpu_1 or xpu_2" -v --tb=short

# Collect only (dry run)
pytest -m "xpu_1" --collect-only

# With VRAM scheduling
pytest --max-vram-gib=48 -n auto -m "gpu_1 and vllm"
```

### Test Structure

- `tests/conftest.py` — Global fixtures (model download, ports, logging)
- `tests/utils/` — Shared utilities: `constants.py`, `port_utils.py`, `managed_process.py`
- Per-component subdirectories with their own `conftest.py`
- `--strict-markers` enforced; all markers must be registered in `pyproject.toml`

## Common Operations

```bash
# Generate Dockerfile
python ./container/render.py --framework vllm --device xpu --target runtime --platform linux/amd64 --show-result --output-short-filename

# Install dev (Python)
uv pip install -e ".[vllm]"

# Build Rust bindings
cd lib/bindings/python && maturin develop --uv

# Pre-commit
pre-commit run --all-files

# Marker report
python tests/report_pytest_markers.py --output json
```
