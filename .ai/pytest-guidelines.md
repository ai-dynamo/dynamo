# Pytest Guidelines

Rules and conventions for Python tests in this repository.

## Running Tests

Always use the venv-aware invocation -- never bare `pytest`:

```bash
export HF_HUB_OFFLINE=1 HF_TOKEN="$(cat ~/.cache/huggingface/token)"
python3 -m pytest -xvv --basetemp=/tmp/pytest_temp --durations=0 tests/
```

- `python3 -m pytest` ensures the venv's pytest runs with the correct `sys.path`.
  The system `pytest` at `/usr/local/bin/pytest` is **outside** the venv and cannot
  see venv-installed packages (like `dynamo`).
- `-xvv` stops at first failure with verbose output.
- `--durations=0` shows timing for all tests (helps detect slow/flaky tests).

### Filtering by markers

```bash
python3 -m pytest -m "vllm and gpu_1 and pre_merge" -v
python3 -m pytest -m "vllm and e2e and gpu_1" -v
python3 -m pytest -m "vllm and unit and gpu_0" -v
python3 -m pytest tests/serve/ -m "vllm and gpu_1 and pre_merge" -vv --tb=short
```

Use `--durations=10` locally to find the 10 slowest tests.

### Filtering by keyword (`-k`)

Use `-k` to select tests by name pattern. Be aware that `-k` matches substrings:

```bash
# BAD -- also matches "disaggregated" tests
python3 -m pytest tests/serve/ -k "aggregated" -v

# GOOD -- excludes disaggregated
python3 -m pytest tests/serve/ -k "aggregated and not disagg" -v --tb=short
```

## Critical Rules

These are the most common sources of flaky, non-hermetic tests. Violating any of
these will block your PR.

Hardcoded values in tests are cringy code. They signal that the author didn't
think about parallel execution, reproducibility, or the next person who has to
debug a phantom CI failure at 2 AM. Don't be that person.

### DO NOT hardcode ports

Never use literal port numbers (e.g. `port=8000`, `port=8081`) in test code. Two
tests that share a port will collide when run in parallel, causing mysterious
failures that only reproduce in CI.

**Instead:** Use the `dynamo_dynamic_ports` fixture (allocates `frontend_port` +
`system_ports` per test) or call `allocate_port()` / `allocate_ports()` directly
from `tests.utils.port_utils`.

```python
# BAD
resp = requests.get("http://localhost:8000/v1/models")

# GOOD
def test_example(dynamo_dynamic_ports):
    port = dynamo_dynamic_ports.frontend_port
    resp = requests.get(f"http://localhost:{port}/v1/models")
```

### DO NOT hardcode temp paths

Never write to fixed paths like `/tmp/my-test.log` or `/tmp/output/`. The next test
(or a parallel worker) will pick up stale files, creating subtle side-effects.

**Instead:** Use Python's `tempfile` module or pytest's `tmp_path` fixture. Both
provide unique paths and auto-cleanup.

```python
# BAD
with open("/tmp/test-output.json", "w") as f:
    json.dump(result, f)

# GOOD
def test_example(tmp_path):
    out = tmp_path / "test-output.json"
    out.write_text(json.dumps(result))
```

Also: never dump output files into the repo working tree. This pollutes the repo and
risks clobbering real files, especially in dev (root user) containers.

### DO NOT write custom engine start/stop logic

Never write your own subprocess management code to launch, health-check, or tear down
engines (vLLM, SGLang, TRT-LLM) or infrastructure (NATS, etcd, frontends). Homegrown
lifecycle code inevitably leaks processes, misses cleanup on failure, or races with
parallel tests.

**Instead:** Use the existing fixtures and context managers:

- **Fixtures:** `runtime_services_dynamic_ports`, `start_services_with_http`,
  `start_services_with_grpc`, `start_services_with_mocker`
- **Context managers:** `DynamoFrontendProcess`, `DynamoWorkerProcess`,
  `ManagedProcess`, `EtcdServer`, `NatsServer`

These handle health-checking, port allocation, log capture, straggler cleanup,
and graceful teardown automatically. If your test needs something the existing
infrastructure doesn't support, extend the shared fixtures -- don't reinvent them
in your test file.

```python
# BAD -- hand-rolled subprocess management
proc = subprocess.Popen(["python3", "-m", "dynamo.mocker", ...])
time.sleep(10)  # hope it's ready
try:
    run_test()
finally:
    proc.kill()

# GOOD -- use the provided fixture
def test_example(start_services_with_mocker):
    frontend_port = start_services_with_mocker
    # engine is already up, health-checked, and will be cleaned up automatically
```

### DO NOT copy-paste test infrastructure -- reuse and refactor

Do not duplicate setup logic, helper functions, or fixture code across test files.
Copy-pasted code means the same bug gets fixed in one place but not the others, and
changing a shared pattern requires hunting down every copy.

**Instead:**

- **Reuse existing fixtures and helpers.** Check `tests/conftest.py`,
  subdirectory `conftest.py` files, and `tests/utils/` before writing anything new.
- **Extract shared logic into fixtures or utility functions.** If two or more tests
  need the same setup, it belongs in a `conftest.py` or `tests/utils/`.
- **Parametrize rather than duplicate.** If tests differ only in config (model,
  backend, port count), use `@pytest.mark.parametrize` with indirect fixtures
  instead of writing separate test functions.

```python
# BAD -- same setup copy-pasted across three test files
def test_vllm_chat():
    proc = start_engine("vllm", model="Qwen/Qwen3-0.6B")
    wait_for_ready(proc)
    resp = send_chat_request(proc.port)
    assert resp.status_code == 200

def test_vllm_completion():       # 90% identical to above
    proc = start_engine("vllm", model="Qwen/Qwen3-0.6B")
    wait_for_ready(proc)
    resp = send_completion_request(proc.port)
    assert resp.status_code == 200

# GOOD -- shared fixture, parametrized payloads
@pytest.mark.parametrize("payload_fn", [chat_payload_default, completion_payload_default])
def test_vllm_requests(start_serve_deployment, payload_fn):
    resp = send_request(start_serve_deployment.port, payload_fn())
    assert resp.status_code == 200
```

When the existing infrastructure doesn't fit your needs, extend the shared code
(add a parameter to a fixture, add a helper to `tests/utils/`) rather than forking
a private copy.

---

## Markers

`--strict-markers` and `--strict-config` are enforced in `pyproject.toml`. Using an
undefined marker **fails collection**. All markers must be registered in both
`pyproject.toml [tool.pytest.ini_options].markers` and `tests/conftest.py:pytest_configure`.

### Required markers

Every test must have **at least**:

1. **A scheduling marker** -- when the test runs in CI:
   - `pre_merge` -- runs on every PR before merge
   - `post_merge` -- runs after merge to main
   - `nightly` -- runs nightly
   - `weekly` -- runs weekly
   - `release` -- runs on release pipelines

2. **A GPU marker** -- how many GPUs are needed:
   - `gpu_0` -- no GPU required
   - `gpu_1` -- single GPU
   - `gpu_2`, `gpu_4`, `gpu_8` -- multi-GPU

3. **A type marker** -- what kind of test:
   - `unit` -- unit test
   - `integration` -- integration test
   - `e2e` -- end-to-end test

### Scheduling marker guidance

CI compute is finite. Choose placement carefully:

- Only use `pre_merge` for tests that are **absolutely critical** -- every pre-merge
  test slows down every PR for every contributor.
- E2E tests involve more components and tend to be flakier. Prefer `post_merge` for
  E2E tests unless they guard a critical path.
- Consider `nightly` or `weekly` for expensive, GPU-heavy, or stress tests.

### Framework markers

Apply when the test depends on a specific inference backend:
- `vllm`, `trtllm`, `sglang`

### Timeouts

Tests that run longer than 30 seconds **must** have `@pytest.mark.timeout(<seconds>)`.
Set the timeout to **3x the measured average duration** to absorb variance.

Measure your test 5-10 times, then add a timing comment:

```python
@pytest.mark.timeout(300)  # ~100s average, 3x buffer
def test_vllm_aggregated(...):
    # on average this test takes about 1.5 minutes
    ...
```

Timing comments let AI/automation understand requirements when shuffling test suites.

### Other commonly used markers

- `model("org/model-name")` -- declares the HF model used; the `predownload_models`
  fixture reads these to download only what's needed.
- `slow` -- known slow test.
- `parallel` -- safe to run with pytest-xdist.
- `h100` -- requires H100 hardware.
- `fault_tolerance`, `deploy`, `router`, `planner`, `kvbm` -- component markers.
- `k8s` -- requires Kubernetes.

### Example

```python
@pytest.mark.pre_merge
@pytest.mark.gpu_1
@pytest.mark.e2e
@pytest.mark.vllm
@pytest.mark.model("Qwen/Qwen3-0.6B")
@pytest.mark.timeout(300)
def test_vllm_aggregated(start_serve_deployment):
    ...
```

## Async Tests

`asyncio_mode = "auto"` is configured in `pyproject.toml`. Do **not** add
`@pytest.mark.asyncio` manually -- all `async def test_*` functions are collected
automatically.

## Hermetic Testing

Tests must be isolated and must not interfere with each other. Every test should:

- Run reliably in any order, on any machine, at any time.
- Produce deterministic results.
- Not create side-effects for other tests.
- Clean up properly after itself.
- Fail fast when something is wrong.

Given enough resources, multiple tests must be able to execute in parallel without
conflicts or race conditions. See the **Critical Rules** section at the top for the
three most important requirements (dynamic ports, temp paths, shared fixtures).

### Additional anti-patterns

- **Reusing namespace/component/endpoint names** across tests that share a
  registration service. Use unique names per test.
- **Leaking environment variables**. Use `monkeypatch.setenv()` or save/restore
  patterns so env changes don't persist across tests.

### Optimization tips

- Combine multiple assertions in one engine launch/teardown cycle when tests share
  the same deployment config.
- Use the mock engine (`dynamo.mocker`) instead of a real vLLM/SGLang/TRT-LLM engine
  when the test doesn't need real inference.
- Mock external services (APIs, databases, etc.) to keep tests fast and deterministic.

## Fixtures

### Service infrastructure

- **`runtime_services_dynamic_ports`** -- preferred for xdist-safe tests. Spins up
  per-test NATS and etcd on dynamic ports, sets `NATS_SERVER` / `ETCD_ENDPOINTS`
  env vars, cleans up after.
- **`runtime_services`** -- simpler, uses default ports. Not xdist-safe.
- **`runtime_services_session`** -- session-scoped, shared across xdist workers via
  file locks. Good for large test suites where per-test instances are too expensive.

### Port allocation

- **`dynamo_dynamic_ports`** -- allocates `frontend_port` + `system_ports` per test.
  Never hardcode ports (8000, 8081, etc.) in tests.
- **`num_system_ports`** -- defaults to 1. Use indirect parametrize for more:
  `@pytest.mark.parametrize("num_system_ports", [2], indirect=True)`

### Model management

- **`predownload_models`** (session-scoped) -- downloads full models. Reads
  `@pytest.mark.model(...)` from collected tests to download only what's needed.
  Sets `HF_HUB_OFFLINE=1` after download so workers skip redundant API calls.
- **`predownload_tokenizers`** (session-scoped) -- same, but skips weight files.

### Backend-specific parametrize

- **`discovery_backend`** -- defaults to `"etcd"`. Parametrize with `["file", "etcd"]`.
- **`request_plane`** -- defaults to `"nats"`. Parametrize with `["nats", "tcp"]`.
- **`durable_kv_events`** -- defaults to `False`. Set `[True]` for JetStream mode.

### Logging

An autouse `logger` fixture writes per-test logs to `test_output/<test_name>/test.log.txt`.
Some sub-suites (e.g. `tests/planner/`) override this with a no-op fixture.

## xdist / Parallel Safety

- Use `runtime_services_dynamic_ports` + `dynamo_dynamic_ports` for port isolation.
- Use `SharedEtcdServer` / `SharedNatsServer` (via `runtime_services_session`) for
  session-scoped shared services with file-lock coordination.
- Never rely on fixed ports or global state across workers.
- Each xdist worker is a separate process -- env vars don't leak.

## Warnings

`filterwarnings = ["error"]` is set globally, with specific ignores for known
third-party deprecations (CUDA, protobuf, pynvml, torchao, etc.). If your test
triggers a new warning, either fix the root cause or add a targeted ignore in
`pyproject.toml` with a comment explaining why.

## Error Handling in Tests

- No blanket `except Exception` -- let failures propagate.
- Catch only specific exceptions you can actually handle.
- Prefer fixtures for setup/teardown over try/finally in test bodies.

## Test File Organization

```
tests/
  conftest.py              # Root fixtures: services, ports, model downloads, logging
  serve/                   # Backend serve tests (vllm, trtllm, sglang)
    conftest.py            # Image server, MinIO LoRA fixtures
  frontend/                # Frontend HTTP/gRPC tests
    conftest.py            # HTTP/gRPC service fixtures, mocker workers
    grpc/                  # gRPC-specific tests
  planner/                 # Planner component tests
    unit/                  # Planner unit tests
  router/                  # Router E2E tests
  fault_tolerance/         # Fault tolerance tests
    cancellation/
    migration/
    etcd_ha/
    gpu_memory_service/
    deploy/
  kvbm_integration/        # KV block manager integration tests
  deploy/                  # Deployment tests
  basic/                   # Basic smoke tests (wheel contents, CUDA version)
  dependencies/            # Import/dependency tests
  utils/                   # Shared test utilities (NOT test files)
    constants.py           # Model IDs, default ports
    managed_process.py     # ManagedProcess for subprocess lifecycle
    port_utils.py          # Dynamic port allocation
    test_output.py         # Test output path resolution
```

## Serve Tests Pattern

Backend serve tests (`tests/serve/test_vllm.py`, etc.) follow a config-driven pattern:

```python
vllm_configs = {
    "aggregated": VLLMConfig(
        name="aggregated",
        directory=vllm_dir,
        script_name="agg.sh",
        marks=[pytest.mark.gpu_1, pytest.mark.pre_merge, pytest.mark.timeout(300)],
        model="Qwen/Qwen3-0.6B",
        request_payloads=[...],
    ),
}
```

Configs are parametrized into test functions via `params_with_model_mark()`, which
auto-applies the `model` marker from the config's model field.

