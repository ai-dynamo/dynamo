# Pytest Guidelines

Rules and conventions for Python tests in this repository.

## Running Tests

**Always run your tests locally before pushing to CI.** Every failed CI run
wastes shared GPU minutes and blocks other PRs. A local run catches most
failures in seconds.

### Reproducing CI failures locally

Don't guess from log snippets. Pull the same container image CI built and
reproduce the failure locally:

```bash
# 1. Find the image:tag in the CI job's "docker build" or "docker push" step.

# 2. Run it with GPU access:
docker run --rm -it --gpus all <ci-image>:<tag> bash

# 3. Run the failing test:
python3 -m pytest -xvv tests/path/to/test_that_failed.py::test_name
```

If the test passes locally in the CI container, the failure is likely a
resource or timing issue in CI. If it fails, you have an exact reproduction.

Pull the container, reproduce, fix, verify -- in that order. Don't make
speculative fixes from logs.

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

These are the most common sources of flaky, non-hermetic tests. Violating any
will block your PR.

### DO NOT hardcode ports

**Always flag** literal port numbers (`port=8000`, `port=8081`) in test code.
Parallel tests sharing any resource (port, file, env var, etc.) will collide.

Use `dynamo_dynamic_ports` (allocates `frontend_port` + `system_ports` per test)
or `allocate_port()` / `allocate_ports()` from `tests.utils.port_utils`.

```python
# BAD
resp = requests.get("http://localhost:8000/v1/models")

# GOOD
def test_example(dynamo_dynamic_ports):
    port = dynamo_dynamic_ports.frontend_port
    resp = requests.get(f"http://localhost:{port}/v1/models")
```

### DO NOT hardcode temp paths

**Always flag** fixed paths like `/tmp/my-test.log` or `/tmp/output/` in test code.
Parallel workers will clobber each other's files.

Use pytest's `tmp_path` fixture or Python's `tempfile` module -- both provide
unique paths with auto-cleanup.

```python
# BAD -- hardcoded path collides with parallel tests
with open("/tmp/test-output.json", "w") as f:
    json.dump(result, f)

# BAD -- "ghost fixture": accepts tmp_path but ignores it and writes to /tmp anyway.
# Flag any test that requests tmp_path but still references /tmp/ or hardcoded paths.
def test_example(tmp_path):
    with open("/tmp/test-output.json", "w") as f:
        json.dump(result, f)

# GOOD
def test_example(tmp_path):
    out = tmp_path / "test-output.json"
    out.write_text(json.dumps(result))
```

### DO NOT write output files into the repository tree

**Always flag** any test that writes to paths relative to `__file__` or the repo
root. This pollutes the working tree and creates untracked noise in `git status`.

**Exception:** The autouse `logger` fixture writes to `test_output/<test_name>/`
by design -- this is sanctioned shared infra, not ad-hoc test output. Do not
flag it.

```python
# BAD -- writes into the repo alongside the test file; flag this
output = os.path.join(os.path.dirname(__file__), "scratch_output.txt")
with open(output, "w") as f:
    f.write("debug output\n")

# GOOD -- use tmp_path; cleaned up automatically
def test_example(tmp_path):
    output = tmp_path / "scratch_output.txt"
    output.write_text("debug output\n")
```

### DO NOT write custom engine start/stop logic

**Always flag** hand-rolled `subprocess.Popen` / `os.system` / `time.sleep`
patterns for engine or infra lifecycle. Homegrown lifecycle code leaks processes,
misses cleanup on failure, and races with parallel tests.

Use the existing fixtures and context managers:

- **Fixtures:** `runtime_services_dynamic_ports`, `start_services_with_http`,
  `start_services_with_grpc`, `start_services_with_mocker`
- **Context managers:** `DynamoFrontendProcess`, `DynamoWorkerProcess`,
  `ManagedProcess`, `EtcdServer`, `NatsServer`

These handle health-checking, port allocation, log capture, and graceful teardown
automatically. Extend the shared fixtures if needed -- don't reinvent them.

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

**Always flag** duplicated setup logic, helpers, or fixture code across test files.
Copy-pasted infra means bugs get fixed in one copy but not the others.

- Check `tests/conftest.py`, subdirectory `conftest.py` files, and `tests/utils/`
  before writing anything new.
- If two or more tests share setup, extract it into a fixture or `tests/utils/`.
- If tests differ only in config, use `@pytest.mark.parametrize` with indirect
  fixtures instead of separate functions.

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

Extend shared code rather than forking a private copy.

---

## Markers

`--strict-markers` and `--strict-config` are enforced. Using an undefined marker
**fails collection**. Register all markers in both `pyproject.toml` and
`tests/conftest.py:pytest_configure`.

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
- **Tests averaging over 60 seconds should default to `post_merge`** unless they
  guard a critical path that justifies blocking every PR. If a test must stay
  `pre_merge` despite being slow, add a comment explaining why.
- E2E tests involve more components and tend to be flakier. Prefer `post_merge` for
  E2E tests unless they guard a critical path.
- Consider `nightly` or `weekly` for expensive, GPU-heavy, or stress tests.

### Framework markers

Apply when the test depends on a specific inference backend:
- `vllm`, `trtllm`, `sglang`

### Timeouts

Tests over 30 seconds **must** have `@pytest.mark.timeout(<seconds>)`. Set the
timeout to **3x measured average** to absorb variance.

**Always flag** any test that runs over 30 seconds or contains `time.sleep()`,
polling loops, network calls, or subprocess waits but lacks a
`@pytest.mark.timeout(...)` marker. This is a **required change, not a style
suggestion** -- a missing timeout can hang CI indefinitely.

**Also flag** real `time.sleep()` in `pre_merge` + `unit` tests. Unit tests
should not burn wall-clock time. Mock the sleep, use shared fixtures, or
reclassify as `integration`/`e2e` with `@pytest.mark.slow`.

```python
# BAD -- sleeps and loops with no timeout marker; can hang CI forever
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_poll_server():
    for _ in range(50):
        time.sleep(0.1)
    assert True

# GOOD -- timeout prevents infinite hangs
@pytest.mark.timeout(300)  # ~100s average, 3x buffer
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.unit
def test_vllm_aggregated(...):
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

## Hermetic Testing

Tests must be isolated. Every test must run in any order, on any machine, and
produce deterministic results with no side-effects. Multiple tests must be able
to execute in parallel without conflicts.

### Additional anti-patterns

- **Module-level mutable state.** **Always flag** any mutable object (`{}`, `[]`,
  `set()`) at module scope that tests read or write. This makes tests
  order-dependent and produces phantom xdist failures.

  ```python
  # BAD -- module-level dict shared across all tests; flag this
  _shared_results = {}

  def test_a():
      _shared_results["worker-1"] = "registered"

  def test_b():
      # Passes only if test_a ran first!
      assert _shared_results["worker-1"] == "registered"

  # GOOD -- each test gets its own state
  @pytest.fixture
  def results():
      return {}

  def test_a(results):
      results["worker-1"] = "registered"
      assert results["worker-1"] == "registered"
  ```

- **Colliding `dyn://` registration paths across tests.** Dynamo workers register
  under `dyn://{namespace}.{component}.{endpoint}` in etcd/NATS. Hardcoding
  namespace, component, and endpoint strings is fine on its own -- the problem
  is when two tests that share an etcd/NATS instance use the **same full path**,
  causing flaky collisions under parallel execution.

  **Always flag** tests whose full `dyn://` path can collide with another test's.
  The simplest fix is to randomize at least one segment (typically namespace).

  ```python
  # BAD -- two tests using this identical path will collide
  namespace = "dynamo"
  component = "backend"
  endpoint = f"dyn://{namespace}.{component}.generate"

  # GOOD -- unique namespace prevents collisions; component/endpoint can stay fixed
  from tests.router.common import generate_random_suffix
  namespace = f"dynamo-{generate_random_suffix()}"
  component = "backend"
  endpoint = f"dyn://{namespace}.{component}.generate"
  ```

- **Leaking environment variables.** **Always flag** direct `os.environ[...] = ...`
  or `os.environ.update(...)` in tests. These mutations persist into subsequent
  tests and cause order-dependent failures.

  ```python
  # BAD -- env var leaks into every test that runs after this one; flag this
  def test_service_discovery():
      os.environ["NATS_SERVER"] = "nats://rogue-server:4222"
      assert connect()

  # GOOD -- monkeypatch auto-restores after each test
  def test_service_discovery(monkeypatch):
      monkeypatch.setenv("NATS_SERVER", "nats://rogue-server:4222")
      assert connect()
  ```

- **Mutable default arguments in test helpers.** **Always flag** any function
  with a mutable default (`[]`, `{}`, `set()`). Defaults are evaluated once and
  shared across calls, so mutations accumulate silently between invocations.

  ```python
  # BAD -- registry list is shared across all calls; flag this
  def register_workers(new_worker, registry=[]):
      registry.append(new_worker)
      return registry

  # GOOD -- None sentinel, fresh list each call
  def register_workers(new_worker, registry=None):
      if registry is None:
          registry = []
      registry.append(new_worker)
      return registry
  ```

  See also: `python-guidelines.md` > "Mutable default arguments" for the general rule.

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

## Linter Suppression (`# noqa`)

**Always flag** `# noqa` that suppresses warnings for anti-patterns documented in
these guidelines. If the linter caught a real problem (`E711` for `== None`,
`E712` for `== True`, `F841` for unused variable), fix the code.

```python
# BAD -- noqa hides the very bug the linter caught; flag this
assert error == None  # noqa: E711
result = compute()    # noqa: F841

# GOOD -- fix the code
assert error is None
result = compute()
assert result == expected
```

The only acceptable `# noqa` is for genuine false positives. Always explain:
`# noqa: F401 -- imported for side-effects`.

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

