# TESTING.md - Test Structure and Practices

## Overview

Multi-language test suite covering Rust (unit/integration), Python (pytest), and E2E infrastructure tests. Tests are distributed across library crates, component packages, and a top-level `tests/` directory.

---

## Rust Testing

### Framework
- **Built-in**: Rust `#[test]` and `#[cfg(test)]` modules
- **rstest**: Fixture-based parameterized tests (`rstest` crate)
- **proptest**: Property-based testing for invariant validation
- **tokio::test**: Async test runtime for async code

### Test Organization
Tests live alongside source code in `#[cfg(test)]` modules or in separate `tests.rs` / `tests/` files:

```
lib/kvbm-logical/src/
  pools/
    tests.rs          ← unit tests
    block_proptest.rs ← property-based tests
  registry/tests.rs
  manager/tests.rs
  events/tests.rs

lib/kv-router/src/
  indexer/tests.rs
  test_utils.rs       ← shared test helpers

lib/kvbm-connector/src/connector/worker/tests.rs
lib/kvbm-physical/src/transfer/testing.rs
lib/llm/tests/        ← integration-style tests
```

### Test Utilities
- Testing utilities are gated behind a `testing` feature flag: `cargo build -p kvbm-logical --features testing`
- Shared helpers in `src/testing/` modules: `TestBlockBuilder`, `BlockSequenceBuilder`, `create_test_manager()`
- `test_utils.rs` files export helpers for use across test modules

### Running Rust Tests
```bash
# Run all tests for a crate
cargo test -p kvbm-logical --lib

# Run a specific test
cargo test -p kvbm-logical --lib test_name

# Run tests in a specific module
cargo test -p kvbm-logical --lib registry::tests

# Run with testing feature enabled
cargo test -p kvbm-logical --lib --features testing
```

### Patterns
- **RAII/state-machine tests**: Tests verify type-state transitions (e.g., `MutableBlock` → `CompleteBlock` → `ImmutableBlock`)
- **Property-based**: `proptest` in `block_proptest.rs` for pool invariants
- **Fixture modules**: `pub(crate) mod fixtures` with shared block/pool builders

---

## Python Testing

### Framework
- **pytest**: Primary test runner
- **pytest-asyncio**: Async test support
- **pytest-xdist**: Parallel test execution
- **unittest.mock**: Mocking (standard library)
- **filelock**: Test isolation for shared resources

### Test Organization
```
tests/                         ← top-level integration/E2E tests
  conftest.py                  ← global fixtures, port allocation, managed processes
  utils/
    constants.py               ← TEST_MODELS, DefaultPort
    managed_process.py         ← subprocess lifecycle management
    port_utils.py              ← ServicePorts, allocate_port/ports
    test_output.py

components/src/dynamo/
  vllm/tests/
    conftest.py
    test_vllm_unit.py
    test_vllm_kv_events_api.py
    test_vllm_prompt_embeds.py
  trtllm/tests/
    conftest.py
    test_trtllm_unit.py
    test_trtllm_main_init.py
    test_trtllm_handler_base.py
    test_trtllm_additional_metrics.py
    test_trtllm_autodeploy.py
    request_handlers/
      test_trtllm_prefill_handler.py
      test_trtllm_request_handler_factory.py
      test_trtllm_aggregated_handler.py
    multimodal/
      test_trtllm_embedding_fetcher.py
      test_trtllm_cuda_ipc.py
  sglang/tests/
    conftest.py
    test_sglang_unit.py
    test_sglang_memory_occupation_handlers.py
    test_sglang_image_diffusion_handler.py
    test_sglang_prometheus_utils.py
  frontend/tests/
    test_sglang_tool_calls.py
    test_sglang_processor_unit.py
    test_sglang_processor_api.py
  common/tests/
    test_storage.py
    configuration/test_utils.py
    multimodal/test_async_encoder_cache.py
    multimodal/test_embedding_transfer.py
    memory/test_multimodal_embedding_cache_manager.py
```

### Pytest Markers
Defined in `pyproject.toml` and mirrored in `tests/conftest.py`:
- `pre_merge` — run before merging PRs
- `post_merge` — run after merge
- `parallel` — safe to run with pytest-xdist
- `nightly` — nightly CI
- `weekly` — weekly CI
- `gpu_0` — tests that don't require GPU

### Running Python Tests
```bash
# Run all tests
pytest tests/

# Run specific component tests
pytest components/src/dynamo/vllm/tests/

# Run by marker
pytest -m pre_merge

# Run in parallel
pytest -n auto -m parallel
```

### Patterns
- **conftest.py fixtures**: Port allocation, managed process lifecycle, temp directories
- **Managed processes**: `ManagedProcess` utility wraps subprocess lifecycle for integration tests
- **Port management**: `allocate_port`/`deallocate_port` with `FileLock` for parallel test isolation

---

## E2E Tests

Located in `tests/` subdirectories for specific scenarios:
- `tests/fault_tolerance/` — GPU memory service and deploy fault tolerance
- `tests/planner/` — Planner component tests
- `tests/serve/` — Serving integration tests
- `tests/frontend/` — Frontend API tests
- `tests/deploy/` — Deployment tests

Each subdirectory has its own `conftest.py` for scenario-specific fixtures.

---

## Go Testing (Operator)

The Kubernetes operator (`deploy/Kubernetes/operator/`) uses:
- `testing` package — standard Go tests
- **Ginkgo/Gomega** — BDD-style controller tests
- `httptest` — HTTP server mocking
- `sigs.k8s.io/controller-runtime/pkg/client/fake` — Kubernetes fake client
- Table-driven tests with `[]struct{ name, input, expected }` pattern

---

## Coverage and CI

- Rust: `cargo test` run per crate in CI
- Python: pytest with markers controlling which tests run per CI stage
- No explicit coverage thresholds found, but marker-gated tests ensure critical paths run pre-merge
