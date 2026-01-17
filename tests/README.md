# Dynamo Testing Guidelines

This document provides instructions for organizing, marking, and running tests in the Dynamo project. Follow these guidelines to ensure consistency and maintainability across the test suite.

---

## Test Organization: Where to Store Tests

### Directory Structure
```
dynamo/
├── lib/
│   ├── runtime/
│   │   ├── src/
│   │   │   └── lib.rs          # Rust code + unit tests inside
│   │   └── tests/              # Rust integration tests for runtime
│   ├── llm/
│   │   └── src/
│   │       └── lib.rs          # Rust code + unit tests inside
│   │   └── tests/              # Rust integration tests for llm
│   └── ...
├── components/
│   ├── src/dynamo/
│   |   └── planner/
│   │   │   └── tests/              # Python unit/integration tests for planner
│   |   └── router/
│   │   │   └── tests/
│   |   └── ...
│   ├── backend/
│   │   └── vllm
│   │   │    └── tests/             # Python unit/integration tests for backend
│   │   └── trtllm
│   │   │    └── tests/
│   │   └── trtllm
│   │   │    └── tests/
│   └── ...
├── tests/                      # End-to-end and cross-component tests
│   ├── serve/
│   ├── kvbm/
│   ├── benchmark/
│   ├── fault_tolerance/
│   └── ...
```
- Place **unit/integration tests** for a component in its `tests/` subfolder.
- Place **end-to-end (E2E) tests** and cross-component tests in `dynamo/tests/`.
- Name test files as `test_<component>_<flow>.py` for clarity.

### Test Types and Locations
| Type         | Description                              | Location              |
|--------------|------------------------------------------|-----------------------|
| Unit         | Single function/class, isolated          | `<component>/tests/`  |
| Integration  | Interactions between modules/services    | `<component>/tests/`  |
| End-to-End   | User workflows, CLI, API                 | `tests/serve/`, etc.  |
| Benchmark    | Performance/load                         | `tests/benchmark/`    |
| Stress       | Chaos, long-run, resource limits         | `tests/fault_tolerance/` |

---

## Test Marking: How to Mark Tests

Markers are required for all tests. They are used for test selection in CI and local runs.

### Marker Requirements
- Every test must have at least one **Lifecycle** marker, and **test type** and **Hardware** markers.
- **component** markers are required as applicable.

### Marker Table
| Category                | Marker(s)                | Description                        |
|-------------------------|--------------------------|------------------------------------|
| Lifecycle [required]    | pre_merge, post_merge, nightly,  weekly, release   | When the test should run           |
| Test Type [required]    | unit, integration, e2e, benchmark, stress, multimodal   | Nature of the test                 |
| Hardware [required]     | gpu_0, gpu_1, gpu_2,  gpu_4, gpu_8, h100      | Number/type of GPUs required       |
| Component/Framework     | vllm, trtllm, sglang, kvbm, planner, router    | Backend or component specificity   |
| Execution               | parallel                 | Test can run in parallel with pytest-xdist |
| Other                   | slow, skip, xfail, mypy, custom_build        | Special handling                   |

### Example
```python
@pytest.mark.integration
@pytest.mark.gpu_2
@pytest.mark.vllm
def test_kv_cache_multi_gpu_behavior():
    ...
```

### Lifecycle Marker Note
Use the marker for the earliest pipeline stage where the test must run (e.g., `@pytest.mark.pre_merge`). This ensures the test is included in that stage and all subsequent ones (e.g., nightly, release), as CI pipelines select tests marked for earlier stages.

**Example:**
If a test is marked with `@pytest.mark.pre_merge`, and the nightly pipeline runs:
```bash
pytest -m "e2e and (pre_merge or post_merge or nightly)"
```
then this test will be included in the nightly run as well.

---

## Test Execution: How to Run Tests Locally and in CI

### Environment Setup
- Use the dev container for consistency.
- Install dependencies as specified in `pyproject.toml`.
- Set the `HF_TOKEN` environment variable for HuggingFace downloads:
  ```bash
  export HF_TOKEN=your_token_here
  ```
- Model cache is located at `~/.cache/huggingface` to avoid repeated downloads.

### Running Tests
- Run all tests:
  ```bash
  pytest
  ```
- Run by marker:
  ```bash
  pytest -m "unit"
  pytest -m "integration and gpu_1"
  pytest -m "e2e and pre_merge"
  pytest -m "benchmark and vllm"
  ```
- Run by component:
  ```bash
  pytest -m planner
  pytest -m kvbm
  ```
- Show print/log output:
  ```bash
  pytest -s
  ```
- Run in container:
  ```bash
  ./container/build.sh --framework <backend>
  ./container/run.sh --mount-workspace -it -- pytest
  ./container/run.sh --mount-workspace -it -- pytest -m [optional markers]
  ```
- CI runs use the similar instructions as running inside the container. For example, running E2E tests as part of the nightly suite inside the Dynamo-VLLM container (which requires a single GPU) can be done with:
  ```bash
  ./container/run.sh --image $VLLM_IMAGE_NAME --name $VLLM_CONTAINER_NAME -- pytest -m "e2e and gpu_1 and (pre_merge or post_merge or nightly) "
  ```

#### Running tests locally outside of a container

To run tests outside of the development container, ensure that you have properly setup your environment and have installed the following dependencies in your `venv`:

```bash
uv pip install pytest-mypy
uv pip install pytest-asyncio
```
---

## Rust Testing: Organization and Execution

Rust tests in Dynamo are organized as follows:
- **Unit tests** are placed within the corresponding Rust source files (e.g., `lib.rs`) using `#[cfg(test)]` modules.
- **Integration tests** are placed in the crate's `tests/` directory and must be gated behind the `integration` feature.

### Test Segmentation by Features
- Use Cargo features to enable or disable groups of tests. For example:
  ```bash
  cargo test --features planner
  ```
- Place all integration tests behind the `integration` feature gate. This ensures they are only run when explicitly enabled:
  ```bash
  cargo test --features integration
  cargo test --all-features
  ```

### Marking Slow or Special-Case Tests
- Use `#[ignore]` to mark slow or special-case tests. These tests will not run by default and must be explicitly included:
  ```bash
  cargo test -- --ignored
  ```

### Example
```rust
#[cfg(test)]
mod kv_cache_tests {
    #[test]
    fn test_kv_cache_basic() {
        // ...
    }

    #[test]
    #[ignore]
    fn test_kv_cache_long_running() {
        // ...
    }
}
```

### CI Integration
- CI runs integration tests using either `cargo test --features integration` or `cargo test --all-features`.
- Use feature gates to control which tests are included in each CI pipeline.

---

## Additional Requirements and Troubleshooting

- Tests must be deterministic; flaky tests are not permitted.
- Performance targets:
  - Unit: <15 seconds per suite
  - Integration: <5 minutes (premerge), <5 minutes (postmerge)
  - E2E: <15 minutes (premerge).
- If a test is not running, verify the filename, markers, and folder location.
- For flaky tests, remove sources of randomness or set a fixed seed. Avoid unnecessary network calls.
- For slow tests, profile and optimize, or mark as `@pytest.mark.slow`.
- If model downloads fail, ensure `HF_TOKEN` is set and network access is available.
- If coverage is insufficient, add more tests or refactor code for better testability.

---

## References
- [pytest documentation](https://docs.pytest.org/en/stable/)

For further assistance, contact the Dynamo development team.
