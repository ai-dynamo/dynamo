# GPU Memory Service Fault Tolerance Tests

## Overview

These tests validate the GPU Memory Service shadow engine architecture for fault tolerance. The GPU Memory Service enables VA-stable (virtual address stable) sleep/wake for model weights, allowing engines to be suspended and resumed without losing their GPU memory mappings.

Both vLLM and SGLang backends are supported through parametrized tests that automatically skip if the backend is not installed.

## Architecture

The GPU Memory Service shadow engine pattern works as follows:

1. **GPU Memory Service Servers**: One per GPU device, manages physical memory allocations with connection-based RW/RO locking
2. **Shadow Engine**: A vLLM or SGLang engine that loads weights via GPU Memory Service, then sleeps to release GPU memory while keeping weights in shared memory
3. **Primary Engine**: The active engine serving inference requests
4. **Failover**: When the primary engine fails, the shadow engine can be woken up with VA-stable weights to take over

## Test Files

### `test_shadow_engine_failover.py`

Contains parametrized tests that run for both vLLM and SGLang backends:

1. **`test_gpu_memory_service_shadow_engine_failover[vllm]`** / **`test_gpu_memory_service_shadow_engine_failover[sglang]`**: Full end-to-end test of the shadow engine failover flow:
   - Start GPU Memory Service for each GPU device
   - Start a shadow engine and put it to sleep
     - vLLM: `/engine/sleep`
     - SGLang: `/engine/release_memory_occupation`
   - Start a primary engine and run inference
   - Kill the primary engine (simulating failure)
   - Wake the shadow engine and verify it can serve inference
     - vLLM: `/engine/wake`
     - SGLang: `/engine/resume_memory_occupation`

2. **`test_gpu_memory_service_basic_sleep_wake[vllm]`** / **`test_gpu_memory_service_basic_sleep_wake[sglang]`**: Simpler test validating basic sleep/wake functionality:
   - Start engine with GPU Memory Service
   - Run initial inference
   - Sleep the engine
   - Wake the engine
   - Run inference after wake

## Backend Detection

Tests automatically detect which backends are available using `importlib.util.find_spec()`:
- If vLLM is not installed, `[vllm]` tests are skipped with "vLLM not installed"
- If SGLang is not installed, `[sglang]` tests are skipped with "SGLang not installed"

This allows the same test file to work in both vLLM and SGLang devcontainers without failing.

## Markers

- `vllm` - Applied to vLLM-specific test parameters
- `sglang` - Applied to SGLang-specific test parameters
- `gpu_2` - Requires 2 GPUs (for TP=2 configurations)
- `gpu_1` - Can run with single GPU (basic test)
- `e2e` - End-to-end test
- `nightly` - Resource-intensive, runs in nightly CI
- `fault_tolerance` - Fault tolerance test category

## Running the Tests

### Run all tests (skips unavailable backends)

```bash
pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py -s
```

### Run only vLLM tests

```bash
pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py -k vllm -s
```

### Run only SGLang tests

```bash
pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py -k sglang -s
```

### Basic Sleep/Wake Test (Single GPU)

```bash
pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py::test_gpu_memory_service_basic_sleep_wake -s
```

### Full Failover Test (2 GPUs)

```bash
pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py::test_gpu_memory_service_shadow_engine_failover -s
```

### With TP=2 (requires 2+ GPUs)

```bash
GPU_MEMORY_SERVICE_TP=2 pytest -v tests/fault_tolerance/gpu_memory_service/test_shadow_engine_failover.py -s
```

### With a different model

```bash
GPU_MEMORY_SERVICE_TEST_MODEL="Qwen/Qwen3-14B" GPU_MEMORY_SERVICE_TP=2 pytest -v tests/fault_tolerance/gpu_memory_service/ -s
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_MEMORY_SERVICE_TEST_MODEL` | `Qwen/Qwen3-0.6B` | Model to use for testing |
| `GPU_MEMORY_SERVICE_TP` | `1` | Tensor parallelism degree |

## Notes

- **MoE Models**: Due to meta-model creation, GPU Memory Service does not work with MoE models because of `.item()` calls in `__init__`.
- **Port Conflicts**: Tests automatically allocate dynamic ports to avoid conflicts.
- **Socket Files**: Tests clean up Unix socket files automatically.
- **Backend-specific ports**: vLLM uses NIXL ports, SGLang uses `--port` and `--disaggregation-bootstrap-port`.

## Related Documentation

- vLLM manual testing instructions: `components/src/dynamo/vllm/gpu_memory_service_adapters/TESTING.md`
- SGLang manual testing instructions: `components/src/dynamo/sglang/gpu_memory_service_adapters/TESTING.md`
- GPU Memory Service implementation: `lib/gpu_memory_service/`
- vLLM GPU Memory Service adapters: `components/src/dynamo/vllm/gpu_memory_service_adapters/`
- SGLang GPU Memory Service adapters: `components/src/dynamo/sglang/gpu_memory_service_adapters/`
