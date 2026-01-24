# GPU Memory Service Tests

## Overview

These tests validate the GPU Memory Service integration with vLLM. The GPU Memory Service enables VA-stable (virtual address stable) sleep/wake for model weights, allowing engines to be suspended and resumed without reloading weights from disk.

## Test Files

### `test_gms_sleep_wake.py`

Basic sleep/wake test:
- Start engine with GPU Memory Service
- Run initial inference
- Sleep the engine (verify memory is freed)
- Wake the engine
- Run inference after wake

### `test_gms_shadow_failover.py`

Full shadow engine failover test:
- Start GPU Memory Service for each GPU device
- Start a shadow engine and put it to sleep
- Start a primary engine and run inference
- Kill the primary engine (simulating failure)
- Wake the shadow engine and verify it can serve inference

## Running the Tests

### Basic Sleep/Wake Test

```bash
pytest -v tests/gpu_memory_service/test_gms_sleep_wake.py -s
```

### Shadow Engine Failover Test

```bash
pytest -v tests/gpu_memory_service/test_gms_shadow_failover.py -s
```

### With TP=2 (requires 2+ GPUs)

```bash
GPU_MEMORY_SERVICE_TP=2 pytest -v tests/gpu_memory_service/ -s
```

### With a different model

```bash
GPU_MEMORY_SERVICE_TEST_MODEL="Qwen/Qwen3-14B" GPU_MEMORY_SERVICE_TP=2 pytest -v tests/gpu_memory_service/ -s
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GPU_MEMORY_SERVICE_TEST_MODEL` | `Qwen/Qwen3-0.6B` | Model to use for testing |
| `GPU_MEMORY_SERVICE_TP` | `1` | Tensor parallelism degree |

## Markers

- `vllm` - Requires vLLM
- `gpu_1` - Requires 1 GPU (sleep/wake test)
- `gpu_2` - Requires 2 GPUs (failover test)
- `e2e` - End-to-end test
- `fault_tolerance` - Fault tolerance test category

## Related Documentation

- GPU Memory Service implementation: `lib/gpu_memory_service/`
- vLLM integration: `lib/gpu_memory_service/vllm_integration/`
