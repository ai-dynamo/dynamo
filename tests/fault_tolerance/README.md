# Fault Tolerance Tests

This directory contains end-to-end tests for Dynamo's fault tolerance capabilities.

## Tests

### `test_request_migration.py`

Tests worker fault tolerance with migration support using the `test_request_migration_vllm` function. This test:

0. Downloads the DeepSeek-R1-Distill-Llama-8B model from HuggingFace if not already cached
1. Starts a Dynamo frontend using `python -m dynamo.frontend` with round-robin routing
2. Starts 2 workers sequentially using `python3 -m dynamo.vllm` with specific configuration:
   - Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
   - `--enforce-eager`, `--gpu-memory-utilization 0.45`
   - `--max-model-len 8192`, `--migration-limit 3`
3. Waits for both workers to be fully ready (looking for "Reading Events from" messages)
4. Sends a test request ("Who are you?", 100 tokens) to determine which worker handles requests
5. Determines primary/backup worker roles based on round-robin routing and log analysis
6. Sends a long completion request ("Tell me a long long long story about yourself?", 8000 tokens) in a separate thread
7. Waits 0.5 seconds, then kills the primary worker using SIGKILL process group termination
8. Verifies the request completes successfully despite the worker failure (with 240s timeout)
9. Checks that the frontend logs contain "Stream disconnected... recreating stream..." indicating migration occurred

## Prerequisites

- vLLM backend installed (`pip install ai-dynamo-vllm`)
- NATS and etcd services running (provided by `runtime_services` fixture)
- Access to DeepSeek-R1-Distill-Llama-8B model (automatically downloaded from HuggingFace)
- Sufficient GPU memory (test uses 0.45 GPU memory utilization)

## Running the Tests

To run the fault tolerance tests:

```bash
# Run all fault tolerance tests
pytest /workspace/tests/fault_tolerance

# Run specific test with verbose output
pytest /workspace/tests/fault_tolerance/test_request_migration.py::test_request_migration_vllm -v

# Run with specific markers
pytest -m "e2e and vllm" /workspace/tests/fault_tolerance

# Run with debug logging
pytest /workspace/tests/fault_tolerance/test_request_migration.py::test_request_migration_vllm -v -s
```

## Test Markers

- `@pytest.mark.e2e`: End-to-end test
- `@pytest.mark.vllm`: Requires vLLM backend
- `@pytest.mark.gpu_1`: Requires single GPU access
- `@pytest.mark.slow`: Known to be slow (due to model loading and inference)

## Environment Variables

- `DYN_LOG`: Set to `debug` or `trace` for verbose logging (automatically set to `debug` by worker processes)
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are used for testing

## Expected Test Duration

The test typically takes 2-3 minutes to complete, including:
- Model download/loading time (if not cached) - can take 1-2 minutes for first run
- Worker startup and registration
- Request processing and response validation
- Worker failure simulation and migration
- Cleanup

## Troubleshooting

If tests fail:

1. Check that NATS and etcd services are running
2. Verify vLLM backend is properly installed
3. Ensure sufficient GPU memory is available (test requires ~45% GPU memory)
4. Check internet connectivity for model download from HuggingFace
5. Review test logs for specific error messages
6. Verify that the DeepSeek-R1-Distill-Llama-8B model can be accessed
