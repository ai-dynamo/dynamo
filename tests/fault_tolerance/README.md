# Fault Tolerance Tests

This directory contains end-to-end tests for Dynamo's fault tolerance capabilities.

## Tests

### `test_request_migration.py`

Tests worker fault tolerance with migration support using the `test_request_migration_vllm` function. This test:

1. Starts a Dynamo frontend using `python -m dynamo.frontend` with round-robin routing
2. Starts 2 workers sequentially using `python3 -m dynamo.vllm` with specific configuration
3. Waits for both workers to be fully ready (looking for "Reading Events from" messages)
4. Sends a test request ("Who are you?", 100 tokens) to determine which worker handles requests
5. Determines primary/backup worker roles based on round-robin routing and log analysis
6. Sends a long completion request (8000 tokens) in a separate thread
7. Waits 0.5 seconds, then kills the primary worker using SIGKILL process group termination
8. Verifies the request completes successfully despite the worker failure
9. Checks that the frontend logs contain "Stream disconnected... recreating stream..." indicating migration occurred

## Prerequisites

- vLLM backend installed (`pip install ai-dynamo-vllm`)
- NATS and etcd services running (provided by `runtime_services` fixture)
- Access to Meta-Llama-3.1-8B-Instruct model (may require HuggingFace token for gated models)

## Running the Tests

To run the fault tolerance tests:

```bash
# Run all fault tolerance tests
pytest tests/fault_tolerance/

# Run specific test with verbose output
pytest tests/fault_tolerance/test_request_migration.py::test_request_migration_vllm -v

# Run with specific markers
pytest -m "e2e and vllm" tests/fault_tolerance/

# Run with debug logging
DYN_LOG=debug pytest tests/fault_tolerance/test_request_migration.py::test_request_migration_vllm -v -s
```

## Test Markers

- `@pytest.mark.e2e`: End-to-end test
- `@pytest.mark.vllm`: Requires vLLM backend
- `@pytest.mark.slow`: Known to be slow (due to model loading and inference)

## Environment Variables

- `HF_TOKEN`: Required for accessing gated models like Meta-Llama-3.1-8B-Instruct
- `DYN_LOG`: Set to `debug` or `trace` for verbose logging
- `CUDA_VISIBLE_DEVICES`: Control which GPUs are used for testing

## Expected Test Duration

The test typically takes 1.5-2 minutes to complete, including:
- Model download/loading time (if not cached)
- Worker startup and registration
- Request processing and response validation
- Worker failure simulation and migration
- Cleanup

## Troubleshooting

If tests fail:

1. Check that NATS and etcd services are running
2. Verify vLLM backend is properly installed
3. Ensure sufficient GPU memory is available
4. Check HuggingFace token if using gated models
5. Review test logs for specific error messages
