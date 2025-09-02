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
3. Waits for both workers to be fully ready (health check returns "ready" status)
4. Sends a test request ("Who are you?", 100 tokens) to determine which worker handles requests
5. Determines primary/backup worker roles based on round-robin routing and log analysis
6. Sends a long completion request ("Tell me a long long long story about yourself?", 8000 tokens) in a separate thread
7. Waits 0.5 seconds, then kills the primary worker using SIGKILL process group termination
8. Verifies the request completes successfully despite the worker failure (with 240s timeout)
9. Checks that the frontend logs contain "Stream disconnected... recreating stream..." indicating migration occurred

### `test_request_cancellation.py`

Tests request cancellation functionality across multiple API endpoints and deployment configurations. Contains three test functions:

#### `test_request_cancellation_vllm`
Tests basic request cancellation with a single worker:

0. Downloads the DeepSeek-R1-Distill-Llama-8B model from HuggingFace if not already cached
1. Starts a Dynamo frontend using `python -m dynamo.frontend` with debug logging enabled
2. Starts a single worker using `python3 -m dynamo.vllm` with specific configuration:
   - Model: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
   - `--enforce-eager`, `--gpu-memory-utilization 0.45`, `--max-model-len 8192`, `--migration-limit 3`
   - Debug logging enabled on port 8081
3. Tests request cancellation across three scenarios:
   - **Completion API**: `/v1/completions` endpoint cancellation
   - **Chat Completion API (non-streaming)**: `/v1/chat/completions` endpoint cancellation
   - **Chat Completion API (streaming)**: `/v1/chat/completions` with streaming cancellation
4. For each scenario:
   - Sends a long request with 1-second timeout to trigger cancellation
   - Validates that cancellation messages appear in both frontend and worker logs
   - Uses incremental log offset tracking to avoid false positives from previous tests
5. Checks for specific cancellation patterns:
   - Frontend log: "issued control message Kill to sender"
   - Worker log: "Aborted Request ID: <request_id>" matching the "New Request ID: <request_id>"

#### `test_request_cancellation_vllm_decode`
Tests request cancellation during disaggregated decode phase:

0. Downloads the DeepSeek-R1-Distill-Llama-8B model from HuggingFace if not already cached
1. Starts a Dynamo frontend using `python -m dynamo.frontend` with debug logging enabled
2. Starts a prefill worker using `python3 -m dynamo.vllm --is-prefill-worker` on port 8082
3. Starts a decode worker using `python3 -m dynamo.vllm` on port 8081
4. Tests completion request cancellation in the disaggregated setup
5. Validates cancellation messages appear in prefill worker, decode worker, and frontend logs
6. Checks for specific patterns:
   - Frontend log: "issued control message Kill to sender"
   - Decode worker log: "Aborted Request ID: <request_id>"
   - Prefill worker log: "New Prefill Request ID: <request_id>"

#### `test_request_cancellation_vllm_prefill`
Tests request cancellation during disaggregated prefill phase:

- (Skipped until request cancellation can cancel before receiving the first response)

## Prerequisites

- vLLM backend installed
- NATS and etcd services running (provided by `runtime_services` fixture)
- Access to DeepSeek-R1-Distill-Llama-8B model (automatically downloaded from HuggingFace)
- Sufficient GPU memory

## Running the Tests

To run the fault tolerance tests:

```bash
# Run all fault tolerance tests
pytest -m "e2e and vllm" /workspace/tests/fault_tolerance

# Run specific test functions with debug logging
pytest /workspace/tests/fault_tolerance/test_request_migration.py::test_request_migration_vllm -v -s
pytest /workspace/tests/fault_tolerance/test_request_cancellation.py::test_request_cancellation_vllm -v -s
pytest /workspace/tests/fault_tolerance/test_request_cancellation.py::test_request_cancellation_vllm_decode -v -s
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

The tests typically take 2-3 minutes to complete each, including:
- Model download/loading time (if not cached) - can take 1-2 minutes for first run
- Worker startup and registration
- Request processing and response validation
- Worker failure simulation and migration (for migration test) / Request cancellation validation (for cancellation tests)
- Cleanup

## Troubleshooting

If tests fail:

1. Check that NATS and etcd services are running
2. Verify vLLM backend is properly installed
3. Ensure sufficient GPU memory is available
4. Check internet connectivity for model download from HuggingFace
5. Review test logs for specific error messages
6. Verify that the DeepSeek-R1-Distill-Llama-8B model can be accessed
7. For cancellation tests: Check that timeout-based cancellation is working properly and cancellation patterns appear in logs
8. For migration tests: Verify worker process termination and stream recreation behavior
9. For disaggregated cancellation tests: Ensure both prefill and decode workers are properly started and cancellation works across the disaggregated setup
