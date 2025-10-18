# Fault Tolerance Tests

## Migration Tests

The migration directory contains tests for worker fault tolerance with migration support.

### test_request_migration_vllm_worker_failure
Tests worker fault tolerance when a worker is killed during request processing:

```bash
pytest tests/fault_tolerance/migration/test_vllm.py::test_request_migration_vllm_worker_failure -v -s
```

1. Starts a Dynamo frontend with round-robin routing
2. Starts 2 workers sequentially with vLLM backend
3. Sends a long completion request in a separate thread
4. Uses parallel polling to determine which worker received the request by checking for
   "New Request ID:" in logs
5. Kills the worker processing the request using SIGKILL
6. Verifies the request completes successfully despite the worker failure
7. Checks that migration occurred by looking for "Stream disconnected... recreating stream..."
   in frontend logs

### test_request_migration_vllm_graceful_shutdown
Tests worker fault tolerance with graceful shutdown (SIGTERM) during request processing:

```bash
pytest tests/fault_tolerance/migration/test_vllm.py::test_request_migration_vllm_graceful_shutdown -v -s
```

1. Starts a Dynamo frontend and 2 workers with the same configuration as above
2. Sends a long completion request in a separate thread
3. Uses parallel polling to determine which worker received the request
4. Gracefully shuts down the worker processing the request using SIGTERM with 10s timeout
5. Verifies the request completes successfully despite the graceful shutdown
6. Verifies migration occurred by checking frontend logs

### test_no_request_migration_vllm_worker_failure
Tests worker fault tolerance when migration is disabled (migration_limit=0):

```bash
pytest tests/fault_tolerance/migration/test_vllm.py::test_no_request_migration_vllm_worker_failure -v -s
```

1. Starts a Dynamo frontend with round-robin routing
2. Starts 2 workers sequentially with vLLM backend and migration_limit=0 (migration disabled)
3. Sends a long completion request in a separate thread
4. Uses parallel polling to determine which worker received the request
5. Kills the worker processing the request using SIGKILL
6. Verifies the request fails with status 500 (as expected without migration)
7. Verifies migration did NOT occur by checking for "Migration limit exhausted" in frontend logs

This test validates that when migration is disabled, requests fail when the processing worker
is killed, which is the expected behavior.

### test_no_request_migration_vllm_graceful_shutdown
Tests worker fault tolerance with graceful shutdown when migration is disabled (migration_limit=0):

```bash
pytest tests/fault_tolerance/migration/test_vllm.py::test_no_request_migration_vllm_graceful_shutdown -v -s
```

1. Starts a Dynamo frontend and 2 workers with vLLM backend and migration_limit=0
2. Sends a long completion request in a separate thread
3. Uses parallel polling to determine which worker received the request
4. Gracefully shuts down the worker processing the request using SIGTERM with 10s timeout
5. Verifies the request fails with status 500 (as expected without migration)
6. Verifies migration did NOT occur by checking frontend logs

This test validates that even with graceful shutdown, requests fail when migration is disabled,
confirming that migration_limit=0 properly disables the migration feature.

## Cancellation Tests

The cancellation directory contains tests for request cancellation functionality across multiple
API endpoints and deployment configurations.

### vLLM Cancellation Tests

#### test_request_cancellation_vllm_aggregated
Tests request cancellation in aggregated mode (single worker handles both prefill and decode):

```bash
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_aggregated -v -s
```

1. Starts a frontend and single vLLM worker
2. Tests cancellation across three scenarios:
   - Completion request
   - Chat completion request (non-streaming)
   - Chat completion request (streaming - reads 5 responses before cancelling)
3. For each scenario, polls for request ID in worker logs, cancels the request, and verifies
   cancellation in both worker and frontend logs

#### test_request_cancellation_vllm_decode_cancel
Tests request cancellation during decode phase in disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_decode_cancel -v -s
```

1. Starts a frontend, prefill worker, and decode worker
2. Sends a streaming chat completion request
3. Polls for request ID in decode worker and verifies it reached prefill worker
4. Reads 5 streaming responses (decode phase) before cancelling
5. Verifies cancellation messages in decode worker and frontend logs

#### test_request_cancellation_vllm_remote_prefill_cancel
Tests request cancellation during remote prefill phase in disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_remote_prefill_cancel -v -s
```

1. Starts a frontend, prefill worker, and decode worker
2. Sends a completion request with a very long prompt
3. Polls for request ID in both workers
4. Cancels during the prefill phase (before decode starts)
5. Verifies cancellation messages in both workers and frontend logs

### TRT-LLM Cancellation Tests

#### test_request_cancellation_trtllm_aggregated
Tests request cancellation in aggregated mode with TRT-LLM backend:

```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_aggregated -v -s
```

1. Starts a frontend and single TRT-LLM worker in `prefill_and_decode` mode
2. Tests cancellation across three scenarios:
   - Completion request
   - Chat completion request (non-streaming)
   - Chat completion request (streaming - reads 5 responses before cancelling)
3. For each scenario, polls for request ID in worker logs, cancels the request, and verifies
   cancellation in both worker and frontend logs

#### test_request_cancellation_trtllm_decode_first_decode_cancel
Tests cancellation during decode phase in decode-first disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_decode_first_decode_cancel -v -s
```

1. Starts a frontend with decode-first strategy (decode worker receives requests first)
2. Starts prefill worker, then decode worker
3. Sends a streaming chat completion request
4. Polls for request ID in decode worker and verifies it reached prefill worker
5. Reads 5 streaming responses during decode phase before cancelling
6. Verifies cancellation messages in decode worker and frontend logs

#### test_request_cancellation_trtllm_decode_first_remote_prefill_cancel
Tests cancellation during remote prefill in decode-first disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_decode_first_remote_prefill_cancel -v -s
```

1. Starts a frontend with decode-first strategy
2. Starts prefill worker, then decode worker
3. Sends a completion request with a very long prompt to ensure prefill phase
4. Polls for request ID in decode worker, then prefill worker (remote prefill)
5. Cancels during the prefill phase before decode starts
6. Verifies "Aborted Request ID" in prefill worker and "Aborted Remote Request ID" in decode
   worker

#### test_request_cancellation_trtllm_prefill_first_prefill_cancel
Tests cancellation during prefill phase in prefill-first disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_prefill_first_prefill_cancel -v -s
```

1. Starts a frontend with prefill-first strategy (prefill worker receives requests first)
2. Starts decode worker, then prefill worker
3. Sends a completion request with a very long prompt
4. Polls for request ID in prefill worker (local prefill)
5. Cancels during the prefill phase before reaching decode worker
6. Verifies cancellation in prefill worker and frontend logs

#### test_request_cancellation_trtllm_prefill_first_remote_decode_cancel
Tests cancellation during remote decode in prefill-first disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_prefill_first_remote_decode_cancel -v -s
```

1. Starts a frontend with prefill-first strategy
2. Starts decode worker, then prefill worker
3. Sends a streaming chat completion request
4. Polls for request ID in prefill worker, then decode worker (remote decode)
5. Reads 5 streaming responses during remote decode phase before cancelling
6. Verifies "Aborted Request ID" in decode worker and "Aborted Remote Request ID" in prefill
   worker

### SGLang Cancellation Tests

#### test_request_cancellation_sglang_aggregated
Tests request cancellation in aggregated mode with SGLang backend:

```bash
pytest tests/fault_tolerance/cancellation/test_sglang.py::test_request_cancellation_sglang_aggregated -v -s
```

1. Starts a frontend and single SGLang worker in aggregated mode
2. Tests cancellation across three scenarios:
   - Completion request
   - Chat completion request (non-streaming)
   - Chat completion request (streaming - reads 1 response before cancelling)
3. For each scenario, polls for Dynamo request ID, waits for SGLang to start processing,
   cancels the request, and verifies cancellation in both worker and frontend logs
4. Note: Currently flaky due to SGLang limitations with prefill cancellation

#### test_request_cancellation_sglang_decode_cancel
Tests request cancellation during remote decode phase in disaggregated setup:

```bash
pytest tests/fault_tolerance/cancellation/test_sglang.py::test_request_cancellation_sglang_decode_cancel -v -s
```

1. Starts a frontend, decode worker, and prefill worker (requires 2 GPUs)
2. Sends a streaming chat completion request
3. Polls for request ID in decode worker and verifies it reached prefill worker
4. Reads 1 streaming response to trigger SGLang ID logging
5. Waits for SGLang to start processing in decode worker
6. Cancels the request and verifies cancellation messages in all workers and frontend logs
