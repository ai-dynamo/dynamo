# Fault Tolerance Tests

## Migration Tests

The migration directory contains tests for worker fault tolerance with migration support.

### Test Matrix

| Test | Shutdown Method | Migration Enabled | Expected Result | Verification |
|------|----------------|-------------------|-----------------|--------------|
| `test_request_migration_vllm_worker_failure` | SIGKILL (immediate) | Yes (default) | Request succeeds | "Stream disconnected... recreating stream..." in logs |
| `test_request_migration_vllm_graceful_shutdown` | SIGTERM (10s timeout) | Yes (default) | Request succeeds | "Stream disconnected... recreating stream..." in logs |
| `test_no_request_migration_vllm_worker_failure` | SIGKILL (immediate) | No (migration_limit=0) | Request fails (500) | "Migration limit exhausted" in logs |
| `test_no_request_migration_vllm_graceful_shutdown` | SIGTERM (10s timeout) | No (migration_limit=0) | Request fails (500) | "Migration limit exhausted" in logs |

### Common Test Flow

All migration tests follow this pattern:

1. Start a Dynamo frontend with round-robin routing
2. Start 2 vLLM workers sequentially
3. Send a long completion request (max_tokens=8192) in a separate daemon thread
4. Use parallel polling to determine which worker received the request (checks for "New Request ID:" in logs)
5. Terminate the worker processing the request (method varies by test)
6. Validate the request outcome (success or failure based on migration setting)
7. Verify migration behavior in frontend logs

**Run examples:**
```bash
# With migration enabled
pytest tests/fault_tolerance/migration/test_vllm.py::test_request_migration_vllm_worker_failure -v -s
pytest tests/fault_tolerance/migration/test_vllm.py::test_request_migration_vllm_graceful_shutdown -v -s

# With migration disabled
pytest tests/fault_tolerance/migration/test_vllm.py::test_no_request_migration_vllm_worker_failure -v -s
pytest tests/fault_tolerance/migration/test_vllm.py::test_no_request_migration_vllm_graceful_shutdown -v -s
```

## Cancellation Tests

The cancellation directory contains tests for request cancellation functionality across multiple
API endpoints, backends, and deployment configurations.

### Test Overview by Backend

#### vLLM Cancellation Tests

| Test | Mode | Cancellation Phase | Request Type | Setup |
|------|------|-------------------|--------------|-------|
| `test_request_cancellation_vllm_aggregated` | Aggregated | During generation | 3 scenarios: completion, chat, streaming chat | 1 worker |
| `test_request_cancellation_vllm_decode_cancel` | Disaggregated | Remote decode | Streaming chat (5 responses read) | Prefill + Decode workers |
| `test_request_cancellation_vllm_remote_prefill_cancel` | Disaggregated | Remote prefill | Completion (long prompt) | Prefill + Decode workers |

**Run examples:**
```bash
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_aggregated -v -s
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_decode_cancel -v -s
pytest tests/fault_tolerance/cancellation/test_vllm.py::test_request_cancellation_vllm_remote_prefill_cancel -v -s
```

#### TRT-LLM Cancellation Tests

| Test | Mode | Cancellation Phase | Request Type | Setup |
|------|------|--------------------|--------------|-------|
| `test_request_cancellation_trtllm_aggregated` | Aggregated | During generation | 3 scenarios: completion, chat, streaming chat | 1 worker (prefill_and_decode) |
| `test_request_cancellation_trtllm_disagg_decode_cancel` | Disaggregated | Remote decode | Streaming chat (5 responses read) | Prefill + Decode workers |
| `test_request_cancellation_trtllm_disagg_prefill_cancel` | Disaggregated | Remote prefill | Completion (long prompt) | Prefill + Decode workers |

**Run examples:**
```bash
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_aggregated -v -s
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_disagg_decode_cancel -v -s
pytest tests/fault_tolerance/cancellation/test_trtllm.py::test_request_cancellation_trtllm_disagg_prefill_cancel -v -s
```

#### SGLang Cancellation Tests

| Test | Mode | Cancellation Phase | Request Type | Setup | Notes |
|------|------|-------------------|--------------|-------|-------|
| `test_request_cancellation_sglang_aggregated` | Aggregated | During generation | 3 scenarios: completion, chat, streaming chat (1 response read) | 1 worker | ⚠️ Flaky: SGLang prefill cancellation issues |
| `test_request_cancellation_sglang_decode_cancel` | Disaggregated | Remote decode | Streaming chat (1 response read) | Decode + Prefill workers | Requires 2 GPUs |

**Run examples:**
```bash
pytest tests/fault_tolerance/cancellation/test_sglang.py::test_request_cancellation_sglang_aggregated -v -s
pytest tests/fault_tolerance/cancellation/test_sglang.py::test_request_cancellation_sglang_decode_cancel -v -s
```

### Common Cancellation Test Pattern

1. Start frontend and workers (configuration varies by test)
2. Send request (type varies by test scenario)
3. Poll for request ID in worker logs
4. For streaming: read N responses before cancellation
5. Cancel the request via API
6. Verify cancellation messages in worker and frontend logs

**Verification patterns:**
- Aggregated mode: "Aborted Request ID" in worker logs
- Disaggregated - prefill cancellation: "Aborted Request ID" in prefill worker (cancellation during prefill)
- Disaggregated - decode cancellation: "Aborted Request ID" in decode worker (cancellation during decode)
