# Fault Tolerance Tests

## Migration Tests

The migration directory contains tests for worker fault tolerance with migration support across multiple backends (vLLM, SGLang, TRT-LLM) in both aggregated and disaggregated modes.

### Test Parameterization

All migration tests are parameterized with the following dimensions:

| Parameter IDs | Description |
|---------------|-------------|
| `migration_enabled`, `migration_disabled` | Controls whether migration is allowed |
| `worker_failure` (SIGKILL), `graceful_shutdown` (SIGTERM) | Worker termination method |
| `chat`, `completion` (skipped) | API endpoint to test |
| `stream`, `unary` (skipped) | Streaming vs unary responses |
| `nats`, `tcp` | Request plane transport |

### Test Matrix

Each backend (vLLM, SGLang, TRT-LLM) has the following test types:

| Test | Mode | Setup |
|------|------|-------|
| `test_request_migration_{backend}_aggregated` | Aggregated | 2 workers |
| `test_request_migration_{backend}_prefill` | Disaggregated | 1 decode + 2 prefill |
| `test_request_migration_{backend}_kv_transfer` | Disaggregated | 1 prefill + 2 decode |
| `test_request_migration_{backend}_decode` | Disaggregated | 1 prefill + 2 decode |

Where `{backend}` is one of: `vllm`, `sglang`, `trtllm`

### Common Test Flow

1. Start a Dynamo frontend with round-robin routing
2. Start workers (configuration varies by mode: aggregated or disaggregated)
3. Send a request (chat/completion, streaming/unary) in a background thread
4. Determine which worker received the request via log polling
5. For decode tests: wait for initial responses before termination
6. Terminate the worker processing the request (SIGKILL or SIGTERM)
7. Validate the request outcome based on `migration_limit`:
   - `migration_limit > 0`: Request succeeds, verify TTFT/TPOT if streaming and migration metrics
   - `migration_limit = 0`: Request fails with expected error
8. Verify migration behavior in frontend logs

**Run examples:**
```bash
# Run all vLLM migration tests
pytest tests/fault_tolerance/migration -m vllm -v -s

# Run aggregated or decode tests for SGLang
pytest tests/fault_tolerance/migration -m sglang -k "aggregated or decode" -v -s

# Run specific parameter combination
pytest tests/fault_tolerance/migration -m trtllm -k "aggregated and nats and stream and chat and worker_failure and migration_enabled" -v -s
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
