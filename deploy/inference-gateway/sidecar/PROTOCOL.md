# vLLM NIXL P/D handoff protocol lock

This file pins the wire contract the decode sidecar's `vllm_nixl` adapter
implements. Nothing here is inferred: every field, flag, and nullable structure
below was read from the pinned vLLM source, and the fixtures are the shape that
source emits.

- Repository the adapter lives in: `ai-dynamo/dynamo`
- Adapter source: `deploy/inference-gateway/sidecar/src/vllm_nixl.rs`
- **Pinned vLLM release: `v0.29.0`**
- Protocol revision string: `vllm-v0.29.0-nixl-pull`
  (`DYN_VLLM_NIXL_PROTOCOL_VERSION`; a mismatch is refused at startup)

## Sources reviewed at that tag

| What | Path in `vllm-project/vllm` at `v0.29.0` |
|---|---|
| Producer-side handoff construction | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py` |
| Block clipping used before the exchange | `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py` |
| `BlockIds` definition | `vllm/distributed/kv_transfer/kv_connector/utils.py` |
| Reference (non-production) P/D HTTP rewriter | `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` |

The toy proxy is a **protocol reference only**. The adapter deliberately does
not copy three of its behaviours: it does not continue to decode when the
handoff is missing, it does not force every response through a streaming
wrapper, and it does not buffer a request or response without a bound.

## 1. Fixtures

### 1.1 Derived prefill request

Produced by the adapter from the client request. The caller's `stream`,
`stream_options`, `min_tokens`, and `min_completion_tokens` are replaced or
removed; everything else is copied verbatim.

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 1,
  "temperature": 0.7,
  "stream": false,
  "kv_transfer_params": {
    "do_remote_decode": true,
    "do_remote_prefill": false,
    "remote_engine_id": null,
    "remote_block_ids": null,
    "remote_host": null,
    "remote_port": null
  }
}
```

### 1.2 Successful prefill response (sanitized)

The prefill worker answers with an ordinary one-token completion **plus** the
handoff. The completion content is discarded; only `kv_transfer_params` is
forwarded.

```json
{
  "id": "chatcmpl-prefill",
  "object": "chat.completion",
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": ""},
      "finish_reason": "length"
    }
  ],
  "usage": {"prompt_tokens": 128, "completion_tokens": 1, "total_tokens": 129},
  "kv_transfer_params": {
    "do_remote_prefill": true,
    "do_remote_decode": false,
    "remote_block_ids": [[1, 2, 3], [7]],
    "remote_engine_id": "engine-abc",
    "remote_request_id": "req-123",
    "remote_host": "10.0.0.5",
    "remote_port": 5600,
    "tp_size": 1,
    "dcp_size": 1,
    "pp_size": 1,
    "remote_num_tokens": 128,
    "remote_blocks_expiry_time": null,
    "transfer_mode": "pull"
  }
}
```

### 1.3 Final decode request

The **original** client request with the validated handoff inserted. Note that
`stream`, `stream_options`, and `min_tokens` are the caller's values, not the
prefill leg's.

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "hello"}],
  "max_tokens": 64,
  "min_tokens": 4,
  "temperature": 0.7,
  "stream": true,
  "stream_options": {"include_usage": true},
  "kv_transfer_params": {
    "do_remote_prefill": true,
    "do_remote_decode": false,
    "remote_block_ids": [[1, 2, 3], [7]],
    "remote_engine_id": "engine-abc",
    "remote_request_id": "req-123",
    "remote_host": "10.0.0.5",
    "remote_port": 5600,
    "tp_size": 1,
    "dcp_size": 1,
    "pp_size": 1,
    "remote_num_tokens": 128,
    "remote_blocks_expiry_time": null,
    "transfer_mode": "pull"
  }
}
```

## 2. Handoff field contract

Every field below is read by `NixlPullConnectorScheduler`
(`pull_scheduler.py::get_num_new_matched_tokens` /
`update_state_after_alloc`). The adapter validates the first block and
preserves the rest.

| Field | Type | Required by adapter | Notes |
|---|---|---|---|
| `do_remote_prefill` | bool | yes, must be `true` | `get_num_new_matched_tokens` only pulls when this is set |
| `do_remote_decode` | bool | yes, must be `false` | `true` is the producer-side flag |
| `remote_block_ids` | array of arrays of non-negative int | yes, present | Per KV cache group. **Nesting is significant.** |
| `remote_engine_id` | non-empty string | yes | Producer engine identity |
| `remote_request_id` | non-empty string | yes | Producer-side request id; forwarded verbatim |
| `remote_host` | non-empty string | yes | Side-channel host, **not** an HTTP destination |
| `remote_port` | int in `1..=65535` | yes | Side-channel port, **not** an HTTP destination |
| `tp_size` | int | preserved | Producer tensor-parallel size |
| `dcp_size` | int | preserved | Producer decode-context-parallel size |
| `pp_size` | int | preserved | Producer pipeline-parallel size |
| `remote_num_tokens` | int | preserved | Tokens the producer actually computed |
| `remote_blocks_expiry_time` | float or `null` | preserved | `null` for the prefill side |
| `transfer_mode` | string | preserved | Connector transfer mode |
| any other field | any | preserved | Unknown fields are forwarded unchanged |

`remote_block_ids` is typed `BlockIds = tuple[list[int], ...] | list[list[int]]`
in `kv_connector/utils.py`, which serializes as an array of arrays. Flattening it
would mis-address the KV cache groups.

### Required, optional, and nullable

- **Required**: the fields marked *yes* above. A missing one is a protocol error
  (`invalid_prefill_handoff`) and the decode leg does not run.
- **Optional**: `tp_size`, `dcp_size`, `pp_size`, `remote_num_tokens`,
  `transfer_mode`.
- **Nullable**: `remote_blocks_expiry_time` (the producer sets `null` on the
  prefill side).
- **Legitimately empty**: `remote_block_ids` may be `[]`, `[[]]`, or `[[], []]`.
  This is what a full local prefix-cache hit looks like — the connector still
  needs the notification so the producer frees its blocks. An empty structure is
  therefore **not** treated as a missing handoff.

## 3. Request identity

- The HTTP `x-request-id` header is forwarded unchanged on both legs. It is
  client-controlled and is **not** assumed unique.
- `remote_request_id` is the producer's own request id and is forwarded verbatim
  inside the handoff. The adapter never rewrites it to match the HTTP id, and
  never derives one from the other.
- No process-wide request-to-handoff cache exists: the handoff lives only for
  the duration of one adapter call.

## 4. Support status of request variants

| Variant | Status | Behaviour |
|---|---|---|
| `stream: false` | supported | decode response relayed, non-streaming |
| `stream: true` (+ `stream_options`) | supported | decode response streamed with backpressure |
| `messages`, `tools`, `response_format`, sampling params | supported | forwarded verbatim, unmodified |
| unknown JSON fields / vendor extensions | supported | forwarded verbatim on both legs |
| `n == 1` or absent | supported | forwarded verbatim |
| `n != 1` | **rejected before prefill** | `400 unsupported_pd_variant`; the protocol defines one handoff per request, and the adapter will not silently set `n = 1` |
| client-supplied non-empty `kv_transfer_params` | **rejected before prefill** | `400 invalid_pd_request`; the adapter owns this field on the P/D path |
| `kv_transfer_params: null` or `{}` | tolerated | treated as absent |
| request body over the configured cap | **rejected while reading** | `413 pd_request_too_large` |
| prefill response over the configured cap | **rejected while reading** | `502 prefill_response_too_large` |
| `/v1/completions`, Responses API | out of scope | only `POST /v1/chat/completions` is routed |

## 5. Error taxonomy

| Stage | Situation | Status | `error.code` |
|---|---|---|---|
| pre-flight | invalid EPP header | 502 | `invalid_epp_metadata` (unchanged) |
| pre-flight | body not a JSON object | 400 | `invalid_pd_request` |
| pre-flight | unsupported variant (`n != 1`) | 400 | `unsupported_pd_variant` |
| pre-flight | client-injected handoff | 400 | `invalid_pd_request` |
| pre-flight | body over cap | 413 | `pd_request_too_large` |
| pre-flight | request body did not arrive in time | 408 | `pd_request_timeout` |
| prefill | connect failure | 502 | `prefill_upstream_unavailable` |
| prefill | read timeout (one gap between reads) | 504 | `prefill_upstream_timeout` |
| prefill | prefill leg exceeded its total deadline | 504 | `prefill_deadline_exceeded` |
| prefill | HTTP error status | upstream status | `prefill_upstream_error` |
| prefill | response over cap | 502 | `prefill_response_too_large` |
| prefill | non-JSON / truncated / SSE | 502 | `invalid_prefill_handoff` |
| handoff | missing, null, or non-object | 502 | `invalid_prefill_handoff` |
| handoff | field missing / wrong type / bad port / wrong flags | 502 | `invalid_prefill_handoff` |
| decode | connect failure | 502 | `decode_upstream_error` (unchanged) |
| decode | timeout | 504 | `decode_upstream_timeout` (unchanged) |
| decode | HTTP error before headers | upstream status | upstream body relayed |
| decode | stream error after headers | — | body terminated; no second status, no fabricated `[DONE]` |
| any | cancellation / force shutdown | 502 | `request_cancelled` (unchanged) |
| — | adapter disabled (default) | 501 | `pd_adapter_unavailable` (unchanged) |

An upstream error body is never relayed from the prefill leg: it may quote the
request. The upstream **status** is preserved so the caller can still classify
the failure.

## 6. Version-upgrade checklist

Re-verify all of the following before changing the pin:

1. `pull_scheduler.py::request_finished` — the returned dict's keys and their
   types. This is the authoritative handoff shape.
2. `get_num_new_matched_tokens` — which flags gate the pull, and whether
   `remote_num_tokens` or `kv_recompute_threshold` changed meaning.
3. `BlockIds` in `kv_connector/utils.py` — whether the block structure is still
   an array of arrays.
4. The producer-side request flags in the toy proxy or its successor — whether
   `do_remote_decode` / `do_remote_prefill` and the null placeholders are still
   the way to mark the prefill leg.
5. The integration launch script — connector role names, KV load policy, and any
   new required launch flags.
6. Regenerate the fixtures in section 1 from the new tag and update
   `SUPPORTED_VLLM_VERSION` / `SUPPORTED_PROTOCOL_VERSION` together.
7. Re-run `cargo test -p dynamo-epp-sidecar`, then the two-worker smoke test.
