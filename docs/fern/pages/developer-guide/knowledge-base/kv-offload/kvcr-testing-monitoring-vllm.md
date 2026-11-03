---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Verify and Monitor KVCR with vLLM
subtitle: Confirm a cross-worker KV transfer and watch KVCR in production
---

## Verify a Transfer

Send a long prompt to one worker, then the same prompt to another worker, and check that the second request reuses the first one's KV cache. The routing headers pin the worker for this test only; in normal use, omit them and let the router choose.

The steps below use one worker with two data-parallel ranks, as in [Configure KVCR for vLLM](configure-kvcr-vllm.md). With two separate workers, repeat step 1 until both worker IDs appear, send each request with its worker's ID and without the `x-dynamo-dp-rank` header, and in step 6 read both workers' metrics: ports `8081` and `8082` in the Guard example, or port `9090` of each worker Pod through `kubectl port-forward` on Kubernetes. The source worker shows `source_write` and the target worker shows `remote_deliver`.

1. Get the worker ID:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":1,"nvext":{"extra_fields":["worker_id"]}}'
   ```

   The response contains `"decode_worker_id": <id>`. Use that number as `<worker-id>` below.

2. Create `request.json` with a prompt of at least 1,000 tokens that you have not sent before, for example a few pages of any document:

   ```json
   {"model": "Qwen/Qwen3-0.6B", "prompt": "<at least 1,000 tokens of new text>", "max_tokens": 8}
   ```

3. Send it to rank 0:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-id>' -H 'x-dynamo-dp-rank: 0' -d @request.json
   ```

   The response shows `"cached_tokens": 0`.

4. Wait 15 seconds. Offload, KV events, and router indexing are asynchronous.

5. Send the same request to rank 1:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-id>' -H 'x-dynamo-dp-rank: 1' -d @request.json
   ```

   `cached_tokens` in `usage.prompt_tokens_details` must be positive and a multiple of the block size. If it is `0`, see [Troubleshooting](#troubleshooting).

6. Confirm the transfer on the worker's metrics port (`DYN_SYSTEM_PORT`, `8081` in the configure examples):

   ```bash
   curl -s localhost:8081/metrics | grep '^vllm:kvcr_transfer_blocks_total'
   ```

   Rank 0 (`engine="0"`) shows `operation="source_write"` and rank 1 (`engine="1"`) shows `operation="remote_deliver"` with the same count. With a KVCR pool, the source also shows `local_fill`.

To run the check again, change the prompt in `request.json`; a prompt that is already cached on both ranks does not transfer.

## Monitor

Scrape each worker's `/metrics` (requires `DYN_SYSTEM_PORT` and `"enable_telemetry": true`):

| Metric | Use |
|---|---|
| `vllm:kvcr_transfer_blocks_total{operation}` | Blocks moved: `source_write`, `remote_deliver`, `local_fill`, `local_deliver`. |
| `vllm:kvcr_transfer_bytes_total{operation}` | Bytes moved. |
| `vllm:kvcr_duration_seconds{scope,result}` | Latency and outcome per stage. |
| `vllm:kvcr_state{resource}` | Current state, such as `block_records` and `in_flight_ops`. |

Alert on failed or partial deliveries:

```text
increase(vllm:kvcr_duration_seconds_count{scope=~"remote_deliver|remote_fetch",result=~"partial|failed"}[5m]) > 0
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Worker does not start | `router_capabilities` includes `router_hint`; `self_describing_kv_events` is `true`; every control and KV-events port is unique; `/dev/shm` covers `cpu_bytes_to_use` for every rank. |
| `cached_tokens` stays `0` | The frontend runs `--router-mode kv`; the frontend's `dynamo_component_kv_cache_events_applied` is increasing; the worker log has no `KV routing remains disabled`. Then retry with a new prompt in `request.json`. |
| Worker hangs at startup, or transfers fail across hosts | Set `UCX_NET_DEVICES` to the one NIC attached to the worker's GPU (`PIX` in `nvidia-smi topo -m`); with every NIC visible, workers can stall while exchanging transfer metadata. |
| `remote_deliver` partial or failed counts rise under load | Raise `operation_timeout_ms` (default `1000`) and keep `abandon_timeout_ms` at least twice it. |
| No transfer across hosts | `control_advertise_host` is reachable from the other worker, not `127.0.0.1`. |
| Peers cannot read a crashed worker's Guard | The state agent is running and the worker sets `dynamo_state_agent`. |
| Log shows `kv_offload_tiering_cascade_job_failures` | Expected in peer-only setups, once per new prefix. It does not affect transfers. |

## Related Topics

- [Configure KVCR for vLLM](configure-kvcr-vllm.md)
- [Run KVCR on Kubernetes with vLLM](kvcr-kubernetes-vllm.md)
- [Run KVCR Guard on Kubernetes with vLLM](kvcr-guard-kubernetes-vllm.md)
