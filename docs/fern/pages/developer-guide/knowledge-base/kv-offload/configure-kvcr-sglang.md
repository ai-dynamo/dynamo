---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configure KVCR for SGLang
subtitle: Run two SGLang workers on one GPU server that share KV cache through KVCR
---

> [!WARNING]
> Preview. Needs SGLang [#45647](https://github.com/sgl-project/sglang/pull/45647) (stacked on [#45646](https://github.com/sgl-project/sglang/pull/45646)) and the image recipe in KVCR [#114](https://github.com/ai-dynamo/kvcr/pull/114), both unmerged, and Dynamo with [#16117](https://github.com/ai-dynamo/dynamo/pull/16117). HiCache mode on one node only.

KVCR runs inside SGLang as a HiCache storage backend. SGLang stages KV in HiCache host memory; KVCR keeps it in its own KVCR pool and sends it to another worker when the Dynamo KV router attaches a source hint. To load KV straight into GPU memory instead, or to use Guard, see [Configure KVCR Linker Mode for SGLang](configure-kvcr-sglang-linker.md).

## Before You Begin

- The runtime image built from `Dockerfile.quick-start-sglang` in [KVCR #114](https://github.com/ai-dynamo/kvcr/pull/114) (pins SGLang `d365ebd`, Dynamo `7369d08`, KVCR `7186e7c`). No release image contains it. The image sets file-based discovery and TCP/ZMQ planes, so no etcd or NATS is needed on one host; this matches the tested setup.
- Two GPUs, one per worker.

## Start the Frontend

```bash
python3 -m dynamo.frontend --http-port 8000 --namespace kvcr --router-mode kv
```

## Start Two Workers

Run each worker in its own terminal. They differ only in GPU, ports, and the metrics port. `--max-total-tokens` keeps the example's GPU KV pool small, so the host tier sized from it stays small too.

Worker 0:

```bash
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=24001 python3 -m dynamo.sglang \
  --endpoint dyn://kvcr.backend.generate \
  --model-path Qwen/Qwen3-0.6B --tp-size 1 --page-size 64 --max-total-tokens 65536 \
  --enable-hierarchical-cache --hicache-ratio 1.5 \
  --hicache-mem-layout page_first --hicache-write-policy write_through \
  --hicache-storage-backend kvcr --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"control_advertise_host": "127.0.0.1", "control_port": 24100, "nixl_listen_port": 24200, "local_dram_bytes": 8589934592}' \
  --kv-events-config '{"publisher": "zmq", "topic": "", "endpoint": "tcp://*:24020"}'
```

Worker 1:

```bash
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=24002 python3 -m dynamo.sglang \
  --endpoint dyn://kvcr.backend.generate \
  --model-path Qwen/Qwen3-0.6B --tp-size 1 --page-size 64 --max-total-tokens 65536 \
  --enable-hierarchical-cache --hicache-ratio 1.5 \
  --hicache-mem-layout page_first --hicache-write-policy write_through \
  --hicache-storage-backend kvcr --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config '{"control_advertise_host": "127.0.0.1", "control_port": 24132, "nixl_listen_port": 24232, "local_dram_bytes": 8589934592}' \
  --kv-events-config '{"publisher": "zmq", "topic": "", "endpoint": "tcp://*:24021"}'
```

Required flags: `--enable-hierarchical-cache`, `--hicache-storage-backend kvcr`, `--hicache-mem-layout page_first` (or `page_first_direct`), and `--kv-events-config`, plus `--router-mode kv` on the frontend. For more than one host, set `control_advertise_host` to an address the other workers can reach.

## Check It Works

1. Get the worker IDs. Each response names the worker that served it; repeat until you have seen both:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":1,"nvext":{"extra_fields":["worker_id"]}}'
   ```

   Use the `decode_worker_id` values as `<worker-0-id>` and `<worker-1-id>`.

2. Create `request.json` with a prompt of at least 1,000 tokens that you have not sent before:

   ```json
   {"model": "Qwen/Qwen3-0.6B", "prompt": "<at least 1,000 tokens of new text>", "max_tokens": 8}
   ```

3. Send it to worker 0, wait 15 seconds, then send it to worker 1:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-0-id>' -d @request.json

   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-1-id>' -d @request.json
   ```

4. In the second response, `cached_tokens` is greater than 0 and a multiple of 64; the last partial page is recomputed. Worker 1 logs `HiCache prefetch success ... matched=0 loaded=<n>`: the `loaded` tokens came from worker 0 through KVCR. SGLang skips the KVCR lookup for prefixes shorter than its HiCache prefetch threshold (256 tokens by default), so use a long prompt.

For per-transfer detail, add `"diagnostic_telemetry": true` to the extra config. Worker 1 then logs `KVCR L3 operation` with `"direction":"deliver"` and `"failed_keys":[]`.

## Size the Tiers

Host memory for cached KV is split between two tiers:

| Tier | Set by | Shared with other workers |
|---|---|---|
| HiCache host tier (engine-managed) | `--hicache-ratio` (a multiple of the GPU KV pool) or `--hicache-size` (GB) | No. Peers fetch only from the KVCR pool; the host tier stages blocks between the GPU and KVCR. |
| KVCR pool | `local_dram_bytes`, per rank | Yes, under KVCR's policy |

Give most of the budget to `local_dram_bytes`, but keep the HiCache host tier larger than the GPU KV pool in tokens. Blocks fetched through KVCR are staged in the host tier before they reach the GPU, and with `write_through` the host tier also holds a copy of every block on the GPU. If the host tier is not larger than the GPU pool, SGLang drops KVCR hits and recomputes the prompt. A `--hicache-ratio` of `1.5` leaves half a GPU pool of room for hits; the startup log reports both pool sizes in tokens.

### Write Policy

`--hicache-write-policy` decides when a block is copied from the GPU to the host tier, and from there to KVCR:

| Policy | Blocks reach KVCR | Use with KVCR |
|---|---|---|
| `write_through` | As soon as they are cached | Recommended: peers can fetch a prefix right after the first request computes it |
| `write_through_selective` | After a block has been hit several times | Copies less, but a prefix is shareable only once it is hot |
| `write_back` | Only when the GPU evicts them | Avoid: peers cannot fetch a prefix while it is still on the source GPU |

## KVCR Settings

Fields of `--hicache-storage-backend-extra-config` in HiCache mode. Linker mode uses different names for the same settings (for example `advertise_host`, `nixl_port`, `dram_size_gb`), so do not copy fields between the two pages.

| Field | Required | Default | Meaning |
|---|---|---|---|
| `control_advertise_host` | Yes | - | Address peers use to reach this worker. Without it the worker is not advertised as a source. Not `0.0.0.0`. |
| `control_port` | No; unique per worker on a host | `32100` | Base port. Each rank uses `control_port + dp_rank * tp * cp * pp + rank offset`, so leave a gap between workers. |
| `nixl_listen_port` | No; unique per worker on a host | `control_port + 1024` | Same offsets. Must not overlap control ports. |
| `local_dram_bytes` | No | 8 GiB | KVCR pool per rank, in bytes. Separate from the HiCache host tier. |
| `enable_remote_hint` | No | `true` | Advertise this worker as a hint source. |
| `operation_timeout_ms`, `abandon_timeout_ms` | No | `30000`, `60000` | Transfer deadlines. |
| `diagnostic_telemetry` | No | `false` | Logs per-operation transfer records. |

## Related Topics

- [KV Cache Runner (KVCR)](kvcr.md)
- [Configure KVCR Linker Mode for SGLang](configure-kvcr-sglang-linker.md)
- [Tune Routing for KVCR](kvcr-routing-tuning.md)
- [HiCache](../modular-components/backends/sglang/hicache.md)
