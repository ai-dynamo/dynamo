---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configure KVCR Linker Mode for SGLang
subtitle: Load KV from other workers straight into GPU memory, with optional Guard, on one GPU server
---

> [!WARNING]
> Preview: needs Dynamo [#16827](https://github.com/ai-dynamo/dynamo/pull/16827) and SGLang [#45650](https://github.com/sgl-project/sglang/pull/45650) with the KVCR changes it is stacked on ([#45646](https://github.com/sgl-project/sglang/pull/45646) to [#45649](https://github.com/sgl-project/sglang/pull/45649)), none merged. One host only.

In Linker mode, KVCR loads KV pages directly between GPU memory and its own host-memory pool, layer by layer. Unlike HiCache mode ([Configure KVCR for SGLang](configure-kvcr-sglang.md)), there is no HiCache host tier in between.

## Before You Begin

- A runtime with these builds; no release image contains them:
  - Dynamo with [#16827](https://github.com/ai-dynamo/dynamo/pull/16827)
  - SGLang with the KVCR Linker ([#45650](https://github.com/sgl-project/sglang/pull/45650) and the PRs it is stacked on)
  - KVCR `v0.2.0` and NIXL `1.5.0`
- Two GPUs, one per worker.
- On one host without etcd or NATS, set `DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/dynamo-kvcr DYN_REQUEST_PLANE=tcp DYN_RESPONSE_PLANE=tcp DYN_EVENT_PLANE=zmq` in every terminal; this list matches the tested setup. Without RDMA, also set `UCX_TLS=sm,self,cuda_copy,cuda_ipc,tcp`.

Run each command in its own terminal.

## Start the Frontend

```bash
python3 -m dynamo.frontend --router-mode kv --http-port 8000
```

## Start Two Workers

Worker 0:

```bash
CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=9200 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B --page-size 64 --mem-fraction-static 0.6 \
  --kv-events-config '{"publisher": "zmq", "topic": "", "endpoint": "tcp://*:23000"}' \
  --enable-unified-cache-external-linker --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config '{"advertise_host": "127.0.0.1", "control_port": 26800, "nixl_port": 20500, "kvcr_service_socket_path": null}'
```

Worker 1:

```bash
CUDA_VISIBLE_DEVICES=1 DYN_SYSTEM_PORT=9201 python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B --page-size 64 --mem-fraction-static 0.6 \
  --kv-events-config '{"publisher": "zmq", "topic": "", "endpoint": "tcp://*:23001"}' \
  --enable-unified-cache-external-linker --unified-cache-external-linker-backend kvcr \
  --hicache-storage-backend-extra-config '{"advertise_host": "127.0.0.1", "control_port": 26801, "nixl_port": 20501, "kvcr_service_socket_path": null}'
```

`"kvcr_service_socket_path": null` gives each worker its own in-process KVCR pool. Without it, the worker looks for a Guard at `/run/kvcr/memory.sock`, logs `KVCR-Service is unreachable`, and falls back to the in-process pool.

## Check It Works

1. Get the worker IDs. Each response names the worker that served it; repeat until you have seen both:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hello","max_tokens":1,"nvext":{"extra_fields":["worker_id"]}}'
   ```

   Use the `decode_worker_id` values as `<worker-a-id>` and `<worker-b-id>`.

2. Create `request.json` with a prompt of at least 1,000 tokens that you have not sent before:

   ```json
   {"model": "Qwen/Qwen3-0.6B", "prompt": "<at least 1,000 tokens of new text>", "max_tokens": 8}
   ```

3. Send it to worker A, wait 15 seconds, then send it to worker B:

   ```bash
   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-a-id>' -d @request.json

   curl -s localhost:8000/v1/completions -H 'content-type: application/json' \
     -H 'x-dynamo-worker-instance-id: <worker-b-id>' -d @request.json
   ```

4. In the second response, `cached_tokens` is greater than 0 and a multiple of 64. Worker B logs `Prefill batch, #new-token: <a few>, #cached-token: <n>`: it computed only the last partial page and loaded the rest from worker A.

The first request after a fresh start can take a minute while SGLang builds a kernel extension. For transfer counters, add `"enable_telemetry": true` to the extra config; each worker then logs `KVCR telemetry` every 5 seconds, with `kvcr_transfer_blocks[source_write]` on worker A and the same `kvcr_transfer_blocks[remote_deliver]` count on worker B. Linker mode counts one block per page per layer.

## Add Guard

With Guard, a `kvcr_service` process owns the KVCR pools. If a worker's engine crashes, the restarted worker reattaches to its pool, and the KV state agent keeps the router sending the other worker to that pool while the crashed worker is down. See [The KV State Agent](kvcr.md#the-kv-state-agent).

1. Start one Guard service with one Guard per rank (here, one per worker):

   ```bash
   mkdir -p /run/kvcr /dev/shm/kvcr
   python3 -m kvcr.kvcr_service --socket-path /run/kvcr/memory.sock --pool-dir /dev/shm/kvcr \
     --guard-count 2 --pool-sizes-gb 4 --compatibility-digest qwen3-0.6b-linker-v1 --health-port 9092
   ```

   `curl localhost:9092/ready` returns `200` once the pools are ready.

2. Start the state agent. `--max-slots` is the number of cache owners it tracks, one per rank:

   ```bash
   DYN_SYSTEM_PORT=9091 python3 -m dynamo.kv_state_agent --max-slots 2
   ```

3. Start each worker with the Guard settings and a cache-owner ID. Replace `<host-ip>` with this host's IP address (`hostname -i` shows it). With the state agent, `advertise_host` must not be a loopback address such as `127.0.0.1`, because the state agent connects to it. Worker 0:

   ```bash
   CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=9200 python3 -m dynamo.sglang \
     --model-path Qwen/Qwen3-0.6B --page-size 64 --mem-fraction-static 0.6 \
     --kv-events-config '{"publisher": "zmq", "topic": "", "endpoint": "tcp://*:23000"}' \
     --enable-unified-cache-external-linker --unified-cache-external-linker-backend kvcr \
     --hicache-storage-backend-extra-config '{"advertise_host": "<host-ip>", "control_port": 26800, "nixl_port": 20500, "kvcr_service_socket_path": "/run/kvcr/memory.sock", "guard_index": 0, "compatibility_digest": "qwen3-0.6b-linker-v1"}' \
     --kvcr-state-agent-owner 01010101010101010101010101010101:02020202020202020202020202020202/0303030303030303/00000000000000000000000000000000
   ```

   For worker 1, use `CUDA_VISIBLE_DEVICES=1`, `DYN_SYSTEM_PORT=9201`, events port `23001`, `"control_port": 26801`, `"nixl_port": 20501`, `"guard_index": 1`, and a cache-owner ID ending in `...0001`.

4. To check recovery, send `request.json` to worker 0, wait 15 seconds, and stop worker 0's engine (`nvidia-smi` shows its PID; `kill -9 <pid>`). The Guard service logs `guard_promoted ... recovered_blocks=<n>`. Send `request.json` to worker 1: `cached_tokens` is positive, even after the router has removed worker 0. Restart worker 0 with the same command; it reattaches and serves the prompt from its pool.

## Settings

Fields of `--hicache-storage-backend-extra-config` in Linker mode. Unknown names stop the worker at startup. HiCache mode uses different names for the same settings (for example `control_advertise_host`, `nixl_listen_port`, `local_dram_bytes`), so do not copy fields between the two pages.

| Field | Required | Default | Meaning |
|---|---|---|---|
| `advertise_host` | For router hints | Local IP | Address peers use to reach this worker. Dynamo advertises a worker as a source only when this and `control_port` are set. |
| `control_port`, `nixl_port` | `control_port` for router hints; both unique per worker on a host | `19500`, `20500` | Base ports; the rank is added. The two must differ. |
| `kvcr_service_socket_path` | No | `/run/kvcr/memory.sock` | Guard socket. `null` uses an in-process pool. |
| `guard_index` | With Guard | `0` | Which of the service's Guards this worker uses; the rank is added. |
| `compatibility_digest` | With Guard | Derived from the model layout | Must match the service's `--compatibility-digest`. |
| `dram_size_gb` | No | `1` | In-process pool per rank. With Guard, `--pool-sizes-gb` sizes the pool instead. |
| `operation_timeout_ms`, `abandon_timeout_ms` | No | `1000`, `5000` | Transfer deadlines; the second must be at least twice the first. |
| `enable_telemetry` | No | `false` | Logs KVCR counters every 5 seconds. |

Dynamo worker flag: `--kvcr-state-agent-owner <cache-owner-id>` attaches the worker to the state agent. It requires Linker mode and ZMQ KV events with an empty topic.

Linker mode has no HiCache host tier, so give the host memory you budget for KV cache to `dram_size_gb` or the Guard pool.

## Related Topics

- [KV Cache Runner (KVCR)](kvcr.md)
- [How KV Hints Work](kv-hints.md)
- [Configure KVCR for SGLang](configure-kvcr-sglang.md)
- [Tune Routing for KVCR](kvcr-routing-tuning.md)
