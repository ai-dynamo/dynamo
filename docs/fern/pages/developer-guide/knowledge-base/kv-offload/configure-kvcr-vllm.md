---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Configure KVCR for vLLM
subtitle: Run the Dynamo KV router and a KVCR-enabled vLLM worker on one GPU server
---

These commands run the Dynamo frontend and a vLLM worker with a KVCR tier on one host with two GPUs. Pick a variant:

- **KVCR pool** (recommended): KVCR keeps its own pool of shareable blocks, sized separately from vLLM's CPU cache.
- **Guard**: the pool lives in a separate Guard service and survives a worker crash.
- **Peer-only** (testing only): workers fetch from each other's vLLM CPU cache.

## Before You Begin

- A Linux host with two NVIDIA GPUs.
- A runtime with matching Dynamo, vLLM, KVCR v0.2.0, and NIXL builds. Dynamo release images do not include the KVCR package yet; build one with the [KVCR quick start](https://github.com/ai-dynamo/kvcr/blob/main/docs/quick-start.md) (`Dockerfile.quick-start`) and run the commands below inside it.
- Enough `/dev/shm` for vLLM's CPU cache (`cpu_bytes_to_use` for every rank) and any Guard pools. The quick start gives the container 8 GB, which covers the KVCR pool example; for the Guard example, start the container with about 35 GB of `/dev/shm` per worker (`docker run --shm-size`).
- These environment variables, for one host without etcd or NATS. The quick-start image already sets them:

  | Variable | Value | Purpose |
  |---|---|---|
  | `DYN_DISCOVERY_BACKEND`, `DYN_FILE_KV` | `file`, `/tmp/dynamo-kvcr` | The frontend and workers find each other through files in this directory instead of etcd. |
  | `DYN_REQUEST_PLANE` | `tcp` | Sends requests to workers over TCP instead of NATS. |
  | `DYN_EVENT_PLANE` | `zmq` | Sends KV events to the router over ZeroMQ instead of NATS. |
  | `PYTHONHASHSEED` | `0` | Makes block hashes identical across processes, so the router matches blocks from different workers. |
  | `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES` | `1` | Publishes block hashes as integers, the form the router indexes. |

Each command below runs in the foreground; use a separate terminal for each. A worker is ready when `curl -s localhost:8000/v1/models` lists the model.

## Start the Frontend

KVCR fetches are planned by the KV router, so the frontend must run with `--router-mode kv`:

```bash
python3 -m dynamo.frontend --router-mode kv --http-port 8000
```

## Start a Worker

This worker has two data-parallel ranks, and they act as peers. Each rank has a small vLLM CPU cache (`cpu_bytes_to_use`) and a KVCR pool (`secondary_g2_slots`), and fetches blocks from the other rank's pool.

```bash
export MODEL=Qwen/Qwen3-0.6B

export KV_TRANSFER_CONFIG='{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2000000000,
    "self_describing_kv_events": true,
    "secondary_tiers": [{
      "type": "kvcr",
      "router_capabilities": ["router_hint"],
      "control_ports": [17771, 17772],
      "control_advertise_host": "127.0.0.1",
      "secondary_g2_slots": 1024,
      "enable_telemetry": true
    }]
  }
}'

CUDA_VISIBLE_DEVICES=0,1 DYN_SYSTEM_PORT=8081 python3 -m dynamo.vllm \
  --model "$MODEL" \
  --data-parallel-size 2 \
  --block-size 64 \
  --enable-prefix-caching \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
```

`--kv-events-config` publishes the worker's cache contents to the router; without it the router never sends hints. `DYN_SYSTEM_PORT` exposes the worker's `/metrics`. To confirm a transfer, see [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md).

For a quick functional test you can remove `secondary_g2_slots` (peer-only): ranks then fetch from each other's vLLM CPU cache. Keep the pool for real use: it is faster, because KVCR registers it once at startup and transfers do not wait for vLLM to pin memory.

## Size the Tiers

`secondary_g2_slots` is a number of KV blocks, not bytes. The pool is ordinary host memory in the worker process, not `/dev/shm`. Pool size per rank = slots x bytes per block. Bytes per block for a dense model = 2 x layers x KV heads x head dim x dtype bytes x block size. For Qwen3-0.6B at block size 64 that is 7.34 MB, so 1024 slots is 7.5 GB per rank, 15 GB for this two-rank worker. After one transfer you can also read it from the worker metrics: `vllm:kvcr_transfer_bytes_total / vllm:kvcr_transfer_blocks_total`.

Host memory for cached KV is split between two tiers:

| Tier | Set by | Memory | Shared with other workers |
|---|---|---|---|
| vLLM CPU cache (engine-managed) | `cpu_bytes_to_use`, per rank | `/dev/shm` | Only until vLLM evicts the block |
| KVCR pool | `secondary_g2_slots`, or `--pool-sizes-gb` with Guard | Host memory (Guard: `/dev/shm`) | Yes, under KVCR's policy; survives a crash with Guard |

Keep the vLLM CPU cache small and give the rest of the budget to KVCR. vLLM's cache still stages every block that moves between the GPU and KVCR, and receives blocks fetched from peers, so it cannot be zero. For example, with 200 GiB of host memory per worker for Qwen3.8-27B-FP8, use 25 GiB for `cpu_bytes_to_use` and 175 GiB for KVCR: `secondary_g2_slots: 3584` (50 MiB blocks), or `--pool-sizes-gb 175` with Guard.

## Add Guard

The Guard service (`kvcr_service`) owns the pool. After a worker crash, the restarted worker reattaches to its blocks, and other workers can fetch them in the meantime. The [KV state agent](kvcr.md#the-kv-state-agent) (`dynamo.kv_state_agent`) keeps the router pointing at a crashed worker's blocks.

Run one Guard service per worker process: several worker processes cannot share one Guard service yet. Each data-parallel rank uses the Guard with its rank's index, so `--guard-count` must be at least the worker's data-parallel size.

1. Start the Guard service:

   ```bash
   mkdir -p /run/kvcr /dev/shm/kvcr
   python3 -m kvcr.kvcr_service \
     --socket-path /run/kvcr/memory.sock \
     --pool-dir /dev/shm/kvcr \
     --guard-count 1 \
     --pool-sizes-gb 32 \
     --compatibility-digest qwen3-0.6b-v1 \
     --health-port 9092
   ```

   `curl localhost:9092/ready` returns `200` once the pool is ready. Registering a large pool can take tens of seconds; if the service times out at startup, raise `--startup-timeout-seconds` (default `60`). The pool lives in `/dev/shm`, and each Guard adds a 100 MiB journal to it. `--pool-sizes-gb` is the pool size in GB per Guard; size it as in [Size the Tiers](#size-the-tiers).

2. Start one state agent. All workers in the Dynamo namespace share it, and `--max-slots` is the number of cache owners it tracks, one per rank across those workers:

   ```bash
   DYN_SYSTEM_PORT=9091 python3 -m dynamo.kv_state_agent --max-slots 2
   ```

3. Start the worker with a Guard attachment and a `dynamo_state_agent` block:

   - `raw_advertise_host` is the address the state agent uses to read this worker's KV events. Replace `<host-ip>` with this host's IP address (`hostname -i` shows it). Dynamo rejects loopback addresses here, even on one host; `control_advertise_host` can stay `127.0.0.1`.
   - `cache_owner_ids` names this worker's cache so the router can find it after a crash. It must be unique per worker and stay the same across restarts: copy the example value and change only the last segment for each worker.

   ```bash
   export KV_TRANSFER_CONFIG='{
     "kv_connector": "OffloadingConnector",
     "kv_role": "kv_both",
     "kv_connector_extra_config": {
       "spec_name": "TieringOffloadingSpec",
       "cpu_bytes_to_use": 2147483648,
       "self_describing_kv_events": true,
       "dynamo_state_agent": {
         "raw_advertise_host": "<host-ip>",
         "cache_owner_ids": {"0": "01010101010101010101010101010101:02020202020202020202020202020202/0303030303030303/00000000000000000000000000000000"}
       },
       "secondary_tiers": [{
         "type": "kvcr",
         "kvcr_service_socket_path": "/run/kvcr/memory.sock",
         "compatibility_digest": "qwen3-0.6b-v1",
         "router_capabilities": ["router_hint"],
         "control_ports": [23280],
         "control_advertise_host": "127.0.0.1",
         "enable_telemetry": true
       }]
     }
   }'

   CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT=8081 python3 -m dynamo.vllm \
     --model Qwen/Qwen3-0.6B \
     --block-size 64 \
     --enable-prefix-caching \
     --kv-transfer-config "$KV_TRANSFER_CONFIG" \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'
   ```

   For a second worker on the same host, repeat steps 1 and 3 with its own values:

   | Setting | Worker A | Worker B |
   |---|---|---|
   | `--socket-path`, `kvcr_service_socket_path` | `/run/kvcr/memory.sock` | `/run/kvcrB/memory.sock` |
   | `--pool-dir` | `/dev/shm/kvcr` | `/dev/shm/kvcrB` |
   | `--health-port` | `9092` | `9093` |
   | `CUDA_VISIBLE_DEVICES` | `0` | `1` |
   | `control_ports` | `[23280]` | `[23281]` |
   | KV-events endpoint | `tcp://*:20081` | `tcp://*:20082` |
   | `DYN_SYSTEM_PORT` | `8081` | `8082` |
   | `cache_owner_ids` last segment | `...0000` | `...0001` |

## Configuration Reference

Fields of the KVCR entry in `secondary_tiers`:

| Field | Required | Default | Meaning |
|---|---|---|---|
| `router_capabilities` | Yes | - | Must include `router_hint`; the worker does not start without it. |
| `control_ports` | Yes | - | One port per local data-parallel rank, in rank order. |
| `control_advertise_host` | On more than one host | Hostname | Address peers use to reach this worker. Loopback works only on one host. |
| `control_host` | No | `0.0.0.0` | Bind address for the control port. |
| `secondary_g2_slots` | No | `0` | KVCR pool size, in KV blocks. Ignored when Guard is configured. |
| `kvcr_service_socket_path`, `compatibility_digest` | With Guard, together | - | Attach to a Guard service. The digest must match the service's and change when the model, dtype, or block layout changes. |
| `operation_timeout_ms` | No | `1000` | Deadline for one KVCR operation. |
| `abandon_timeout_ms` | No | `5000` | At least twice `operation_timeout_ms`. |
| `enable_telemetry` | No | `false` | Exposes `vllm:kvcr_*` metrics on the worker's `DYN_SYSTEM_PORT`. |

Fields of `kv_connector_extra_config` that KVCR depends on:

| Field | Required | Default | Meaning |
|---|---|---|---|
| `cpu_bytes_to_use` | Yes | - | Size of vLLM's own CPU cache per rank, in `/dev/shm`. Not a KVCR pool. |
| `self_describing_kv_events` | With a pool, Guard, or disk tier | `false` | Set to `true` in every configuration; the router needs it to index KVCR's tiers. The worker does not start without it when a pool, Guard, or disk tier is configured. |
| `dynamo_state_agent.raw_advertise_host` | With the state agent | - | Non-loopback address where the state agent reads this worker's KV events. |
| `dynamo_state_agent.cache_owner_ids` | With the state agent | - | Stable, unique ID of this worker's cache, per local rank. |

## Related Topics

- [Verify and Monitor KVCR with vLLM](kvcr-testing-monitoring-vllm.md)
- [Run KVCR on Kubernetes with vLLM](kvcr-kubernetes-vllm.md)
- [Run KVCR Guard on Kubernetes with vLLM](kvcr-guard-kubernetes-vllm.md)
- [KV Cache Runner (KVCR)](kvcr.md)
