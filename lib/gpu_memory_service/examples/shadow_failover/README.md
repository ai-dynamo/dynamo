<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone GMS shadow-engine failover (vLLM, single GPU)

Two vLLM engines share one GPU and one set of GPU Memory Service (GMS) -resident
weights. A *primary* (`ENGINE_ID=0`) loads the model once, publishes the weights
into GMS, and serves; a *shadow* (`ENGINE_ID=1`) imports those
**already-resident** weights (zero-copy, no second disk load) and parks itself,
blocked on a shared kernel `flock`. When the primary's process is killed the
kernel releases the lock, so the shadow takes over **without reloading the
weights** — that avoided reload is the point.

> ⚠️ **Experimental.** Single node, single GPU, vLLM only. Recovers from a
> process-level engine crash on a healthy GPU/node — **not** GPU/node loss, and
> it does not preserve in-flight requests or KV cache.

## Prerequisites

- A CUDA GPU + driver (GMS opens CUDA contexts and owns device memory).
- `gpu_memory_service` importable (`pip install -e ../../`).
- `dynamo.vllm` available.
- `etcd` and `nats-server` on `PATH` (NATS must run with JetStream, `-js`) —
  `dynamo.vllm` registers with them on startup.

## Run it

```bash
./run.sh
```

Override any tunable via the environment (defaults shown):

```bash
MODEL=Qwen/Qwen3-0.6B GMS_SOCKET_DIR=/tmp/gms-demo \
CUDA_VISIBLE_DEVICES=0 ./run.sh
```

`run.sh` starts etcd + nats + the GMS server + both engines, waits for them to
warm up, then SIGKILLs the primary and leaves the shadow running. Watch the
takeover in `/tmp/gms-demo-shadow.log` (the shadow acquires the lock and wakes).

## What it shows

- The first GMS client (`--load-format gms`, `ENGINE_ID=0`) loads weights from
  disk and publishes them; the second imports them zero-copy — no second disk
  load, no second copy of the weights in GPU memory.
- Both engines run in autonomous shadow mode (`DYN_VLLM_GMS_SHADOW_MODE`): each
  blocks on the shared `flock`, so exactly one serves and the other waits.
- Killing the primary's process group releases its `flock`; the kernel hands the
  lock to the shadow, which re-attaches to the resident weights and serves
  without reloading them.

## How it works / replicate it yourself

The engine-side mechanics, the manual (control-endpoint) alternative, and what
you must build yourself to run this across more than one node are explained in
[`../../docs/standalone-usage.md`](../../docs/standalone-usage.md).
