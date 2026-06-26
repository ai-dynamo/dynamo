<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone GMS shadow-engine failover (vLLM, single GPU)

A *primary* vLLM engine loads the model once and publishes its weights into a
per-GPU GPU Memory Service (GMS) server. A *shadow* engine imports those
**already-resident** weights (zero-copy, no second disk load) and parks itself.
When the primary's process is killed, the shadow wakes and serves **without
reloading the weights** — that avoided reload is the point.

> ⚠️ **Experimental.** Single node, single GPU, vLLM only. Recovers from a
> process-level engine crash on a healthy GPU/node — **not** GPU/node loss, and
> it does not preserve in-flight requests or KV cache.

## Prerequisites

- A CUDA GPU + driver (GMS opens CUDA contexts and owns device memory).
- `gpu_memory_service` importable (`pip install -e ../../`).
- `dynamo.vllm` and `dynamo.frontend` available.
- `etcd` and `nats-server` on `PATH` (NATS must run with JetStream, `-js`).

## Run it

```bash
./run.sh
```

Override any tunable via the environment (defaults shown):

```bash
MODEL=Qwen/Qwen3-0.6B GMS_SOCKET_DIR=/tmp/gms-demo \
CUDA_VISIBLE_DEVICES=0 FRONTEND_PORT=8000 ./run.sh
```

`run.sh` starts infra + GMS + frontend + both engines, drives the failover by
hand via the engine control endpoints, and always tears down its processes on
exit. On success it prints `TAKEOVER OK`.

## What it shows

- The first GMS client (`--load-format gms`) loads weights from disk and
  publishes them; a second client with `{"gms_read_only": true}` imports them
  zero-copy — no second disk load.
- `--enable-sleep-mode` + `/engine/control/{sleep,wake_up}` let a standby park
  and re-attach to the resident weights instead of reloading.
- Killing the primary's process group frees its GPU memory; the shadow then
  serves through the same frontend.
- A brief allocation backpressure window after the kill is expected, so the
  takeover check retries until the dead primary's memory is reclaimed.

## How it works / replicate it yourself

This recipe drives sleep/wake **manually**. The autonomous (flock-gated) path,
the engine-wrapper responsibilities, and what the Kubernetes operator adds —
i.e. everything you would need to replicate shadow failover with just the GMS
wheel — are explained in
[`../../docs/standalone-usage.md`](../../docs/standalone-usage.md).
