<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone GMS shadow-engine failover (vLLM)

Two vLLM engines share the same GPU Memory Service (GMS) -resident weights. A
*primary* (`ENGINE_ID=0`) loads the model once and publishes the weights into
GMS; a *shadow* (`ENGINE_ID=1`) imports those **already-resident** weights
(zero-copy, no second disk load) and parks itself, blocked on a shared kernel
`flock`. When the primary's process is killed the kernel releases the lock, so
the shadow takes over **without reloading the weights**.

Works on a single node (one or more GPUs) and across multiple nodes; run the same
`run.sh` on every node. No Kubernetes required.

> ⚠️ **Experimental.** vLLM only. Recovers from a process-level engine crash on a
> healthy GPU/node; **not** GPU/node loss, and it does not preserve in-flight
> requests or KV cache. See [the guide](../../docs/standalone-usage.md) for how it
> works and what is GMS-specific vs. ordinary multi-node deployment.

## Prerequisites

- A CUDA GPU + driver.
- `gpu_memory_service` importable (`pip install -e ../../`).
- `dynamo.vllm` available (and on the leader node, `dynamo.frontend`).
- `etcd` and `nats-server` (JetStream, `-js`) on the leader node's `PATH`.

## Run it

Single node:

```bash
./run.sh
```

Multiple nodes (run on each node; `NODE_RANK=0` is the leader):

```bash
# leader node
NNODES=2 NODE_RANK=0 LEADER_ADDR=<leader-ip> TP=2 ./run.sh
# each worker node
NNODES=2 NODE_RANK=1 LEADER_ADDR=<leader-ip> TP=2 ./run.sh
```

Other tunables (defaults shown): `MODEL=Qwen/Qwen3-0.6B`, `TP=1`,
`GMS_SOCKET_DIR=/tmp/gms-demo`, `FRONTEND_PORT=8000`.

`run.sh` starts a GMS server on every node (plus etcd + nats + the frontend on
the leader) and a primary and shadow engine per node, all in autonomous shadow
mode. It does **not** kill anything.

## Trigger failover

Once the engines are up, kill the primary yourself; the kernel releases its
`flock` and the shadow takes over automatically:

```bash
kill -KILL -"$(cat /tmp/gms-demo-engine-0.pgid)"
```

Watch `/tmp/gms-demo-engine-1.log`: the shadow acquires the lock and wakes,
serving without reloading the weights. (In a multi-node run, the primary's
workers on the other nodes are now orphaned; tear them down and relaunch a fresh
shadow group to restore redundancy.)

## What it shows

- The first GMS client loads weights from disk and publishes them; the second
  imports them zero-copy, with no second disk load and no second copy in GPU
  memory.
- Both engines run in autonomous shadow mode (`DYN_VLLM_GMS_SHADOW_MODE`): each
  blocks on the shared `flock`, so exactly one serves and the other waits.
- Killing the primary releases its `flock`; the kernel hands the lock to the
  shadow, which re-attaches to the resident weights and serves without reloading.

## How it works / multi-node details

See [`../../docs/standalone-usage.md`](../../docs/standalone-usage.md).
