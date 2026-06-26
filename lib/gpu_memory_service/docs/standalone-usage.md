<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Memory Service: shadow-engine failover with plain processes

How to get **shadow-engine failover with plain inference-engine processes** (no
Kubernetes) — starting from a single node and adding only what multi-node needs
on top. On Kubernetes the Dynamo operator automates all of this
([k8s workflow](../../../docs/kubernetes/shadow-engine-failover.md)); this is the
from-scratch version. See also [GMS internals](../README.md) and the runnable
[single-GPU recipe](../examples/shadow_failover/README.md).

> ⚠️ **Experimental.** APIs may change; for non-production evaluation only.

## What GMS gives you

GMS is an **out-of-process, per-GPU memory server**: it owns GPU memory (CUDA VMM
mappings) and hands it to engines over a Unix socket, so an engine **attaches to**
GPU-resident weights instead of owning them. That buys you:

- **No weight reload on restart** — a restarted engine re-imports the resident
  weights zero-copy.
- **Warm standbys** — a pre-initialized "shadow" engine shares the *same* resident
  weights and takes over a process-level crash **without reloading them**.

It is **not** GPU/node-loss recovery and does **not** preserve in-flight requests
or KV-cache contents; it recovers from a software/process crash on a healthy GPU.

GMS provides exactly two things: the resident weights and a **`flock` lock
primitive**. It does not decide who is active — that's the engine and you.

## Single node

1. **Run a GMS server on the node:**

   ```bash
   GMS_SOCKET_DIR=/tmp/gms python -m gpu_memory_service.cli.server
   ```

   It serves one socket per `(device, tag)` for `weights` and `kv_cache`; the
   server and all engines share one `GMS_SOCKET_DIR`.

2. **Launch a primary and one or more shadows on the same GPU** as GMS clients
   (`--load-format gms`), sharing that `GMS_SOCKET_DIR` and one lock file. The
   first (`ENGINE_ID=0`) loads the model from disk and publishes the weights; the
   rest import them read-only — no second disk load, no second copy.

3. **Promotion is automatic.** In shadow mode each engine boots, parks, and blocks
   on a kernel **`flock`** over the shared lock file; one acquires it and serves,
   the rest wait. When the active process dies (even on `SIGKILL`) the kernel
   releases the lock and a standby takes over — no health-check needed. (Or drive
   `/engine/control/{sleep,wake_up}` yourself if you'd rather control promotion.)

So on one node you supply: a shared lock file, a way to **route traffic to the
current lock holder** (with `dynamo.vllm` the engine registers with discovery only
*after* acquiring the lock, so a router sees only the active one), and — to restore
redundancy after a failover — a way to relaunch a fresh standby. The runnable
[recipe](../examples/shadow_failover/README.md) shows the whole flow.

## What the engine must support

A vanilla engine can't share GMS weights and stand by: its normal "sleep" copies
GPU buffers to host (which breaks on GMS's `cuMemMap`'d memory) and its memory
accounting assumes it owns the whole GPU. A GMS-standby engine needs:

- **GMS weight load** (`--load-format gms`) with a per-process RW/RO role (one
  writer via `ENGINE_ID`; this also avoids a deadlock under tensor parallelism).
- **GMS-aware sleep/wake**: sleep unmaps the GMS virtual addresses and releases
  physical backing; wake reconnects and **remaps the same virtual addresses** —
  how a standby re-attaches to the resident weights without reloading.
- **Memory-accounting fixes** so a co-resident engine doesn't think the GPU is
  full and doesn't size its KV cache against the active engine's allocation.
- **Scratch KV**: a (re)initializing shadow captures its CUDA graphs against
  placeholder, un-backed KV-cache memory instead of a full real allocation, so it
  doesn't consume GPU memory for KV while the active engine is still running. On
  promotion it swaps in the real GMS-backed KV **at the same virtual addresses**,
  so the already-captured CUDA graphs stay valid.
- **Hold-until-active**: boot, initialize, then wait (on the `flock` or your
  signal) and serve only once promoted.

`dynamo.vllm` implements all of this; integrating another engine means doing this
list (the `flock`, scratch managers, and unmap/remap primitives are in the wheel).

| Engine | Standby lifecycle today |
| --- | --- |
| **vLLM** (`dynamo.vllm`) | **Full** — load, sleep/wake, scratch KV, `flock` activation. See the [recipe](../examples/shadow_failover/README.md). |
| **TensorRT-LLM** | **Weight load only (prototype)** — `LoadFormat.GMS` RW/RO; no sleep/wake, scratch KV, or `flock` activation yet. |
| **SGLang** | GMS weight / memory-saver integration via the Dynamo runtime. |

## Multiple nodes: what's extra

Everything above still holds. The only new fact is that a multi-node engine is one
**distributed group** — a leader plus workers across the nodes, joined by a
rendezvous (master address/port). That changes three things:

- **A standby is a whole group, not a process.** Run the active group plus one or
  more complete standby groups; each spans all the nodes and imports the same
  per-node GMS weights (RO), so standby groups add no weight memory.
- **Each group needs its own rendezvous.** Give every group its own leader
  address/port so its workers pair with *their* leader, not another group's. This
  per-group wiring is the one genuinely new thing to set up.
- **Promotion is still one `flock`, and failover is whole-group.** Only leaders
  serve, so co-locate the groups' leaders on one node and let them share a single
  lock file — the same single-node election; workers just follow their own leader.
  Because a distributed collective can't be partially recovered, the unit you fail
  over is the whole group: when you kill the active leader, also tear down that
  group's orphaned workers and start a fresh standby group to restore redundancy.

The failover trigger is still yours (you kill the primary); GMS just gives each
replacement fast per-rank weight re-materialization (it imports its shard from GMS
instead of reloading from disk). **Wide expert parallelism (WideEP) is simply a
large instance of this** — same per-group rendezvous, same leader `flock`, same
whole-group failover.
