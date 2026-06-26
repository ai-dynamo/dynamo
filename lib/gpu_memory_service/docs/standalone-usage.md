<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Memory Service: standalone usage and shadow-engine failover

This is a quick, steps-oriented sketch of what a user *does* with the GPU Memory
Service (GMS) **without** the Dynamo Kubernetes operator, what they *get*, and
how shadow-engine failover works (including the multi-node / WideEP case). For
GMS internals see [`../README.md`](../README.md); for the runnable recipe see
[`../examples/shadow_failover/README.md`](../examples/shadow_failover/README.md);
for the Kubernetes workflow see
[`../../../docs/kubernetes/shadow-engine-failover.md`](../../../docs/kubernetes/shadow-engine-failover.md).

> ⚠️ **Experimental.** API shape and behavior may change. Use for non-production
> evaluation only.

## What GMS is, and why you'd use it

GMS is an **out-of-process, per-GPU memory server**: it talks to clients over a
Unix socket (RPC + `SCM_RIGHTS` FD passing), owns GPU memory via CUDA VMM
mappings, and hands those mappings to inference engines. The engine **attaches
to** GPU-resident model weights (and optionally KV cache) instead of owning
them.

**Benefit:**

1. **No re-loading weights from disk on engine restart** — a restarted engine
   imports the already-resident weights zero-copy.
2. **Warm standby ("shadow") engines** — a pre-initialized standby shares the
   *same* resident weights and can take over a same-node, process-level engine
   failure **without a weight reload**.

**Non-goals (be clear about these):** GMS is *not* GPU-loss or node-loss
recovery; it does *not* preserve in-flight requests or KV-cache contents; it is
experimental.

## The core user flow (framework-agnostic)

1. **Start the GMS server — one per node.**

   ```bash
   python -m gpu_memory_service.cli.server
   ```

   It auto-discovers GPUs and runs one server process per `(device, tag)` for
   the tags `weights` and `kv_cache`. Every client **and** the server must share
   the same `GMS_SOCKET_DIR`; sockets are named `gms_<GPU-UUID>_<tag>.sock`.

2. **Launch your engine as a GMS client** by passing the backend's GMS load
   flag (e.g. vLLM / SGLang / TRT-LLM `--load-format gms`). The **first** client
   to connect loads weights from disk (RW) and publishes them into GMS; **later**
   clients import them read-only (RO), zero-copy — **no second disk load**.

3. **(For failover) launch one or more standby engines as RO clients** so a warm
   shadow already holds the resident weights.

The single user-visible knob is the **load-format flag**. Pass it again on
restart so a restarted engine re-imports instead of reloading.

## Single-node shadow failover

A runnable, single-GPU, no-Kubernetes example lives at
[`../examples/shadow_failover/README.md`](../examples/shadow_failover/README.md).
Its flow, in ~5 steps:

1. Start the GMS server and supporting infra (etcd / nats / `dynamo.frontend`).
2. Launch the **primary** (RW writer) — it loads the model once and publishes
   weights into GMS.
3. Launch the **shadow** (RO importer) — it attaches to the *same* resident
   weights, then **pauses** (releases GPU memory, keeps VA reservations).
4. **Kill the primary** (SIGKILL — simulates a process crash; GPU/node stay
   healthy).
5. **Wake the shadow** — it re-attaches to the resident weights and serves
   **without reloading them**.

### Who becomes active: two mechanisms

- **Autonomous flock (what the operator uses).** A standby started in shadow
  mode boots, pauses, and blocks on a kernel `flock` over a shared lock file.
  The Linux kernel auto-releases the lock when the holder dies — even on
  SIGKILL — so a standby is **promoted automatically**. This is leader election
  that is **node-local**: the lock file lives on one host.
- **Manual control (what the standalone recipe uses).** You drive the engine's
  `/engine/control/sleep` and `/engine/control/wake_up` endpoints and SIGKILL
  the primary yourself. Simpler to demo; the mechanics are visible.

## Multi-node and WideEP: what's automatic vs. what needs a control plane

The `flock` only does leader election **on a single node** (the leader / rank-0
node). It does **not** coordinate across nodes. In a multi-node engine, two
pieces make cross-node failover work, and **neither is part of GMS**:

1. **Stable per-rank / per-shadow identity + rendezvous** so each standby cohort
   assembles around its own leader.
2. **Cohort-atomic teardown/recreate** so one member's death tears down and
   recreates the whole distributed group cleanly.

In Dynamo these are provided by the **operator + Grove** (gang scheduling,
pod-index-pinned hostnames, a cascade controller), tied to how pods are launched
in Kubernetes. **Outside the operator you must build the equivalent control
plane yourself.**

### WideEP ("fail over many servers at once")

GMS's contribution to WideEP fault tolerance is **fast per-rank weight
re-materialization**: when a rank/server dies, a replacement imports *that
rank's* weight shard from GMS instead of reloading from disk. TRT-LLM's
`weight_sharing/source_identity.py` provides the **per-rank weight-set identity**
("rank N must import the shard produced for rank N") that makes this correct.

GMS does **not**, however:

- detect the dead rank,
- decide to serve degraded (N−1),
- re-spawn the rank, or
- rejoin the collective (back to N).

That detection + coordination + re-spawn control plane is the **engine's /
orchestrator's** responsibility (the operator in Dynamo, or the backend's own
fault-tolerance). GMS alone does **not** "fail over many servers."

## Backend support status

| Backend | Standalone status |
| --- | --- |
| **vLLM** | **Full standalone shadow failover works today** — custom worker, GMS-aware sleep/wake, scratch-KV, flock activation. See the recipe. |
| **TensorRT-LLM** | **Weight-sharing load path only (prototype)** — RW writer / RO readers via `LoadFormat.GMS` + `GmsConfig`. **No** shadow-failover lifecycle yet (no GMS-aware sleep/wake, no scratch-KV, no flock activation). |
| **SGLang** | GMS weight / memory-saver integration via the Dynamo runtime. |

**TRT-LLM current gates to know about:**

- The **RO import path is effectively inert** until the publisher
  source-identity seam is implemented.
- **GMS + the MoE load balancer is rejected at config time.**

So TRT-LLM gets **fast weight sharing now**; shadow / WideEP failover needs
additional orchestration glue — which can reuse primitives already in the GMS
wheel (the `flock`, scratch managers, unmap/remap).

## What the GMS wheel must expose for standalone use

A non-Dynamo engine (e.g. TRT-LLM) builds its own integration against these
import surfaces, so the **published wheel must include them**:

- `gpu_memory_service.client.torch.{allocator, module, tensor}`
- `gpu_memory_service.common.{locks, utils}` — including `get_socket_path`
- `gpu_memory_service.server` and `gpu_memory_service.cli` — server entrypoint
  is `python -m gpu_memory_service.cli.server`
- `gpu_memory_service.failover_lock` — for failover
- `gpu_memory_service.integrations.common.{patches, utils}` — e.g.
  `patch_empty_cache`, `finalize_gms_write`

> The framework-specific subpackages
> `gpu_memory_service.integrations.{vllm, sglang, trtllm}` are **Dynamo-runtime
> glue and are NOT part of the standalone surface.** An external engine
> implements its own integration against the primitives above.

> **Caveat:** `finalize_gms_write` (in `integrations.common.utils`) returns a
> `GMSCommittedMemoryStats` dataclass, so external callers must read
> `.committed_bytes` rather than `int(...)`.

(The exact extras / packaging mechanism is being decided separately; this only
describes the surface that must be importable.)
