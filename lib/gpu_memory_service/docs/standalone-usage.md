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

- **Manual control (what the standalone recipe uses).** You drive the engine's
  `/engine/control/sleep` and `/engine/control/wake_up` endpoints and SIGKILL
  the primary yourself. Simpler to demo; the mechanics are visible.
- **Autonomous flock (what the operator uses).** A standby blocks on a kernel
  `flock` and is promoted automatically when the holder dies. See the
  orchestration section below for how this fits together.

## Orchestration: who does what (and how to replicate it)

Shadow failover is split across three layers. GMS supplies the memory
primitives; an engine wrapper turns those into a sleep/wake-able standby; an
orchestrator decides who is active and recovers a dead cohort. Knowing which
layer owns what tells you exactly how much you must build to do this yourself.

### Layer 1 — the GMS wheel (primitives)

The wheel gives you an out-of-process, per-GPU memory server and the building
blocks to share its memory:

- **RW-writer / RO-importer weight sharing** — the first client loads from disk
  and publishes; later clients import the resident weights zero-copy.
- a **torch mempool / pluggable allocator** so engine allocations are routed
  through GMS;
- a kernel **`flock`-based failover lock** (`gpu_memory_service.failover_lock`)
  whose holder is auto-released by the kernel on process death;
- **scratch-allocation managers** and **unmap/remap of VA reservations** for
  tearing down and re-attaching backing at stable virtual addresses.

GMS does **not** decide who is active, when to fail over, or how to co-schedule
engines. Those are the next two layers.

### Layer 2 — the engine wrapper (what makes a standby possible)

A vanilla engine cannot just share GMS weights and stand by — vanilla sleep
copies GPU buffers to host and breaks on GMS's `cuMemMap`'d memory, and the
engine's memory accounting assumes it owns the whole GPU. Dynamo's vLLM wrapper
(`components/src/dynamo/vllm` + `lib/gpu_memory_service/integrations/vllm`) adds:

- the `--load-format gms` loader and auto-selection of the GMS worker class;
- a **GMS-aware sleep/wake**: sleep unmaps the VAs and releases physical backing;
  wake reconnects and **remaps the same virtual addresses** — this is how a
  standby re-attaches to the resident weights instead of reloading them;
- **memory-accounting fixes** so a co-resident engine doesn't think the GPU is
  full, and so KV-cache sizing ignores the active engine's allocation;
- **scratch KV** so a standby can fully initialize while co-resident with the
  active engine (throwaway KV backing now, real GMS-backed KV on wake);
- **RW/RO role selection by `ENGINE_ID`** (engine `0` writes weights, others
  import) to avoid a multi-rank deadlock;
- the **activation/standby behavior**: in autonomous *shadow mode* the engine
  boots, pauses, blocks on the shared `flock`, and only registers with service
  discovery (becomes routable) **after** it acquires the lock — so the router
  only ever sees the active engine, and the kernel releasing the lock on process
  death promotes a standby automatically. The standalone recipe instead drives
  the equivalent sleep/wake by hand via the engine control endpoints.

**Takeaway:** if you integrate GMS into another engine, *this* is the layer you
reimplement — sleep/wake-as-unmap/remap, scratch KV, `ENGINE_ID` RW/RO, and
either a flock-gated activation or your own promote/serve trigger.

### Layer 3 — the Dynamo Kubernetes operator (cluster orchestration)

The operator supplies what GMS and the wrapper don't:

- **co-location and GPU sharing** — it puts the GMS server and engine(s) on the
  same GPU and shares that GPU across pods/containers via DRA, with a shared
  volume so every engine sees the same lock file and GMS sockets;
- **failover env injection** — `ENGINE_ID` from the pod index, the shared lock
  path, shadow mode, scratch KV, and the socket dir;
- **failure handling** — failover engine pods use `restartPolicy: Never` plus a
  cascade controller, so when *any* engine in a failover group terminates the
  **whole group** is torn down and recreated cleanly (no half-dead distributed
  group);
- **multi-node** — one GMS server per rank, pod-index-pinned leader hostnames so
  each standby *cohort* (same pod index across ranks) rendezvouses around its own
  leader, headless workers, and `flock` leader election running only on the
  rank-0/leader node (it is node-local by design).

**Takeaway:** doing shadow failover **without** the operator is easy on a single
node — run a GMS server, launch a writer plus one or more RO standbys on the
same GPU sharing one lock file, and the kernel `flock` handles promotion. That
is exactly this recipe. **Multi-node is the hard part you must build yourself:**
stable per-cohort rendezvous + cohort-atomic teardown/recreate + failure
detection. There is no GMS-level shortcut for it.

### WideEP ("fail over many servers at once")

This is the same split at scale. GMS's only contribution is **fast per-rank
weight re-materialization**: when a rank/server dies, its replacement imports
*that rank's* weight shard from GMS instead of reloading from disk. TRT-LLM's
`weight_sharing/source_identity.py` supplies the per-rank weight-set identity
("rank N imports the shard produced for rank N") that keeps this correct.

GMS does **not** detect the dead rank, decide to serve degraded (N−1), re-spawn
the rank, or rejoin the collective (back to N). That detect / serve-degraded /
respawn / rejoin control plane is the orchestrator's job — Layer 3 above, or the
backend's own fault tolerance. GMS alone does **not** "fail over many servers."

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
