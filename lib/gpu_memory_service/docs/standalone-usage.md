<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPU Memory Service: shadow-engine failover with plain processes

This describes how to use the GPU Memory Service (GMS) to get **shadow-engine
failover when you run plain inference-engine processes yourself** — on one node
or a couple of nodes, launched however you like (Slurm, `ssh`, a supervisor,
`systemd`; it doesn't matter). It explains what GMS gives you, what your engine
must support, and — the important part — **what you have to wire up yourself**.

If you happen to run on Kubernetes, the Dynamo operator automates everything in
the "what you wire up" sections below (see
[`../../../docs/kubernetes/shadow-engine-failover.md`](../../../docs/kubernetes/shadow-engine-failover.md)).
This doc is the from-scratch version. For GMS internals see
[`../README.md`](../README.md); for a runnable single-GPU example see
[`../examples/shadow_failover/README.md`](../examples/shadow_failover/README.md).

> ⚠️ **Experimental.** API shape and behavior may change. Use for non-production
> evaluation only.

## What GMS is, and why you'd use it

GMS is an **out-of-process, per-GPU memory server**. It owns GPU memory via CUDA
VMM mappings and hands those mappings to engine processes over a Unix socket
(RPC + `SCM_RIGHTS` FD passing). The engine **attaches to** GPU-resident model
weights (and optionally KV cache) instead of owning them.

Two benefits:

1. **No weight reload on engine restart** — a restarted engine imports the
   already-resident weights zero-copy instead of reading them from disk again.
2. **Warm standby ("shadow") engines** — a pre-initialized standby shares the
   *same* resident weights and can take over a process-level engine crash
   **without a weight reload**.

**Non-goals:** GMS is *not* GPU-loss or node-loss recovery; it does *not*
preserve in-flight requests or KV-cache contents. It recovers from a
**software/process-level engine failure on a healthy GPU and node**.

## The pieces involved

Shadow failover is the cooperation of three things. GMS gives you only the
first; you supply (or your engine supplies) the other two:

1. **GMS (the wheel)** — holds the weights in GPU memory and provides a lock
   primitive. *Does not* decide who is active or restart anything.
2. **A standby-capable engine** — an engine that can boot, attach to the
   resident weights, sit idle, and later "wake" and serve without reloading.
   vLLM via `dynamo.vllm` does this today; see [what the engine must
   support](#what-the-engine-itself-must-support).
3. **Your orchestration** — start the processes, route traffic to whichever
   engine is active, detect a death, and restart the dead one. This is what you
   own; the rest of this doc is mostly about it.

## Single node: what you do

On one node with one (or a few) GPUs:

1. **Start a GMS server on the node.**

   ```bash
   GMS_SOCKET_DIR=/tmp/gms python -m gpu_memory_service.cli.server
   ```

   It auto-discovers GPUs and runs one server per `(device, tag)` for tags
   `weights` and `kv_cache`. The server and every engine must share the same
   `GMS_SOCKET_DIR` (sockets are `gms_<GPU-UUID>_<tag>.sock`).

2. **Launch a primary and one or more shadows on the same GPU**, all as GMS
   clients (`--load-format gms`), all pointing at the same `GMS_SOCKET_DIR` and
   the same lock file. The **first** to connect (`ENGINE_ID=0`) loads the model
   from disk and publishes the weights into GMS; the rest import them read-only,
   zero-copy — **no second disk load**, and no second copy of the weights in GPU
   memory.

3. **Decide who serves — two options:**

   - **Autonomous (recommended).** Run the engines in *shadow mode* with a shared
     lock file (`FAILOVER_LOCK_PATH`). Each engine boots, parks itself, and
     blocks on a kernel **`flock`** over that file. Exactly one acquires it and
     starts serving; the others wait. When the active engine's process dies —
     even on `SIGKILL` — **the kernel releases the lock automatically**, a
     waiting standby acquires it and takes over. No health-checker required for
     the promotion itself. This is what the runnable
     [recipe](../examples/shadow_failover/README.md) does.
   - **Manual.** Drive the engine's `/engine/control/sleep` and `wake_up`
     endpoints yourself and decide when to promote — useful if you want an
     external controller to own promotion instead of the kernel `flock`.

**What you must provide on a single node:**

- **A shared lock file path** reachable by all the engines (a normal local path;
  `flock` is a kernel lock on that file).
- **Routing to the active engine.** Whatever fronts your engines (a load
  balancer, your own router, or the Dynamo frontend if you use `dynamo.vllm`)
  must send traffic only to the engine that currently holds the lock. In
  `dynamo.vllm`, the engine only registers with discovery *after* it acquires
  the lock, so the router naturally sees only the active one; with a plain
  engine you point your LB at the active endpoint (or health-check the standbys
  out until they take over).
- **Restarting the dead engine.** Promotion is automatic, but you are now down
  one standby. Your launcher (Slurm, a supervisor, `systemd`, a shell loop)
  should relaunch a fresh standby to restore redundancy.

That's the whole single-node story, and it is genuinely simple: a GMS server, a
writer, one or more readers sharing a lock file, plus a way to route and a way
to relaunch.

## What the engine itself must support

A *vanilla* engine can't just share GMS weights and stand by — its normal
"sleep" copies GPU buffers to host (which breaks on GMS's `cuMemMap`'d memory),
and its memory accounting assumes it owns the whole GPU. To be a GMS standby, an
engine needs:

- **GMS weight load** (`--load-format gms`): RW writer publishes, RO readers
  import; choose the role per process (e.g. by an `ENGINE_ID` env var so
  exactly one writes — this also avoids a deadlock when the engine is itself
  multi-GPU/tensor-parallel).
- **GMS-aware sleep/wake**: on sleep, *unmap* the GMS virtual addresses and
  release physical backing; on wake, reconnect and *remap the same virtual
  addresses*. This remap is how a standby re-attaches to the resident weights
  instead of reloading them.
- **Memory-accounting fixes** so a co-resident engine doesn't think the GPU is
  full, and so KV-cache sizing ignores the active engine's allocation.
- **"Scratch KV"** so a standby can fully initialize while the active engine
  still owns the real KV layout (throwaway backing now, real GMS-backed KV on
  wake).
- **Hold-until-active behavior**: boot, initialize, then wait (on the `flock`,
  or on your signal) and only start serving once promoted.

For vLLM this is all implemented in `dynamo.vllm`
(`components/src/dynamo/vllm` + `lib/gpu_memory_service/integrations/vllm`), so
you get it by running that engine. **If you are integrating GMS into a different
engine, this list is exactly the work to do** — and the primitives you need
(the `flock`, scratch managers, unmap/remap) are already in the GMS wheel.

| Engine | Standby lifecycle today |
| --- | --- |
| **vLLM** (`dynamo.vllm`) | **Full** — GMS load, GMS-aware sleep/wake, scratch KV, `flock` activation. Use the [recipe](../examples/shadow_failover/README.md). |
| **TensorRT-LLM** | **Weight load only (prototype)** — `LoadFormat.GMS` RW writer / RO readers. **No** sleep/wake, scratch KV, or `flock` activation yet; shadow failover needs the lifecycle above added (reusing the wheel's primitives). |
| **SGLang** | GMS weight / memory-saver integration via the Dynamo runtime. |

## Multiple nodes: what you have to build

When the engine itself spans nodes (tensor/pipeline parallel, or wide expert
parallel), the same idea applies but **you** own more of it. The key facts:

- **Run complete engine groups, not single processes.** A multi-node engine is
  one distributed group (a leader + workers across the nodes, joined by a
  rendezvous address). To have a warm standby, you run the whole active group
  **plus one or more whole standby groups** — every group spans all the nodes.
  All groups attach to the same GMS-resident weights (one server per node), so
  the standby groups cost no extra weight memory.
- **Each group needs its own rendezvous, and members must pair up by slot.**
  Give each group its own leader address/port (master addr/port), and make the
  *same-slot* members across nodes join the *same* leader. Concretely: standby
  group #1's workers on every node rendezvous with standby group #1's leader —
  not with the active leader. This pairing is what makes a standby group a
  self-contained, ready-to-serve engine rather than a pile of stray processes.
- **Promotion is still a single-node lock.** Only the *leader* processes serve /
  register, and the leaders all live on one node (the leader node). So the
  `flock` election happens among the leader processes on that one node — it is
  node-local, and that's fine. Whichever leader holds the lock is the active
  group; its workers on the other nodes are already bound to it. Workers never
  contend the lock.

**What you must build (your launcher / control plane):**

1. **Failure detection** — notice when any process in the active group dies.
2. **Cohort-atomic teardown** — kill the *entire* active group across all nodes,
   not just the dead process. A multi-node group shares NCCL collectives and a
   `torch.distributed` rendezvous; once one rank dies the rest are wedged and
   **cannot be partially recovered**. (Do not try to restart a single rank
   in-place.)
3. **Promotion** — let a standby group take over. With autonomous `flock` shadow
   mode this is automatic: the dead leader released the lock, a standby leader
   acquires it and serves, and its workers were already paired to it.
4. **Re-create a standby** — relaunch a fresh group to restore redundancy.

You can do all four with whatever you already use — a Slurm job array with a
prolog/epilog, a supervisor process, a small controller script. **The Dynamo
Kubernetes operator is simply an implementation of these four steps** (gang
scheduling for the groups, a cascade controller for the atomic teardown); off
Kubernetes you implement the equivalent in your launcher. GMS gives you the fast
weight re-materialization and the lock; the detect / tear-down / promote /
relaunch loop is yours.

The same applies to wide expert parallelism (WideEP) — it is just a large
multi-node deployment. When a rank/server dies, its replacement imports *that
rank's* weight shard from GMS instead of reloading from disk (TRT-LLM's
`weight_sharing/source_identity.py` provides the per-rank identity that keeps
"rank N imports the shard produced for rank N" correct). GMS still does **not**
detect the dead rank, serve degraded (N−1), re-spawn it, or rejoin the
collective — that's the same detect / tear-down / promote / relaunch loop above,
just across more processes.
