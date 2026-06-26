<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Standalone GMS Shadow-Engine Failover (vLLM, single GPU, no Kubernetes)

This recipe demonstrates **GPU Memory Service (GMS) shadow-engine failover for
vLLM on a single node and a single GPU, with no Dynamo Kubernetes operator**.

A *primary* vLLM engine loads the model from disk once and publishes its weights
into a per-GPU GMS server. A pre-initialized *shadow* engine then imports those
**already-resident** weights through GMS (a zero-copy attach — no second disk
load) and parks itself with vLLM sleep mode. When the primary suffers a
process-level crash, the shadow wakes and takes over **without reloading the
model weights**. That avoided weight reload is the whole point: recovery from a
same-node engine/process failure is fast because the GPU-resident weights never
left the GPU.

Everything here runs as plain local processes that you can also launch by hand,
one script per step.

## What you get / when to use it

Use this to evaluate same-node recovery from an unknown vLLM engine or
software-process failure, where the cost you are trying to avoid is loading a
second independent copy of the model weights into GPU memory.

**Non-goals (this is NOT):**

- recovery from GPU loss, node loss, or hardware failure;
- in-flight request preservation or KV-cache preservation;
- a general checkpoint/restore system;
- production-ready — this is an **experimental** preview.

These limitations mirror the Kubernetes workflow. Rather than duplicate them,
see [`docs/kubernetes/shadow-engine-failover.md`](../../../../docs/kubernetes/shadow-engine-failover.md)
("Limitations" and "When to Use It Today"). The same constraints apply to this
standalone recipe.

## Prerequisites

- A **CUDA GPU + driver** (the GMS server opens CUDA contexts and allocates
  device memory; this cannot run on a CPU-only host).
- The **`gpu-memory-service`** package importable as `gpu_memory_service`
  (this recipe lives under
  [`lib/gpu_memory_service`](../../) — install it from there, e.g.
  `pip install -e lib/gpu_memory_service`).
- **`dynamo.vllm`** and **`dynamo.frontend`** available (the Dynamo vLLM
  runtime).
- **`pynvml`** — GMS derives stable per-GPU socket paths from the GPU UUID via
  NVML.
- External binaries on `PATH`:
  - **`etcd`** — Dynamo's discovery/coordination store.
  - **`nats-server`** — message bus; **must be started with JetStream (`-js`)**,
    which Dynamo requires for KV-event streaming.

## The user flow in 6 steps

1. **Start the GMS server** — `./start_gms.sh` launches the production
   supervisor (`python -m gpu_memory_service.cli.server`), which auto-discovers
   GPUs and runs one server per `(device, tag)` for `weights` and `kv_cache`.
2. **Start etcd + nats + frontend** — `./start_infra.sh` brings up `etcd` and
   `nats-server -js`; the frontend is `python -m dynamo.frontend --http-port 8000`.
3. **Launch the primary (writer)** — `./run_engine.sh primary` starts the RW
   engine; it loads the model **once** and publishes weights into GMS.
4. **Launch the shadow (reader), then pause it** — `./run_engine.sh shadow`
   starts an RO engine that **imports the same resident weights** (no disk
   reload); then `POST /engine/control/sleep {"level":2}` parks it.
5. **Kill the primary** — `./kill_primary.sh` SIGKILLs the primary's process
   group (simulating a process crash; GPU and node stay healthy).
6. **Wake the shadow → it serves without a weight reload** —
   `POST /engine/control/wake_up {}`, then `./verify.sh` confirms the frontend
   keeps returning completions.

## Run it

```bash
cd lib/gpu_memory_service/examples/shadow_failover
./run_demo.sh
```

`run_demo.sh` runs all ten steps with `==> STEP N ...` banners and **always
cleans up on exit** (see [Cleanup](#cleanup)).

Override any tunable via environment variables (defaults shown):

```bash
MODEL="Qwen/Qwen3-0.6B" \
GMS_SOCKET_DIR="/tmp/gms-demo" \
CUDA_VISIBLE_DEVICES="0" \
FRONTEND_PORT="8000" \
PRIMARY_SYSTEM_PORT="8081" SHADOW_SYSTEM_PORT="8082" \
  ./run_demo.sh
```

Other overridable knobs (see `common.sh`): `ETCD_ENDPOINTS`, `NATS_SERVER`,
`DYN_LOG`, `GPU_MEM_UTIL`, the per-engine `*_NIXL_PORT` / `*_KV_EVENT_PORT`
ports, and the local `ETCD_DATA_DIR` / `NATS_STORE_DIR` / `RUN_DIR` paths.

All three GMS clients (server, primary, shadow) **must agree on
`GMS_SOCKET_DIR`** — that is how they find the same per-GPU sockets.

## How it works

- **`--load-format gms`** makes the first engine the **RW writer**: it loads the
  model from disk and publishes the weights into the per-GPU GMS `weights`
  server. Subsequent engines that pass
  `--model-loader-extra-config '{"gms_read_only": true}'` become **RO
  importers**: they attach to the resident weights (zero-copy `cuMemMap` at
  stable virtual addresses) — **no disk reload**. In this recipe `run_engine.sh
  primary` is the writer and `run_engine.sh shadow` is the RO importer.
- **`--enable-sleep-mode`** plus the engine control endpoints let the shadow
  stay warm and then take over:
  - pause: `POST http://localhost:<system_port>/engine/control/sleep` with
    `{"level": 2}`
  - resume: `POST http://localhost:<system_port>/engine/control/wake_up` with
    `{}`

  These hit the **engine's** `DYN_SYSTEM_PORT`, not the frontend.
- On takeover the shadow **re-attaches to the resident weights** instead of
  reloading them, then resumes serving.

### Manual control vs. the operator's autonomous flock

This recipe uses **manual control-endpoint orchestration**: a human (or
`run_demo.sh`) decides when to `sleep`, `kill`, and `wake_up`. It deliberately
does **not** set `DYN_VLLM_GMS_SHADOW_MODE`, `ENGINE_ID`, `FAILOVER_LOCK_PATH`,
or `DYN_GMS_SCRATCH_KV_ENABLED`.

The Dynamo **Kubernetes operator** instead uses an **autonomous flock**: a
standby engine started with `DYN_VLLM_GMS_SHADOW_MODE` blocks on a kernel
`flock` over a shared lock file and is **promoted automatically** when the
primary dies, because the kernel releases the lock when the dead process exits.
That path is documented in
[`docs/kubernetes/shadow-engine-failover.md`](../../../../docs/kubernetes/shadow-engine-failover.md).
We use the simpler manual flavor here so the mechanics are visible and you can
drive them by hand.

## Verifying takeover

`verify.sh` POSTs a completion to the **frontend**
(`POST /v1/completions`) and asserts an HTTP 200 with a non-empty `choices`
array, retrying a number of times. Success looks like:

- the frontend **keeps returning completions** after the primary is killed;
- the **shadow log** (`${RUN_DIR}/shadow.log`) shows it **imported** weights
  through GMS and did **not** perform a fresh disk load;
- a brief window of **allocation backpressure** is expected immediately after
  the kill: the shadow's KV cache may not allocate until the dead primary's GPU
  memory is reclaimed, so `verify.sh` retries until the takeover settles.

On success `verify.sh` prints `TAKEOVER OK`.

## Cleanup

`run_demo.sh` installs an `EXIT`/`INT`/`TERM` trap that tears down **everything**
it started: both engine process groups, the frontend, the GMS supervisor (whose
process group reaps every per-`(device, tag)` child), and the etcd/nats
processes.

**Always confirm full cleanup.** Stray engine or GMS processes hold GPU memory
and will block the next run. The GMS test suite ships a leak guard for exactly
this reason. If you ran the steps by hand and need to reclaim the GPU, kill the
recorded process groups, for example:

```bash
RUN_DIR="${RUN_DIR:-/tmp/gms-demo-run}"
for f in shadow primary frontend gms; do
  [ -f "${RUN_DIR}/${f}.pgid" ] && kill -KILL -"$(cat "${RUN_DIR}/${f}.pgid")" 2>/dev/null || true
done
for s in nats etcd; do
  [ -f "${RUN_DIR}/${s}.pid" ] && kill -TERM "$(cat "${RUN_DIR}/${s}.pid")" 2>/dev/null || true
done
```

Then verify with `nvidia-smi` that no demo processes still hold device memory.

## Scope & caveats

- **Single node only.** This recipe covers a single-node, single-GPU process
  failure on a healthy GPU/node.
- **Multi-node failover is out of scope here.** Distributed topologies
  (cross-node TP/PP, WideEP) need cross-node cohort rendezvous plus
  cohort-atomic teardown/recreate. The Dynamo Kubernetes operator provides that
  (Grove + the cascade controller); there is **no standalone equivalent** in
  this recipe.
- **vLLM-specific today.** **TensorRT-LLM** currently integrates GMS only as a
  weight-sharing load path (a prototype) and does **not** yet implement the
  shadow-failover lifecycle, so this failover recipe is vLLM-only for now.
- **Experimental.** API shape and behavior may change; use for non-production
  evaluation only.

## Files in this recipe

| File              | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `run_demo.sh`     | One-shot orchestrator that runs the whole flow end to end.     |
| `start_infra.sh`  | Start `etcd` + `nats-server -js`.                              |
| `start_gms.sh`    | Start the production GMS server supervisor.                    |
| `run_engine.sh`   | Shared engine launcher; `primary` vs `shadow` differ by env.   |
| `kill_primary.sh` | Process-group SIGKILL of the primary (failure injection).      |
| `verify.sh`       | Send a frontend completion and assert takeover.                |
| `common.sh`       | Shared env defaults + `wait_for_ready` / `wait_for_port`.      |
