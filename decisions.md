# GPU Failover POC: Decisions

## Decision 1: K8s Layout

**Intra-pod with shared GPU access via DRA.**

Primary engine, shadow engine, and GMS run as containers within a single pod. GPUs are shared across containers using a DRA `ResourceClaimTemplate` where all containers reference the same claim ([reference example](https://github.com/NVIDIA/k8s-dra-driver-gpu/blob/8effb048f94b3f18338a6f93527352cda82ee385/demo/specs/quickstart/v1/gpu-test2.yaml)). UDS socket sharing between containers uses an `emptyDir` volume.

### GMS Container

- All GMS processes are bundled into a single sidecar container: `devices × {weights}` processes per container. KV cache is allocated directly by the engine, not mediated through GMS (see Decision 7, Option B).
- Any child process death causes the container to exit (`wait -n` pattern), triggering a kubelet restart.
- A `startupProbe` gates on all GMS sockets being ready before kubelet unblocks engine containers. Uses sidecar init container (`restartPolicy: Always`, K8s 1.29+).

### Lock Server Container

- A lightweight UDS-based lock server runs as a sidecar container.
- Shares the `emptyDir` volume for its UDS socket.
- Must be ready before engine containers start (startup probe gates on socket availability).
- Persists lock state to a file on the shared volume for crash recovery (see Decision 8).

### Engine Restart on GMS Failure

- Engines must restart when GMS fails. Options under evaluation:
  - Application-level handler: engine detects GMS connection loss (broken pipe on UDS) and self-terminates.
  - Canary liveness probe on engine containers checking GMS socket availability (less reliable if GMS restarts faster than probe interval).
- **TODO**: Verify that engines reliably restart on GMS failure in practice.

## Decision 2: Probes and Failure Detection

### Current State

- Workers expose `/live` and `/health` via the Rust runtime's system status server. Both are aliases for the same handler (`health_handler`), which checks `SystemHealth` endpoint status.
- The operator sets `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]`, so both probes gate on the `generate` endpoint being `Ready`.
- `generate` transitions to `Ready` only when `serve_endpoint()` registers the handler with the ingress server.
- The canary health check system exists but is disabled by the operator (`DYN_HEALTH_CHECK_ENABLED=false`).

### Shadow Mode Sequencing Bug (Existing)

The current shadow mode code calls `sleep()` (which calls `unregister_endpoint_instance()`) **before** `serve_endpoint()` (which does the actual discovery registration). The unregister is a no-op since nothing is registered yet, and `serve_endpoint()` then unconditionally registers the instance in etcd. The sleeping shadow ends up discoverable and receiving traffic it cannot serve.

### Shadow Mode: Skip `serve_endpoint` at Init

`serve_endpoint()` performs no engine logic — it is purely infrastructure (ingress handler registration, etcd discovery registration, health status flip). For shadow mode, `serve_endpoint()` is deferred entirely to the wake path. The engine is fully initialized by the time `setup_vllm_engine()` returns.

### Engine State Machine

Four states. No cycling back to Standby after Active (shadow is single-use for the POC).

```
Init -> Standby -> Waking -> Active -> Dead
```

- **Init**: Model loading, CUDA graph capture.
- **Standby**: Init complete, weights offloaded, sleeping. Not discoverable. Blocking on lock server `acquire()` — the lock grant is the wake trigger (see Decision 4).
- **Waking**: Transition period. Repopulating weights (GMS remap), allocating KV cache (direct), calling `serve_endpoint()`.
- **Active**: Fully operational. Discoverable, serving inference.
- **Dead**: Process crashed or killed by kubelet.

### Role-Aware Liveness Probe

The liveness probe behavior depends on engine state:

- **Init**: `/live` returns 503. Startup probe covers this phase (2h budget via `FailureThreshold: 720`).
- **Standby**: `/live` returns 200 unconditionally. Engine is alive, intentionally idle.
- **Waking**: `/live` returns 200, but the handler checks whether the transition has exceeded a configurable timeout. If exceeded, returns 503 — container is killed. This prevents a hung wake from going undetected.
- **Active**: `/live` checks `generate` endpoint health in `SystemHealth`. Normal operation.

### Startup Probe for Shadow

The startup probe waits for the engine to leave `Init` (i.e., enter `Standby`). This signals that model loading and CUDA graph capture are complete.

**TODO**: Determine the signaling mechanism — options include setting `SystemHealth` directly or introducing an engine state field the probe handler reads.

### State Transition Ordering

The state is set to `Active` **after** `engine.wake_up()` + `serve_endpoint()` complete, not before. Setting it before risks the container being killed during a slow but healthy wake (the liveness probe would start checking endpoint health prematurely). The waking timeout handles the inverse risk (hung wake going undetected).

### `serve_endpoint` Restructuring

With the lock server as the wake trigger (Decision 4), the shadow engine's main loop is linear:

```
init_weights() → sleep() → acquire() [blocks] → wake_up() → serve_endpoint()
```

No HTTP wake endpoint is needed. `serve_endpoint()` is called directly from the main thread after `wake_up()` completes, so there is no need to spawn it as a background task from an HTTP handler. The `/engine/wake_up` system endpoint is not required for failover — the lock server mediates the transition.

### Alternative Implementation: Implicit State via Existing Health APIs

Instead of an explicit engine state enum, the same probe behavior can be achieved by leveraging the existing three-branch fallback in `SystemHealth.get_health_status()`:

1. Shadow container omits `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` (empty) and starts with `DYN_SYSTEM_STARTING_HEALTH_STATUS=notready`.
2. After model init + sleep, Python calls `set_health_status(Ready)` -> startup probe passes (branch 3: simple system health check).
3. While sleeping, no health check targets exist -> branch 3 continues returning 200.
4. On wake, `serve_endpoint()` registers health check targets -> branch 2 automatically takes over, checking `generate` endpoint status.

**Trade-off vs explicit state machine**: Covers Init, Standby, and Active implicitly through existing APIs. Does not cover the Waking state — a hung wake goes undetected by the probe. Mitigation: add a timeout inside the `wake_up()` handler that self-terminates the process on expiry.

**Requires**: Exposing `set_health_status` to Python (not currently available in bindings).

## Decision 3: Coordinatorless Weight Loading

Both engines can start in shadow mode and race to init simultaneously without a dedicated coordinator. The GMS `RW_OR_RO` lock mode handles this automatically:

- The first engine to connect gets the RW lock and loads weights normally.
- The second engine's `RW_OR_RO` request sees RW is unavailable, falls back to waiting for RO. It blocks in `_acquire_lock` until the first engine calls `commit()`.
- Once committed, the second engine gets an RO lock and imports the same weights (shared physical memory, separate VA mappings). No duplicate weight loading occurs.

This is handled entirely by the existing GMS server lock FSM (`server/locking.py`) and `_acquire_lock` in `server/rpc.py` (lines 226-255). No additional coordination logic is needed for the weight init phase.

**Update**: Decision 3's original analysis assumed TP=1. With TP > 1, the independent `RW_OR_RO` per-device approach has a deadlock. See Decision 5 for the revised weight loading design.

## Decision 4: Global Lock Mechanism for Coordinatorless Failover

A global lock determines which engine wakes up and becomes active without an external coordinator. Both engines init and sleep independently; the lock is the barrier to wake.

### Lock Purpose

The lock serves two coupled functions:
1. **Leader election** — which engine is active (serving inference)
2. **Resource safety** — the GPU is actually available for the new leader

Because GMS force eviction cannot reclaim physical GPU memory while the old engine's processes are alive (Decision 7), these concerns are inseparable: the lock can only release when the old engine's processes are dead and GPU memory is genuinely freed.

### Lock Implementation: UDS Lock Server

A lightweight sidecar process exposing a UDS socket on the shared `emptyDir` volume (see Decision 1). The connection IS the lock.

**API**:
```
acquire(engine_id) → blocks until lock is granted to engine_id
```

No explicit `release()`. Process death closes the socket, triggering automatic release on the server side. The lock server tracks one active engine at a time; `acquire()` calls from other engines block until the active engine's connection closes.

**The engine process is the lock holder**, not individual workers. The engine process outlives all TP workers — it only exits after all children are dead. This means:
- Lock release = engine process dead = all workers dead = all GPU memory freed
- No timing gap, no grace period needed regardless of TP count
- Single connection per engine, no per-rank deadlock risk

### Wake Trigger

The blocking `acquire()` call IS the standby-to-waking transition. Both engines call `acquire()` after init + sleep. The first to be granted wakes up; the second blocks until the first dies. No separate HTTP wake endpoint or external signal is needed for failover — the lock server mediates it. The engine main loop becomes linear:

```
init_weights() → sleep() → acquire() [blocks] → wake_up() → serve_endpoint()
```

### Fencing

No GMS-level force eviction. Process death IS the fencing mechanism:

1. Old engine dies (crash, OOM, kubelet kill) → lock server connection closes.
2. Lock server detects closure → releases the lock.
3. Standby engine's `acquire()` unblocks → returns.
4. By this point, old engine's processes are dead and all GPU memory (KV cache allocated directly, weights imported via GMS) is freed by CUDA driver cleanup.

### Failure Detection

- **Primary**: Engine process monitors lock server socket (POLLHUP detection via dedicated thread). On broken pipe, attempts reconnection (see Decision 8). If reconnection fails, self-terminates.
- **Secondary**: Liveness probe failure → kubelet kills container → socket closes → lock released.
- **Engine hang without process death**: Liveness probe's waking timeout catches hung wake → returns 503 → kubelet kill → process death → socket close. Acceptable latency for the POC.

### TP > 1

The engine parent process holds the single lock connection. Worker process deaths propagate to the parent (vLLM process group teardown), causing the parent to exit and the lock connection to close. No per-worker lock management needed.

**TODO** (carried from Decision 6): Verify vLLM process group teardown behavior on single-worker death.

## Decision 5: Weight Loading Roles (TP > 1 Deadlock Prevention)

### The Deadlock

With TP > 1, if both engines independently use `RW_OR_RO` per device, workers from different engines can cross-acquire RW locks on different devices:

- A0 gets RW on device 0, B1 gets RW on device 1.
- A1 waits for RO on device 1 (needs B1 to commit). B0 waits for RO on device 0 (needs A0 to commit).
- vLLM's executor waits for all workers to complete `init_device()` before calling `load_model()`.
- A0 can't commit (executor blocked waiting for A1). B1 can't commit (executor blocked waiting for B0).
- Circular dependency. Deadlock.

GMS connection happens in `init_device()` (at `worker.py:72`), before `load_model()`. The executor synchronization at `init_device()` is the structural cause — the deadlock occurs whenever workers from different engines interleave, which is near-certain with simultaneous starts.

### Solution: Deterministic Roles

- **Engine A** (primary): connects with `RW_OR_RO` on all devices. Gets RW where weights aren't committed, RO where they are.
- **Engine B** (shadow): connects with `RO` on all devices. Waits for committed state.
- Roles assigned via static configuration (e.g., `ENGINE_ROLE` env var set by the DGD operator).

Engine B only uses RO, so it can never hold an RW lock on any device. No circular dependency possible. Engine A's workers all get RW (or RO on committed devices) without contention from Engine B.

The global lock is **not** involved in weight loading. Both engines init independently with their assigned roles. The global lock is only the barrier to wake.

### Partial Commit Resilience

If Engine A crashes mid-commit (e.g., device 0 committed, device 1 not):

1. Device 0: COMMITTED state persists (RO disconnect doesn't clear allocations). Device 1: `RW_ABORT` fires, allocations cleared.
2. Engine B's workers: B0 gets RO on device 0 (imports). B1 keeps waiting for RO on device 1 — the GMS server holds the connection, re-evaluating `can_acquire_ro()` on every state change notification.
3. Engine A restarts. `RW_OR_RO` per device: device 0 is committed → gets RO (imports, fast). Device 1 is empty → gets RW (loads from disk, slow).
4. Engine A commits device 1. GMS notifies B1. `can_acquire_ro()` returns True. B1 gets RO, imports.
5. Engine B finishes init without ever restarting. Only the lost work is redone.

This works because Engine B's waiting workers are held in the GMS server's `condition.wait_for()` loop. Each state change (`RW_ABORT`, `RW_CONNECT`, `RW_COMMIT`) triggers `notify_all()`, and the workers re-evaluate. They go back to waiting if the condition isn't met, and proceed when it is. Engine A can crash and restart multiple times — Engine B just waits patiently.

A fail-fast timeout on Engine B's RO acquisition is still needed as a safety net for unrecoverable situations (GMS crash, Engine A permanently gone), but is not the primary recovery mechanism.

## Decision 6: GMS Crash During Sleep

### The Problem

`sleep()` closes the weight GMS connection: `unmap()` releases the RO weight lock and closes the socket. After sleep, the engine has zero open sockets to any GMS process. No heartbeat, no background poll. (KV cache is not mediated through GMS — see Decision 7, Option B.)

If GMS crashes and restarts while engines are sleeping, engines cannot detect the crash. Decision 1's broken-pipe detection requires an open socket — it is a no-op for sleeping engines.

### On Wake After GMS Restart

`remap()` connects to the fresh GMS and requests an RO lock. The fresh GMS has `_committed = False` (all state lost on process death). `can_acquire_ro()` requires `_committed = True` — it never returns True. Without a timeout, the engine hangs indefinitely. No writer will ever commit because both engines are sleeping with stale state.

### Solution: Fail-Fast Remap

`remap()` must always be called with a finite `timeout_ms` on the wake path. Two existing detection layers cover GMS restart scenarios:

1. **RO lock timeout**: Fresh GMS has no committed state → RO lock can never be granted → times out. Catches the common case.
2. **`StaleMemoryLayoutError`**: If a different engine loaded new weights to the fresh GMS before the sleeping engine wakes, the memory layout hash won't match. Catches the case where committed state exists but belongs to a different init cycle.

Both produce exceptions. The `wake_up()` handler must treat any remap failure as fatal and exit the process (`sys.exit(1)`), triggering a container restart and full re-init.

The healthy-case RO lock is granted in under a millisecond (weights committed, no writer active), so a timeout of 30–60s is safe from false positives.

### Interaction with the Global Lock

If the lock holder wakes up and hits the remap failure:
- Lock holder dies → lock server connection closes → lock released (Decision 4).
- Standby engine's `acquire()` unblocks → same remap failure → also dies.
- Both restart → full re-init against the fresh GMS. Clean recovery.

The lock server is independent of GMS and survives a GMS crash. The remap failure on wake is the detection mechanism, with delay equal to the remap timeout.

### TP > 1

Each worker independently remaps against its own device's weight GMS. If only one device's GMS restarted, only that worker's `remap()` fails. The failing worker must crash the entire engine. This relies on vLLM's existing process group behavior — when one worker dies, the driver tears down all workers. Consistent with how vLLM handles other per-worker fatal errors (GPU faults, OOM).

**TODO**: Verify vLLM process group teardown behavior on single-worker death.

## Decision 7: CUDA VMM Reference Counting and Force Eviction

### Discovery

Tested whether `cuMemRelease` on the GMS (creator) process frees physical GPU memory while another process (engine worker) holds an imported handle. The test (`test_cuda_vmm_refcount.py`) uses two processes communicating via UDS with `SCM_RIGHTS` to isolate the CUDA VMM behavior, measuring nvidia-smi at each stage.

Results (512 MiB allocation, nvidia-smi as ground truth):

| Step | nvidia-smi used | Delta | Meaning |
|------|-----------------|-------|---------|
| Baseline | 3 MiB | — | Nothing on GPU |
| After alloc + client import | 1198 MiB | +1195 MiB | 512 MiB alloc + CUDA context overhead |
| After server `cuMemRelease` | 1198 MiB | +0 MiB | **Memory NOT freed** |
| After client releases handle | 686 MiB | -512 MiB | Memory freed on last ref drop |
| After process death | 3 MiB | +0 MiB | Back to baseline |

CUDA VMM uses reference counting across processes. Physical memory is freed only when ALL handles (original + imported) are released. There is no CUDA API to force-free a shared allocation (`cuMemRelease`, `cuMemSetAccess`, etc. — none bypass the ref count). This is by design in the CUDA driver.

### Implication: Force Eviction via GMS Is Insufficient

GMS calling `clear_all` / `cuMemRelease` during a force eviction (e.g., `RW_ABORT` in the KV cache path) will not reclaim physical GPU memory if the old engine's worker processes are still alive holding imported handles. The new engine would attempt to allocate KV cache on a GPU that still has the old allocation pinned, leading to OOM.

### Options Explored

**A. Process death as the fencing mechanism. ← CHOSEN (fencing strategy).** Don't attempt force eviction. Ensure the old engine's processes are fully dead before the new engine allocates. The lock release must be tied to actual process death, not just GMS disconnect. Implemented via the UDS lock server (Decision 4).

**B. Don't use GMS for KV cache. ← CHOSEN (KV cache strategy).** Each engine allocates KV cache directly (not through GMS). Process death automatically frees it. GMS is only used for weights (read-only, shared, long-lived). Simplifies the design: no KV cache RW lock management, no force eviction logic.

**C. Accept the OOM gap (over-provision).** Size KV cache so two instances fit simultaneously. Wasteful — permanently halved capacity.

**D. Cooperative release (fragile).** GMS signals the old engine to release handles before eviction. Only works if the old engine is responsive — the failure scenario where it's not.

### Lock Purpose Redefined

The global lock serves two functions:
1. **Leader election** — which engine is active (serving inference)
2. **Resource safety** — the GPU is actually available for the new leader

Because force eviction doesn't work, these two concerns must be coupled: the lock can only release when the old engine's processes are dead and GPU memory is genuinely freed.

### Lock Implementation: Ref-Counted UDS/TCP Server

A lightweight lock server where the connection IS the lock:

- `acquire(engine_id)` — blocks until this engine is active. The returned socket must be kept open.
- No `release()` — process death closes the socket, the server auto-decrements. Lock releases when ref count hits 0.
- The lock server tracks one active engine at a time. Workers from other engines block until the active engine's count drops to zero, then the next engine's waiters are promoted.

**The engine process is the lock holder**, not individual workers. The engine process outlives all TP workers — it only exits after all children are dead. This means:
- Lock release = engine process dead = all workers dead = all GPU memory freed
- No timing gap, no grace period needed regardless of TP count
- Single ref per engine, no per-rank deadlock risk

For single-node: UDS on shared `emptyDir` volume. For multi-node (engine spans nodes): same protocol over TCP, lock server exposed as a K8s Service. API and semantics are identical; only the transport changes.

### RW_GLOBAL (Explored, Deferred)

An alternative design where GMS itself mediates the global lock was explored: a new `RW_GLOBAL` lock type where GMS contacts the lock service on behalf of the worker, with per-rank ref counting through GMS connections. This integrates the lock into the existing GMS protocol and solves the CUDA VMM cleanup ordering (by the time the ref count drops to 0, both imported and creator handles are released). However, it adds significant complexity to GMS (new lock type, lock service dependency, GMS crash handling) and is deferred in favor of the simpler engine-process-as-lock-holder design for the POC.

## Decision 8: Lock Server Crash Recovery

### The Problem

The lock server is an SPOF within the pod. If it crashes while Engine A holds the lock:

1. Lock server restarts with no in-memory state.
2. Engine B calls `acquire("engine-b")`. Fresh server has no record of Engine A — grants the lock.
3. Engine B wakes up. Engine A is still alive and serving. Split-brain.

### Design: Persistent State + Reconnect Window

The lock server persists its state to a file on the shared `emptyDir` volume:

```json
{"holder": "engine-a", "granted_at": "2026-02-17T10:05:30Z"}
```

On startup (fresh or restart), the lock server reads this file:

- **No persisted state (first boot)**: Grant immediately to the first `acquire()` caller.
- **Persisted state exists (restart)**:
  1. Enter a **reconnect window** (configurable, e.g., 10s). All `acquire()` calls block.
  2. If the previous holder reconnects within the window → re-grant the lock. No failover triggered. The active engine survives the lock server blip transparently.
  3. If the reconnect window expires → previous holder is presumed dead. Lock becomes available, next `acquire()` is granted.

### Engine-Side Behavior on Lock Server Crash

The engine process monitors its lock server socket via a dedicated thread (`poll()` for POLLHUP or blocking `recv()` returning EOF). On detection:

1. Immediately begins reconnection attempts to the lock server socket path.
2. If reconnection succeeds and the lock is re-granted, the engine continues operating without disruption.
3. If reconnection fails within a configurable timeout (e.g., lock server not back yet, or another engine was granted the lock), the engine self-terminates.

Detection time for the broken pipe is milliseconds. The reconnection timeout on the engine side should be slightly longer than the lock server's reconnect window to allow for lock server restart time.

### Failure Scenarios

| Scenario | Lock Server | Engine A | Engine B | Outcome |
|----------|-------------|----------|----------|---------|
| Lock server crash, Engine A healthy | Restarts, reads state, waits for reconnect | Detects POLLHUP, reconnects | Blocked in acquire() | A reconnects, continues. No failover. |
| Lock server crash, Engine A dead | Restarts, reads state, waits for reconnect | Dead | Blocked in acquire() | Reconnect window expires, B acquires. |
| Lock server crash, Engine A hung | Restarts, reads state, waits for reconnect | Can't detect POLLHUP | Blocked in acquire() | Reconnect window expires, B acquires. Liveness probe eventually kills A. |
| Both crash simultaneously | Restarts, reads state, waits for reconnect | Restarts, full re-init | Restarts, full re-init | Reconnect window expires (old holder dead). First to finish init + acquire() wins. |

### State File Semantics

- **Written** on every `acquire()` grant.
- **Cleared** (holder set to null) when the holder's connection closes and no reconnect window is active.
- **Location**: `emptyDir` volume (survives container restart within the same pod; lost on pod reschedule — acceptable since pod reschedule restarts everything).
- **Concurrency**: `flock()` protects against partial writes during crash.
