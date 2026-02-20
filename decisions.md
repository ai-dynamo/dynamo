# GPU Failover POC: Decisions

## Decision 1: K8s Layout

**Intra-pod with shared GPU access via DRA.**

Primary engine, shadow engine, and GMS run as containers within a single pod. GPUs are shared across containers using a DRA `ResourceClaimTemplate` where all containers reference the same claim ([reference example](https://github.com/NVIDIA/k8s-dra-driver-gpu/blob/8effb048f94b3f18338a6f93527352cda82ee385/demo/specs/quickstart/v1/gpu-test2.yaml)). UDS socket sharing between containers uses an `emptyDir` volume.

### GMS Container

- All GMS processes are bundled into a single sidecar container: `devices × {weights, kv_cache}` processes per container.
- Any child process death causes the container to exit (`wait -n` pattern), triggering a kubelet restart.
- A `startupProbe` gates on all GMS sockets being ready before kubelet unblocks engine containers. Uses sidecar init container (`restartPolicy: Always`, K8s 1.29+).

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
- **Standby**: Init complete, weights offloaded, sleeping. Not discoverable. Waiting for wake signal via `/engine/wake_up` on the system port.
- **Waking**: Transition period. Repopulating weights, allocating KV cache, calling `serve_endpoint()`.
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

`serve_endpoint()` is a long-lived blocking await (runs the request loop). The `wake_up()` handler is an HTTP handler that must return a response. For shadow mode, `serve_endpoint()` must be spawned as a background task from within `wake_up()`. The handler needs access to all endpoint objects and the health check payload.

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
