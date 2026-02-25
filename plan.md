# GPU Failover POC: Implementation Plan

Reference: [decisions.md](decisions.md)

## Stack

1. **Failover lock** — `FailoverLock` interface + `flock`-based implementation
2. **Deterministic weight loading** — ENGINE_ID env var (0 = RW_OR_RO, 1+ = RO)
3. **GMS failure handling** — remap timeout, StaleMemoryLayoutError detection, broken-pipe detection on active GMS connections
4. **Shadow main loop rewrite** — init → sleep → acquire → wake → serve, linear serve_endpoint, fixes the sequencing bug
5. **System health + probes** — state machine (Init, Standby, Waking, Active, Dead), role-aware liveness probe, startup probe signaling, waking timeout guard
6. **Operator + DGD** — shadow field, pod spec (3 containers: primary, shadow, GMS sidecar), DRA, volumes, probes, env vars

---

## Diff 1: Failover Lock

### Overview

A file-lock-based leader election mechanism for engine failover. Uses `flock(LOCK_EX)` on a shared file (`emptyDir` volume) as the lock primitive. No server process, no sidecar, no protocol — the Linux kernel is the lock manager.

Both engines call `acquire()` after init + sleep. The first to acquire the flock wakes up and serves. The second blocks in a polling loop. When the active engine dies, the kernel releases the flock, and the blocked engine acquires it.

Decisions covered: D4 (lock implementation, wake trigger).

### Why flock

- **No SPOF.** A UDS lock server is itself a single point of failure that needs crash recovery logic (Decision 8). With flock, the "lock server" is the Linux kernel. It doesn't crash.
- **Automatic release on process death.** `flock()` is released when the process dies (even SIGKILL). This couples leader election to resource safety — by the time the standby acquires, the old engine's GPU memory is freed.
- **Cross-container via emptyDir.** `flock()` operates at the kernel VFS layer. All containers in a pod share the same kernel. A lock held by the engine container is visible to any other container accessing the same file on the shared volume.
- **~25 lines of implementation.** No server, no protocol, no sidecar, no state file, no reconnect logic.

Single-node only. For multi-node (engine spanning nodes), a UDS/TCP server or K8s Lease implementation of the same `FailoverLock` interface can be swapped in without changing engine code.

### File Layout

```
lib/gpu_memory_service/failover_lock/
├── __init__.py              # re-exports FailoverLock
├── interface.py             # FailoverLock ABC
└── flock/
    ├── __init__.py          # re-exports FlockFailoverLock
    └── lock.py              # FlockFailoverLock implementation
```

### Files

#### `lib/gpu_memory_service/failover_lock/interface.py`

The abstract interface. Implementation-agnostic.

```python
"""Failover lock for GPU engine leader election.

In a failover deployment, two engines (primary and shadow) initialize
weights and then race to become the active engine — the one serving
inference. The FailoverLock gates this transition: acquire() blocks
until this engine is granted the active role.

The lock couples two concerns:
  1. Leader election — which engine is active
  2. Resource safety — GPU memory is free for the new leader

Release happens on process death (implicit) or explicit call. By the
time a standby engine acquires, the old engine's GPU memory (KV cache,
CUDA contexts) has been reclaimed by the OS/driver.
"""

import asyncio
from abc import ABC, abstractmethod


class FailoverLock(ABC):
    @abstractmethod
    async def acquire(self) -> asyncio.Event:
        """Block until this engine is granted the active role.

        Returns an asyncio.Event that is set if the lock is lost
        (implementation-dependent; e.g., lock server crash for a
        UDS-based impl). For flock, the event never fires — the
        lock cannot be lost while the process is alive.
        """
        ...

    @abstractmethod
    async def release(self) -> None:
        """Release the lock (give up the active role).

        Called on graceful shutdown. Also happens implicitly on
        process death (kernel closes the file descriptor, releasing
        the flock).
        """
        ...

    @abstractmethod
    async def owner(self) -> str | None:
        """Return the engine_id of the current lock holder, or None.

        Reads the lock file contents. The holder writes its engine_id
        into the file after acquiring. Useful for logging and
        observability. Callable from any process with access to the
        shared volume.
        """
        ...
```

#### `lib/gpu_memory_service/failover_lock/flock/lock.py`

The `flock`-based implementation.

```python
import asyncio
import fcntl
import logging
import os

from gpu_memory_service.failover_lock.interface import FailoverLock

logger = logging.getLogger(__name__)


class FlockFailoverLock(FailoverLock):
    def __init__(self, lock_path: str, engine_id: str):
        self._lock_path = lock_path
        self._engine_id = engine_id
        self._fd: int | None = None

    async def acquire(self, poll_interval: float = 0.05) -> asyncio.Event:
        self._fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR)

        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(poll_interval)

        # Write identity for observability (owner() reads this)
        os.ftruncate(self._fd, 0)
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.write(self._fd, self._engine_id.encode())

        logger.info(f"Failover lock acquired: {self._engine_id}")

        # flock cannot be lost externally — event never fires
        return asyncio.Event()

    async def release(self) -> None:
        if self._fd is not None:
            logger.info(f"Failover lock released: {self._engine_id}")
            os.close(self._fd)
            self._fd = None

    async def owner(self) -> str | None:
        try:
            with open(self._lock_path, 'r') as f:
                content = f.read().strip()
                return content if content else None
        except FileNotFoundError:
            return None
```

**How it works:**

1. `acquire()` opens the lock file with `O_CREAT | O_RDWR` and attempts `flock(LOCK_EX | LOCK_NB)` in a polling loop.
   - `LOCK_EX`: exclusive lock — only one process can hold it.
   - `LOCK_NB`: non-blocking — raises `BlockingIOError` instead of blocking the thread, keeping the asyncio event loop responsive.
   - The loop sleeps 50ms between attempts. When the other engine dies and the kernel releases the flock, the next poll succeeds.
2. After acquiring, the holder writes its `engine_id` into the file. `owner()` reads this from any process.
3. `release()` closes the file descriptor. The kernel releases the flock. If the process dies without calling `release()`, the kernel closes all FDs automatically — same effect.

**Polling vs blocking thread:** The polling approach (`LOCK_NB` + `asyncio.sleep`) is preferred over `asyncio.to_thread(flock, LOCK_EX)` because it's cancellable (the coroutine yields at each sleep), easy to extend with timeouts, and stays in pure asyncio with no thread pool. The 50ms poll interval adds negligible latency to failover (weight remap takes seconds).

#### `lib/gpu_memory_service/failover_lock/__init__.py`

```python
from gpu_memory_service.failover_lock.interface import FailoverLock

__all__ = ["FailoverLock"]
```

#### `lib/gpu_memory_service/failover_lock/flock/__init__.py`

```python
from gpu_memory_service.failover_lock.flock.lock import FlockFailoverLock

__all__ = ["FlockFailoverLock"]
```

### Usage (preview of diff 4)

```python
from gpu_memory_service.failover_lock.flock import FlockFailoverLock

lock = FlockFailoverLock("/shared/failover.lock", engine_id="engine-a")

init_weights()
sleep()

lost = await lock.acquire()  # blocks until flock granted

wake_up()
serve_endpoint()

# On graceful shutdown:
await lock.release()
```

### Failover Trace

```
Engine A                         Kernel                         Engine B
   |                               |                               |
   | flock(LOCK_EX|LOCK_NB) -----> |                               |
   |                               | granted (no holder)           |
   | <--- success -----------------|                               |
   | write "engine-a" to file      |                               |
   |                               |                               |
   | [lock held, serving]          |                               |
   |                               |  flock(LOCK_EX|LOCK_NB) ----> |
   |                               |  EWOULDBLOCK (A holds it)     |
   |                               |  <--- BlockingIOError --------|
   |                               |         ... sleep 50ms ...    |
   |                               |  flock(LOCK_EX|LOCK_NB) ----> |
   |                               |  EWOULDBLOCK                  |
   |                               |         ... sleep 50ms ...    |
   |                               |                               |
   | [Engine A dies]               |                               |
   | --- close(fd) --------------> |                               |
   |                               | flock released (last FD closed)|
   |                               |                               |
   |                               |  flock(LOCK_EX|LOCK_NB) ----> |
   |                               |  granted                      |
   |                               |  <--- success ----------------|
   |                               |  write "engine-b" to file     |
   |                               |  [lock held, serving]         |
```

### Tests

Tests live alongside the implementation or under `tests/`. Use `pytest-asyncio` and `tmp_path` for isolated lock files.

**Test cases:**

1. **`test_acquire_release`** — One lock acquires and releases. Verify file contains engine_id after acquire, FD is closed after release.

2. **`test_two_engines_contention`** — Engine A acquires. Engine B calls acquire (runs as a separate asyncio task). Close Engine A's FD. Verify Engine B acquires within a few poll intervals.

3. **`test_process_death_releases`** — Fork a child process that acquires the lock. Kill the child (SIGKILL). Verify parent can acquire. Confirms kernel releases flock on process death.

4. **`test_owner`** — Engine A acquires. Call `owner()` from a separate instance. Verify returns "engine-a". Release. Call `owner()` — file still contains "engine-a" (stale but acceptable; flock is the authority, not the file content).

5. **`test_cross_process`** — Two separate processes (via `multiprocessing`) race to acquire. Verify exactly one wins, the other blocks, and on winner death the other acquires.

### Known Limitations

- **Single-node only.** `flock()` works on local filesystems. Multi-node requires a UDS/TCP or K8s Lease implementation of `FailoverLock`.
- **No lock loss event.** The `asyncio.Event` returned by `acquire()` never fires for the flock implementation. A UDS-based implementation would set this event on server crash.
- **50ms acquisition latency.** The polling interval adds up to 50ms between the old engine dying and the new engine acquiring. Negligible relative to wake-up time.
- **Brief `owner()` inconsistency.** Between `ftruncate` and `write`, `owner()` may return empty. Window is microseconds.

### Not Included in This Diff

- Integration with engine main.py (diff 4)
- K8s deployment / emptyDir volume configuration (diff 6)
- UDS-based `FailoverLock` implementation (deferred; same interface, swap when multi-node needed)

---

## Diff 2: Deterministic Weight Loading

### Overview

Prevent the TP > 1 deadlock (Decision 5) by assigning deterministic GMS lock modes based on `ENGINE_ID`. Engine 0 uses `RW_OR_RO` (can load weights from disk or import). Engine 1+ uses `RO` (import only, blocks until weights are committed).

Decisions covered: D5 (weight loading roles).

### Current State

`RW_OR_RO` is hardcoded in two vLLM call sites:

1. `worker.py:73` — `init_device()` creates the singleton GMS connection
2. `model_loader.py:68` — `load_model()` reuses the singleton via `get_or_create`

Since `init_device()` always runs before `load_model()`, the singleton is created by worker.py. Both call sites must be consistent.

### The Change

**`ENGINE_ID` env var:** Integer. `0` = primary (default, backward compatible). `1+` = shadow (RO only).

This is the same identifier the lock uses for `acquire()` — one env var serves both weight loading roles and lock identity.

### Files

#### `lib/gpu_memory_service/common/utils.py` — Add helper

```python
def get_weight_lock_type() -> "RequestedLockType":
    """Determine weight GMS lock type from ENGINE_ID.

    ENGINE_ID=0 (default): RW_OR_RO — can load weights or import.
    ENGINE_ID=1+: RO — import only, blocks until committed.
    """
    from gpu_memory_service.common.types import RequestedLockType

    engine_id = int(os.environ.get("ENGINE_ID", "0"))
    if engine_id == 0:
        return RequestedLockType.RW_OR_RO
    return RequestedLockType.RO
```

#### `lib/gpu_memory_service/integrations/vllm/worker.py` — Use helper

```python
# Line 73, in init_device():
# Before:
mode=RequestedLockType.RW_OR_RO
# After:
mode=get_weight_lock_type()
```

#### `lib/gpu_memory_service/integrations/vllm/model_loader.py` — Use helper

```python
# Line 68, in load_model():
# Before:
mode=RequestedLockType.RW_OR_RO
# After:
mode=get_weight_lock_type()
```

### Behavior by ENGINE_ID

| ENGINE_ID | Lock type requested | Server grants | Behavior |
|-----------|-------------------|---------------|----------|
| 0 (default) | RW_OR_RO | RW (if no writer) or RO (if committed) | Loads weights from disk OR imports. Backward compatible with single-engine. |
| 1+ | RO | RO (blocks until committed) | Always imports. Never holds RW on any device. No deadlock possible. |

### Partial Commit Resilience

If Engine 0 crashes mid-commit (e.g., device 0 committed, device 1 not):

1. Engine 1's workers on device 0 already have RO (imported). Device 1's worker blocks in `condition.wait_for()`.
2. Engine 0 restarts. Its device 0 worker gets RO (committed → fast import). Device 1 worker gets RW (empty → loads from disk).
3. Engine 0 commits device 1. GMS notifies Engine 1's waiting worker. It imports.
4. Engine 1 finishes init without restarting. Only the lost work is redone.

This works because Engine 1's RO acquisition blocks in the GMS server's `condition.wait_for()` loop, re-evaluating on every state change. No timeout needed for this path (timeout is for the GMS-crash-during-sleep scenario in diff 3).

### Not in Scope

- sglang integration (`memory_saver.py:84` also hardcodes `RW_OR_RO` — same pattern, deferred)
- Who sets `ENGINE_ID` in K8s (diff 6, operator injects per container)

### Tests

Adapt `test_shadow_sleep_wake.sh`:

1. **TP=1, two engines**: Engine 0 (`ENGINE_ID=0`) and Engine 1 (`ENGINE_ID=1`) start simultaneously. Verify Engine 0 gets RW, Engine 1 blocks on RO, Engine 0 commits, Engine 1 imports.
2. **TP=2, two engines**: Same but with 2 devices. Verify no deadlock — Engine 1 never holds RW on any device.
3. **Engine 0 crash mid-commit (TP=2)**: Engine 0 commits device 0, crashes before device 1. Verify Engine 1 waits patiently. Engine 0 restarts, commits device 1, Engine 1 completes.

---

## Diff 3: GMS Failure Handling (remap timeout + stale layout)

### Overview

Make the `wake_up()` → `remap()` path robust against two failure modes that are currently unhandled:

1. **GMS unreachable / RO lock timeout** — `remap()` calls `_connect(lock_type=RO)` which opens a new UDS connection and performs a handshake. If GMS is dead, mid-restart, or the RW lock is held indefinitely by a crashed writer, this blocks forever. We add a `timeout_ms` to bound the wait.

2. **Stale memory layout** — While the engine was unmapped (sleeping), a writer could have changed the allocation structure (different sizes, different tensor layouts). `remap()` already detects this via a hash comparison and raises `StaleMemoryLayoutError`. The worker currently doesn't catch it.

Both failures are **fatal** for the engine: it cannot serve inference with missing or wrong weight mappings. The correct response is to exit so the orchestrator (K8s) can restart the pod.

Decisions covered: D3 (GMS failure handling), D7 (remap is the only reconnection point).

### What Already Exists

The `GMSClientMemoryManager.remap()` method already has the full machinery:

```python
# memory_manager.py, lines 415-485

def remap(self, timeout_ms: Optional[int] = None) -> bool:
    # ...
    self._connect(lock_type=RequestedLockType.RO, timeout_ms=eff_timeout)
    #   ↑ raises TimeoutError if timeout_ms expires

    current_hash = self._client_rpc.get_memory_layout_hash()
    if self._last_memory_layout_hash and current_hash != self._last_memory_layout_hash:
        raise StaleMemoryLayoutError(...)
    #   ↑ raises StaleMemoryLayoutError if layout changed

    # ... remap preserved VAs ...
```

The `GMSRPCClient.__init__` → `_connect()` → `HandshakeRequest(timeout_ms=...)` sends the timeout to the server, which enforces it during lock acquisition. If the server is unreachable entirely (socket doesn't exist, connection refused), the `socket.connect()` call itself raises `ConnectionError` immediately.

**No changes needed in `memory_manager.py` or `rpc.py`.**

### The Change

A single file change in `GMSWorker.wake_up()`:

#### `lib/gpu_memory_service/integrations/vllm/worker.py`

**Before** (current code, lines 196-201):

```python
if "weights" in tags:
    manager = get_gms_client_memory_manager()
    assert manager is not None, "GMS client is not initialized"
    assert manager.is_unmapped, "GMS weights are not unmapped"
    manager.remap()
    torch.cuda.synchronize()
```

**After:**

```python
if "weights" in tags:
    manager = get_gms_client_memory_manager()
    assert manager is not None, "GMS client is not initialized"
    assert manager.is_unmapped, "GMS weights are not unmapped"

    try:
        manager.remap(timeout_ms=30_000)
    except TimeoutError:
        logger.error(
            "Fatal: timed out waiting for GMS RO lock during remap "
            "(GMS may be down or RW lock held indefinitely)"
        )
        sys.exit(1)
    except StaleMemoryLayoutError as e:
        logger.error(
            "Fatal: weight layout changed while unmapped, cannot remap: %s", e
        )
        sys.exit(1)
    except ConnectionError as e:
        logger.error(
            "Fatal: cannot connect to GMS during remap: %s", e
        )
        sys.exit(1)

    torch.cuda.synchronize()
```

**New import at the top of `worker.py`:**

```python
import sys
from gpu_memory_service.client.memory_manager import StaleMemoryLayoutError
```

### Why `sys.exit(1)` and not an exception

`wake_up()` is called by vLLM's engine core via `collective_rpc` on each worker. An exception here would propagate through vLLM's RPC machinery, which may or may not surface it cleanly. `sys.exit(1)` is definitive: the worker process exits, vLLM's process group detects the failure, and the engine dies. In K8s, the pod restarts.

This matches the existing pattern in the codebase — `handlers.py` already uses `os._exit(1)` for `EngineDeadError`.

### Timeout Value (30s)

30 seconds is chosen as a reasonable upper bound:

- **GMS restart**: If the GMS container crashes and restarts, it typically takes a few seconds. 30s covers slow restarts.
- **RW lock contention**: If another writer is holding RW (e.g., Engine 0 reloading weights), 30s is generous. Normal weight commit takes 1-5s.
- **Not infinite**: Without a timeout, a dead GMS would hang the wake path forever, preventing the engine from ever recovering or exiting.

For POC this is a hardcoded constant. A future enhancement could make it configurable via env var (e.g., `GMS_REMAP_TIMEOUT_MS`).

### Failure Scenarios

| Scenario | What happens | Exception | Outcome |
|----------|-------------|-----------|---------|
| GMS is healthy, no writer | `remap()` succeeds immediately | none | Engine wakes, serves |
| GMS is healthy, RW lock held briefly | `remap()` blocks until RO granted (< timeout) | none | Engine wakes after brief delay |
| GMS is healthy, RW lock held > 30s | `_connect()` times out | `TimeoutError` | Engine exits |
| GMS container is dead (socket missing) | `socket.connect()` fails | `ConnectionError` | Engine exits |
| GMS container is restarting (socket exists, no listener) | `socket.connect()` fails | `ConnectionError` | Engine exits |
| GMS restarted with new/empty state | Hash mismatch after connect | `StaleMemoryLayoutError` | Engine exits |
| GMS restarted, same weights re-committed | Hash matches | none | Engine wakes (correct) |

### Deferred: Active Connection Monitoring

Detecting GMS death **during** inference (while the RO socket is idle) is a separate concern. Options discussed:

- `asyncio.add_reader(fd, callback)` on the GMS socket — works when an asyncio loop is available (rank-0 / main process)
- Daemon thread with blocking `socket.recv(1)` — universal but heavier
- `SIGIO` / `O_ASYNC` on the fd — kernel-level callback, fragile in Python

This is tabled for now. The remap-time checks in this diff handle the critical path (wake from sleep). Active monitoring can be added as a follow-up.

### Tests

1. **`test_remap_timeout`** — Start a GMS server, connect a writer that holds RW lock. Shadow engine calls `remap(timeout_ms=1000)`. Verify `TimeoutError` is raised after ~1s, not hanging.

2. **`test_remap_stale_layout`** — Engine A connects RO, unmaps. Engine B connects RW, commits different allocations. Engine A calls `remap()`. Verify `StaleMemoryLayoutError`.

3. **`test_remap_gms_dead`** — Engine connects RO, unmaps. Kill GMS (remove socket file). Call `remap()`. Verify `ConnectionError`.

4. **`test_remap_success`** — Normal unmap/remap cycle with no changes. Verify `remap()` returns `True`, weights are accessible.

### Not in Scope

- Active GMS connection monitoring during inference (deferred, see above)
- Retry logic (intentionally omitted — exit and let K8s restart is simpler and safer for POC)
- Changes to `memory_manager.py` or `rpc.py` (already have the right behavior)

---

## Diff 4+5: Shadow Main Loop Rewrite + Health Probes

### Overview

Rewrite the shadow engine control flow in `init_decode_worker` to use the failover lock for leader election, and drive health probe behavior through the existing `SystemHealth` infrastructure without adding new state enums or routes.

Today the shadow flow is: `register → sleep → serve_endpoint` (with wake_up as a passive HTTP handler). The new flow is: `sleep → set_health_status(Ready) → acquire lock → wake → register → serve_endpoint` — linear, lock-driven, no side-effect HTTP wake handler.

Decisions covered: D2 (state machine via implicit health APIs), D4 (lock-driven wake trigger).

### Current Flow (today)

```
init_decode_worker():
  setup_vllm_engine()
  DecodeWorkerHandler(...)
  register_engine_route("sleep", ...)
  register_engine_route("wake_up", ...)
  register_vllm_model(...)               # registers with discovery BEFORE sleep
  if shadow:
      handler.sleep()                     # unregisters, puts vLLM to sleep
      log("call /engine/wake_up to activate")
  health_check_payload = ...
  await asyncio.gather(serve_endpoint(...))  # entered while sleeping
```

**Problems:**
- Engine is briefly visible in discovery before sleeping (premature registration).
- `serve_endpoint` is entered while the engine is asleep — wake is a side-effect from an HTTP call.
- No lock gates who wakes — any caller can hit `/engine/wake_up`.
- If `wake_up` fails, the engine is stuck sleeping inside the serve loop.

### New Flow (shadow mode only)

```
init_decode_worker():
  setup_vllm_engine()
  DecodeWorkerHandler(...)
  register_engine_route("sleep", ...)
  register_engine_route("wake_up", ...)

  if normal mode:
      register_vllm_model(...)
      await serve_endpoints(...)           # unchanged

  if shadow mode:
      handler.sleep()                      # unmap weights, free KV
      set_health_status(Ready)             # startup probe passes (branch 3)
      lock = FlockFailoverLock(path, engine_id)
      await lock.acquire()                 # BLOCKS until flock granted
      handler.wake_up(...)                 # remap weights + allocate KV (diff 3 error handling)
      register_vllm_model(...)             # NOW register — engine is ready
      await serve_endpoints(...)           # serve (linear, entered only when active)
```

### Health Probe Strategy

Uses the existing three-branch fallback in `SystemHealth.get_health_status()` without any new routes or state enums.

**Configuration (shadow container env vars):**
- `DYN_SYSTEM_STARTING_HEALTH_STATUS=notready`
- `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` — **not set**

**Branch selection logic (from `system_health.rs`):**
- Branch 1: If `use_endpoint_health_status` is non-empty → check those endpoints. **Not active** (env var not set).
- Branch 2: Else if `health_check_targets` is non-empty → check registered targets. **Active once `serve_endpoint` is called** (ingress registers targets).
- Branch 3: Else → `system_health == Ready`. **Active during INIT, STANDBY, and WAKING** (no targets registered yet).

**Probe behavior by state:**

| State | What happens | Active branch | `system_health` field | Health check targets | Probe result |
|-------|-------------|---------------|----------------------|---------------------|-------------|
| **INIT** | Model loading, GMS connection | Branch 3 | `NotReady` (starting config) | none | **503** |
| **STANDBY** | Sleeping, waiting for lock | Branch 3 | `Ready` (set after sleep) | none | **200** |
| **WAKING** | Lock acquired, remap + KV alloc | Branch 3 | `Ready` (unchanged) | none | **200** |
| **ACTIVE** | `serve_endpoint` running | Branch 2 | `Ready` (irrelevant) | registered, ingress sets `Ready` | **200** |
| **DEAD** | Process exited | — | — | — | connection refused |

**Transition from branch 3 → branch 2 (WAKING → ACTIVE):**
When `serve_endpoint` is called, `endpoint.rs` registers health check targets (inserting `NotReady` into `endpoint_health`). Immediately after, the ingress serve loop sets the endpoint to `Ready`. No 503 blip — the two calls happen within the same `start()` method.

**Why this works for K8s:**
- During INIT: startup probe fails (503). K8s waits without killing.
- During STANDBY: startup probe passes (200). Liveness passes. K8s keeps the pod alive.
- During WAKING: liveness still passes (200). Engine is transitioning.
- During ACTIVE: ingress-driven health. Probe reflects real serving ability.
- Traffic routing is gated by **discovery registration** (`register_vllm_model`), not by the probe. The frontend doesn't know about the shadow until `serve_endpoint` registers it. So the probe returning 200 during STANDBY/WAKING doesn't cause premature traffic.

### Known Limitation: Hung Wake

During WAKING, the probe returns 200 (branch 3). If `wake_up()` hangs beyond `remap()` (which has a 30s timeout from diff 3) — e.g., in KV cache allocation or CUDA sync — the probe cannot detect this. The process would be stuck in WAKING with a healthy probe indefinitely.

**Mitigation options (not implemented in this diff):**
- Watchdog timer (`threading.Timer` + `os._exit`) around the wake call
- Branch 1 approach: set `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS=["generate"]` and flip to `NotReady` during waking, making the K8s liveness probe act as the timeout
- Both are straightforward to add later if needed

For POC, `remap(timeout_ms=30_000)` covers the primary risk (GMS unreachable). The remaining wake steps are local GPU operations that are unlikely to hang in practice.

### Prerequisite: `set_health_status` Python Binding

`SystemHealth.set_health_status()` exists in Rust but is not exposed to Python. A small binding is needed.

**Files:**

#### `lib/bindings/python/src/dynamo/_core.pyi` — Add type stub

```python
class DistributedRuntime:
    # ... existing methods ...

    def set_health_status(self, ready: bool) -> None:
        """Set the system health status.

        Args:
            ready: True for Ready, False for NotReady.
        """
        ...
```

#### Rust binding implementation (in the PyO3 DistributedRuntime impl)

```rust
fn set_health_status(&self, ready: bool) -> PyResult<()> {
    let status = if ready {
        HealthStatus::Ready
    } else {
        HealthStatus::NotReady
    };
    self.inner.system_health().lock().set_health_status(status);
    Ok(())
}
```

### Files Changed

#### `components/src/dynamo/vllm/main.py` — Shadow branch rewrite

The shadow branch in `init_decode_worker` changes from:

```python
# Current: register → sleep → serve
await register_vllm_model(...)

if config.gms_mode == "shadow":
    await handler.sleep({"level": 1})
    logger.info("[Shadow] Engine is now sleeping...")

health_check_payload = ...
await asyncio.gather(serve_endpoints(...))
```

To:

```python
if config.gms_mode == "shadow":
    # Sleep: unmap weights, free KV
    await handler.sleep({"level": 1})

    # Signal startup probe: model loaded, sleeping, ready to be activated
    runtime.set_health_status(True)
    logger.info("[Shadow] Engine sleeping, startup probe passing, waiting for lock")

    # Acquire failover lock (blocks until flock granted)
    lock_path = os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock")
    engine_id = os.environ.get("ENGINE_ID", "0")
    lock = FlockFailoverLock(lock_path, engine_id=f"engine-{engine_id}")
    await lock.acquire()
    logger.info("[Shadow] Lock acquired, waking engine")

    # Wake: remap weights + allocate KV (diff 3 error handling in worker.py)
    await handler.wake_up({"tags": None})
    logger.info("[Shadow] Engine awake, registering with discovery")

    # NOW register — engine is fully ready to serve
    await register_vllm_model(...)

    # Serve (linear entry, not gathered with sleep)
    health_check_payload = VllmHealthCheckPayload(
        engine_client, use_text_input=config.use_vllm_tokenizer
    ).to_dict()
    await serve_endpoints(...)

else:
    # Normal mode: unchanged
    await register_vllm_model(...)
    health_check_payload = ...
    await serve_endpoints(...)
```

(The `serve_endpoints(...)` calls are shorthand for the existing `asyncio.gather(generate_endpoint.serve_endpoint(...), ...)` block.)

#### `components/src/dynamo/vllm/main.py` — New imports

```python
from gpu_memory_service.failover_lock.flock import FlockFailoverLock
```

### What Doesn't Change

- `handlers.py` — `sleep()` and `wake_up()` methods unchanged (wake_up error handling is in worker.py from diff 3)
- `memory_manager.py` / `rpc.py` — no changes (diffs 1-3 are complete)
- Normal (non-shadow) engine flow — completely untouched
- The `/engine/sleep` and `/engine/wake_up` HTTP routes — still registered for operational use, just not the wake trigger for shadow mode

### Tests

1. **`test_shadow_flow_sequence`** — Mock the failover lock. Verify the shadow branch calls sleep → set_health_status → acquire → wake_up → register_vllm_model → serve in that order.

2. **`test_no_premature_registration`** — Verify `register_vllm_model` is not called before `wake_up` completes in shadow mode.

3. **`test_health_probe_init`** — Start shadow engine with `DYN_SYSTEM_STARTING_HEALTH_STATUS=notready`. Verify probe returns 503 during init.

4. **`test_health_probe_standby`** — After sleep + `set_health_status(Ready)`, verify probe returns 200.

5. **`test_health_probe_active`** — After serve_endpoint starts, verify probe still returns 200 (branch 2 seamless takeover).

6. **`test_normal_mode_unchanged`** — Verify non-shadow engines follow the original flow with no behavioral changes.

---

## Diff 6: Operator + DGD — K8s Deployment

### Overview

Add a `failover` field to the DGD worker spec. When enabled, the operator transforms a single-container worker pod into a multi-container failover pod: two engine containers (ENGINE_ID=0 and ENGINE_ID=1), a GMS weight sidecar, and shared volumes for UDS + flock. Both engines run the same shadow flow (init → sleep → acquire → wake → serve); the only asymmetry is the weight loading role.

Decisions covered: D1 (K8s layout), D2 (probe config), D4 (flock file on shared volume), D5 (ENGINE_ID injection).

### User-Facing API

A single boolean on the worker service:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-failover
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag

    VllmWorker:
      componentType: worker
      replicas: 1
      failover:
        enabled: true
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          workingDir: /workspace/examples/backends/vllm
          command: ["python3", "-m", "dynamo.vllm"]
          args: ["--model", "Qwen/Qwen3-0.6B"]
```

When `failover.enabled: true`, the operator generates a pod with 3 containers instead of 1. The DGD user doesn't need to know about GMS, ENGINE_ID, flock, or sidecar configuration — the operator handles all of it.

### CRD Changes

#### `deploy/operator/api/v1alpha1/dynamocomponentdeployment_types.go`

Add to `DynamoComponentDeploymentSharedSpec`:

```go
// Failover enables active-passive failover for this worker.
// When enabled, two engine containers run in the same pod with a
// shared GMS weight sidecar. The first engine to acquire the failover
// lock becomes active; the second sleeps as a hot standby.
// Only valid for componentType "worker".
// +optional
Failover *FailoverSpec `json:"failover,omitempty"`
```

#### `deploy/operator/api/v1alpha1/common.go`

Add the new type:

```go
// FailoverSpec configures active-passive failover for a worker component.
// This struct is the configuration surface for all failover-related settings.
// For the POC only `Enabled` is defined; future fields (e.g. custom lock
// timeout, GMS image override, standby count) will be added here.
type FailoverSpec struct {
    // Enabled activates failover mode. Two engine containers are created
    // in the same pod, sharing GPUs via DRA and coordinating via a
    // file-based lock on a shared emptyDir volume.
    Enabled bool `json:"enabled"`
}
```

### Pod Generation Changes

#### `deploy/operator/internal/dynamo/graph.go` — `GenerateBasePodSpec()`

When `component.Failover != nil && component.Failover.Enabled`:

**1. Clone the main container into two engine containers:**

```
engine-0: same image/command/args, ENGINE_ID=0
engine-1: same image/command/args, ENGINE_ID=1
```

Both get the same resource requests/limits, the same command/args, and the same base env vars. The only differences are:
- `ENGINE_ID` env var (0 vs 1)
- Container name suffix (`-engine-0`, `-engine-1`)

**2. Add GMS weight sidecar (init container with `restartPolicy: Always`):**

```go
corev1.Container{
    Name:          "gms-weights",
    Image:         mainContainer.Image,  // same image
    Command:       []string{"python3", "-m", "gpu_memory_service"},
    Args:          gmsArgsForDevices(gpuCount),
    RestartPolicy: ptr(corev1.ContainerRestartPolicyAlways),
    VolumeMounts: []corev1.VolumeMount{
        {Name: "failover-shared", MountPath: "/shared"},
    },
    StartupProbe: &corev1.Probe{
        ProbeHandler: corev1.ProbeHandler{
            Exec: &corev1.ExecAction{
                Command: []string{"python3", "-c", gmsReadyCheck},
            },
        },
        PeriodSeconds:    2,
        FailureThreshold: 150,  // 5 minutes
    },
}
```

The GMS sidecar uses the same container image as the engine (it has the `gpu_memory_service` package installed). The startup probe checks that all GMS UDS sockets exist on the shared volume. kubelet starts the sidecar first (init container ordering), waits for the startup probe to pass, then starts the engine containers.

**3. Add shared volume:**

```go
corev1.Volume{
    Name: "failover-shared",
    VolumeSource: corev1.VolumeSource{
        EmptyDir: &corev1.EmptyDirVolumeSource{},
    },
}
```

Mounted at `/shared` in all three containers. Holds:
- GMS UDS sockets (`/shared/gms_<gpu_uuid>.sock`)
- Flock file (`/shared/failover.lock`)

**4. Mount shared volume in both engine containers:**

```go
corev1.VolumeMount{
    Name: "failover-shared", MountPath: "/shared",
}
```

**5. Env var injection — identical on both engines except ENGINE_ID and port:**

Both containers share the pod network namespace, so they **cannot both bind to the same port**. The operator assigns staggered system ports:

```
ENGINE_ID=0                                 # or 1
DYN_SYSTEM_PORT=9090                        # engine-0: 9090, engine-1: 9091
GMS_SOCKET_DIR=/shared
FAILOVER_LOCK_PATH=/shared/failover.lock
DYN_SYSTEM_STARTING_HEALTH_STATUS=notready
DYN_SYSTEM_ENABLED=true
```

`DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` is intentionally **omitted** (activates Branch 3 in `SystemHealth`). Both engines race symmetrically through init → sleep → set_health_status(Ready) → acquire lock → wake → serve.

Each engine container declares its own `containerPort` and named port so probes target the correct port per container.

**6. Probes — same readiness probe on both engines:**

Both engine containers get the **same readiness probe** that is normally set on workers (`GET /health` on the system port). This is critical because the discovery mechanism uses the readiness probe to determine which pods can receive traffic.

```yaml
readinessProbe:
  httpGet:
    path: /health
    port: dynamo-system
  periodSeconds: 10
  timeoutSeconds: 4
  failureThreshold: 3
```

Both engines also get the same liveness and startup probes as a normal worker.

Both engines are **identical** — same env vars, same probes, same code path. They race symmetrically:

- `DYN_SYSTEM_STARTING_HEALTH_STATUS=notready` — `/health` returns 503 at boot
- `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` **omitted** — activates Branch 3 in `SystemHealth`

The lifecycle for both: init → sleep → `set_health_status(Ready)` (probe now passes) → block on flock → winner wakes → `serve_endpoint` (Branch 2 takes over seamlessly). The only env var difference between the two containers is `ENGINE_ID` (0 vs 1), which only affects weight loading roles in GMS.

**Known POC limitation:** K8s considers a pod Ready only when **all** containers pass their readiness probe. If the active engine crashes and its container restarts, its readiness probe fails during restart, marking the entire pod not-ready even while the standby engine is actively serving. This is acceptable for the POC since Dynamo's transport layer doesn't rely on K8s Service endpoints for routing (see comment in `component_worker.go`: "worker registration is done through external KvStore and Transport does not use Kubernetes Service"). External consumers that depend on pod readiness (e.g. Kubernetes Service selectors) will see a brief disruption.

**7. GPU sharing via DRA:**

When failover is enabled and the cluster supports DRA, the operator generates a `ResourceClaimTemplate` and references it from all containers:

```go
pod.Spec.ResourceClaims = []corev1.PodResourceClaim{
    {
        Name:                      "shared-gpu",
        ResourceClaimTemplateName: &claimTemplateName,
    },
}
```

Each container references the claim:
```go
container.Resources.Claims = []corev1.ResourceClaim{
    {Name: "shared-gpu"},
}
```

The `ResourceClaimTemplate` requests the appropriate number of GPUs. All containers in the pod share access.

**Fallback for non-DRA clusters:** If DRA is not available (older K8s, no NVIDIA DRA driver), the operator falls back to `nvidia.com/gpu` resource limits on the GMS sidecar and engine containers. This requires the NVIDIA device plugin to be configured for GPU sharing (time-slicing or MPS). For the POC, DRA is the primary path.

### File Layout

```
deploy/operator/
├── api/v1alpha1/
│   ├── common.go                          # + FailoverSpec type
│   └── dynamocomponentdeployment_types.go # + Failover field
├── internal/dynamo/
│   ├── graph.go                           # + failover pod generation logic
│   ├── component_worker.go                # + failover probe/env overrides
│   └── failover.go                        # NEW: failover-specific helpers
├── config/crd/bases/
│   └── nvidia.com_dynamographdeployments.yaml  # regenerated
└── internal/webhook/
    └── dynamocomponentdeployment_validator.go   # + failover validation
```

### New File: `deploy/operator/internal/dynamo/failover.go`

Helper functions isolated from the main graph.go:

```go
package dynamo

// buildFailoverPod transforms a single-container worker pod spec into
// a multi-container failover pod with two engines and a GMS sidecar.
func buildFailoverPod(basePod *corev1.PodSpec, mainContainer corev1.Container, gpuCount int) *corev1.PodSpec

// buildGMSSidecar creates the GMS weight server init container.
func buildGMSSidecar(image string, gpuCount int, sharedMountPath string) corev1.Container

// buildEngineContainer clones the main container with ENGINE_ID and failover env vars.
func buildEngineContainer(base corev1.Container, engineID int, sharedMountPath string) corev1.Container

// gmsReadyCheckScript returns the Python one-liner that checks GMS socket readiness.
func gmsReadyCheckScript(gpuCount int, socketDir string) string
```

### Validation

In the webhook validator (`dynamocomponentdeployment_validator.go`):

- `failover` is only valid on `componentType: worker`
- **`failover` + `multinode` → reject with error.** If both `failover.enabled` and `multinode` are set, the webhook returns an admission error: `"failover is not supported with multinode deployments"`. Multi-node failover requires cross-node lock coordination (TCP lock server) and cross-node DRA, neither of which exist yet.
- `failover` requires `resources.limits.gpu >= 1`

### Discovery and Delivery

**Problem:** Kube-based discovery doesn't work with two engine containers in the same pod. The headless service exposes one endpoint per pod (`pod-ip:9090`) via EndpointSlices. With failover, we have two engines on different ports (9090 and 9091) behind the same pod IP. K8s-based discovery (which keys on the pod and a single well-known port) cannot distinguish active from standby, and the `TargetPort` in the headless service can only point to one port.

**POC approach:** Use **etcd-based discovery** for failover workers. With etcd, each engine registers itself independently when it calls `serve_endpoint`, using its own address (`pod-ip:port`). Only the active engine registers, so the frontend sees exactly one endpoint. On failover, the old registration expires/is cleaned up and the new active engine registers with its address.

The operator should **not set** the kube discovery labels (`dynamo-discovery-backend: kubernetes`, `dynamo-discovery-enabled: true`) on failover pods. This forces etcd-based discovery for these workers. Non-failover workers are unaffected and continue using whichever discovery backend is configured.

**TODO (post-POC):**

1. **Fix kube-based discovery for failover.** Possible approaches: (a) a sidecar proxy that forwards to the currently-active engine port, (b) a controller that updates the Service targetPort on failover, (c) per-container EndpointSlice entries (not natively supported by K8s).
2. **Assess TCP-based delivery.** Verify that the Dynamo TCP transport correctly handles: (a) two engines on the same pod IP with different ports, (b) deregistration of the crashed engine's transport address, (c) re-registration of the new active engine's transport address. Specifically check that the NATS/transport layer doesn't cache stale addresses and that the frontend picks up the new endpoint promptly.

### What the Generated Pod Looks Like

For a single-GPU failover worker:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-failover-worker-xxxxx
spec:
  initContainers:
  - name: gms-weights
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    restartPolicy: Always
    command: ["python3", "-m", "gpu_memory_service"]
    args: ["--socket-dir", "/shared", "--devices", "0"]
    volumeMounts:
    - name: failover-shared
      mountPath: /shared
    startupProbe:
      exec:
        command: ["test", "-S", "/shared/gms_0.sock"]
      periodSeconds: 2
      failureThreshold: 150
    resources:
      claims:
      - name: shared-gpu

  containers:
  # Both engines are identical except ENGINE_ID and system port.
  # DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS is intentionally omitted → Branch 3.
  # Ports are staggered (9090, 9091) because containers share the pod network namespace.
  - name: engine-0
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    command: ["python3", "-m", "dynamo.vllm"]
    args: ["--model", "Qwen/Qwen3-0.6B"]
    ports:
    - name: system-0
      containerPort: 9090
      protocol: TCP
    env:
    - name: ENGINE_ID
      value: "0"
    - name: GMS_SOCKET_DIR
      value: /shared
    - name: FAILOVER_LOCK_PATH
      value: /shared/failover.lock
    - name: DYN_SYSTEM_STARTING_HEALTH_STATUS
      value: notready
    - name: DYN_SYSTEM_ENABLED
      value: "true"
    - name: DYN_SYSTEM_PORT
      value: "9090"
    volumeMounts:
    - name: failover-shared
      mountPath: /shared
    startupProbe:
      httpGet:
        path: /live
        port: system-0
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 720
    livenessProbe:
      httpGet:
        path: /live
        port: system-0
      periodSeconds: 5
      timeoutSeconds: 4
      failureThreshold: 1
    readinessProbe:
      httpGet:
        path: /health
        port: system-0
      periodSeconds: 10
      timeoutSeconds: 4
      failureThreshold: 3
    resources:
      claims:
      - name: shared-gpu

  - name: engine-1
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    command: ["python3", "-m", "dynamo.vllm"]
    args: ["--model", "Qwen/Qwen3-0.6B"]
    ports:
    - name: system-1
      containerPort: 9091
      protocol: TCP
    env:
    - name: ENGINE_ID
      value: "1"
    - name: GMS_SOCKET_DIR
      value: /shared
    - name: FAILOVER_LOCK_PATH
      value: /shared/failover.lock
    - name: DYN_SYSTEM_STARTING_HEALTH_STATUS
      value: notready
    - name: DYN_SYSTEM_ENABLED
      value: "true"
    - name: DYN_SYSTEM_PORT
      value: "9091"
    volumeMounts:
    - name: failover-shared
      mountPath: /shared
    startupProbe:
      httpGet:
        path: /live
        port: system-1
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 720
    livenessProbe:
      httpGet:
        path: /live
        port: system-1
      periodSeconds: 5
      timeoutSeconds: 4
      failureThreshold: 1
    readinessProbe:
      httpGet:
        path: /health
        port: system-1
      periodSeconds: 10
      timeoutSeconds: 4
      failureThreshold: 3
    resources:
      claims:
      - name: shared-gpu

  volumes:
  - name: failover-shared
    emptyDir: {}

  resourceClaims:
  - name: shared-gpu
    resourceClaimTemplateName: vllm-failover-worker-gpu
```

### Dependencies

- **Diff 1** (failover lock): `FAILOVER_LOCK_PATH` env var consumed by engine
- **Diff 2** (deterministic weights): `ENGINE_ID` env var consumed by GMS client
- **Diff 3** (GMS failure handling): `GMS_SOCKET_DIR` for remap timeout
- **Diff 4** (shadow main loop): Engine code reads `ENGINE_ID`, `FAILOVER_LOCK_PATH`, triggers shadow flow
- **Diff 5** (system health + probes): Shadow probe behavior, `DYN_SYSTEM_STARTING_HEALTH_STATUS`

### Tests

1. **Unit: `buildFailoverPod`** — Given a base container spec and GPU count, verify the output pod has 2 engine containers + 1 GMS sidecar, correct env vars, shared volume, resource claims.
2. **Unit: validation webhook** — Verify failover rejected on frontend, rejected with multinode, accepted on worker with GPU.
3. **Integration: DGD reconciliation** — Apply a DGD with `failover.enabled: true`. Verify the generated DCD/Deployment has the expected pod spec.
4. **E2E: deploy and failover** — Apply DGD on a real cluster with DRA. Verify primary serves, kill engine-0 container, verify engine-1 wakes and takes over.

### Not in Scope

- Multi-node failover (DRA sharing across nodes, TCP lock server) — webhook rejects `failover` + `multinode`
- Kube-based discovery for failover workers (POC uses etcd; TODO above)
- TCP delivery verification (TODO above)
- Custom GMS image (uses same image as engine for POC)
- Configurable flock path or GMS socket directory (hardcoded `/shared` for POC)
- Rolling updates with failover (failover pods are restarted, not rolled)
- KV cache GMS sidecar (KV cache is direct allocation, not through GMS)
