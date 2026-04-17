# MM kwargs Transfer — Design & Changes

## Overview

When a multimodal request arrives at the frontend, vLLM's HF processor runs the
vision encoder and produces `mm_kwargs` — a dict of tensors (`pixel_values`,
`image_grid_thw`, etc.) that the backend needs to run prefill.  Without
transfer, the backend must re-run the same expensive vision encoder.

This doc covers:
1. The three transport modes added and their tradeoffs
2. The NIXL sender buffer pool (eliminates per-request registration overhead)
3. Benchmark tooling
4. Files changed

---

## Transfer Modes

Three modes are selectable via `DYNAMO_MM_TRANSFER` (default: `shm`).

### SHM — Shared Memory (default, same-node only)

**How it works:** Frontend pickles each `MultiModalKwargsItem`, writes it to a
named POSIX shared memory segment, and sends the segment name+size in
`extra_args["mm_kwargs_shm"]`. The backend opens the segment by name, reads the
bytes, unpickles, and injects into vLLM's mm processor cache.

**Performance:** ~2ms total for a 3.7 MB image tensor (pickle + memcpy + open).

**Constraint:** Same node only. Cross-node backends fail silently and fall back
to running the HF processor themselves — no crash.

**Control vars:**
```
DYNAMO_MM_TRANSFER=shm   # default
```

---

### NIXL — RDMA via UCX (cross-node)

**How it works:** Frontend registers pickled bytes with NIXL as a readable
buffer and sends the RDMA metadata in `extra_args["mm_kwargs_nixl"]`. The
backend pulls the data via NIXL READ using a pre-allocated descriptor pool.

**Performance (with pool):** ~0–1 ms per-request (memcpy into pre-registered
buffer + metadata re-serialization). The pool amortizes the one-time NIXL
registration cost (~34ms per buffer) over all requests.

**Performance (without pool, original code):** ~34ms per-request because every
request re-registers a new descriptor.

**UCX transport note:** NIXL defaults to the best available transport. On many
nodes without CMA/XPMEM support, UCX falls back to TCP loopback (~12ms) or
InfiniBand HCA (adds ~25ms round-trip). Setting `UCX_MM_ERROR_HANDLING=y` with
`UCX_TLS=shm,self` enables the SysV/POSIX shared-memory AM transport for
same-node transfers, bringing latency close to SHM mode.

**Control vars:**
```
DYNAMO_MM_TRANSFER=nixl
UCX_MM_ERROR_HANDLING=y    # enable SHM AM transport on nodes that need it
UCX_TLS=shm,self           # force SHM transport (same-node only)
```

---

### TCP — Embedded in Request Payload

**How it works:** Frontend pickles each item and base64-encodes it, embedding
the result directly in `extra_args["mm_kwargs_tcp"]["items_b64"]`. No separate
memory segment or RDMA channel. The backend decodes and unpickles inline when
processing the request.

**Performance:** ~12ms total (pickle ~1ms + base64 encode ~3ms + TCP framing +
decode ~3ms + unpickle ~1ms). Overhead scales with payload size.

**Use case:** Fallback for environments where neither SHM nor NIXL is available
(cross-node without IB, heterogeneous clusters, debugging). Up to ~50MB
practical limit before TCP framing becomes a bottleneck.

**Control vars:**
```
DYNAMO_MM_TRANSFER=tcp
```

---

## NIXL Sender Buffer Pool

### Problem

The original NIXL sender called `nixl_connect.create_readable()` per request,
which registers a new descriptor with NIXL each time. Registration cost: ~34ms.
At any non-trivial QPS this dominates the entire MM transfer path.

### Solution

Pre-register a fixed-size pool of CPU buffers at sender startup. Per-request
work is reduced to:
1. `pool.get()` — borrow a pre-registered slot
2. `np.frombuffer` + bulk copy — memcpy pickled bytes into the slot's buffer
3. Update `desc._data_size` and clear cached serialization state
4. `op.metadata()` — re-serialize the RDMA metadata (reflects new payload size)
5. After backend confirms receipt: `pool.put()` — return slot

**Pool defaults:**
- 8 sender slots (covers 8 concurrent in-flight images)
- 8 MB per slot (covers most single-image models; Qwen3-VL 2B ≈ 3.7 MB)

**Key implementation detail:** `ReadableOperation._release()` normally
deregisters the NIXL descriptor when the operation is GC'd. The pool subclasses
it with a no-op `_release()` so the descriptor stays registered across reuse:

```python
class _NonReleasingOp(ReadableOperation):
    def _release(self) -> None:
        pass  # pool owns descriptor lifetime
```

**Slow path (pool exhausted or item > 8 MB):** falls back to dynamic allocation
(original behavior). No dropped requests, just higher latency for those items.

---

## Performance Summary

Measured on a single-node setup (Qwen3-VL-2B-Instruct, 3.7 MB image tensor):

| Mode | Sender prepare | Receiver total | E2E (sender + recv) | Notes |
|---|---|---|---|---|
| SHM | ~1.5ms | ~0.5ms | ~2ms | Same-node only |
| NIXL (pool, SHM transport) | ~0.4ms | ~0.8ms | ~1.2ms | `UCX_MM_ERROR_HANDLING=y UCX_TLS=shm,self` |
| NIXL (pool, TCP transport) | ~0.4ms | ~12ms | ~12ms | Default on nodes without CMA |
| TCP | ~4ms | ~8ms | ~12ms | Works anywhere |

---

## Files Changed

### `components/src/dynamo/common/multimodal/mm_kwargs_transfer.py`

- **Added `MmKwargsTcpSender`**: pickles + base64-encodes each item into
  `extra_args["mm_kwargs_tcp"]["items_b64"]`; no cleanup needed (data in
  request payload); `[TIMING][TCP-Sender]` prints per item and total.
- **Rewrote `MmKwargsNixlSender`** with sender buffer pool:
  - Pool of pre-registered `(buf, desc, op, cap)` tuples allocated at init
  - `_make_slot()`: allocates buffer, registers descriptor once, wraps in
    `_NonReleasingOp`
  - `_slot_load()`: memcpy + reset `desc._data_size`, `desc._serialized`,
    `op._serialized_request`, `op._status` for a clean re-arm
  - `_slot_restore()`: resets `desc._data_size` to full capacity before pool
    return
  - `prepare()`: fast path (pool hit) vs slow path (pool exhausted / oversized)
  - `[TIMING][NIXL-Sender]` prints per item and total
- NVTX annotations on all hot paths for Nsight profiling.

### `components/src/dynamo/frontend/vllm_processor.py`

- Added `MmKwargsTcpSender` to imports.
- Stored `self._transfer_mode = transfer_mode` from `DYNAMO_MM_TRANSFER` env var.
- Updated lazy sender instantiation in `_prepare_mm_routing()`:
  ```python
  if self._transfer_mode == "tcp":
      self._sender = MmKwargsTcpSender()
  elif self.use_shm_transfer:
      self._sender = MmKwargsShmSender()
  else:
      self._sender = MmKwargsNixlSender()
  ```

### `components/src/dynamo/vllm/handlers.py`

- Added TCP check at the top of `_try_receive_mm_kwargs_nixl()` (before SHM
  and NIXL checks):
  ```python
  tcp_meta = extra_args.get("mm_kwargs_tcp")
  if tcp_meta:
      return await self._receive_mm_kwargs_tcp(request, tcp_meta)
  ```
- Added `_receive_mm_kwargs_tcp()`: base64-decode → unpickle → validate
  `MultiModalKwargsItem` → build engine input → inject into mm processor cache.
  `[TIMING][TCP-Receiver]` prints per item (decode + unpickle ms) and total.

### `examples/backends/vllm/launch/bench_mm_transfer.sh` (new)

Benchmark script that wraps `agg_multimodal_router.sh`, sends N image requests
across selected modes (shm / nixl / tcp), and reports per-stage timing from
`[TIMING]` log lines.

Features:
- Mode selection via CLI args (`bash bench_mm_transfer.sh shm nixl tcp`)
- Configurable via env vars: `MODEL`, `NUM_REQUESTS`, `IMAGE_URL`,
  `GPU_MEMORY_UTILIZATION`, `NUM_WORKERS`, `SINGLE_GPU`
- UCX override passthrough for NIXL mode (`UCX_TLS`, `UCX_MM_ERROR_HANDLING`)
- Per-stage averages (pickle, encode, create_write, register, begin_read, wait,
  open_read, decode, unpickle, gather)
- Sender prepare total, receiver total, and e2e totals per mode
- Side-by-side comparison table when multiple modes run

Usage:
```bash
# Both modes (default)
SINGLE_GPU=true bash bench_mm_transfer.sh

# NIXL with SHM transport fix
SINGLE_GPU=true UCX_MM_ERROR_HANDLING=y UCX_TLS=shm,self bash bench_mm_transfer.sh nixl

# All three modes, 10 requests each
SINGLE_GPU=true NUM_REQUESTS=10 bash bench_mm_transfer.sh shm nixl tcp
```

### `examples/backends/vllm/launch/check_ucx_transport.sh` (new)

Diagnostic script that probes which UCX transport NIXL selects for same-node
CPU-to-CPU transfers.  Runs a minimal NIXL initialization with
`UCX_LOG_LEVEL=info`, extracts the `self cfg` line, and classifies the
transport (IB/CMA/SysV/POSIX/TCP) with performance interpretation and remediation
advice.

Usage:
```bash
bash check_ucx_transport.sh
UCX_MM_ERROR_HANDLING=y bash check_ucx_transport.sh
UCX_TLS=cma,self bash check_ucx_transport.sh
```

---

## Env Var Reference

| Variable | Values | Effect |
|---|---|---|
| `DYNAMO_MM_TRANSFER` | `shm` (default), `nixl`, `tcp` | Select transfer mode |
| `DYNAMO_DISABLE_NIXL_MM` | `1` | Disable all mm_kwargs transfer; backend re-runs vision encoder |
| `UCX_MM_ERROR_HANDLING` | `y` | Enable peer failure handler; allows NIXL to use SysV/POSIX SHM AM transport |
| `UCX_TLS` | `shm,self` / `cma,self` / etc. | Override UCX transport selection for NIXL |

---

## Graceful Degradation

All transfer paths fail gracefully:
- SHM: cross-node open failure → backend runs HF processor (no crash)
- NIXL: any receive error → backend runs HF processor (no crash)
- TCP: any decode/unpickle error → backend runs HF processor (no crash)
- Pool exhausted: NIXL falls back to dynamic allocation for that item

The backend's `_try_receive_mm_kwargs_nixl()` returns `None` on any failure;
the caller treats `None` as "no pre-rendered input" and proceeds normally.
