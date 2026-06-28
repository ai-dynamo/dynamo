# GMS Multiple Device Enablement — Design Proposal

## 1. Overview

GPU Memory Service (GMS) manages cross-process virtual memory on accelerators for weight /KV-cache sharing in Dynamo inference clusters.

This design introduces a **device-agnostic VMM abstraction layer** so that Intel XPU can plug in alongside the existing CUDA path without touching the server, client, or snapshot logic.

---

## 2. Goals

| # | Goal | Status |
|---|------|--------|
| G1 | Define a vendor-neutral `VMMDevice` ABC covering all device operations GMS needs | ✅ Phase 1 |
| G2 | Wrap existing CUDA helpers in a `CudaVMM` class that inherits the ABC | ✅ Phase 1 |
| G3 | Process-global VMM singleton via `init_vmm()` / `get_vmm()`, CLI `--device-type` | ✅ Phase 1 |
| G4 | Implement `XpuVMM` | ⬜ Phase 2 |
| G5 | Add XPU torch mempool dispatch via `torch.xpu` APIs | ⬜ Phase 2 |
| G6 | Enable snapshot save/load for XPU | ⬜ Phase 2 |

---

## 3. Architecture (Phase 1 — Delivered)

```
┌──────────────────────────────────────────────────────┐
│                  CLI / Supervisor                     │
│  server.py  ──→  --device-type {cuda,xpu}            │
└────────────────────────┬─────────────────────────────┘
                         │ VMMDeviceType enum
                         ▼
┌──────────────────────────────────────────────────────┐
│            common/vmm/__init__.py                     │
│  init_vmm(device_type) — once per process            │
│  get_vmm() → VMMDevice  (singleton)                  │
│                                                      │
│  ┌───────────────┐          ┌───────────────┐        │
│  │  CudaVMM      │          │  XpuVMM (TBD) │        │
│  │  (cuda_utils) │          │  (sycl_utils) │        │
│  └───────────────┘          └───────────────┘        │
└──────────────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   GMSRPCServer       GMS          GMSAllocationManager
   (server/rpc.py)  (server/gms.py) (server/allocations.py)
          │
          ▼
   GMSClientMemoryManager  →  GMSStorageClient (snapshot)
   (client/memory_manager.py)  (snapshot/storage_client.py)
```

### 3.1 New Module: `common/vmm/`

| File | Purpose |
|------|---------|
| `__init__.py` | `VMMDeviceType` enum, `init_vmm()` singleton initializer, `get_vmm()` accessor, `get_vmm_device_type()` |
| `device.py` | `VMMDevice` — `abc.ABC` with 24 `@abstractmethod` vendor-neutral methods |
| `cuda_utils.py` | Module-level CUDA helpers (unchanged logic) + `CudaVMM(VMMDevice)` class |

### 3.2 `VMMDevice` ABC Surface (24 methods)

```
Category                  Method
───────────────────────── ─────────────────────────────────────
Driver lifecycle          ensure_initialized()
                          synchronize()
Discovery / sizing        list_devices() → list[int]
                          get_allocation_granularity(device) → int
Physical memory           create_tolerate_oom(size, device) → (bool, int)
                          release(handle)
Shareable handles         export_to_shareable_handle(handle) → int (FD)
                          import_shareable_handle_close_fd(fd) → int
VA space + mapping        address_reserve(size, granularity) → int
                          address_free(va, size)
                          map(va, size, handle)
                          unmap(va, size)
                          set_access(va, size, device, access)
Monitoring / sizing       device_memory_info(device) → (free_bytes, total_bytes)
Pointer validation        validate_pointer(va)
Runtime helpers           runtime_check_result(result, name)
                          runtime_set_device(device)
                          host_register(ptr, size)
                          host_unregister(ptr)
Stream management         stream_create_nonblocking() → opaque
                          stream_destroy(stream)
                          stream_synchronize(stream)
Async copy                memcpy_h2d_async(dst, src, size, stream)
                          memcpy_d2h_async(dst, src, size, stream)
```

### 3.3 Singleton Lifecycle

```
Process startup (cli/server.py, cli/runner.py, snapshot loader/saver):
    init_vmm(VMMDeviceType.from_str(args.device_type))

Any module needing VMM:
    vmm = get_vmm()          # returns cached singleton
    vmm.runtime_set_device(device)
    vmm.list_devices()
    ...
```

The singleton is immutable after initialization. Conflicting re-initialization
raises `RuntimeError`. Thread-safe via `threading.Lock`.

### 3.4 CLI Argument Propagation

```
gms-server (supervisor)        --device-type cuda|xpu
  └─→ init_vmm(device_type)
  └─→ gms-server (per-device)  --device-type  (forwarded to child process)
       └─→ init_vmm(device_type)
       └─→ get_vmm() used throughout server, allocations, client
```

No constructor threading — every module calls `get_vmm()` directly.

## 4. Files Changed (Phase 1)

| File | Change |
|------|--------|
| `cli/server.py` | Parse `--device-type`, call `init_vmm()`, use `get_vmm().list_devices()` |
| `cli/runner.py` | Call `init_vmm(config.device_type)` at startup |
| `cli/snapshot/loader.py` | Call `init_vmm(device_type)`, `vmm = get_vmm()` in helpers |
| `cli/snapshot/saver.py` | Call `init_vmm(device_type)`, `vmm = get_vmm()` in helpers |
| `client/memory_manager.py` | `self._vmm = get_vmm()`, property `device_type` via `get_vmm_device_type()` |
| `client/torch/allocator.py` | CUDA-only guards via `get_vmm_device_type()` |
| `common/vmm/__init__.py` | **NEW** — enum + singleton |
| `common/vmm/device.py` | **NEW** — VMMDevice ABC |
| `common/vmm/cuda_utils.py` | **NEW** (moved from `common/cuda_utils.py`) — CUDA helpers + CudaVMM class |
| `common/utils.py` | Add `align_to_granularity()` utility |
| `server/allocations.py` | `self._vmm = get_vmm()` |
| `server/gms.py` | No longer forwards device params — singleton |
| `server/rpc.py` | No longer forwards device params — singleton |
| `snapshot/storage_client.py` | No longer accepts device_type — uses singleton |
| `snapshot/disk.py` | Fix import path: `common.vmm` |
| `snapshot/backends/nixl_staging.py` | Fix import path: `common.vmm` |
| `snapshot/backends/pinned_host.py` | Fix import path: `common.vmm` |
| `tests/report_pytest_markers.py` | Update stub module list |

---

## 5. Phase 2 — XPU Implementation

### 5.1 `XpuVMM(VMMDevice)`

Implement the methods on the oneAPI SYCL runtime.

### 5.2 Torch Allocator Dispatch

The PyTorch front door (`client/torch/`) routes torch tensor allocations through GMS via a pluggable allocator inside a `gms_use_mem_pool(tag)` context; the dispatch key is the existing `get_vmm_device_type()`.

- **`extensions/allocator.cpp` + `setup.py` — no change.**  `my_malloc(ssize_t,  int, void* stream)` forwards the opaque stream/queue to Python *without dereferencing it*, so it is ABI-compatible with `XPUPluggableAllocator`'s
  `void* alloc_fn(size_t, int, sycl::queue*)`. The **same** `_allocator_ext.so`
  and `"my_malloc"` / `"my_free"` symbols serve both backends.
- **`client/torch/allocator.py` — only change.** Swap the four CUDA-hardcoded
  spots for device dispatch (`torch.cuda.*`, `torch.xpu.*`):

  | Spot | CUDA | XPU |
  |------|------|-----|
  | `_ensure_callbacks_initialized` | `torch.cuda.CUDAPluggableAllocator` | `torch.xpu.memory.XPUPluggableAllocator` |
  | `_create_mem_pool` | `torch.cuda.memory.MemPool` | `torch.xpu.memory.MemPool` |
  | `gms_use_mem_pool` | `torch.cuda.use_mem_pool` | `torch.xpu.memory.use_mem_pool` |
  | `prune_allocations` | `torch.cuda.synchronize` | `torch.xpu.synchronize` |

  The torch APIs are symmetric, so a small accessor returning the active device's
  `(PluggableAllocator, MemPool, use_mem_pool, synchronize)` is sufficient.

---

## 6. Backward Compatibility

CUDA path remains unchanged. All existing `--device-type cuda` deployments work identically. The default value is `cuda` everywhere.
