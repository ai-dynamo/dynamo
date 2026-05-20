# GMS Multiple Device Enablement — Design Proposal

## 1. Overview

GPU Memory Service (GMS) manages cross-process virtual memory on accelerators
for weight/KV-cache sharing in Dynamo inference clusters.

This design introduces a **device-agnostic VMM abstraction layer** so that
Intel XPU can plug in alongside the existing CUDA path without
touching the server, client, or snapshot logic.

---

## 2. Goals

| # | Goal | Status |
|---|------|--------|
| G1 | Define a vendor-neutral `VMMDevice` Protocol covering all device operations GMS needs | ✅ Phase 1 |
| G2 | Wrap existing CUDA helpers in a `CudaVMM` class that satisfies the Protocol | ✅ Phase 1 |
| G3 | Thread `--device-kind` through CLI → server → client with a factory function | ✅ Phase 1 |
| G4 | Implement `XpuVMM` with Level-zero | ⬜ Phase 2 |
| G5 | Add XPU torch mempool dispatch via `torch.xpu` APIs | ⬜ Phase 2 |
| G6 | Enable snapshot save/load for XPU | ⬜ Phase 2 |

---

## 3. Architecture (Phase 1 — Delivered)

```
┌──────────────────────────────────────────────────────┐
│                  CLI / Supervisor                     │
│  server.py  ──→  --device-kind {cuda,xpu}            │
└────────────────────────┬─────────────────────────────┘
                         │ VMMDeviceType enum
                         ▼
┌──────────────────────────────────────────────────────┐
│            common/vmm/__init__.py                     │
│  get_vmm_device(device_kind) → VMMDevice             │
│                                                      │
│  ┌───────────────┐          ┌───────────────┐        │
│  │  CudaVMM      │          │  XpuVMM (TBD) │        │
│  │  (cuda_utils) │          │  (l0_utils)   │        │
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
| `__init__.py` | `VMMDeviceType` enum (`cuda` / `xpu`), `get_vmm_device()` factory |
| `device.py` | `VMMDevice` — `typing.Protocol` with 24 vendor-neutral methods |
| `cuda_utils.py` | Module-level CUDA helpers (unchanged logic) + `CudaVMM` class |

### 3.2 `VMMDevice` Protocol Surface (24 methods)

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
Monitoring / sizing      device_memory_info(device) → (free_bytes, total_bytes)
                          # Unified for CUDA (NVML) and XPU

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
### 3.3 CLI Argument Propagation

```
gms-server (supervisor)        --device-kind cuda|xpu
  └─→ gms-server (per-device)  --device-kind  (forwarded)
       └─→ Config.device_kind  ──→ GMSRPCServer ──→ GMS ──→ GMSAllocationManager
```

Client-side:
```
GMSClientMemoryManager(socket, device, tag, device_kind=...)
GMSStorageClient(..., device_kind=...)
```

## 4. Files Changed (Phase 1)

| File | Change |
|------|--------|
| `cli/args.py` | Add `--device-kind` argument, `Config.device_kind` field |
| `cli/runner.py` | Pass `device_kind` to `GMSRPCServer` constructor |
| `cli/server.py` | Parse `--device-kind`, use `vmm.list_devices()` instead of CUDA import |
| `cli/snapshot/loader.py` | Accept + forward `device_kind` |
| `cli/snapshot/saver.py` | Accept + forward `device_kind` |
| `client/memory_manager.py` | Store `_device_kind`, expose `device_kind` property |
| `client/torch/allocator.py` | CUDA-only guards, forward `device_kind` to manager/pool |
| `common/vmm/__init__.py` | **NEW** — enum + factory |
| `common/vmm/device.py` | **NEW** — VMMDevice Protocol |
| `common/vmm/cuda_utils.py` | **NEW** (moved from `common/cuda_utils.py`) — CUDA helpers + CudaVMM class |
| `common/utils.py` | Add `align_to_granularity()` utility |
| `server/allocations.py` | Accept `device_kind`, instantiate `_vmm` |
| `server/gms.py` | Forward `device_kind` to allocations |
| `server/rpc.py` | Forward `device_kind` to GMS |
| `snapshot/storage_client.py` | Accept + forward `device_kind` to `GMSClientMemoryManager` |
| `snapshot/disk.py` | Fix import path: `common.vmm` |
| `snapshot/backends/nixl_staging.py` | Fix import path: `common.vmm` |
| `snapshot/backends/pinned_host.py` | Fix import path: `common.vmm` |
| `tests/report_pytest_markers.py` | Update stub module list |

---

## 5. Phase 2 — XPU Implementation
Depend on Phase 1 completion.

---

## 6. Backward Compatibility

CUDA path should remain unchanged. All existing `--device-kind cuda` deployments should continue to work identically.
