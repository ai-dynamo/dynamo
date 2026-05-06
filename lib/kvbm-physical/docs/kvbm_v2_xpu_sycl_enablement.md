# KVBM v2 — XPU/SYCL enablement

How Intel XPU (SYCL/oneAPI) was integrated into KVBM v2 alongside the
existing NVIDIA CUDA backend: the trait surfaces that were extracted, the
SYCL implementations that were added, and the crate-level wiring that keeps
KVBM v2 engine-agnostic and framework-agnostic.

This document covers the state of the branch, the evolution from the
CUDA-only baseline, and the relationships between the eight crates under
`lib/` that make up KVBM v2.

## KVBM v2 modules and scope

KVBM v2 is a set of eight Rust crates under `lib/` that together form an
**engine-agnostic, framework-agnostic KV-cache block manager**. The goal is
for any inference framework (vLLM, SGLang, TensorRT-LLM, …) to plug KVBM v2
in under its own runtime, with no dependency flowing from KVBM into a
specific framework.

**Ownership per crate:**

| Crate | Responsibility | XPU/SYCL in scope? |
|---|---|---|
| `kvbm-common` | Shared primitives: `BlockId`, `SequenceHash`, `LogicalLayoutHandle` (G1/G2/G3/G4 tier enum). No device deps. | No — device-neutral by design. |
| `kvbm-config` | Static configuration for caches, discovery, NIXL, object storage, offload, onboard policies, tokio/rayon runtimes, messengers. Pure config structs. | No. |
| `kvbm-logical` | Logical block lifecycle: registries, pools, sequence tracking, metrics, TinyLFU cache, pub/sub adapters, framework integrations. Works on logical handles; never touches devices directly. | No. |
| `dynamo-memory` (`lib/memory`) | `MemoryDescriptor`, `DeviceAllocator` / `PinnedAllocator` traits, `DeviceStorage` / `PinnedStorage`, `NumaWorkerPool`, `SyclMemPool`, `CudaMemPool`. Backend-agnostic storage layer — no device-SDK types in the public API. | **Yes** — houses `SyclMemPool`, SYCL-aware NUMA discovery, and the allocator traits that downstream device wrappers implement. |
| `kvbm-kernels` | CUDA and SYCL kernel launchers: `vectorized_copy`, `memcpy_batch`, `sycl_vectorized_copy`, `sycl_*_from_block`. Built from `.cu` via nvcc and `.cpp` via `icpx -fsycl`. | **Yes** — SYCL kernel sources and FFI wrappers live here. |
| `kvbm-physical` | Device abstraction traits (`DeviceContextOps`, `DeviceStreamOps`, `DeviceEventOps`, `DeviceMemPoolOps`), the `DeviceBackend` enum, CUDA and SYCL implementations, and `TransferManager` / `TransferContext` on top of them. | **Yes** — the core of the multi-backend layer; CUDA and SYCL both implement the trait surface here. |
| `kvbm-engine` | Orchestrator: `InstanceLeader`, `PhysicalWorker`, `ReplicatedDataWorker`, sessions, offload pipeline, object tier (G4), `CollectiveOps` with NCCL and **oneCCL** implementations, `OneCclBootstrap`. | **Yes** — adds the `oneccl` feature and the XPU-aware layer-wise onboard path. |
| `kvbm-py3` (`lib/bindings/kvbm`) | Python/FFI bindings consumed by external frameworks. Owns the vLLM connector and its backend-specific `event_sync_blocking` (CUDA / SYCL / fallback). | **Yes** — SYCL variant added alongside CUDA, mutually exclusive by feature. |

The small crates (`kvbm-common`, `kvbm-config`, `kvbm-logical`) are listed
for completeness but contain no device-specific code and are not touched by
XPU/SYCL enablement. The diagrams in this document draw them as
inert upper layers so the reader can see where KVBM v2 sits as a whole.

## KVBM v2 crate stack

This diagram shows the full KVBM v2 module stack under `lib/`, the
dependency direction between crates, and the boundaries that external
frameworks (vLLM, SGLang, …) integrate across. The device-specific
implementations (CUDA / SYCL) sit at the bottom; logical/config/common
crates at the top are framework-agnostic and have no device deps.

```mermaid
graph TB
    subgraph ext["External frameworks (not in this repo)"]
        VLLM["vLLM<br/>(via kvbm-py3 connector)"]
        SGL["SGLang / TRT-LLM / …<br/>(future integrations)"]
    end

    subgraph v2["KVBM v2 (lib/)"]
        subgraph topLayer["Framework-agnostic layer"]
            Common["kvbm-common<br/>BlockId, SequenceHash<br/>LogicalLayoutHandle (G1..G4)"]
            Config["kvbm-config<br/>cache / offload / onboard<br/>discovery / messenger"]
            Logical["kvbm-logical<br/>registries, pools, sequences<br/>TinyLFU, metrics, pubsub"]
        end

        subgraph midLayer["Orchestrator layer"]
            Engine["kvbm-engine<br/>InstanceLeader, PhysicalWorker<br/>sessions, offload, G4 object tier<br/>CollectiveOps: NCCL + oneCCL"]
        end

        subgraph physLayer["Device abstraction layer"]
            Physical["kvbm-physical<br/>DeviceBackend enum<br/>Device{Context,Stream,Event,MemPool}Ops<br/>TransferManager, TransferContext"]
            Kernels["kvbm-kernels<br/>CUDA: vectorized_copy, memcpy_batch<br/>SYCL: sycl_vectorized_copy, sycl_*_from_block"]
        end

        subgraph memLayer["Storage layer"]
            Memory["dynamo-memory<br/>MemoryDescriptor<br/>DeviceAllocator / PinnedAllocator traits<br/>DeviceStorage / PinnedStorage<br/>NumaWorkerPool<br/>SyclMemPool / CudaMemPool"]
        end

        subgraph bindLayer["Bindings layer"]
            PyBind["kvbm-py3 (lib/bindings/kvbm)<br/>Python extension module<br/>vLLM connector, event_sync_blocking"]
        end
    end

    subgraph hw["Device runtimes"]
        CUDA["CUDA driver + cudarc"]
        SYCL["SYCL / oneAPI (icpx)<br/>oneapi-rs bindings"]
        NCCL["NCCL"]
        OneCCL["oneCCL"]
    end

    %% External → bindings
    VLLM -- "Python FFI" --> PyBind
    SGL -. "future" .-> PyBind

    %% Bindings → engine (the glue layer that orchestrates everything)
    PyBind --> Engine

    %% Top layer dependencies
    Logical --> Common
    Logical --> Config
    Config --> Common

    %% Engine depends on logical, physical, and top layer
    Engine --> Logical
    Engine --> Physical
    Engine --> Common
    Engine --> Config

    %% Physical layer
    Physical --> Memory
    Physical --> Common
    Physical --> Kernels
    Kernels -. optional .-> CUDA
    Kernels -. optional .-> SYCL

    %% Memory layer
    Memory -. optional (sycl feature) .-> SYCL
    Memory --> CUDA

    %% Device backends
    Physical -. cuda feature .-> CUDA
    Physical -. xpu-sycl feature .-> SYCL

    %% Collectives
    Engine -. nccl feature .-> NCCL
    Engine -. oneccl feature .-> OneCCL

    %% Styling
    classDef agnostic fill:#eceff1,stroke:#455a64,color:#000,stroke-width:2px
    classDef orchestrator fill:#9c27b0,stroke:#6a1b9a,color:#fff,stroke-width:2px
    classDef physical fill:#ff6b35,stroke:#c44520,color:#fff,stroke-width:2px
    classDef kernels fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
    classDef memory fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:2px
    classDef bindings fill:#9e9e9e,stroke:#616161,color:#fff,stroke-width:2px
    classDef external fill:#fff9c4,stroke:#f57f17,color:#000,stroke-width:2px
    classDef hw fill:#cfd8dc,stroke:#37474f,color:#000,stroke-width:2px

    class Common,Config,Logical agnostic
    class Engine orchestrator
    class Physical physical
    class Kernels kernels
    class Memory memory
    class PyBind bindings
    class VLLM,SGL external
    class CUDA,SYCL,NCCL,OneCCL hw
```

**Key takeaways for framework integration:**

- **No upward dep from KVBM into a framework.** `kvbm-common`, `kvbm-config`,
  and `kvbm-logical` have no knowledge of vLLM, PyTorch, or any inference
  runtime. That's what makes v2 reusable across frameworks.
- **One integration point: `kvbm-py3`.** External frameworks import the
  Python extension module built from `lib/bindings/kvbm`, which internally
  talks to `kvbm-engine`. Adding SGLang or TensorRT-LLM would mean a new
  connector in the bindings crate (or a sibling Python binding) without
  changes to the lower layers.
- **Device/runtime is a compile-time feature, not a crate.** CUDA vs. XPU is
  chosen via Cargo features (`cuda` / `xpu-sycl` on `kvbm-physical` and
  `dynamo-memory`; `nccl` / `oneccl` on `kvbm-engine`). `kvbm-py3` enforces
  that `cuda` and `xpu-sycl` are mutually exclusive.
- **Kernels are optional and FFI-gated.** `kvbm-kernels` produces
  `libkvbm_kernels.so` (nvcc) and `libkvbm_kernels_sycl.so` (icpx); both are
  loaded only when the corresponding backend feature is enabled. SYCL
  kernels are built automatically when the `xpu-sycl` Cargo feature is
  enabled (icpx must be on PATH or `ONEAPI_ROOT` set).

The remaining diagrams and tables zoom in on the **device abstraction** and
**memory layer** inside this stack — the two layers that XPU/SYCL enablement
actually touches.

## Evolution: from CUDA-only to CUDA + XPU/SYCL

Before this work, KVBM v2 assumed CUDA everywhere: device types, allocation,
pools, collectives, pinned memory, and even test helpers called CUDA APIs
directly. Enabling Intel XPU required extracting a small backend-agnostic
trait surface at each boundary, then adding SYCL/oneAPI implementations
behind it. The table below summarizes the shape of each boundary before and
after XPU/SYCL enablement.

| Area | Before (CUDA-only) | After (CUDA + XPU/SYCL) |
|---|---|---|
| **Backend selector** | Implicit — everything was CUDA. | `DeviceBackend::{Cuda, Sycl}` enum in `kvbm-physical`, with runtime `is_available()` probes guarded by `catch_unwind` so a missing `libcuda.so` / `libsycl.so` doesn't abort the process. `DeviceContext::new(backend, id)` dispatches to `CudaContext` or `SyclContext` behind `#[cfg(feature = "cuda")]` / `#[cfg(feature = "xpu-sycl")]`. |
| **Device context / stream / event** | Direct use of `cudarc::driver::{CudaContext, CudaStream, CudaEvent}`. | New traits `DeviceContextOps`, `DeviceStreamOps`, `DeviceEventOps`, `DeviceMemPoolOps` in `lib/kvbm-physical/src/device/traits.rs`. `CudaContext` and `SyclContext` both implement them; the transfer executor, notification loop, and pool wrappers talk to trait objects only. |
| **Copy API on streams** | Direction-named CUDA calls (`cudaMemcpyAsync` with explicit H2D/D2H kinds). | Pattern-based primitives on `DeviceStreamOps`: `batch_copy` (N DMAs, direction auto-detected), `memcpy_htod` / `memcpy_dtoh` (scalar uploads/downloads), `vectorized_copy` (kernel over pointer arrays). The executor picks between them based on op type, not direction. |
| **`TransferStrategy` enum** | `CudaAsyncH2D`, `CudaAsyncD2H`, `CudaAsyncD2D`; `panic!` on `System ↔ Device`. | Two changes — (1) **rename** `CudaAsync*` → `Async*` since the backend-agnostic executor dispatches to either `CudaStreamWrapper` or `SyclStreamWrapper`; (2) **add** `BlockingH2D`, `BlockingD2H` that replace the upstream panic with an async copy + inline `device_stream.synchronize()`. Blocking variants apply on **both** backends because the motivating case (unpinned `System` memory degrading async copies to staged blocking behavior) affects CUDA and SYCL identically. See [`device_executor_flow.md`](./device_executor_flow.md#transferstrategy-vs-upstream--rename-and-additions) for the full before/after mapping. |
| **Memory pool** | Native CUDA pool (`cuMemPoolCreate` / `cuMemAllocFromPoolAsync`). | Two pool implementations behind `DeviceMemPoolOps`: `CudaMemPool` (native API, stream-ordered) and `SyclMemPool` (software free-list over `sycl::malloc_device`, SYCL has no native pool API). `SyclMemPool`'s `PoolInner.active_allocs: HashMap<ptr, real_size>` keeps `cached_bytes` and `release_threshold` accurate when a best-fit returns a block larger than requested. |
| **Device storage** | `DeviceStorage::new(size, device_id)` hard-coded to `cudarc::driver::result::malloc_sync`. | `DeviceStorage::new(size, Arc<dyn DeviceAllocator>)`. The new `DeviceAllocator` trait lives in `dynamo-memory` with *no* CUDA types, and `DeviceContext` in `kvbm-physical` bridges it to the selected backend. The same pattern rewired `PinnedStorage`. |
| **Pinned host allocation** | `PinnedStorage` called `cuMemHostAlloc` inline. | A `PinnedAllocator` trait in `dynamo-memory` is called *on a NUMA-pinned worker thread* managed by `NumaWorkerPool` (one thread per NUMA node, global singleton). `CudaPinnedAllocator` / `SyclPinnedAllocator` in `kvbm-physical` provide the backend implementations; the worker first-touches pages after allocation so Linux binds them to the correct node. End-to-end validation via `move_pages(2)` is available for either backend through the `validate_numa_placement` binary — see [`sycl_pool_and_numa.md`](../../memory/docs/sycl_pool_and_numa.md#validating-first-touch-on-real-hardware) for the correct invocation on CUDA vs. XPU hosts (XPU requires the `xpu-sycl` feature at build time). |
| **NUMA topology / PCI discovery** | CUDA-specific: PCI BDF pulled from `cuDeviceGet*` attributes. | Backend-agnostic. `DeviceContextOps::pci_bdf_address()` returns `"DDDD:BB:DD.F"` — CUDA queries `CU_DEVICE_ATTRIBUTE_PCI_*`; SYCL reads `SyclDevice::info()?.pci_address`. `dynamo_memory::numa` resolves NUMA nodes via sysfs (with `nvidia-smi` / `xpu-smi` fallbacks) and subdivides CPU sets fairly across *all* GPUs of the same vendor, using NVML for CUDA and a sysfs PCI scan for Intel (class `0x03xxxx`, vendor `0x8086`). |
| **Event reuse** | Single-shot events — `CudaEvent::record(&stream)`. | New `DeviceEventOps::record_on_stream(stream_handle)` lets the same `DeviceEvent` be re-recorded on a later op. Required by `PhysicalWorker::execute_local_layerwise_onboard` in `kvbm-engine`, which records one event per layer on a shared H2D stream. |
| **Kernels** | CUDA `.cu` sources compiled by nvcc in `kvbm-kernels`: `vectorized_copy`, `memcpy_batch`, optional permute kernels. | Same CUDA kernels, *plus* SYCL `.cpp` sources (`sycl/vectorized_copy_kernel.cpp`, `sycl/tensor_permute_kernel.cpp`) compiled by `icpx -fsycl` into `libkvbm_kernels_sycl.so` when `xpu-sycl` is enabled. FFI wrappers in `src/tensor_kernels_sycl.rs` expose them as `sycl_vectorized_copy`, `sycl_universal_from_block`, `sycl_block_from_universal`. SYCL kernels pass a `sycl::queue*` instead of a `cudaStream_t` and use byte-size element dispatch rather than dtype templates. |
| **Collectives (`kvbm-engine`)** | NCCL only (`feature = "nccl"`, `cudarc::nccl`). | `CollectiveOps` trait is now implemented by `NcclCollectives` **and** `OneCclCollectives` (`feature = "oneccl"`, `oneapi-rs::ccl`). oneCCL supports both a from-scratch bootstrap (`OneCclBootstrap` — 8 B world_size + 256 B KVS address rendezvous) and borrowed handles from PyTorch / vLLM. Broadcasts use `ccl_rs_group_start/end` with a single `event_wait` on the last submitted op. |
| **vLLM connector event sync (`kvbm-py3`)** | `event_sync_blocking(u64)` called `cuEventSynchronize` and asserted on failure. | Three cfg-gated implementations of the same `pub fn event_sync_blocking(u64) -> anyhow::Result<()>`: CUDA (`cuEventSynchronize`), SYCL (`oneapi_rs::sys::sycl_rs_event_wait`), and a "no backend" stub that `bail!`s. The call site now `?`-propagates the error instead of swallowing it. |
| **Test helpers (`kvbm-physical`)** | Direct `cudaMemcpy` with explicit D2H/H2D kinds in `fill.rs` / `checksum.rs`. | Backend-agnostic `sync_memcpy_dtoh` / `sync_memcpy_htod` helpers on `DeviceContext` that construct a throwaway stream and synchronize. Tests pick a backend via `test_device_backend()` which prefers SYCL when compiled in and available, falling back to CUDA. |
| **Benchmark tooling** | `bench_engine` and `bench_transfer` hard-coded `cuda_device_id`. | Both accept `--backend {auto,cuda,sycl}` and plumb the selected backend through `TransferManager::builder().device_backend(..).device_id(..)`. `bench_engine` resolves NUMA affinity via PCI BDF for both backends using the shared `get_device_cpu_set(backend_kind, bdf)` API. |

The rest of the document shows the resulting trait surface, the full
cross-crate graph, and the memory layer in detail.

## System overview — all crates and their abstractions

The first diagram is the 10 000-foot view: which crate owns which abstraction
and how components connect end-to-end. Each color is a single *layer of
ownership*:

- 🟩 **Green** — `dynamo-memory`: storage + allocator traits, NUMA worker pool, SYCL pool implementation.
- 🟧 **Orange** — `kvbm-physical`: device abstraction traits + CUDA / SYCL wrappers.
- 🟦 **Blue** — `kvbm-kernels`: GPU kernel launchers (CUDA + SYCL).
- 🟪 **Purple** — `kvbm-engine`: collective ops (NCCL + oneCCL) and workers that drive transfers.
- Grey — `kvbm-py3` / vLLM binding layer.

```mermaid
graph TB
    subgraph bindings["kvbm-py3 (bindings)"]
        PyConnector["PyKvConnectorWorker"]
        EvSync["event_sync_blocking(u64)<br/>cfg: cuda / xpu-sycl"]
        PyConnector --> EvSync
    end

    subgraph engine["kvbm-engine"]
        PW["PhysicalWorker<br/>execute_local_layerwise_onboard"]
        CollTrait["trait CollectiveOps<br/>broadcast(layout, blocks, layers)"]
        NcclImpl["NcclCollectives<br/>(feature = nccl)"]
        OneCclImpl["OneCclCollectives<br/>(feature = oneccl)"]
        OneCclBS["OneCclBootstrap<br/>KVS rendezvous"]
        CollTrait -.-> NcclImpl
        CollTrait -.-> OneCclImpl
        OneCclImpl --> OneCclBS
    end

    subgraph physical["kvbm-physical"]
        TM["TransferManager<br/>TransferContext<br/>stream pools × 4"]
        Exec["execute_device_transfer<br/>whole-block vs FC↔LW"]
        DC["DeviceContext"]
        DS["DeviceStream"]
        DE["DeviceEvent"]
        DP["DeviceMemPool"]
        CudaCtx["CudaContext / Wrappers<br/>device/cuda/mod.rs"]
        SyclCtx["SyclContext / Wrappers<br/>device/sycl/mod.rs"]
        CtxOps["trait DeviceContextOps"]
        StrOps["trait DeviceStreamOps"]
        EvOps["trait DeviceEventOps"]
        PoolOps["trait DeviceMemPoolOps"]

        TM --> DC
        TM --> Exec
        Exec --> DS
        Exec --> DP
        DC -.-> CtxOps
        DS -.-> StrOps
        DE -.-> EvOps
        DP -.-> PoolOps
        CudaCtx -.implements.-> CtxOps
        CudaCtx -.implements.-> StrOps
        CudaCtx -.implements.-> EvOps
        CudaCtx -.implements.-> PoolOps
        SyclCtx -.implements.-> CtxOps
        SyclCtx -.implements.-> StrOps
        SyclCtx -.implements.-> EvOps
        SyclCtx -.implements.-> PoolOps
    end

    subgraph kernels["kvbm-kernels"]
        CudaKern["vectorized_copy<br/>memcpy_batch<br/>CUDA (.cu)"]
        SyclKern["sycl_vectorized_copy<br/>sycl_*_from_block<br/>SYCL (.cpp via icpx)"]
    end

    subgraph memory["dynamo-memory"]
        MD["trait MemoryDescriptor"]
        DevS["DeviceStorage"]
        PinS["PinnedStorage"]
        DAlloc["trait DeviceAllocator"]
        PAlloc["trait PinnedAllocator"]
        NumaWP["NumaWorkerPool<br/>one thread per NUMA node"]
        SMPool["SyclMemPool<br/>software free-list<br/>active_allocs: ptr→size"]
        CMPool["CudaMemPool<br/>native pool API"]

        DevS -.implements.-> MD
        PinS -.implements.-> MD
        DevS --> DAlloc
        PinS --> DAlloc
        NumaWP --> PAlloc
    end

    %% Cross-crate edges
    PyConnector -. via DLPack .-> TM
    PW --> TM
    PW -. uses .-> CollTrait

    NcclImpl --> DS
    OneCclImpl -. sycl::queue .-> DS

    CudaCtx -.implements.-> DAlloc
    SyclCtx -.implements.-> DAlloc

    CudaCtx -. owns inner .-> CudaPinAlloc["CudaPinnedAllocator<br/>cuMemHostAlloc"]
    SyclCtx -. owns inner .-> SyclPinAlloc["SyclPinnedAllocator<br/>sycl::malloc_host"]
    CudaPinAlloc -.implements.-> PAlloc
    SyclPinAlloc -.implements.-> PAlloc
    CudaCtx --> NumaWP
    SyclCtx --> NumaWP

    CudaCtx --> CMPool
    SyclCtx --> SMPool
    DP --> CMPool
    DP --> SMPool

    CudaCtx -. FFI .-> CudaKern
    SyclCtx -. FFI .-> SyclKern

    classDef mem fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:2px
    classDef phys fill:#ff6b35,stroke:#c44520,color:#fff,stroke-width:2px
    classDef kern fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:2px
    classDef eng fill:#9c27b0,stroke:#6a1b9a,color:#fff,stroke-width:2px
    classDef bind fill:#9e9e9e,stroke:#616161,color:#fff,stroke-width:2px

    class MD,DevS,PinS,DAlloc,PAlloc,NumaWP,SMPool,CMPool,CudaPinAlloc,SyclPinAlloc mem
    class TM,Exec,DC,DS,DE,DP,CudaCtx,SyclCtx,CtxOps,StrOps,EvOps,PoolOps phys
    class CudaKern,SyclKern kern
    class PW,CollTrait,NcclImpl,OneCclImpl,OneCclBS eng
    class PyConnector,EvSync bind
```

### How a transfer flows through the stack

1. A caller in `kvbm-engine` (e.g. `PhysicalWorker::execute_local_layerwise_onboard`)
   asks `TransferManager::execute_transfer` for an onboard.
2. `TransferContext` round-robins a `DeviceStream` out of the H2D or D2H
   pool (`next_h2d_stream` / `next_d2h_stream`) and the executor
   (`transfer::executor::device`) picks `batch_copy` (FC→FC) or
   `vectorized_copy` (FC↔LW). Both whole-block DMAs and kernel launches
   share the same direction pool; see `device_executor_flow.md` for the
   rationale.
3. `batch_copy` / `vectorized_copy` go through `DeviceStreamOps`, which
   dispatches to either `CudaStreamWrapper` or `SyclStreamWrapper`.
4. `vectorized_copy` implementations call into `kvbm-kernels`
   (`kvbm_kernels::vectorized_copy` for CUDA, `kvbm_kernels::sycl_vectorized_copy`
   for SYCL — the latter links against `libkvbm_kernels_sycl.so` built by
   `icpx -fsycl`).
5. Allocation flows the other way: `PhysicalLayout::builder().allocate_device(ctx)`
   in `kvbm-physical` hands `DeviceStorage::new` an `Arc<dyn DeviceAllocator>`;
   `DeviceStorage` lives in `dynamo-memory` and calls the trait.
6. Pinned host allocation passes through `NumaWorkerPool` in `dynamo-memory`,
   which runs a backend-specific `PinnedAllocator` on a NUMA-pinned worker
   thread and first-touches pages before handing the pointer back.

## Trait surface (drill-down)

```mermaid
classDiagram
    direction TB

    class DeviceBackend {
        <<enum>>
        Cuda
        Sycl
        +detect_backend() Result~Self~
        +list_available() Vec~Self~
        +is_available() bool
    }

    class DeviceContext {
        -backend: DeviceBackend
        -device_id: u32
        -ops: Box~dyn DeviceContextOps~
        +new(backend, device_id) Result~Self~
        +pci_bdf_address() Option~String~
        +create_stream() Result~DeviceStream~
        +allocate_device(size) Result~u64~
        +free_device(ptr) Result
        +allocate_pinned(size) Result~u64~
        +free_pinned(ptr) Result
        +create_memory_pool(reserve, threshold) Result~DeviceMemPool~
    }

    class DeviceStream {
        -backend: DeviceBackend
        +ops: Box~dyn DeviceStreamOps~
        +batch_copy(src, dst, size) Result
        +memcpy_htod(dst_device, src_host) Result
        +memcpy_dtoh(src_device, dst_host) Result
        +vectorized_copy(src_dev, dst_dev, chunk, count) Result
        +record_event() Result~DeviceEvent~
        +synchronize() Result
    }

    class DeviceEvent {
        -backend: DeviceBackend
        +ops: Box~dyn DeviceEventOps~
        +is_complete() Result~bool~
        +synchronize() Result
        +record_on(stream: DeviceStream) Result
    }

    class DeviceMemPool {
        -backend: DeviceBackend
        -ops: Box~dyn DeviceMemPoolOps~
        +alloc_async(size, stream) Result~u64~
        +free_async(ptr, stream) Result
    }

    class DeviceContextOps {
        <<trait>>
        +device_id() u32
        +create_stream() Result~Box dyn DeviceStreamOps~
        +allocate_device(size) Result~u64~
        +free_device(ptr) Result
        +allocate_pinned(size) Result~u64~
        +free_pinned(ptr) Result
        +bind_to_thread() Result*
        +disable_event_tracking() Result*
        +create_memory_pool(reserve, threshold) Result~Box dyn DeviceMemPoolOps~
        +raw_handle() Option~u64~*
        +pci_bdf_address() Option~String~*
    }

    class DeviceStreamOps {
        <<trait>>
        +batch_copy(src, dst, size) Result
        +memcpy_htod(dst_device, src_host) Result
        +memcpy_dtoh(src_device, dst_host) Result
        +vectorized_copy(src_dev, dst_dev, chunk, count) Result
        +record_event() Result~Box dyn DeviceEventOps~
        +synchronize() Result
        +raw_handle() Option~u64~*
    }

    class DeviceEventOps {
        <<trait>>
        +is_complete() Result~bool~
        +synchronize() Result
        +record_on_stream(stream_handle: u64) Result
        +raw_handle() Option~u64~*
    }

    class DeviceMemPoolOps {
        <<trait>>
        +alloc_async(size, stream) Result~u64~
        +free_async(ptr, stream) Result
    }

    class DeviceAllocator {
        <<trait in dynamo-memory>>
        +allocate_device(size) Result~u64~
        +free_device(ptr) Result
        +allocate_pinned(size) Result~u64~
        +free_pinned(ptr) Result
        +device_id() u32
    }

    class CudaContext {
        <<cuda backend>>
        +create_stream() CudaStreamWrapper
        +allocate_device()
        +allocate_pinned() NUMA-aware
        +create_memory_pool() CudaMemPoolWrapper
        +pci_bdf_address()
    }

    class CudaStreamWrapper {
        <<cuda backend>>
        +batch_copy()
        +memcpy_htod()
        +memcpy_dtoh()
        +vectorized_copy() kernel via kvbm-kernels
        +record_event() CudaEventWrapper
    }

    class CudaEventWrapper {
        <<cuda backend>>
        +is_complete()
        +synchronize()
        +record_on_stream()
    }

    class CudaMemPoolWrapper {
        <<cuda backend>>
        -pool: CudaMemPool
        +alloc_async() stream-ordered
        +free_async()
    }

    class SyclContext {
        <<xpu backend>>
        -cache: SyclContextCache
        +create_stream() round-robin
        +allocate_device() tracked
        +allocate_pinned() NUMA-aware
        +create_memory_pool() SyclMemPoolWrapper
        +pci_bdf_address()
    }

    class SyclStreamWrapper {
        <<xpu backend>>
        +batch_copy()
        +memcpy_htod()
        +memcpy_dtoh()
        +vectorized_copy() SYCL kernel via kvbm-kernels
        +record_event() SyclEventWrapper barrier
    }

    class SyclEventWrapper {
        <<xpu backend>>
        +is_complete()
        +synchronize()
        +record_on_stream()
    }

    class SyclMemPoolWrapper {
        <<xpu backend>>
        -pool: SyclMemPool
        -pending_frees: Vec~PendingFree~
        +alloc_async()
        +free_async() event-deferred
    }

    DeviceContext --> DeviceBackend : has
    DeviceContext *-- DeviceContextOps : ops
    DeviceContext ..> DeviceStream : creates
    DeviceContext ..> DeviceMemPool : creates
    DeviceContext ..|> DeviceAllocator : implements

    DeviceStream --> DeviceBackend : has
    DeviceStream *-- DeviceStreamOps : ops
    DeviceStream ..> DeviceEvent : creates

    DeviceEvent --> DeviceBackend : has
    DeviceEvent *-- DeviceEventOps : ops

    DeviceMemPool --> DeviceBackend : has
    DeviceMemPool *-- DeviceMemPoolOps : ops

    DeviceContextOps ..> DeviceStreamOps : creates
    DeviceContextOps ..> DeviceMemPoolOps : creates
    DeviceStreamOps ..> DeviceEventOps : creates

    CudaContext ..|> DeviceContextOps : implements
    CudaStreamWrapper ..|> DeviceStreamOps : implements
    CudaEventWrapper ..|> DeviceEventOps : implements
    CudaMemPoolWrapper ..|> DeviceMemPoolOps : implements

    SyclContext ..|> DeviceContextOps : implements
    SyclStreamWrapper ..|> DeviceStreamOps : implements
    SyclEventWrapper ..|> DeviceEventOps : implements
    SyclMemPoolWrapper ..|> DeviceMemPoolOps : implements

    style DeviceAllocator fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style DeviceContext fill:#fff3e0,stroke:#ef6c00,stroke-width:3px

    style DeviceStreamOps fill:#ff6b35,stroke:#c44520,color:#fff,stroke-width:3px
    style DeviceStream fill:#ff8c5a,stroke:#c44520,color:#fff,stroke-width:2px
    style CudaStreamWrapper fill:#ffb088,stroke:#c44520,stroke-width:2px
    style SyclStreamWrapper fill:#ffb088,stroke:#c44520,stroke-width:2px

    style DeviceEventOps fill:#9c27b0,stroke:#6a1b9a,color:#fff,stroke-width:2px

    style DeviceMemPoolOps fill:#2196f3,stroke:#1565c0,color:#fff,stroke-width:3px
    style DeviceMemPool fill:#64b5f6,stroke:#1565c0,color:#fff,stroke-width:2px
    style CudaMemPoolWrapper fill:#90caf9,stroke:#1565c0,stroke-width:2px
    style SyclMemPoolWrapper fill:#90caf9,stroke:#1565c0,stroke-width:2px
```

## Memory layer (`dynamo-memory`)

The storage side is deliberately separated from the device abstraction so
`dynamo-memory` stays free of device-SDK deps. Storage types expose themselves
as `MemoryDescriptor` (size, address, storage kind, optional NIXL descriptor)
and delegate allocation to a backend-agnostic `DeviceAllocator`. Pinned host
memory additionally goes through a NUMA worker pool that owns a
`PinnedAllocator` per backend.

```mermaid
classDiagram
    direction TB

    class MemoryDescriptor {
        <<trait>>
        +addr() usize
        +size() usize
        +storage_kind() StorageKind
        +as_any() Any
        +nixl_descriptor() Option~NixlDescriptor~
    }

    class StorageKind {
        <<enum>>
        System
        Pinned
        Device(u32)
        Disk(u64)
    }

    class DeviceAllocator {
        <<trait>>
        +allocate_device(size) Result~u64~
        +free_device(ptr) Result
        +allocate_pinned(size) Result~u64~
        +free_pinned(ptr) Result
        +device_id() u32
    }

    class PinnedAllocator {
        <<trait>>
        +alloc_pinned(size) Result~*mut u8~
        +free_pinned(ptr) Result
    }

    class DeviceStorage {
        -ctx: Arc~dyn DeviceAllocator~
        -ptr: u64
        -device_id: u32
        -len: usize
        +new(len, ctx) Result~Self~
    }

    class PinnedStorage {
        -ctx: Arc~dyn DeviceAllocator~
        -ptr: usize
        -len: usize
        +new(len, ctx) Result~Self~
    }

    class SystemStorage {
        +new(len) Result~Self~
    }

    class DiskStorage {
        +new(len) Result~Self~
    }

    class NumaWorkerPool {
        <<singleton>>
        +global() &static Self
        +allocate_pinned_for_gpu(size, pci_bdf, allocator) Result~Option~*mut u8~~
        +allocate_pinned_on_node(size, node, allocator) Result
    }

    class NumaWorker {
        -node: NumaNode
        -thread: pinned via sched_setaffinity
        +allocate(size, allocator) Result
    }

    class SyclMemPool {
        -queue: Arc~SyclQueue~
        -inner: Mutex~PoolInner~
        -release_threshold: u64
        +alloc(size) Result~u64~
        +free(ptr, size_hint) Result
    }

    class PoolInner {
        -free_list: BTreeMap~size, Vec~FreeBlock~~
        -active_allocs: HashMap~ptr, real_size~
        -cached_bytes: usize
    }

    class CudaMemPool {
        -pool: CUmemoryPool
        +alloc_async_raw(size, stream) Result
        +free_async_raw(ptr, stream) Result
    }

    class CudaPinnedAllocator {
        <<kvbm-physical>>
        -context: CudaContext
        +alloc_pinned() cuMemHostAlloc
    }

    class SyclPinnedAllocator {
        <<kvbm-physical>>
        -queue: SyclQueue
        +alloc_pinned() sycl::malloc_host
    }

    class DeviceContext {
        <<kvbm-physical>>
        implements DeviceAllocator
    }

    SystemStorage ..|> MemoryDescriptor
    DiskStorage ..|> MemoryDescriptor
    DeviceStorage ..|> MemoryDescriptor
    PinnedStorage ..|> MemoryDescriptor

    DeviceStorage --> DeviceAllocator : Arc dyn
    PinnedStorage --> DeviceAllocator : Arc dyn

    DeviceContext ..|> DeviceAllocator : bridges to kvbm-physical

    NumaWorkerPool *-- NumaWorker : per-node
    NumaWorker --> PinnedAllocator : calls dyn

    CudaPinnedAllocator ..|> PinnedAllocator
    SyclPinnedAllocator ..|> PinnedAllocator

    DeviceContext ..> NumaWorkerPool : allocate_pinned
    DeviceContext ..> CudaPinnedAllocator : wraps
    DeviceContext ..> SyclPinnedAllocator : wraps

    SyclMemPool *-- PoolInner
    DeviceContext ..> SyclMemPool : create_memory_pool
    DeviceContext ..> CudaMemPool : create_memory_pool

    StorageKind <.. MemoryDescriptor

    style DeviceAllocator fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style PinnedAllocator fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
    style MemoryDescriptor fill:#66bb6a,stroke:#2e7d32,color:#fff,stroke-width:2px
    style NumaWorkerPool fill:#81c784,stroke:#2e7d32,color:#fff,stroke-width:2px
    style SyclMemPool fill:#81c784,stroke:#2e7d32,color:#fff,stroke-width:2px
    style CudaMemPool fill:#81c784,stroke:#2e7d32,color:#fff,stroke-width:2px
    style DeviceContext fill:#ffb088,stroke:#c44520,stroke-width:3px,color:#000
    style CudaPinnedAllocator fill:#ffccbc,stroke:#c44520
    style SyclPinnedAllocator fill:#ffccbc,stroke:#c44520
```

Key points:

- **`MemoryDescriptor`** is the only type-erasable trait — everything downstream
  (layouts, NIXL registration) handles `Arc<dyn MemoryDescriptor>`.
- **`DeviceAllocator`** and **`PinnedAllocator`** are both backend-agnostic. The
  `DeviceContext` in `kvbm-physical` implements `DeviceAllocator` directly, and
  it *owns* small `CudaPinnedAllocator` / `SyclPinnedAllocator` structs that
  implement `PinnedAllocator` for the NUMA worker pool.
- **`NumaWorkerPool`** is a process-global singleton. It never does backend
  calls itself — it just hands a `PinnedAllocator` to a NUMA-pinned worker
  thread, which does the allocation and first-touches each page.
- **`SyclMemPool`** is the only software memory pool in the tree. `PoolInner`
  keeps `active_allocs: HashMap<ptr, real_size>` so that a best-fit reuse that
  returns a larger block doesn't corrupt `cached_bytes` or the
  `release_threshold` budget when the caller later frees with a smaller size
  hint. The CUDA pool uses the native CUDA pool APIs and needs none of this.

## Key design decisions

### Backend selection

`DeviceBackend` is a plain enum over the compiled-in backends. Availability is
probed at runtime with `catch_unwind` so a missing `libcuda.so` or
`libsycl.so` does not abort the process.

```rust
pub enum DeviceBackend { Cuda, Sycl }
impl DeviceBackend {
    pub fn detect_backend() -> Result<Self>;  // Cuda-first, then Sycl
    pub fn list_available() -> Vec<Self>;
    pub fn is_available(&self) -> bool;
}
```

`DeviceContext::new(backend, device_id)` dispatches to `CudaContext::new` or
`SyclContext::new` behind `#[cfg(feature = "cuda")]` / `#[cfg(feature = "xpu-sycl")]`.

### Copy API — pattern-based, not direction-based

Three primitives on `DeviceStreamOps`:

- **`batch_copy(src_ptrs, dst_ptrs, size)`** — N independent DMA copies of the
  same size. Direction (H2D, D2H, D2D) is auto-detected from pointer addresses
  by the runtime (`cudaMemcpyDefault` / SYCL `queue.memcpy()`). Used for
  whole-block FC→FC transfers.
- **`memcpy_htod(dst_device, src_host)`** / **`memcpy_dtoh(src_device, dst_host)`** —
  stream-ordered scalar copies used to upload/download pointer arrays for
  `vectorized_copy` and by the backend-agnostic `sync_memcpy_*` test helpers.
- **`vectorized_copy(src_ptrs_device, dst_ptrs_device, chunk_size, count)`** —
  N independent copies executed in parallel by a GPU kernel. Both pointer
  arrays live in device memory (previously uploaded via `memcpy_htod`).
  Implemented by `kvbm-kernels` for CUDA (`kvbm_kernels_launch_vectorized_copy`)
  and SYCL (`kvbm_kernels_sycl_launch_vectorized_copy` in
  `sycl/vectorized_copy_kernel.cpp`). Used for FC↔LW per-chunk transfers.

**Engine selection is no longer explicit.** The executor picks `batch_copy`
or `vectorized_copy` based purely on layout shape (whole-block vs. FC↔LW).
The legacy `EngineHint` enum was removed along with the Level-Zero backend
it served.

### Stream pools

`TransferContext` creates **two** stream pools per device — one for H2D and
one for D2H — each `num_streams` wide (default 4) with round-robin
acquisition. Whole-block DMAs and kernel launches share the same direction
pool; neither CUDA nor SYCL binds queues to distinct engine classes today,
so splitting copy vs. compute into separate pools gave no concurrency
benefit. The two-pool layout matches upstream `ai-dynamo/dynamo` CUDA.

On SYCL the `SyclContextCache` holds a bounded pool of in-order
`sycl::queue`s, all built on the same `sycl::context` as the alloc queue so
USM pointers are valid across the pool (SYCL spec requires this). Pool size
is controlled by `KVBM_SYCL_STREAM_POOL_SIZE` (default 4).

### Events

`DeviceEventOps` now includes `record_on_stream(stream_handle)` so a single
event can be re-recorded across successive ops — used by
`PhysicalWorker::execute_local_layerwise_onboard` in `kvbm-engine`, which
records one event per layer on a shared H2D stream so the caller can
stream-wait per-layer.

### Memory pool

`DeviceMemPoolOps::{alloc_async, free_async}` take a `&dyn DeviceStreamOps`
for stream ordering.

- **CUDA**: `CudaMemPoolWrapper` uses `cuMemAllocFromPoolAsync` /
  `cuMemFreeAsync` with the raw `CUstream` handle — GPU-side ordering is
  implicit.
- **SYCL**: SYCL has no native pool API. `SyclMemPoolWrapper` wraps
  `dynamo_memory::SyclMemPool` (a software free-list over
  `sycl::malloc_device`). `free_async` defers the return-to-pool behind a
  recorded `DeviceEvent` via `pending_frees`; the deferred frees are drained
  on the next `alloc_async`. The inner `SyclMemPool` tracks every outstanding
  ptr → real allocated size in `active_allocs`, so reusing a larger block for
  a smaller request does not corrupt `cached_bytes` / `release_threshold`.

### PCI BDF for NUMA discovery

`DeviceContextOps::pci_bdf_address()` returns the device's PCI BDF as a
`"DDDD:BB:DD.F"` string — CUDA queries `CU_DEVICE_ATTRIBUTE_PCI_*`; SYCL
reads `SyclDevice::info()?.pci_address`. `dynamo_memory::numa` consumes this
to pick a NUMA node via sysfs (`/sys/bus/pci/devices/<bdf>/numa_node`), with
`nvidia-smi` / `xpu-smi` fallbacks.

### Allocation bridging

The `dynamo-memory` crate no longer depends on any device SDK for the
storage abstraction. Instead it defines:

```rust
pub trait DeviceAllocator: Send + Sync + Debug {
    fn allocate_device(&self, size: usize) -> Result<u64>;
    fn free_device(&self, ptr: u64) -> Result<()>;
    fn allocate_pinned(&self, size: usize) -> Result<u64>;
    fn free_pinned(&self, ptr: u64) -> Result<()>;
    fn device_id(&self) -> u32;
}
```

`DeviceContext` in `kvbm-physical` implements this trait, and both
`DeviceStorage::new(size, Arc<dyn DeviceAllocator>)` and
`PinnedStorage::new(size, Arc<dyn DeviceAllocator>)` accept it. This keeps
backend-specific dependencies out of `dynamo-memory` while letting the
physical-layout builder allocate through whichever backend was selected at
`TransferManager` construction time.

### NUMA-aware pinned allocation

`dynamo_memory::numa::worker_pool::NumaWorkerPool` is a global lazy pool
of threads, one per NUMA node, each pinned with `sched_setaffinity`. Host
memory is allocated via a backend-specific `PinnedAllocator`:

```rust
pub trait PinnedAllocator: Send + Sync + 'static {
    fn alloc_pinned(&self, size: usize) -> Result<*mut u8, String>;
    fn free_pinned(&self, ptr: *mut u8) -> Result<(), String>;
}
```

`CudaPinnedAllocator` (in `device/cuda/mod.rs`) binds the CUDA context to the
pinned worker thread before calling `cuMemHostAlloc`. `SyclPinnedAllocator`
(in `device/sycl/mod.rs`) calls `queue.malloc_host(size)` — no context
binding is required. After the backend allocation returns, the worker writes
one byte per page to force first-touch on the correct node.

### Collectives

A parallel trait `CollectiveOps` lives in `kvbm-engine`:

- **`NcclCollectives`** — CUDA via `cudarc::nccl`.
- **`OneCclCollectives`** — SYCL via `oneapi-rs::ccl`. Broadcasts use
  `ccl_rs_group_start/end` with a single `event_wait` on the last submitted
  op; construction supports both a from-scratch bootstrap
  (`OneCclBootstrap` — KVS address serialized to 8B world_size + 256B KVS
  address) and borrowed handles from PyTorch / vLLM.

### Feature graph

| Crate | Feature | Effect |
|---|---|---|
| `dynamo-memory` | `sycl` | Enables `SyclMemPool`, SYCL NUMA enumerators. Requires `oneapi-rs`. |
| `kvbm-kernels` | `xpu-sycl` | Compiles SYCL `.cpp` sources via icpx and links `libkvbm_kernels_sycl.so`. |
| `kvbm-kernels` | `xpu-sycl-permute` | Enables SYCL permute kernel re-exports (implies `xpu-sycl`). |
| `kvbm-physical` | `cuda` (default) | Compiles `device/cuda`. Pulls in `kvbm-kernels`. |
| `kvbm-physical` | `xpu-sycl` | Compiles `device/sycl`. Pulls in `kvbm-kernels/xpu-sycl`, `oneapi-rs`, and `dynamo-memory/xpu-sycl`. |
| `kvbm-engine` | `nccl` | Compiles `collectives::nccl`. |
| `kvbm-engine` | `oneccl` | Compiles `collectives::oneccl` + `oneccl_bootstrap`. |
| `kvbm-py3` (bindings) | `cuda` / `xpu-sycl` | Selects which `event_sync_blocking` is emitted. Mutually exclusive. |
| `kvbm-py3` | `nccl` / `oneccl` | Imply the corresponding device feature. |

CUDA and SYCL backends can coexist in a single `kvbm-physical` build (both
features enabled), but the bindings crate assumes exactly one of
`cuda` / `xpu-sycl`.
