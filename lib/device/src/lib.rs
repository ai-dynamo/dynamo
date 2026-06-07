// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-agnostic device handle traits and CUDA / SYCL implementations.
//!
//! This crate provides the device-side counterpart to `dynamo-memory`'s
//! storage-layer traits: the [`DeviceBackend`] tag enum, the
//! `Device{Context,Stream,Event,MemPool}Ops` trait surface, the
//! trait-object wrappers callers actually hold, and the concrete CUDA
//! and SYCL implementations behind feature gates.
//!
//! `dynamo-device` is a tier-1 Dynamo primitive: it does not depend on
//! any `kvbm-*` crate, so non-KVBM consumers (GMS, future profilers,
//! etc.) can use it without pulling in subsystem code. KVBM crates
//! depend on `dynamo-device`, never the reverse.
//!
//! Multi-backend dispatch uses the Static Enum + Trait Objects pattern:
//! [`DeviceBackend`] is the runtime tag; the concrete implementations
//! are constructed via [`DeviceContext::new`] and held as
//! `Box<dyn DeviceContextOps>`.

pub mod traits;
pub mod topology;

pub use topology::get_device_cpu_set;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "xpu-sycl")]
pub mod sycl;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub use traits::{
    DeviceContextOps, DeviceStreamOps, DeviceEventOps, DeviceMemPoolOps,
    DeviceDtype, DeviceSliceOps, HtodDtype,
    DeviceGraphExec, DeviceGraphOps,
};

/// Device backend type selector.
///
/// Tag enum identifying which hardware backend a device handle targets.
/// The runtime probes (`is_available`, `detect_backend`, `list_available`)
/// are inherent methods below — they dispatch to feature-gated FFI in
/// the `cuda` / `sycl` submodules of this crate.
///
/// # Adding a new backend
///
/// Adding e.g. ROCm is a single-crate change: append a `Rocm` variant,
/// add a `rocm` Cargo feature, gate a `rocm` submodule with the FFI
/// impl, and extend the inherent methods below. No other crate changes
/// are required for the backend tag itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DeviceBackend {
    /// NVIDIA CUDA backend.
    #[default]
    Cuda,
    /// SYCL backend (Intel XPU via SYCL).
    Sycl,
}

impl DeviceBackend {
    /// Human-readable name for logs and diagnostics.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Sycl => "SYCL (XPU)",
        }
    }

    /// Return true if this backend is compiled in AND a device is
    /// physically present.
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cuda => {
                #[cfg(feature = "cuda")]
                { cuda::is_available() }
                #[cfg(not(feature = "cuda"))]
                { false }
            }
            Self::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                { sycl::is_available() }
                #[cfg(not(feature = "xpu-sycl"))]
                { false }
            }
        }
    }

    /// Auto-detect the best available device backend.
    ///
    /// Priority order: CUDA then SYCL (XPU).
    pub fn detect_backend() -> Result<Self> {
        #[cfg(feature = "cuda")]
        {
            if Self::Cuda.is_available() {
                tracing::info!("Auto-detected CUDA backend");
                return Ok(Self::Cuda);
            }
        }
        #[cfg(feature = "xpu-sycl")]
        {
            if Self::Sycl.is_available() {
                tracing::info!("Auto-detected SYCL (XPU) backend");
                return Ok(Self::Sycl);
            }
        }
        bail!("No supported device backend available on this system")
    }

    /// List every backend that is both compiled in and physically
    /// present on the current system.
    pub fn list_available() -> Vec<Self> {
        let mut backends = Vec::new();
        #[cfg(feature = "cuda")]
        if Self::Cuda.is_available() {
            backends.push(Self::Cuda);
        }
        #[cfg(feature = "xpu-sycl")]
        if Self::Sycl.is_available() {
            backends.push(Self::Sycl);
        }
        backends
    }
}

impl FromStr for DeviceBackend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cuda" | "gpu" | "nvidia" => Ok(Self::Cuda),
            "sycl" | "xpu" | "intel" => Ok(Self::Sycl),
            _ => Err(format!("Unknown device backend: {s}")),
        }
    }
}

/// Unified device context holding polymorphic implementation.
pub struct DeviceContext {
    backend: DeviceBackend,
    device_id: u32,
    ops: Box<dyn DeviceContextOps>,
}

impl DeviceContext {
    /// Create a new device context for the specified backend and device.
    pub fn new(backend: DeviceBackend, device_id: u32) -> Result<Self> {
        let ops: Box<dyn DeviceContextOps> = match backend {
            DeviceBackend::Cuda => {
                #[cfg(feature = "cuda")]
                { Box::new(cuda::CudaDeviceContext::new(device_id)?) }
                #[cfg(not(feature = "cuda"))]
                { bail!("CUDA backend not compiled (enable 'cuda' feature)") }
            }
            DeviceBackend::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                { Box::new(sycl::SyclDeviceContext::new(device_id)?) }
                #[cfg(not(feature = "xpu-sycl"))]
                { bail!("SYCL backend not compiled (enable 'xpu-sycl' feature)") }
            }
        };

        Ok(Self { backend, device_id, ops })
    }

    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }

    /// Get the PCI BDF address.
    ///
    /// Returns `None` if unavailable.
    pub fn pci_bdf_address(&self) -> Option<String> {
        self.ops.pci_bdf_address()
    }

    /// Disable per-stream event tracking on the underlying device context.
    ///
    /// # Safety
    ///
    /// Only safe when the caller manually manages event synchronization.
    /// Mirrors [`DeviceContextOps::disable_event_tracking`].
    pub unsafe fn disable_event_tracking(&self) -> Result<()> {
        unsafe { self.ops.disable_event_tracking() }
    }

    pub fn create_stream(&self) -> Result<DeviceStream> {
        let stream_ops = self.ops.create_stream()?;
        Ok(DeviceStream {
            backend: self.backend,
            ops: stream_ops,
        })
    }

    pub fn allocate_device(&self, size: usize) -> Result<u64> {
        self.ops.allocate_device(size)
    }

    pub fn free_device(&self, ptr: u64) -> Result<()> {
        self.ops.free_device(ptr)
    }

    pub fn allocate_pinned(&self, size: usize) -> Result<u64> {
        self.ops.allocate_pinned(size)
    }

    pub fn free_pinned(&self, ptr: u64) -> Result<()> {
        self.ops.free_pinned(ptr)
    }

    /// Create a memory pool for stream-ordered async allocations.
    pub fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<DeviceMemPool> {
        let ops = self.ops.create_memory_pool(reserve_size, release_threshold)?;
        Ok(DeviceMemPool {
            backend: self.backend,
            ops,
        })
    }

    /// Recover the underlying `cudarc::driver::CudaContext` if this
    /// device context wraps the CUDA backend.
    ///
    /// Returns `None` for non-CUDA backends. CUDA-typed callers (planner,
    /// benchmark cache) use this to call cudarc APIs that take an
    /// `Arc<CudaContext>` directly, avoiding a duplicate `cuCtxCreate`.
    #[cfg(feature = "cuda")]
    pub fn cuda_context(&self) -> Option<std::sync::Arc<cudarc::driver::CudaContext>> {
        self.ops
            .as_any()
            .downcast_ref::<cuda::CudaDeviceContext>()
            .map(|c| c.inner().clone())
    }

    /// Attempt to get graph record/replay operations for this context.
    ///
    /// Returns `Some` if the backend implements `DeviceGraphOps`
    /// (SYCL with SYCL graph support). Returns `None` for
    /// backends that do not yet implement the trait (CUDA uses raw
    /// `cuGraph` APIs directly in the planner; Phase 5 would refactor).
    pub fn graph_ops(&self) -> Option<&dyn DeviceGraphOps> {
        match self.backend {
            DeviceBackend::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                {
                    self.ops
                        .as_any()
                        .downcast_ref::<sycl::SyclDeviceContext>()
                        .map(|ctx| ctx as &dyn DeviceGraphOps)
                }
                #[cfg(not(feature = "xpu-sycl"))]
                { None }
            }
            DeviceBackend::Cuda => None, // CUDA uses raw cuGraph APIs directly
        }
    }
}

impl std::fmt::Debug for DeviceContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContext")
            .field("backend", &self.backend)
            .field("device_id", &self.device_id)
            .finish()
    }
}

/// Device stream wrapper.
pub struct DeviceStream {
    backend: DeviceBackend,
    ops: Box<dyn DeviceStreamOps>,
}

impl DeviceStream {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    /// Access the underlying stream operations trait object.
    ///
    /// Used by `DeviceGraphOps` callers that need to pass the stream
    /// as `&dyn DeviceStreamOps` (e.g. `record_memcpy_graph`, `launch`).
    pub fn stream_ops(&self) -> &dyn DeviceStreamOps {
        &*self.ops
    }

    pub fn bind_to_thread(&self) -> Result<()> {
        self.ops.bind_to_thread()
    }

    pub fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()> {
        self.ops.batch_copy(src_ptrs, dst_ptrs, size)
    }

    pub fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()> {
        self.ops.memcpy_htod(dst_device, src_host)
    }

    pub fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
        self.ops.memcpy_dtoh(src_device, dst_host)
    }

    pub fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()> {
        self.ops.vectorized_copy(src_ptrs_device, dst_ptrs_device, chunk_size, count)
    }

    pub fn record_event(&self) -> Result<DeviceEvent> {
        let event_ops = self.ops.record_event()?;
        Ok(DeviceEvent {
            backend: self.backend,
            ops: event_ops,
        })
    }

    /// Stream-ordered host-to-device copy returning an owned device buffer.
    ///
    /// Allocates a device buffer of `host.len()` `T`s, enqueues a copy
    /// of the host bytes onto this stream, and returns a [`DeviceSlice`]
    /// whose Drop frees the device memory.
    ///
    /// On CUDA the underlying allocation/free is stream-ordered via
    /// the cudarc API. On SYCL the free runs synchronously because no
    /// stream-ordered free exists in the safe oneapi-rs surface — this
    /// is a known perf gap on the transform-scratch path.
    pub fn clone_htod<T>(&self, host: &[T]) -> Result<DeviceSlice<T>>
    where
        T: HtodDtype,
    {
        match self.backend {
            DeviceBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    let cuda_stream_handle = self
                        .ops
                        .as_any()
                        .downcast_ref::<cuda::CudaDeviceStream>()
                        .ok_or_else(|| anyhow::anyhow!(
                            "DeviceStream::clone_htod: backend tagged Cuda but stream is not a CudaDeviceStream"
                        ))?;
                    let slice_ops: Box<dyn DeviceSliceOps<T>> = Box::new(
                        cuda::CudaDeviceSlice::<T>::clone_htod(cuda_stream_handle, host)?,
                    );
                    Ok(DeviceSlice {
                        backend: DeviceBackend::Cuda,
                        len: host.len(),
                        ops: slice_ops,
                    })
                }
                #[cfg(not(feature = "cuda"))]
                { bail!("DeviceStream::clone_htod: CUDA backend not compiled") }
            }
            DeviceBackend::Sycl => {
                #[cfg(feature = "xpu-sycl")]
                {
                    let sycl_stream = self
                        .ops
                        .as_any()
                        .downcast_ref::<sycl::SyclDeviceStream>()
                        .ok_or_else(|| anyhow::anyhow!(
                            "DeviceStream::clone_htod: backend tagged Sycl but stream is not a SyclDeviceStream"
                        ))?;
                    let slice_ops: Box<dyn DeviceSliceOps<T>> = Box::new(
                        sycl::SyclDeviceSlice::<T>::clone_htod(sycl_stream, host)?,
                    );
                    Ok(DeviceSlice {
                        backend: DeviceBackend::Sycl,
                        len: host.len(),
                        ops: slice_ops,
                    })
                }
                #[cfg(not(feature = "xpu-sycl"))]
                { bail!("DeviceStream::clone_htod: SYCL backend not compiled") }
            }
        }
    }

    /// Insert a wait-for-event dependency on this stream.
    ///
    /// Subsequent work submitted to this stream will not begin until `event`
    /// has completed. The host call itself is non-blocking.
    pub fn wait_event(&self, event: &DeviceEvent) -> Result<()> {
        self.ops.wait_event(&*event.ops)
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ops.synchronize()
    }

    /// Recover the underlying `Arc<cudarc::driver::CudaStream>` if this
    /// stream wraps the CUDA backend.
    ///
    /// C-FFI handle escape hatch for raw kernel launch sites that need
    /// the underlying `cudaStream_t` (graph capture/replay, transform
    /// kernel FFI, benchmark direct DMA). Returns `None` for non-CUDA
    /// backends. Prefer the device-agnostic `DeviceStream` API
    /// (`batch_copy`, `vectorized_copy`, `clone_htod`, `with_device_ptr`)
    /// at all non-FFI call sites.
    #[cfg(feature = "cuda")]
    pub fn cuda_stream_arc(&self) -> Option<std::sync::Arc<cudarc::driver::CudaStream>> {
        self.ops
            .as_any()
            .downcast_ref::<cuda::CudaDeviceStream>()
            .map(|s| s.inner().clone())
    }

    /// Recover the underlying [`sycl::SyclDeviceStream`] if this stream
    /// wraps the SYCL backend.
    ///
    /// C-FFI handle escape hatch for raw kernel launch sites that need
    /// the underlying `sycl::queue*` (transform kernel FFI; benchmark
    /// direct DMA, when wired). Mirrors [`Self::cuda_stream_arc`] on
    /// the SYCL side and exists so kvbm-physical can launch
    /// elem-size-typed C kernels without leaking `oneapi-rs` types
    /// through the device abstraction. Prefer the device-agnostic
    /// `DeviceStream` API at all non-FFI call sites.
    #[cfg(feature = "xpu-sycl")]
    pub fn sycl_stream(&self) -> Option<&sycl::SyclDeviceStream> {
        self.ops
            .as_any()
            .downcast_ref::<sycl::SyclDeviceStream>()
    }
}

/// Synchronous device-to-host memcpy using the device abstraction.
///
/// Creates a temporary stream, enqueues the copy, and synchronizes.
/// Used by test helpers (checksum, fill) that need backend-agnostic device access.
pub fn sync_memcpy_dtoh(device_id: u32, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
    let backend = DeviceBackend::detect_backend()?;
    let ctx = DeviceContext::new(backend, device_id)?;
    let stream = ctx.create_stream()?;
    stream.memcpy_dtoh(src_device, dst_host)?;
    stream.synchronize()?;
    Ok(())
}

/// Synchronous host-to-device memcpy using the device abstraction.
///
/// Creates a temporary stream, enqueues the copy, and synchronizes.
/// Used by test helpers (checksum, fill) that need backend-agnostic device access.
pub fn sync_memcpy_htod(device_id: u32, dst_device: u64, src_host: &[u8]) -> Result<()> {
    let backend = DeviceBackend::detect_backend()?;
    let ctx = DeviceContext::new(backend, device_id)?;
    let stream = ctx.create_stream()?;
    stream.memcpy_htod(dst_device, src_host)?;
    stream.synchronize()?;
    Ok(())
}

impl std::fmt::Debug for DeviceStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceStream")
            .field("backend", &self.backend)
            .finish()
    }
}

/// Device event wrapper.
pub struct DeviceEvent {
    backend: DeviceBackend,
    ops: Box<dyn DeviceEventOps>,
}

impl DeviceEvent {
    /// Wrap a raw `cudarc::driver::CudaEvent` in the device-agnostic
    /// `DeviceEvent`.
    ///
    /// Used by callers that obtained a `CudaEvent` directly from
    /// `cudarc` (e.g. NCCL stream `record_event`) and need to hand it
    /// to a backend-agnostic registrar.
    #[cfg(feature = "cuda")]
    pub fn from_cuda_event(event: cudarc::driver::CudaEvent) -> Self {
        Self {
            backend: DeviceBackend::Cuda,
            ops: Box::new(cuda::CudaDeviceEvent { event }),
        }
    }

    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    pub fn is_complete(&self) -> Result<bool> {
        self.ops.is_complete()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ops.synchronize()
    }

    /// Re-record this event on a stream.
    pub fn record_on(&self, stream: &DeviceStream) -> Result<()> {
        self.ops.record_on_stream(&*stream.ops)
    }

    /// Re-record this event on a stream identified by its raw FFI handle.
    ///
    /// Used at the Python/torch boundary, where the connector receives
    /// a raw `u64` (CUstream / sycl_rs_queue_t pointer) and does not own
    /// a `DeviceStream` wrapper for it.
    pub fn record_on_raw(&self, stream_handle: u64) -> Result<()> {
        self.ops.record_on_raw_stream(stream_handle)
    }

    /// Make the stream identified by `stream_handle` wait on this event.
    ///
    /// Symmetric to [`DeviceStream::wait_event`] for the FFI boundary case.
    pub fn wait_on_raw(&self, stream_handle: u64) -> Result<()> {
        self.ops.wait_on_raw_stream(stream_handle)
    }
}

impl std::fmt::Debug for DeviceEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceEvent")
            .field("backend", &self.backend)
            .finish()
    }
}

// ======================================================================
// DeviceSlice<T> — owned device-allocated buffer
// ======================================================================

/// Owned device-allocated buffer of `T` elements.
///
/// Thin wrapper holding a `Box<dyn DeviceSliceOps<T>>` whose concrete
/// type is constructed by the backend submodules (`cuda::CudaDeviceSlice`
/// / `sycl::SyclDeviceSlice`). Drop frees the underlying allocation.
///
/// Use [`Self::with_device_ptr`] at kernel-dispatch sites to keep
/// cudarc's `SyncOnDrop` guard alive across the launch.
pub struct DeviceSlice<T: DeviceDtype> {
    backend: DeviceBackend,
    len: usize,
    ops: Box<dyn DeviceSliceOps<T>>,
}

impl<T: DeviceDtype> DeviceSlice<T> {
    pub fn backend(&self) -> DeviceBackend { self.backend }
    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Run `f` with the underlying device pointer as a `u64`.
    ///
    /// On CUDA, the closure body executes while cudarc's `SyncOnDrop`
    /// guard from `DevicePtr::device_ptr` is still alive, so any kernel
    /// dispatch inside `f` is correctly ordered against the buffer's
    /// owning stream. On SYCL the pointer is intrinsically stable for
    /// the slice's lifetime; the closure shape is purely for API
    /// symmetry.
    ///
    /// This is the preferred accessor at kernel-dispatch sites. Use
    /// [`Self::device_ptr_u64`] only when the surrounding code already
    /// holds the slice alive across a same-stream synchronous launch
    /// and a closure scope is structurally awkward.
    pub fn with_device_ptr<R>(&self, f: impl FnOnce(u64) -> R) -> R {
        // The trait surface is `&mut dyn FnMut(u64)` so it can be a
        // dyn-trait object; we adapt the FnOnce caller to that shape
        // by stashing the FnOnce in an `Option` and `.take()`-ing it
        // on first invocation, then capturing the result through `out`.
        let mut f_once: Option<_> = Some(f);
        let mut out: Option<R> = None;
        let mut adapter = |raw: u64| {
            let f = f_once
                .take()
                .expect("DeviceSliceOps::with_device_ptr invoked the closure more than once");
            out = Some(f(raw));
        };
        self.ops.with_device_ptr(&mut adapter);
        out.expect("DeviceSliceOps::with_device_ptr did not invoke the closure")
    }

    /// Recover the underlying device pointer as a `u64`.
    ///
    /// Prefer [`Self::with_device_ptr`] at kernel-dispatch sites — it
    /// keeps cudarc's `SyncOnDrop` guard alive across the launch. This
    /// accessor drops the guard immediately and is only sound when the
    /// caller dispatches synchronously on the slice's owning stream
    /// before the slice is dropped.
    pub fn device_ptr_u64(&self) -> u64 {
        self.with_device_ptr(|raw| raw)
    }

    /// Recover the underlying `cudarc::driver::CudaSlice` if this slice
    /// was allocated via the CUDA backend.
    ///
    /// Returns `None` for non-CUDA backends. Used by cudarc-typed
    /// callers (e.g. cudarc graph-replay capture) that need the typed
    /// slice handle directly.
    #[cfg(feature = "cuda")]
    pub fn cuda_slice(&self) -> Option<&cudarc::driver::CudaSlice<T::CudaRepr>> {
        self.ops
            .as_any()
            .downcast_ref::<cuda::CudaDeviceSlice<T>>()
            .map(|s| s.inner())
    }
}

impl<T: DeviceDtype> std::fmt::Debug for DeviceSlice<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceSlice")
            .field("backend", &self.backend)
            .field("len", &self.len)
            .finish()
    }
}


// ======================================================================
// DeviceAllocator — bridge to dynamo-memory's trait
// ======================================================================

impl dynamo_memory::DeviceAllocator for DeviceContext {
    fn allocate_device(&self, size: usize) -> dynamo_memory::Result<u64> {
        self.ops
            .allocate_device(size)
            .map_err(|e| dynamo_memory::StorageError::AllocationFailed(e.to_string()))
    }

    fn free_device(&self, ptr: u64) -> dynamo_memory::Result<()> {
        self.ops
            .free_device(ptr)
            .map_err(|e| dynamo_memory::StorageError::OperationFailed(e.to_string()))
    }

    fn allocate_pinned(&self, size: usize) -> dynamo_memory::Result<u64> {
        self.ops
            .allocate_pinned(size)
            .map_err(|e| dynamo_memory::StorageError::AllocationFailed(e.to_string()))
    }

    fn free_pinned(&self, ptr: u64) -> dynamo_memory::Result<()> {
        self.ops
            .free_pinned(ptr)
            .map_err(|e| dynamo_memory::StorageError::OperationFailed(e.to_string()))
    }

    fn device_id(&self) -> u32 {
        self.device_id
    }
}

// ======================================================================
// DeviceMemPool — unified memory pool wrapper
// ======================================================================

/// Unified device memory pool for stream-ordered async allocation.
pub struct DeviceMemPool {
    backend: DeviceBackend,
    ops: Box<dyn DeviceMemPoolOps>,
}

impl DeviceMemPool {
    pub fn backend(&self) -> DeviceBackend {
        self.backend
    }

    /// Allocate memory from the pool, ordered on the given stream.
    pub fn alloc_async(&self, size: usize, stream: &DeviceStream) -> Result<u64> {
        self.ops.alloc_async(size, &*stream.ops)
    }

    /// Free memory back to the pool, ordered on the given stream.
    pub fn free_async(&self, ptr: u64, stream: &DeviceStream) -> Result<()> {
        self.ops.free_async(ptr, &*stream.ops)
    }
}

impl std::fmt::Debug for DeviceMemPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceMemPool")
            .field("backend", &self.backend)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceBackend;

    #[test]
    fn test_detect_backend() {
        match DeviceBackend::detect_backend() {
            Ok(backend) => {
                println!("Detected: {:?}", backend);
                assert!(backend.is_available());
            }
            Err(e) => {
                println!("No backend available: {}", e);
            }
        }
    }

    #[test]
    fn test_list_available() {
        let backends = DeviceBackend::list_available();
        println!("Available backends: {:?}", backends);
    }
}
