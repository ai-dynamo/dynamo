// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device abstraction traits for multi-backend support.
//!
//! Defines the core traits that all hardware backends
//! (CUDA, SYCL/XPU) must implement.

use anyhow::Result;
use std::any::Any;
use std::fmt::Debug;


/// Device context operations — the main interface for device management.
pub trait DeviceContextOps: Send + Sync + Debug {
    /// Downcasting hook for backend-specific escape hatches.
    ///
    /// Lives on the trait so callers holding `Box<dyn DeviceContextOps>`
    /// can recover the concrete type when they need APIs the abstraction
    /// doesn't cover (e.g. cudarc-typed planner calls). Backend-specific
    /// accessors live on the [`crate::DeviceContext`] wrapper, not here.
    fn as_any(&self) -> &dyn Any;
    /// Get the device ID this context is bound to.
    fn device_id(&self) -> u32;

    /// Create a new stream/queue for async operations.
    ///
    /// Creates a new stream/queue for async operations.
    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>>;

    /// Allocate device memory, returning a device pointer.
    fn allocate_device(&self, size: usize) -> Result<u64>;

    /// Free device memory.
    fn free_device(&self, ptr: u64) -> Result<()>;

    /// Allocate pinned (page-locked) host memory.
    fn allocate_pinned(&self, size: usize) -> Result<u64>;

    /// Free pinned host memory.
    fn free_pinned(&self, ptr: u64) -> Result<()>;

    /// Disable automatic event tracking (CUDA-specific optimization).
    ///
    /// # Safety
    /// Only safe when caller manually manages event synchronization.
    unsafe fn disable_event_tracking(&self) -> Result<()> {
        Ok(()) // Default: no-op
    }

    /// Create a memory pool for stream-ordered async allocations.
    ///
    /// # Arguments
    /// * `reserve_size` - Bytes to pre-allocate to warm the pool
    /// * `release_threshold` - Memory above this threshold is returned to system on free
    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>>;

    /// Get raw context handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }

    /// Get the PCI BDF address for this device.
    ///
    /// Used for backend-agnostic NUMA node lookups via sysfs.
    /// Returns `None` if the backend does not support PCI address queries.
    fn pci_bdf_address(&self) -> Option<String> {
        None
    }

}

/// Device stream/queue operations — async execution interface.
///
/// Operations are defined by copy pattern rather than direction:
/// - `batch_copy`: N independent DMA copies, each of the same size.
///   Direction is auto-detected from pointer addresses.
/// - `vectorized_copy`: N independent copies executed in parallel via a GPU kernel.
///   Pointer arrays are uploaded to device memory for kernel consumption.
pub trait DeviceStreamOps: Send + Sync + Debug {
    /// Downcasting hook for backend-specific escape hatches.
    /// See [`DeviceContextOps::as_any`].
    fn as_any(&self) -> &dyn Any;

    /// Bind the underlying device context to the calling thread.
    fn bind_to_thread(&self) -> Result<()> {
        Ok(())
    }

    /// Batch copy: enqueue N independent memcpy operations, each of `size` bytes.
    ///
    /// Direction (H2D, D2H, D2D) is auto-detected from pointer addresses by
    /// the underlying runtime (`cudaMemcpyDefault` / SYCL `queue.memcpy()`).
    ///
    /// Used for whole-block FC→FC transfers (1 entry per block) and per-chunk
    /// transfers when GPU kernel launch is not available.
    fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()>;

    /// Async host-to-device memcpy on this stream.
    ///
    /// Enqueues a copy of `src_host` bytes to `dst_device` (a device pointer).
    /// The copy is stream-ordered: it executes after all preceding operations
    /// on this stream and before any subsequent ones.
    ///
    /// # Safety contract
    /// `dst_device` must point to at least `src_host.len()` bytes of device memory.
    /// `src_host` must remain valid until the copy completes (caller should
    /// record an event and synchronize before dropping the source buffer).
    fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()>;

    /// Async device-to-host memcpy on this stream.
    ///
    /// Enqueues a copy of `dst_host.len()` bytes from `src_device` into `dst_host`.
    /// The copy is stream-ordered. Caller must synchronize the stream
    /// before reading `dst_host`.
    fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()>;

    /// Vectorized copy: N independent copies executed in parallel via a GPU kernel.
    ///
    /// Both `src_ptrs_device` and `dst_ptrs_device` are device pointers to arrays
    /// of `count` device pointers (previously uploaded via [`memcpy_htod`]).
    /// The kernel reads these arrays and copies `chunk_size` bytes per pair.
    ///
    /// Used for FC↔LW per-chunk transfers where many small copies benefit from
    /// GPU-parallel execution rather than sequential DMA enqueues.
    ///
    /// # Arguments
    /// * `src_ptrs_device` - Device pointer to array of `count` source pointers
    /// * `dst_ptrs_device` - Device pointer to array of `count` destination pointers
    /// * `chunk_size` - Bytes to copy per pointer pair
    /// * `count` - Number of pointer pairs
    fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()>;

    /// Record an event on this stream.
    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>>;

    /// Insert a wait-for-event dependency on this stream.
    ///
    /// Subsequent work submitted to this stream will not begin until `event`
    /// has completed. The host call itself is non-blocking — the dependency
    /// is enforced asynchronously by the device.
    ///
    /// Equivalent of `cuStreamWaitEvent(stream, event, 0)` /
    /// SYCL `queue.ext_oneapi_submit_barrier({event})`.
    fn wait_event(&self, event: &dyn DeviceEventOps) -> Result<()>;

    /// Synchronize stream (wait for all operations to complete).
    fn synchronize(&self) -> Result<()>;

    /// Get raw stream handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }

}

/// Device event operations — async completion tracking.
pub trait DeviceEventOps: Send + Sync + Debug {
    /// Check if event has completed (non-blocking).
    fn is_complete(&self) -> Result<bool>;

    /// Wait for event to complete (blocking).
    fn synchronize(&self) -> Result<()>;

    /// Re-record this event on a stream.
    /// The event object is reused — its inner state is overwritten with a new
    /// marker at the current position on the given stream.
    /// Equivalent of cuEventRecord(event, stream).
    fn record_on_stream(&self, stream: &dyn DeviceStreamOps) -> Result<()>;

    /// Re-record this event on a stream identified by its raw handle.
    ///
    /// Used at the Python/torch FFI boundary, where the connector receives
    /// `stream_handle: u64` (a `CUstream` on CUDA, a `sycl_rs_queue_t*`
    /// on SYCL) and does not own a `DeviceStream` wrapper for it.
    fn record_on_raw_stream(&self, stream_handle: u64) -> Result<()>;

    /// Make the stream identified by `stream_handle` wait on this event.
    ///
    /// Symmetric to [`DeviceStreamOps::wait_event`] but for the FFI case
    /// where the consumer holds only the raw handle (e.g., torch's current
    /// stream passed in from Python).
    fn wait_on_raw_stream(&self, stream_handle: u64) -> Result<()>;

    /// Get raw event handle for interop (optional).
    fn raw_handle(&self) -> Option<u64> {
        None
    }
}


/// Device memory pool operations — stream-ordered async allocation.
///
/// Wraps backend-specific memory pools (CUDA `cuMemAllocFromPoolAsync`,
/// SYCL USM pool, etc.) behind a unified interface.
///
/// Allocations and frees are stream-ordered: memory becomes available
/// after all preceding operations on the stream complete.
pub trait DeviceMemPoolOps: Send + Sync + Debug {
    /// Allocate memory from the pool, ordered on the given stream.
    ///
    /// # Arguments
    /// * `size` - Bytes to allocate
    /// * `stream` - Device stream ops for ordering (raw handle used internally)
    ///
    /// # Returns
    /// Device pointer to the allocated memory.
    fn alloc_async(&self, size: usize, stream: &dyn DeviceStreamOps) -> Result<u64>;

    /// Free memory back to the pool, ordered on the given stream.
    ///
    /// # Arguments
    /// * `ptr` - Device pointer previously allocated from this pool
    /// * `stream` - Device stream ops for ordering
    fn free_async(&self, ptr: u64, stream: &dyn DeviceStreamOps) -> Result<()>;
}


// =====================================================================
// DeviceDtype and per-backend representation marker traits.
// =====================================================================

/// Sealed module to keep [`DeviceDtype`] closed to the variants
/// declared in this crate.
mod private {
    pub trait Sealed {}
    impl Sealed for usize {}
}

/// Logical scalar data types that can live in a [`crate::DeviceSlice`].
///
/// Describes the byte/bit shape of a scalar that is legal as a device
/// slice element. The per-backend `*Repr` associated types name the
/// concrete representation each backend's allocator / memcpy expects.
/// For all current impls (only `usize` today) the host shape *is* the
/// device shape, so `to_cuda` / `to_sycl` are the identity and the
/// `clone_htod` fast path passes the host slice straight through with
/// no allocation or per-element conversion.
///
/// New dtypes are a one-line addition: implement the sealed
/// `private::Sealed`, declare the two `*Repr` types and the two
/// converters. If `T == CudaRepr == SyclRepr`, both converters are
/// the identity and no per-element copy is paid on either backend.
pub trait DeviceDtype: Copy + Send + 'static + private::Sealed {
    /// CUDA-side representation of `Self`.
    ///
    /// Must be `cudarc::driver::DeviceRepr` so cudarc's H2D / kernel
    /// APIs accept it. For identity-shaped scalars (e.g. `usize`),
    /// `type CudaRepr = Self`.
    #[cfg(feature = "cuda")]
    type CudaRepr: cudarc::driver::DeviceRepr + Copy;

    /// SYCL-side representation of `Self`.
    ///
    /// Must be `oneapi_rs::sycl::safe::DeviceRepr` so SYCL's USM allocator
    /// and memcpy APIs accept it. Same identity-by-default story as
    /// `CudaRepr`. The trait name mirrors cudarc's `DeviceRepr` to
    /// keep both backends conceptually symmetric.
    #[cfg(feature = "xpu-sycl")]
    type SyclRepr: oneapi_rs::sycl::safe::DeviceRepr + Copy;

    /// Convert a host element into the CUDA representation. Identity
    /// for the common case; reserved for future scalars whose host and
    /// device shapes differ.
    #[cfg(feature = "cuda")]
    fn to_cuda(self) -> Self::CudaRepr;

    /// SYCL counterpart to [`Self::to_cuda`].
    #[cfg(feature = "xpu-sycl")]
    fn to_sycl(self) -> Self::SyclRepr;
}

impl DeviceDtype for usize {
    #[cfg(feature = "cuda")]
    type CudaRepr = usize;
    #[cfg(feature = "xpu-sycl")]
    type SyclRepr = usize;

    #[cfg(feature = "cuda")]
    #[inline(always)]
    fn to_cuda(self) -> usize {
        self
    }
    #[cfg(feature = "xpu-sycl")]
    #[inline(always)]
    fn to_sycl(self) -> usize {
        self
    }
}


// =====================================================================
// DeviceSliceOps — owned, device-allocated buffer of `T` elements.
// =====================================================================

/// Identity-shape constraint used by [`crate::DeviceStream::clone_htod`].
///
/// Asserts that the host element type is also the per-backend
/// representation (no per-element conversion needed). All current
/// `DeviceDtype` impls satisfy this; a future non-identity element
/// type would need a separate API that performs the conversion.
///
/// Defined as a single combined trait because Rust does not allow
/// `#[cfg]` on individual `where`-clause predicates — expressing the
/// per-backend bounds inline at the call site fails to parse. Each
/// cfg variant of this trait inlines the right per-backend bound so
/// callers can write `T: HtodDtype` and get the correct projection.
#[cfg(all(feature = "cuda", feature = "xpu-sycl"))]
pub trait HtodDtype:
    DeviceDtype<CudaRepr = Self, SyclRepr = Self>
    + cudarc::driver::DeviceRepr
{}

#[cfg(all(feature = "cuda", feature = "xpu-sycl"))]
impl<T> HtodDtype for T where
    T: DeviceDtype<CudaRepr = T, SyclRepr = T>
        + cudarc::driver::DeviceRepr
{}

#[cfg(all(feature = "cuda", not(feature = "xpu-sycl")))]
pub trait HtodDtype:
    DeviceDtype<CudaRepr = Self> + cudarc::driver::DeviceRepr
{}

#[cfg(all(feature = "cuda", not(feature = "xpu-sycl")))]
impl<T> HtodDtype for T where
    T: DeviceDtype<CudaRepr = T> + cudarc::driver::DeviceRepr
{}

#[cfg(all(not(feature = "cuda"), feature = "xpu-sycl"))]
pub trait HtodDtype: DeviceDtype<SyclRepr = Self> {}

#[cfg(all(not(feature = "cuda"), feature = "xpu-sycl"))]
impl<T> HtodDtype for T where T: DeviceDtype<SyclRepr = T> {}

#[cfg(not(any(feature = "cuda", feature = "xpu-sycl")))]
pub trait HtodDtype: DeviceDtype {}

#[cfg(not(any(feature = "cuda", feature = "xpu-sycl")))]
impl<T> HtodDtype for T where T: DeviceDtype {}

/// Backend-agnostic surface for an owned device-allocated buffer.
///
/// Concrete impls live in the `cuda` / `sycl` submodules
/// (`CudaDeviceSlice<T>` / `SyclDeviceSlice<T>`). The wrapper struct
/// [`crate::DeviceSlice`] holds `Box<dyn DeviceSliceOps<T>>` and
/// dispatches through this trait.
///
/// This trait is element-typed (`<T>`) on purpose: a `DeviceSlice<u32>`
/// and a `DeviceSlice<usize>` are different concrete types, just as
/// `cudarc::driver::CudaSlice<T>` is generic.
pub trait DeviceSliceOps<T: DeviceDtype>: Send + Sync {
    /// Downcasting hook for backend-specific escape hatches.
    /// Mirrors [`DeviceContextOps::as_any`].
    fn as_any(&self) -> &dyn Any;

    /// Number of `T` elements in this buffer.
    fn len(&self) -> usize;

    /// Run `f` with the underlying device pointer as a `u64`.
    ///
    /// On CUDA the closure body executes while cudarc's `SyncOnDrop`
    /// guard from `DevicePtr::device_ptr` is still alive, so any kernel
    /// dispatch inside `f` is correctly ordered against the buffer's
    /// owning stream. On SYCL the pointer is intrinsically stable for
    /// the slice's lifetime; the closure shape is for API symmetry.
    fn with_device_ptr(&self, f: &mut dyn FnMut(u64));
}


// =====================================================================
// DeviceGraphExec / DeviceGraphOps — record/replay for batch memcpy.
// =====================================================================

/// A recorded and instantiated graph of memcpy operations that can be
/// replayed (launched) multiple times on a stream.
///
/// Conceptually equivalent to a `CUgraphExec` (CUDA) or a finalized
/// `command_graph<executable>` (SYCL). Implementations are
/// created via [`DeviceGraphOps::record_memcpy_graph`].
///
/// # Lifecycle
///
/// ```text
/// record_memcpy_graph(ptrs, size, stream)
///   → DeviceGraphExec  ──┐
///                         │ try_rebind(new_ptrs)  [CUDA: O(N) set-params]
///                         │ launch(stream)        [O(1) kernel launch]
///                         │    ... repeat ...
///                         └─ Drop                [destroy exec handle]
/// ```
pub trait DeviceGraphExec: Send + Sync {
    /// Number of memcpy operations recorded in this graph.
    fn node_count(&self) -> usize;

    /// Attempt to rebind source/destination addresses for all recorded
    /// memcpy nodes *in place*, without re-recording the graph.
    ///
    /// # Arguments
    /// * `src_ptrs` — new source addresses, length must equal `node_count()`
    /// * `dst_ptrs` — new destination addresses, length must equal `node_count()`
    /// * `size` — per-op byte count (must match the size used at record time)
    ///
    /// # Returns
    /// * `Ok(true)` — rebind succeeded; caller may proceed with `launch`.
    /// * `Ok(false)` — backend does not support in-place rebinding
    ///   (SYCL/UR path). Caller must discard this exec and call
    ///   `record_memcpy_graph` again with the new addresses.
    ///
    /// # CUDA
    /// Calls `cuGraphExecMemcpyNodeSetParams` per node — O(N) but
    /// avoids the capture/instantiate/get-nodes overhead of a full
    /// re-recording.
    ///
    /// # SYCL (sycl_ext_oneapi_graph)
    /// UR does not yet support per-node USM memcpy address update
    /// in graph captures; returns `Ok(false)` unconditionally.
    fn try_rebind(
        &self,
        src_ptrs: &[u64],
        dst_ptrs: &[u64],
        size: usize,
    ) -> Result<bool>;

    /// Launch (enqueue) this graph on the given stream.
    ///
    /// The graph executes asynchronously — work is ordered after all
    /// prior operations on `stream` and before any subsequent ones.
    /// Callers track completion via `stream.record_event()` or
    /// `stream.synchronize()`.
    fn launch(&self, stream: &dyn DeviceStreamOps) -> Result<()>;
}

/// Graph record/replay operations — captures a batch of uniform-size
/// memcpy operations into a replayable executable handle.
///
/// Implemented on the device *context* (not stream) because the
/// recorded graph is stream-independent at construction time — it
/// can be launched on any compatible stream.
///
/// # Backends
///
/// | Backend | Record mechanism | Rebind? |
/// |---------|------------------|---------|
/// | CUDA    | `cuStreamBeginCapture` → `cuGraphInstantiate` | Yes (`cuGraphExecMemcpyNodeSetParams`) |
/// | SYCL    | `urGraphCreateExp` → queue capture → `urGraphInstantiateGraphExp` | No (re-record) |
pub trait DeviceGraphOps: Send + Sync {
    /// Record `src_ptrs.len()` memcpy operations of `size` bytes each
    /// into an executable graph.
    ///
    /// The initial addresses in `src_ptrs`/`dst_ptrs` are baked into the
    /// recording. For backends that support rebinding (CUDA), the caller
    /// can later update addresses via [`DeviceGraphExec::try_rebind`];
    /// for others (SYCL), a fresh recording is required for new addresses.
    ///
    /// # Arguments
    /// * `src_ptrs` — source device/host pointers (one per op)
    /// * `dst_ptrs` — destination device/host pointers (one per op)
    /// * `size` — bytes to copy per op (uniform across all ops)
    /// * `stream` — stream used during recording (CUDA capture stream;
    ///   SYCL uses it to derive context/device handles)
    ///
    /// # Errors
    /// Returns an error if the underlying driver API fails during
    /// capture, instantiation, or finalisation.
    fn record_memcpy_graph(
        &self,
        src_ptrs: &[u64],
        dst_ptrs: &[u64],
        size: usize,
        stream: &dyn DeviceStreamOps,
    ) -> Result<Box<dyn DeviceGraphExec>>;

    /// Whether this backend supports in-place address rebinding on an
    /// existing [`DeviceGraphExec`] without re-recording.
    ///
    /// * CUDA: `true` (uses `cuGraphExecMemcpyNodeSetParams`)
    /// * SYCL: `false` (UR does not yet support USM memcpy node update)
    fn supports_address_rebind(&self) -> bool;
}
