// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SYCL XPU backend implementation via `oneapi-rs`.
//!
//! Pure SYCL implementation for Intel XPU where
//! `sycl::queue` (in-order) is the single ordered execution context per stream
//! — matching CUDA stream semantics exactly.
//!
//! # Process-wide shared SyclContext
//!
//! A single multi-device [`SyclContext`] spans every visible **discrete
//! Level Zero GPU** in the process. Every `SyclDeviceContext` binds its
//! queues to this shared context.
//!

use crate::device::traits::*;
use anyhow::Result;
use oneapi_rs::safe::{SyclContext, SyclDevice, SyclDeviceType, SyclEvent, SyclQueue};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

/// Backend string reported by SYCL for Level Zero devices.
const SYCL_BACKEND_LEVEL_ZERO: &str = "level_zero";

/// Decide whether a `SyclDevice` is a discrete Level Zero GPU eligible
/// for inclusion in the shared multi-device context.
///
/// The filter requires:
/// - `backend == level_zero` 
/// - `device_type == GPU` 
/// - `is_integrated == false`
///
pub fn is_discrete_level_zero_gpu(device: &SyclDevice) -> bool {
    let Ok(info) = device.info() else { return false };
    info.backend == SYCL_BACKEND_LEVEL_ZERO
        && info.device_type == SyclDeviceType::Gpu
        && !info.is_integrated
}

/// Returns the process-wide list of discrete Level Zero GPUs, in the
/// stable 0-based order used by `SyclDeviceContext::new(device_id)`.
///
pub fn discrete_gpus() -> Result<&'static [Arc<SyclDevice>]> {
    discrete_gpu_devices()
}

/// Per-process list of discrete Level Zero GPUs in stable 0-based
/// order. Built lazily on first call from the SYCL enumeration,
/// filtered to the discrete-L0-GPU subset. `kvbm-physical`'s
/// `device_id` indexes into this list.
fn discrete_gpu_devices() -> Result<&'static [Arc<SyclDevice>]> {
    // `Mutex<Option<_>>` serialises SYCL enumeration across threads so
    // the Intel L0 driver sees only one concurrent enumeration. 
    static DEVICES: OnceLock<Mutex<Option<&'static [Arc<SyclDevice>]>>> = OnceLock::new();
    let slot = DEVICES.get_or_init(|| Mutex::new(None));
    let mut guard = slot
        .lock()
        .map_err(|e| anyhow::anyhow!("discrete GPU device list mutex poisoned: {}", e))?;
    if let Some(slice) = *guard {
        return Ok(slice);
    }

    let device_count = SyclDevice::count()
        .map_err(|e| anyhow::anyhow!("SyclDevice::count failed: {}", e))?;
    let mut devs: Vec<Arc<SyclDevice>> = Vec::new();
    let mut skipped: Vec<(usize, String, String, bool)> = Vec::new();
    for i in 0..device_count {
        let Ok(dev) = SyclDevice::by_ordinal(i) else { continue };
        if is_discrete_level_zero_gpu(&dev) {
            devs.push(dev);
        } else if let Ok(info) = dev.info() {
            skipped.push((i, info.backend, info.name, info.is_integrated));
        }
    }

    if devs.is_empty() {
        return Err(anyhow::anyhow!(
            "no discrete Level Zero GPUs visible (SYCL enumerated {} device(s))",
            device_count,
        ));
    }

    tracing::info!(
        "Discovered {} discrete Level Zero GPU(s); skipped {} non-discrete-L0-GPU \
         enumeration(s): {:?}",
        devs.len(),
        skipped.len(),
        skipped,
    );

    // Leak the Vec into a 'static slice; keeps the `Arc<SyclDevice>`
    // handles alive for the process lifetime, which matches SYCL's
    // expectation that device handles outlive any context referencing
    // them.
    let slice: &'static [Arc<SyclDevice>] = Box::leak(devs.into_boxed_slice());
    *guard = Some(slice);
    Ok(slice)
}

/// Resolve a `SyclDevice` for the given kvbm-physical `device_id`.
///
/// `device_id` is a 0-based ordinal over the discrete Level Zero GPU
/// subset.
fn sycl_device(device_id: u32) -> Result<Arc<SyclDevice>> {
    let devices = discrete_gpu_devices()?;
    devices
        .get(device_id as usize)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "discrete GPU device {} not found (only {} discrete GPU(s) visible)",
                device_id,
                devices.len(),
            )
        })
}

/// Returns the process-wide multi-device `SyclContext` spanning every
/// discrete Level Zero GPU.
///
/// Built lazily on first call via `SyclContext::new_multi`. USM
/// pointers allocated through this context are addressable from every
/// queue bound to any of its devices (up to the limits imposed by
/// `zeDeviceCanAccessPeer` for direct cross-device memcpy).
fn shared_sycl_context() -> Result<Arc<SyclContext>> {
    // The Mutex serialises context construction across threads so the
    // Intel L0 driver sees only one `SyclContext::new_multi` call at a
    // time; parallel construction can deadlock or waste work. `Option`
    // inside the mutex holds the constructed context — once populated,
    // further callers clone cheaply.
    static SHARED_CTX: OnceLock<Mutex<Option<Arc<SyclContext>>>> = OnceLock::new();
    let slot = SHARED_CTX.get_or_init(|| Mutex::new(None));
    let mut guard = slot
        .lock()
        .map_err(|e| anyhow::anyhow!("shared SyclContext mutex poisoned: {}", e))?;
    if let Some(ctx) = guard.as_ref() {
        return Ok(Arc::clone(ctx));
    }

    let devices = discrete_gpu_devices()?;
    let ctx = SyclContext::new_multi(devices).map_err(|e| {
        anyhow::anyhow!(
            "SyclContext::new_multi failed for {} discrete GPU(s): {}. \
             All discrete GPUs must share an L0 platform.",
            devices.len(),
            e,
        )
    })?;
    tracing::info!(
        "Initialised shared multi-device SyclContext spanning {} discrete GPU(s)",
        devices.len(),
    );

    *guard = Some(Arc::clone(&ctx));
    Ok(ctx)
}

// =====================================================================
// Buffer ownership tracking
// =====================================================================

/// Track device/host allocations. We store the pointer as u64
/// so the HashMap is Send+Sync without wrapper types. The u64 is cast back to
/// *mut c_void when we need to call sycl::free.
static DEVICE_ALLOCS: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
static HOST_ALLOCS: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();

fn track_alloc(map: &OnceLock<Mutex<HashMap<u64, u64>>>, ptr: u64) {
    map.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .insert(ptr, ptr);
}

fn untrack_alloc(map: &OnceLock<Mutex<HashMap<u64, u64>>>, ptr_u64: u64) -> bool {
    map.get()
        .and_then(|m| m.lock().unwrap().remove(&ptr_u64))
        .is_some()
}

// =====================================================================
// SyclDeviceContext — implements DeviceContextOps
// =====================================================================

#[derive(Debug)]
pub struct SyclDeviceContext {
    device_id: u32,
    device: Arc<SyclDevice>,
    /// Process-wide multi-device SYCL context shared across every
    /// discrete Level Zero GPU. See module docs for filtering rationale.
    shared_context: Arc<SyclContext>,
}

impl SyclDeviceContext {
    pub fn new(device_id: u32) -> Result<Self> {
        Ok(Self {
            device_id,
            device: sycl_device(device_id)?,
            shared_context: shared_sycl_context()?,
        })
    }
}

impl DeviceContextOps for SyclDeviceContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        // Matches the CUDA implementation's `create_stream` semantics:
        // every caller gets their own queue.
        let queue = SyclQueue::new_on_device(&self.shared_context, &self.device)
            .map_err(|e| anyhow::anyhow!(
                "SYCL stream creation failed for device {}: {}",
                self.device_id, e
            ))?;
        Ok(Box::new(SyclStreamWrapper {
            queue,
            device_id: self.device_id,
        }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        let ptr = self
            .shared_context
            .malloc_device(&self.device, size)
            .map_err(|e| anyhow::anyhow!(
                "SYCL device allocation failed ({} bytes): {}", size, e
            ))?;
        let addr = ptr as u64;
        track_alloc(&DEVICE_ALLOCS, addr);
        Ok(addr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        if untrack_alloc(&DEVICE_ALLOCS, ptr) {
            self.shared_context
                .free_raw(ptr as *mut c_void)
                .map_err(|e| anyhow::anyhow!("SYCL free_device failed: {}", e))?;
        } else {
            debug_assert!(
                false,
                "SyclDeviceContext::free_device: untracked ptr {:#x} (double-free or foreign ptr)",
                ptr
            );
        }
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        // Try NUMA-aware allocation on Linux unless explicitly disabled
        #[cfg(target_os = "linux")]
        {
            if dynamo_memory::numa::is_numa_enabled() {
                if let Some(pci) = self.pci_bdf_address() {
                    let allocator = Arc::new(SyclPinnedAllocator {
                        context: Arc::clone(&self.shared_context),
                    });
                    match dynamo_memory::numa::worker_pool::NumaWorkerPool::global()
                        .allocate_pinned_for_gpu(size, &pci, allocator)
                    {
                        Ok(Some(ptr)) => {
                            let addr = ptr as u64;
                            track_alloc(&HOST_ALLOCS, addr);
                            tracing::debug!(
                                "Using NUMA-aware allocation for {} bytes on XPU PCI {}",
                                size, pci
                            );
                            return Ok(addr);
                        }
                        Ok(None) => {} // NUMA node unknown, fall through
                        Err(e) => return Err(anyhow::anyhow!(
                            "NUMA-aware pinned allocation failed: {}", e
                        )),
                    }
                }
            }
        }

        // Fallback: non-NUMA SYCL host allocation.
        let ptr = self
            .shared_context
            .malloc_host(size)
            .map_err(|e| anyhow::anyhow!(
                "SYCL host allocation failed ({} bytes): {}", size, e
            ))?;
        let addr = ptr as u64;
        track_alloc(&HOST_ALLOCS, addr);
        Ok(addr)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        if untrack_alloc(&HOST_ALLOCS, ptr) {
            self.shared_context
                .free_raw(ptr as *mut c_void)
                .map_err(|e| anyhow::anyhow!("SYCL free_pinned failed: {}", e))?;
        } else {
            debug_assert!(
                false,
                "SyclDeviceContext::free_pinned: untracked ptr {:#x} (double-free or foreign ptr)",
                ptr
            );
        }
        Ok(())
    }

    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>> {
        let mut builder = dynamo_memory::SyclMemPool::builder(
            Arc::clone(&self.shared_context),
            Arc::clone(&self.device),
            reserve_size,
        );
        if let Some(threshold) = release_threshold {
            builder = builder.release_threshold(threshold);
        }
        let pool = builder.build()?;

        Ok(Box::new(SyclMemPoolWrapper {
            pool,
            buffers: Mutex::new(HashMap::new()),
            pending_frees: Mutex::new(Vec::new()),
        }))
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.device_id as u64)
    }

    fn pci_bdf_address(&self) -> Option<String> {
        self.device.info().ok().and_then(|info| info.pci_address)
    }
}

/// SYCL-specific [`PinnedAllocator`] for NUMA-aware host memory.
///
/// Holds an `Arc<SyclContext>` to call `SyclContext::malloc_host` on the
/// NUMA-pinned worker thread. Pinned USM allocation is context-scoped in
/// SYCL.
struct SyclPinnedAllocator {
    context: Arc<SyclContext>,
}

impl dynamo_memory::PinnedAllocator for SyclPinnedAllocator {
    fn alloc_pinned(&self, size: usize) -> Result<*mut u8, String> {
        self.context
            .malloc_host(size)
            .map(|p| p as *mut u8)
            .map_err(|e| format!("SYCL malloc_host failed: {}", e))
    }

    fn free_pinned(&self, ptr: *mut u8) -> Result<(), String> {
        self.context
            .free_raw(ptr as *mut c_void)
            .map_err(|e| format!("SYCL free_raw failed: {}", e))
    }
}

// =====================================================================
// SyclStreamWrapper — implements DeviceStreamOps
// =====================================================================

pub struct SyclStreamWrapper {
    queue: Arc<SyclQueue>,
    device_id: u32,
}

impl std::fmt::Debug for SyclStreamWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclStreamWrapper")
            .field("device_id", &self.device_id)
            .finish()
    }
}

unsafe impl Send for SyclStreamWrapper {}
unsafe impl Sync for SyclStreamWrapper {}

impl DeviceStreamOps for SyclStreamWrapper {
    fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()> {
        assert_eq!(src_ptrs.len(), dst_ptrs.len(), "batch_copy: src/dst length mismatch");

        for (&src, &dst) in src_ptrs.iter().zip(dst_ptrs.iter()) {
            unsafe {
                self.queue.memcpy_raw_async(
                    dst as *mut c_void,
                    src as *const c_void,
                    size,
                )
            }
            .map_err(|e| anyhow::anyhow!("SYCL batch_copy memcpy failed: {}", e))?;
        }
        Ok(())
    }

    fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()> {
        unsafe {
            self.queue.memcpy_raw_async(
                dst_device as *mut c_void,
                src_host.as_ptr() as *const c_void,
                src_host.len(),
            )
        }
        .map_err(|e| anyhow::anyhow!("SYCL memcpy_htod failed: {}", e))
    }

    fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
        unsafe {
            self.queue.memcpy_raw_async(
                dst_host.as_mut_ptr() as *mut c_void,
                src_device as *const c_void,
                dst_host.len(),
            )
        }
        .map_err(|e| anyhow::anyhow!("SYCL memcpy_dtoh failed: {}", e))
    }

    fn vectorized_copy(
        &self,
        src_ptrs_device: u64,
        dst_ptrs_device: u64,
        chunk_size: usize,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }

        // Launch the SYCL vectorized_copy kernel on the SAME queue.
        // The in-order queue guarantees ordering with preceding memcpy_htod calls.
        #[cfg(feature = "xpu-sycl")]
        {
            let ret = unsafe {
                kvbm_kernels::sycl_vectorized_copy(
                    src_ptrs_device as *mut *mut c_void,
                    dst_ptrs_device as *mut *mut c_void,
                    chunk_size,
                    count as i32,
                    self.queue.raw_queue_ptr(),
                )
            };
            if ret != 0 {
                return Err(anyhow::anyhow!("SYCL vectorized_copy kernel failed (rc={})", ret));
            }
            return Ok(());
        }

        #[cfg(not(feature = "xpu-sycl"))]
        {
            // Fallback: read pointer arrays back to host and use batch_copy.
            self.queue.synchronize()
                .map_err(|e| anyhow::anyhow!("SYCL sync before vectorized_copy readback: {}", e))?;

            let mut src_ptrs = vec![0u64; count];
            let mut dst_ptrs = vec![0u64; count];
            let byte_len = count * std::mem::size_of::<u64>();

            unsafe {
                self.queue.memcpy_raw_async(
                    src_ptrs.as_mut_ptr() as *mut c_void,
                    src_ptrs_device as *const c_void,
                    byte_len,
                )
            }
            .map_err(|e| anyhow::anyhow!("SYCL readback src_ptrs failed: {}", e))?;

            unsafe {
                self.queue.memcpy_raw_async(
                    dst_ptrs.as_mut_ptr() as *mut c_void,
                    dst_ptrs_device as *const c_void,
                    byte_len,
                )
            }
            .map_err(|e| anyhow::anyhow!("SYCL readback dst_ptrs failed: {}", e))?;

            self.queue.synchronize()
                .map_err(|e| anyhow::anyhow!("SYCL sync after vectorized_copy readback: {}", e))?;

            self.batch_copy(&src_ptrs, &dst_ptrs, chunk_size)
        }
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let event = self.queue.submit_barrier()
            .map_err(|e| anyhow::anyhow!("SYCL record_event (barrier) failed: {}", e))?;
        Ok(Box::new(SyclEventWrapper { event }))
    }

    fn synchronize(&self) -> Result<()> {
        self.queue
            .synchronize()
            .map_err(|e| anyhow::anyhow!("SYCL queue synchronize failed: {}", e))
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.queue.handle() as u64)
    }
}

// =====================================================================
// SyclEventWrapper — implements DeviceEventOps
// =====================================================================

pub struct SyclEventWrapper {
    event: SyclEvent,
}

unsafe impl Send for SyclEventWrapper {}
unsafe impl Sync for SyclEventWrapper {}

impl std::fmt::Debug for SyclEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclEventWrapper").finish()
    }
}

impl DeviceEventOps for SyclEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        self.event.is_complete()
            .map_err(|e| anyhow::anyhow!("SYCL event query failed: {}", e))
    }

    fn synchronize(&self) -> Result<()> {
        self.event.wait()
            .map_err(|e| anyhow::anyhow!("SYCL event synchronize failed: {}", e))
    }

    fn record_on_stream(&self, stream_handle: u64) -> Result<()> {
        unsafe {
            oneapi_rs::sys::sycl_rs_event_record_on_queue(
                self.event.handle,
                stream_handle as *mut oneapi_rs::sys::sycl_rs_queue_t,
            ).result()
        }.map_err(|e| anyhow::anyhow!("SYCL event record_on_queue failed: {}", e))
    }
}

// =====================================================================
// SyclMemPoolWrapper — implements DeviceMemPoolOps
// =====================================================================

/// Deferred free entry: pointer + event to wait on before actually freeing.
struct PendingFree {
    ptr: u64,
    size: usize,
    event: Box<dyn DeviceEventOps>,
}

// SAFETY: The pointer is a USM device allocation that is safe to free from any thread.
unsafe impl Send for PendingFree {}

/// Memory pool wrapper for SYCL using `dynamo_memory::SyclMemPool`.
///
/// Bridges the `DeviceMemPoolOps` trait (which uses raw `u64` pointers) with
/// the `SyclMemPool` API. An internal `HashMap` tracks active allocations
/// by pointer so that buffers stay alive until explicitly freed back to the pool.
pub struct SyclMemPoolWrapper {
    pool: dynamo_memory::SyclMemPool,
    /// Active allocations: device pointer → size.
    buffers: Mutex<HashMap<u64, usize>>,
    /// Deferred frees waiting for GPU work to complete.
    pending_frees: Mutex<Vec<PendingFree>>,
}

impl std::fmt::Debug for SyclMemPoolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclMemPoolWrapper")
            .field("pool_cached_bytes", &self.pool.cached_bytes())
            .finish()
    }
}

unsafe impl Send for SyclMemPoolWrapper {}
unsafe impl Sync for SyclMemPoolWrapper {}

impl SyclMemPoolWrapper {
    /// Drain pending frees whose events have signaled, returning buffers to the
    /// pool's free-list so they can be reused by subsequent allocations.
    fn drain_completed_frees(&self) -> Result<()> {
        let mut pending = self
            .pending_frees
            .lock()
            .map_err(|e| anyhow::anyhow!("pending_frees lock poisoned: {}", e))?;
        let mut i = 0;
        while i < pending.len() {
            if pending[i].event.is_complete()? {
                let pf = pending.swap_remove(i);
                self.pool.free(pf.ptr, pf.size)?;
            } else {
                i += 1;
            }
        }
        Ok(())
    }
}

impl Drop for SyclMemPoolWrapper {
    fn drop(&mut self) {
        // Synchronize all pending frees so buffers are returned to the pool.
        if let Ok(mut pending) = self.pending_frees.lock() {
            for pf in pending.drain(..) {
                let _ = pf.event.synchronize();
                let _ = self.pool.free(pf.ptr, pf.size);
            }
        }
    }
}

impl DeviceMemPoolOps for SyclMemPoolWrapper {
    fn alloc_async(&self, size: usize, _stream: &dyn DeviceStreamOps) -> Result<u64> {
        // Drain completed pending frees so their buffers can be reused.
        self.drain_completed_frees()?;
        let ptr = self.pool.alloc(size)?;
        self.buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .insert(ptr, size);
        Ok(ptr)
    }

    fn free_async(&self, ptr: u64, stream: &dyn DeviceStreamOps) -> Result<()> {
        let size = self
            .buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .remove(&ptr)
            .ok_or_else(|| anyhow::anyhow!("SyclMemPoolWrapper: ptr {:#x} not found in buffer map", ptr))?;

        let event = match stream.record_event() {
            Ok(event) => event,
            Err(e) => {
                // Restore tracking on failure.
                self.buffers
                    .lock()
                    .map_err(|lock_err| anyhow::anyhow!("buffer map poisoned: {}", lock_err))?
                    .insert(ptr, size);
                return Err(e);
            }
        };

        self.pending_frees
            .lock()
            .map_err(|e| anyhow::anyhow!("pending_frees lock poisoned: {}", e))?
            .push(PendingFree { ptr, size, event });

        Ok(())
    }
}

// =====================================================================
// Availability check
// =====================================================================

/// Check if the SYCL XPU backend is available.
///
/// Uses `catch_unwind` defensively in case the SYCL runtime panics
/// when its shared libraries are not present.
pub fn is_available() -> bool {
    std::panic::catch_unwind(|| SyclDevice::count().map(|n| n > 0).unwrap_or(false)).unwrap_or(false)
}
