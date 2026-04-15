// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SYCL XPU backend implementation via `oneapi-rs`.
//!
//! Replaces the Level Zero (ze) backend with a pure SYCL implementation where
//! `sycl::queue` (in-order) is the single ordered execution context per stream
//! — matching CUDA stream semantics exactly.
//!
//! All device operations route through `oneapi_rs::safe` (RAII wrappers).
//! No `oneapi_rs::sys` calls remain after the P1–P6 oneapi-rs patches.

use crate::device::traits::*;
use anyhow::Result;
use oneapi_rs::safe::{SyclDevice, SyclEvent, SyclQueue};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

// =====================================================================
// Context cache (one per device_id, shared across streams)
// =====================================================================

struct SyclContextCache {
    device_id: u32,
    /// A "context-level" queue used for allocation/free (never for stream work).
    /// SYCL ties USM allocation to a device+context; we use this queue's context.
    alloc_queue: Arc<SyclQueue>,
    /// Bounded pool of SYCL queues for stream use. All queues share the same
    /// SYCL context as `alloc_queue` so USM pointers are valid across them.
    /// Pool size is controlled by `KVBM_SYCL_STREAM_POOL_SIZE` (default 4).
    stream_pool: Vec<Arc<SyclQueue>>,
    /// Round-robin counter for stream pool assignment.
    next_stream: AtomicUsize,
}

unsafe impl Send for SyclContextCache {}
unsafe impl Sync for SyclContextCache {}

impl std::fmt::Debug for SyclContextCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyclContextCache")
            .field("device_id", &self.device_id)
            .finish()
    }
}

fn get_or_create_sycl_context(device_id: u32) -> Result<Arc<SyclContextCache>> {
    static CONTEXT_CACHE: OnceLock<Mutex<HashMap<u32, Arc<SyclContextCache>>>> =
        OnceLock::new();

    let mut cache = CONTEXT_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();

    if let Some(cached) = cache.get(&device_id) {
        return Ok(Arc::clone(cached));
    }

    let alloc_queue = SyclQueue::new_for_device_ordinal(device_id as usize)
        .map_err(|e| anyhow::anyhow!("Failed to create SYCL queue for device {}: {}", device_id, e))?;

    // Build stream pool on the SAME SYCL context as alloc_queue.
    // This ensures USM pointers allocated via alloc_queue are valid on every
    // stream queue (SYCL spec: USM allocations are bound to a context).
    let pool_size: usize = std::env::var("KVBM_SYCL_STREAM_POOL_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4)
        .max(1);

    let sycl_context = alloc_queue.context();
    let stream_pool: Vec<Arc<SyclQueue>> = (0..pool_size)
        .map(|_| {
            SyclQueue::new(sycl_context)
                .map_err(|e| anyhow::anyhow!("Failed to create SYCL stream queue: {}", e))
        })
        .collect::<Result<Vec<_>>>()?;

    tracing::info!(
        "Created SYCL context cache for device {} (stream pool size: {})",
        device_id, pool_size,
    );

    let entry = Arc::new(SyclContextCache {
        device_id,
        alloc_queue,
        stream_pool,
        next_stream: AtomicUsize::new(0),
    });

    cache.insert(device_id, Arc::clone(&entry));
    Ok(entry)
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
// SyclContext — implements DeviceContextOps
// =====================================================================

#[derive(Debug)]
pub struct SyclContext {
    device_id: u32,
    cache: Arc<SyclContextCache>,
}

impl SyclContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let cache = get_or_create_sycl_context(device_id)?;
        Ok(Self { device_id, cache })
    }
}

impl DeviceContextOps for SyclContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self, _hint: EngineHint) -> Result<Box<dyn DeviceStreamOps>> {
        // Return a queue from the bounded pool (round-robin). All pool queues
        // share the same SYCL context as alloc_queue, so USM pointers are valid.
        // EngineHint is ignored — SYCL runtime selects engine automatically.
        let pool = &self.cache.stream_pool;
        let idx = self.cache.next_stream.fetch_add(1, Ordering::Relaxed) % pool.len();
        Ok(Box::new(SyclStreamWrapper {
            queue: Arc::clone(&pool[idx]),
            device_id: self.device_id,
        }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        let ptr = self.cache.alloc_queue
            .malloc_device(size)
            .map_err(|e| anyhow::anyhow!("SYCL device allocation failed ({} bytes): {}", size, e))?;
        let addr = ptr as u64;
        track_alloc(&DEVICE_ALLOCS, addr);
        Ok(addr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        if untrack_alloc(&DEVICE_ALLOCS, ptr) {
            self.cache.alloc_queue
                .free_raw(ptr as *mut c_void)
                .map_err(|e| anyhow::anyhow!("SYCL free_device failed: {}", e))?;
        }
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        let ptr = self.cache.alloc_queue
            .malloc_host(size)
            .map_err(|e| anyhow::anyhow!("SYCL host allocation failed ({} bytes): {}", size, e))?;
        let addr = ptr as u64;
        track_alloc(&HOST_ALLOCS, addr);
        Ok(addr)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        if untrack_alloc(&HOST_ALLOCS, ptr) {
            self.cache.alloc_queue
                .free_raw(ptr as *mut c_void)
                .map_err(|e| anyhow::anyhow!("SYCL free_pinned failed: {}", e))?;
        }
        Ok(())
    }

    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>> {
        let mut builder = dynamo_memory::SyclMemPool::builder(
            Arc::clone(&self.cache.alloc_queue),
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
                kvbm_kernels::xpu_vectorized_copy(
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
pub fn is_available() -> bool {
    SyclDevice::count().map(|n| n > 0).unwrap_or(false)
}
