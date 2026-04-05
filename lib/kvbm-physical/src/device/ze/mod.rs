// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! XPU (Level-Zero) backend implementation.
//!
//! Wraps Level-Zero API types with the device abstraction traits.

use crate::device::traits::*;
use anyhow::Result;
use level_zero::{self as ze, ZE_EVENT_SCOPE_FLAG_HOST};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, Ordering};

/// Global initialization state for Level-Zero runtime.
fn ensure_ze_initialized() -> Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    let result = INIT.get_or_init(|| {
        ze::init().map_err(|e| format!("Level-Zero initialization failed: {:?}", e))
    });
    match result {
        Ok(()) => Ok(()),
        Err(msg) => Err(anyhow::anyhow!("{}", msg)),
    }
}

/// Device context cache entry.
/// Event pool wrapper with atomic index counter for safe slot allocation.
///
/// Level-Zero event pools are fixed-size, index-based. Each `create_event(index)`
/// occupies a slot; `zeEventDestroy` frees the slot. This wrapper assigns monotonically
/// increasing indices (mod pool_size) so concurrent callers never alias the same slot.
///
/// With 8192 slots and ~4 events per transfer, up to ~2048 concurrent in-flight
/// transfers are supported before indices could theoretically wrap into a live slot.
/// Events are short-lived (destroyed after synchronize or drain), so this is safe.
struct SharedEventPool {
    pool: ze::EventPool,
    next_index: AtomicU32,
    pool_size: u32,
}

impl SharedEventPool {
    fn new(pool: ze::EventPool, pool_size: u32) -> Self {
        Self {
            pool,
            next_index: AtomicU32::new(0),
            pool_size,
        }
    }

    /// Allocate a unique event from the pool with the next available index.
    fn create_event(
        &self,
        signal_scope: u32,
        wait_scope: u32,
    ) -> ze::Result<ze::Event> {
        let index = self.next_index.fetch_add(1, Ordering::Relaxed) % self.pool_size;
        self.pool.create_event(index, signal_scope, wait_scope)
    }
}

// SAFETY: EventPool is Send+Sync, AtomicU32 is Send+Sync.
unsafe impl Send for SharedEventPool {}
unsafe impl Sync for SharedEventPool {}

struct DeviceContextCache {
    _driver: ze::Driver,
    device: ze::Device,
    context: Arc<ze::Context>,
    event_pool: Arc<SharedEventPool>,
    /// Queue group ordinal for compute engine (CCS).
    compute_ordinal: u32,
    /// Queue group ordinal for copy engine (BCS); equals compute_ordinal if no dedicated BCS.
    copy_ordinal: u32,
}

unsafe impl Send for DeviceContextCache {}
unsafe impl Sync for DeviceContextCache {}

impl std::fmt::Debug for DeviceContextCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContextCache")
            .field("device", &"<ze::Device>")
            .field("context", &"<Arc<ze::Context>>")
            .finish()
    }
}

/// Global cache of Level-Zero contexts (one per device_id).
fn get_or_create_ze_context(device_id: u32) -> Result<Arc<DeviceContextCache>> {
    ensure_ze_initialized()?;

    static CONTEXT_CACHE: OnceLock<Mutex<HashMap<u32, Arc<DeviceContextCache>>>> = OnceLock::new();

    let mut cache = CONTEXT_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();

    if let Some(cached) = cache.get(&device_id) {
        return Ok(Arc::clone(cached));
    }

    let drivers = ze::Driver::get()
        .map_err(|e| anyhow::anyhow!("Failed to get Level-Zero drivers: {:?}", e))?;

    let driver = drivers
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("No Level-Zero drivers found"))?;

    let devices = driver
        .devices()
        .map_err(|e| anyhow::anyhow!("Failed to enumerate devices: {:?}", e))?;

    let device = devices
        .get(device_id as usize)
        .ok_or_else(|| anyhow::anyhow!("Device {} not found ({} available)", device_id, devices.len()))?;

    let context = Arc::new(
        ze::Context::new(driver)
            .map_err(|e| anyhow::anyhow!("Failed to create Level-Zero context: {:?}", e))?,
    );

    const EVENT_POOL_SIZE: u32 = 8192;
    let raw_pool = context
        .create_event_pool(&[*device], EVENT_POOL_SIZE, ZE_EVENT_SCOPE_FLAG_HOST)
        .map_err(|e| anyhow::anyhow!("Failed to create event pool: {:?}", e))?;
    let event_pool = Arc::new(SharedEventPool::new(raw_pool, EVENT_POOL_SIZE));

    // Probe queue group ordinals for engine selection.
    let compute_ordinal = context.compute_queue_ordinal(device)?;
    let copy_ordinal = context.copy_queue_ordinal(device)
        .unwrap_or(compute_ordinal);

    let cache_entry = Arc::new(DeviceContextCache {
        _driver: driver.clone(),
        device: device.clone(),
        context,
        event_pool,
        compute_ordinal,
        copy_ordinal,
    });

    cache.insert(device_id, Arc::clone(&cache_entry));
    Ok(cache_entry)
}

/// XPU device context wrapping Level-Zero Context and Device.
#[derive(Debug)]
pub struct ZeContext {
    device_id: u32,
    cache: Arc<DeviceContextCache>,
}

unsafe impl Send for ZeContext {}
unsafe impl Sync for ZeContext {}

impl ZeContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let cache = get_or_create_ze_context(device_id)?;
        Ok(Self { device_id, cache })
    }

    pub fn inner(&self) -> &ze::Context {
        &self.cache.context
    }

    pub fn device(&self) -> &ze::Device {
        &self.cache.device
    }
}

impl DeviceContextOps for ZeContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self, hint: EngineHint) -> Result<Box<dyn DeviceStreamOps>> {
        let ordinal = match hint {
            EngineHint::Copy => self.cache.copy_ordinal,
            EngineHint::Compute => self.cache.compute_ordinal,
        };
        let cmd_list = self
            .cache
            .context
            .create_immediate_command_list_with_ordinal(&self.cache.device, ordinal)
            .map_err(|e| anyhow::anyhow!("Failed to create immediate command list: {:?}", e))?;

        Ok(Box::new(ZeStreamWrapper {
            cmd_list: Arc::new(cmd_list),
            event_pool: Arc::clone(&self.cache.event_pool),
            device_id: self.device_id,
        }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        let buffer = self
            .cache
            .context
            .alloc_device(&self.cache.device, size, 1)
            .map_err(|e| anyhow::anyhow!("Level-Zero device allocation failed: {:?}", e))?;
        let ptr = buffer.as_mut_ptr() as u64;
        store_device_buffer(ptr, buffer);
        Ok(ptr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        remove_device_buffer(ptr);
        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        let buffer = self
            .cache
            .context
            .alloc_host(size, 1)
            .map_err(|e| anyhow::anyhow!("Level-Zero host allocation failed: {:?}", e))?;
        let ptr = buffer.as_mut_ptr() as u64;
        store_host_buffer(ptr, buffer);
        Ok(ptr)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        remove_host_buffer(ptr);
        Ok(())
    }

    fn create_memory_pool(
        &self,
        reserve_size: usize,
        release_threshold: Option<u64>,
    ) -> Result<Box<dyn DeviceMemPoolOps>> {
        let mut builder = dynamo_memory::ZeMemPool::builder(
            Arc::clone(&self.cache.context),
            self.cache.device.clone(),
            reserve_size,
        );
        if let Some(threshold) = release_threshold {
            builder = builder.release_threshold(threshold);
        }
        let pool = builder.build()?;
        Ok(Box::new(ZeMemPoolWrapper {
            pool,
            buffers: Mutex::new(HashMap::new()),
            pending_frees: Mutex::new(Vec::new()),
        }))
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.device_id as u64)
    }
}

/// A deferred free: buffer removed from active map but not yet returned to pool.
/// Held until the associated event signals that all prior stream work (including
/// any kernel reading this memory) has completed.
struct PendingFree {
    buffer: ze::DeviceBuffer,
    ptr: u64,
    size: usize,
    event: Box<dyn DeviceEventOps>,
}

/// Memory pool wrapper for Level-Zero using `dynamo_memory::ZeMemPool`.
///
/// Bridges the `DeviceMemPoolOps` trait (which uses raw `u64` pointers) with
/// the `ZeMemPool` API (which uses RAII `DeviceBuffer` for ownership).
/// An internal `HashMap` tracks `DeviceBuffer` ownership by pointer so that
/// buffers stay alive until explicitly freed back to the pool.
///
/// Unlike CUDA's `cuMemFreeAsync` (which is natively stream-ordered), Level-Zero
/// has no stream-ordered free API. `free_async` records an event on the stream
/// and defers the actual return-to-pool until the event signals completion.
/// Completed pending frees are drained opportunistically on each `alloc_async`.
struct ZeMemPoolWrapper {
    pool: dynamo_memory::ZeMemPool,
    /// Maps device pointer -> (DeviceBuffer, size) for ownership tracking.
    /// Entries are inserted on alloc and removed on free.
    buffers: Mutex<HashMap<u64, (ze::DeviceBuffer, usize)>>,
    /// Deferred frees waiting for GPU work to complete before returning to pool.
    pending_frees: Mutex<Vec<PendingFree>>,
}

impl std::fmt::Debug for ZeMemPoolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeMemPoolWrapper")
            .field("pool", &self.pool)
            .finish()
    }
}

impl ZeMemPoolWrapper {
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
                self.pool.free(pf.buffer, pf.ptr, pf.size)?;
                // Don't increment i — swap_remove moved the last element here
            } else {
                i += 1;
            }
        }
        Ok(())
    }
}

impl Drop for ZeMemPoolWrapper {
    fn drop(&mut self) {
        // Synchronize all pending frees so buffers are returned to the pool
        // (and not just dropped via zeMemFree, which would leak pool accounting).
        if let Ok(mut pending) = self.pending_frees.lock() {
            for pf in pending.drain(..) {
                // Best-effort: wait for event, then return to pool
                let _ = pf.event.synchronize();
                let _ = self.pool.free(pf.buffer, pf.ptr, pf.size);
            }
        }
    }
}

impl DeviceMemPoolOps for ZeMemPoolWrapper {
    fn alloc_async(&self, size: usize, _stream: &dyn DeviceStreamOps) -> Result<u64> {
        // Reclaim any completed pending frees before allocating, so the pool's
        // free-list is up to date and can satisfy this request without new memory.
        self.drain_completed_frees()?;
        let (buffer, ptr) = self.pool.alloc(size)?;
        self.buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .insert(ptr, (buffer, size));
        Ok(ptr)
    }

    fn free_async(&self, ptr: u64, stream: &dyn DeviceStreamOps) -> Result<()> {
        // Remove the buffer from the active map.
        let (buffer, size) = self
            .buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .remove(&ptr)
            .ok_or_else(|| anyhow::anyhow!("ZeMemPoolWrapper: ptr {:#x} not found in buffer map", ptr))?;

        // Record an event on the stream: it will signal after all prior commands
        // (including any kernel reading this memory) have completed.
        let event = stream.record_event()?;

        // Defer the actual return-to-pool until the event signals.
        self.pending_frees
            .lock()
            .map_err(|e| anyhow::anyhow!("pending_frees lock poisoned: {}", e))?
            .push(PendingFree { buffer, ptr, size, event });

        Ok(())
    }
}

// Buffer storage to prevent premature drop.
// level-zero-rc uses RAII: DeviceBuffer/HostBuffer call zeMemFree on Drop.
// We store them in global maps keyed by pointer, removing = drop = zeMemFree.

static DEVICE_BUFFERS: OnceLock<Mutex<HashMap<u64, ze::DeviceBuffer>>> = OnceLock::new();
static HOST_BUFFERS: OnceLock<Mutex<HashMap<u64, ze::HostBuffer>>> = OnceLock::new();

fn store_device_buffer(ptr: u64, buffer: ze::DeviceBuffer) {
    DEVICE_BUFFERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .insert(ptr, buffer);
}

fn remove_device_buffer(ptr: u64) {
    if let Some(map) = DEVICE_BUFFERS.get() {
        map.lock().unwrap().remove(&ptr);
    }
}

fn store_host_buffer(ptr: u64, buffer: ze::HostBuffer) {
    HOST_BUFFERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap()
        .insert(ptr, buffer);
}

fn remove_host_buffer(ptr: u64) {
    if let Some(map) = HOST_BUFFERS.get() {
        map.lock().unwrap().remove(&ptr);
    }
}

/// XPU stream wrapper (immediate command list).
pub struct ZeStreamWrapper {
    pub cmd_list: Arc<ze::CommandList>,
    event_pool: Arc<SharedEventPool>,
    device_id: u32,
}

impl std::fmt::Debug for ZeStreamWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeStreamWrapper")
            .field("device_id", &self.device_id)
            .finish()
    }
}

unsafe impl Send for ZeStreamWrapper {}
unsafe impl Sync for ZeStreamWrapper {}

impl DeviceStreamOps for ZeStreamWrapper {
    fn batch_copy(&self, src_ptrs: &[u64], dst_ptrs: &[u64], size: usize) -> Result<()> {
        assert_eq!(src_ptrs.len(), dst_ptrs.len(), "batch_copy: src/dst length mismatch");

        for (&src, &dst) in src_ptrs.iter().zip(dst_ptrs.iter()) {
            self.cmd_list
                .append_memcpy(
                    dst as *mut std::ffi::c_void,
                    src as *const std::ffi::c_void,
                    size,
                )
                .map_err(|e| anyhow::anyhow!("XPU batch_copy (append_memcpy) failed: {:?}", e))?;
        }

        Ok(())
    }

    fn memcpy_htod(&self, dst_device: u64, src_host: &[u8]) -> Result<()> {
        self.cmd_list
            .append_memcpy(
                dst_device as *mut std::ffi::c_void,
                src_host.as_ptr() as *const std::ffi::c_void,
                src_host.len(),
            )
            .map_err(|e| anyhow::anyhow!("XPU memcpy_htod (append_memcpy) failed: {:?}", e))?;
        Ok(())
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

        // TODO: Replace with SYCL JIT kernel for GPU-parallel copy.
        // For now, read pointer arrays back to host and use batch_copy.
        let ptr_bytes = count * std::mem::size_of::<u64>();

        let mut src_ptrs = vec![0u64; count];
        let mut dst_ptrs = vec![0u64; count];

        // Synchronize to ensure the H2D uploads of pointer arrays have completed.
        // the host_synchronize at line 344 inside vectorized_copy is actually redundant given that Step 4 already waited 
        // via upload_event.synchronize(). The uploads are guaranteed done before vectorized_copy is even called. 
        self.cmd_list
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU sync before vectorized_copy readback failed: {:?}", e))?;

        // Read pointer arrays from device back to host.
        // SAFETY: src_ptrs_device/dst_ptrs_device point to count * 8 bytes of device memory.
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_ptrs_device as *const u64,
                src_ptrs.as_mut_ptr(),
                count,
            );
            std::ptr::copy_nonoverlapping(
                dst_ptrs_device as *const u64,
                dst_ptrs.as_mut_ptr(),
                count,
            );
        }

        // Fall back to batch_copy with the recovered host pointers.
        self.batch_copy(&src_ptrs, &dst_ptrs, chunk_size)
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let event = self
            .event_pool
            .create_event(ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST)
            .map_err(|e| anyhow::anyhow!("Failed to create event: {:?}", e))?;

        event
            .host_reset()
            .map_err(|e| anyhow::anyhow!("Failed to reset event: {:?}", e))?;

        self.cmd_list
            .append_signal_event(&event)
            .map_err(|e| anyhow::anyhow!("Failed to signal event: {:?}", e))?;

        Ok(Box::new(ZeEventWrapper { event }))
    }

    fn synchronize(&self) -> Result<()> {
        self.cmd_list
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU stream synchronization failed: {:?}", e))?;
        Ok(())
    }
}

/// XPU event wrapper.
pub struct ZeEventWrapper {
    pub event: ze::Event,
}

unsafe impl Send for ZeEventWrapper {}
unsafe impl Sync for ZeEventWrapper {}

impl std::fmt::Debug for ZeEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeEventWrapper").finish()
    }
}

impl DeviceEventOps for ZeEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        match self.event.query_status() {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.event
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU event synchronization failed: {:?}", e))?;
        Ok(())
    }
}

/// Check if Level-Zero / XPU is available.
pub fn is_available() -> bool {
    ensure_ze_initialized().is_ok()
}
