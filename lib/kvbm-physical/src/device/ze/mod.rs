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


/// Embedded SPIR-V binary for the vectorized_copy kernel.
/// Compiled offline from kvbm-kernels/opencl/vectorized_copy.cl:
///   ocloc compile -file vectorized_copy.cl -spv_only -options "-cl-std=CL2.0"
/// The resulting .spv is embedded at compile time — no runtime path lookup.
static VECTORIZED_COPY_SPIRV: &[u8] =
    include_bytes!("../../../../kvbm-kernels/opencl/vectorized_copy.spv");

const COPY_KERNEL_NAME: &str = "vectorized_copy";
const COPY_WG_SIZE: u32 = 128;
const COPY_MAX_WGS: u32 = 65535;


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
/// Event pool wrapper with a free-list for safe, reusable slot allocation.
///
/// Level-Zero event pools are fixed-size, index-based. Each `create_event(index)`
/// occupies a slot; `zeEventDestroy` frees the slot.  This wrapper tracks available
/// indices in a `Mutex<Vec<u32>>` so that:
///   - A slot is never handed out while still in use (no aliasing).
///   - Destroyed events return their index to the free-list via `ZeEventWrapper::Drop`.
///   - Pool exhaustion surfaces as an explicit error instead of silent corruption.
struct SharedEventPool {
    pool: ze::EventPool,
    free_indices: Mutex<Vec<u32>>,
}

impl SharedEventPool {
    fn new(pool: ze::EventPool, pool_size: u32) -> Self {
        Self {
            pool,
            free_indices: Mutex::new((0..pool_size).collect()),
        }
    }

    /// Allocate a unique event from the pool, taking the next free index.
    ///
    /// Returns `(event, index)` so the caller can return `index` on drop.
    fn create_event(
        &self,
        signal_scope: u32,
        wait_scope: u32,
    ) -> Result<(ze::Event, u32)> {
        let index = self
            .free_indices
            .lock()
            .expect("SharedEventPool lock poisoned")
            .pop()
            .ok_or_else(|| anyhow::anyhow!("Event pool exhausted — all slots in use"))?;
        let event = self
            .pool
            .create_event(index, signal_scope, wait_scope)
            .map_err(|e| anyhow::anyhow!("zeEventCreate failed for index {}: {:?}", index, e))?;
        Ok((event, index))
    }

    /// Return a slot index to the free-list.
    fn return_index(&self, index: u32) {
        self.free_indices
            .lock()
            .expect("SharedEventPool lock poisoned")
            .push(index);
    }
}

// SAFETY: EventPool is Send+Sync, Mutex<Vec<u32>> is Send+Sync.
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
    /// Compiled SPIR-V module for vectorized_copy kernel (None if .spv invalid).
    copy_module: Option<ze::Module>,
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

    let drivers = ze::drivers()
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
        ze::Context::create(driver)
            .map_err(|e| anyhow::anyhow!("Failed to create Level-Zero context: {:?}", e))?,
    );

    const EVENT_POOL_SIZE: u32 = 8192;
    let raw_pool = context
        .create_event_pool(&[device.clone()], EVENT_POOL_SIZE, ZE_EVENT_SCOPE_FLAG_HOST)
        .map_err(|e| anyhow::anyhow!("Failed to create event pool: {:?}", e))?;
    let event_pool = Arc::new(SharedEventPool::new(raw_pool, EVENT_POOL_SIZE));

    // Probe queue group ordinals for engine selection.
    let compute_ordinal = context.compute_queue_ordinal(device)?;
    let copy_ordinal = context.copy_queue_ordinal(device)
        .unwrap_or(compute_ordinal);

    // Try to load the vectorized_copy SPIR-V module. Falls back gracefully
    // if the .spv is a placeholder or invalid (e.g. dev machine without SYCL compiler).
    let copy_module = match context.create_module_from_spirv(device, VECTORIZED_COPY_SPIRV, None) {
        Ok(module) => {
            tracing::info!("Loaded vectorized_copy SPIR-V module for device {}", device_id);
            Some(module)
        }
        Err(e) => {
            tracing::warn!(
                "Failed to load vectorized_copy SPIR-V for device {}: {:?}. Using host-readback fallback.",
                device_id, e
            );
            None
        }
    };


    let cache_entry = Arc::new(DeviceContextCache {
        _driver: driver.clone(),
        device: device.clone(),
        context,
        event_pool,
        compute_ordinal,
        copy_ordinal,
        copy_module,
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

        // Create a per-stream kernel instance (avoids set_arg races between streams).
        let copy_kernel = self.cache.copy_module.as_ref().and_then(|module| {
            match module.create_kernel(COPY_KERNEL_NAME) {
                Ok(k) => {
                    if let Err(e) = k.set_indirect_access(
                        ze::KERNEL_INDIRECT_ACCESS_FLAG_HOST
                        | ze::KERNEL_INDIRECT_ACCESS_FLAG_DEVICE,
                    ) {
                        tracing::warn!("Failed to set indirect access on copy kernel: {:?}", e);
                    }
                    Some(k)
                }
                Err(e) => {
                    tracing::warn!("Failed to create copy kernel: {:?}", e);
                    None
                }
            }
        });

        Ok(Box::new(ZeStreamWrapper {
            cmd_list: Arc::new(cmd_list),
            event_pool: Arc::clone(&self.cache.event_pool),
            device_id: self.device_id,
            copy_kernel,
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
///
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
        // Drain completed pending frees so their buffers can be reused.
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
        let event = match stream.record_event() {
            Ok(event) => event,
            Err(e) => {
                // Keep ownership tracking intact on event-record failure so callers
                // can retry free_async without losing the buffer from the pool map.
                self.buffers
                    .lock()
                    .map_err(|lock_err| anyhow::anyhow!("buffer map poisoned: {}", lock_err))?
                    .insert(ptr, (buffer, size));
                return Err(e);
            }
        };

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
    pub cmd_list: Arc<ze::ImmediateCommandList>,
    event_pool: Arc<SharedEventPool>,
    device_id: u32,
    /// Per-stream kernel instance for vectorized_copy (None = fallback to host readback).
    copy_kernel: Option<ze::Kernel>,
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

    fn memcpy_dtoh(&self, src_device: u64, dst_host: &mut [u8]) -> Result<()> {
        self.cmd_list
            .append_memcpy(
                dst_host.as_mut_ptr() as *mut std::ffi::c_void,
                src_device as *const std::ffi::c_void,
                dst_host.len(),
            )
            .map_err(|e| anyhow::anyhow!("XPU memcpy_dtoh (append_memcpy) failed: {:?}", e))?;
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


        // SPIR-V path: launch GPU kernel from pre-compiled SPIR-V.
        if let Some(ref kernel) = self.copy_kernel {
            let num_pairs = count as i32;
            let grid_dim = std::cmp::min(count as u32, COPY_MAX_WGS);

            kernel.set_group_size(COPY_WG_SIZE, 1, 1)
                .map_err(|e| anyhow::anyhow!("set_group_size failed: {:?}", e))?;
            kernel.set_arg(0, &src_ptrs_device)
                .map_err(|e| anyhow::anyhow!("set_arg(0) failed: {:?}", e))?;
            kernel.set_arg(1, &dst_ptrs_device)
                .map_err(|e| anyhow::anyhow!("set_arg(1) failed: {:?}", e))?;
            kernel.set_arg(2, &chunk_size)
                .map_err(|e| anyhow::anyhow!("set_arg(2) failed: {:?}", e))?;
            kernel.set_arg(3, &num_pairs)
                .map_err(|e| anyhow::anyhow!("set_arg(3) failed: {:?}", e))?;

            self.cmd_list
                .append_launch_kernel(kernel, ze::GroupCount { x: grid_dim, y: 1, z: 1 })
                .map_err(|e| anyhow::anyhow!("append_launch_kernel failed: {:?}", e))?;

            return Ok(());
        }

        // Fallback: read pointer arrays back to host via DMA and use batch_copy.
        // Used when SPIR-V module is unavailable (placeholder .spv or load failure).

        // Synchronize to ensure the H2D uploads of pointer arrays have completed.
        self.cmd_list
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU sync before vectorized_copy readback failed: {:?}", e))?;

        let mut src_ptrs = vec![0u64; count];
        let mut dst_ptrs = vec![0u64; count];
        let byte_len = count * std::mem::size_of::<u64>();

        // Copy pointer arrays from device to host via append_memcpy (safe on discrete GPUs).
        self.cmd_list
            .append_memcpy(
                src_ptrs.as_mut_ptr() as *mut std::ffi::c_void,
                src_ptrs_device as *const std::ffi::c_void,
                byte_len,
            )
            .map_err(|e| anyhow::anyhow!("XPU readback src_ptrs failed: {:?}", e))?;
        self.cmd_list
            .append_memcpy(
                dst_ptrs.as_mut_ptr() as *mut std::ffi::c_void,
                dst_ptrs_device as *const std::ffi::c_void,
                byte_len,
            )
            .map_err(|e| anyhow::anyhow!("XPU readback dst_ptrs failed: {:?}", e))?;

        // Wait for readback to complete before accessing host buffers.
        self.cmd_list
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU sync after vectorized_copy readback failed: {:?}", e))?;

        self.batch_copy(&src_ptrs, &dst_ptrs, chunk_size)
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let (event, index) = self
            .event_pool
            .create_event(ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST)?;

        event
            .host_reset()
            .map_err(|e| anyhow::anyhow!("Failed to reset event: {:?}", e))?;

        self.cmd_list
            .append_signal_event(&event)
            .map_err(|e| anyhow::anyhow!("Failed to signal event: {:?}", e))?;

        Ok(Box::new(ZeEventWrapper {
            event: std::mem::ManuallyDrop::new(event),
            index,
            pool: Arc::clone(&self.event_pool),
        }))
    }

    fn synchronize(&self) -> Result<()> {
        self.cmd_list
            .host_synchronize(u64::MAX)
            .map_err(|e| anyhow::anyhow!("XPU stream synchronization failed: {:?}", e))?;
        Ok(())
    }
}

/// XPU event wrapper.
///
/// On drop, destroys the Level-Zero event first (freeing the pool slot in the
/// driver), then returns the index to the free-list so another caller can reuse it.
/// Using `ManuallyDrop` ensures the event is destroyed before the index is recycled,
/// preventing another thread from calling `zeEventCreate` on a slot whose old event
/// still exists.
pub struct ZeEventWrapper {
    event: std::mem::ManuallyDrop<ze::Event>,
    index: u32,
    pool: Arc<SharedEventPool>,
}

impl ZeEventWrapper {
    /// Access the underlying event (e.g. for query / synchronize).
    pub fn event(&self) -> &ze::Event {
        &self.event
    }
}

impl Drop for ZeEventWrapper {
    fn drop(&mut self) {
        // SAFETY: `event` is valid and has not been dropped yet (ManuallyDrop
        // guarantees this is the only drop site). Destroying the event calls
        // zeEventDestroy, which releases the pool slot in the driver.
        unsafe { std::mem::ManuallyDrop::drop(&mut self.event); }
        // Now the slot is free in the driver — safe to recycle the index.
        self.pool.return_index(self.index);
    }
}

unsafe impl Send for ZeEventWrapper {}
unsafe impl Sync for ZeEventWrapper {}

impl std::fmt::Debug for ZeEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeEventWrapper")
            .field("index", &self.index)
            .finish()
    }
}

impl DeviceEventOps for ZeEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        self.event
            .is_signaled()
            .map_err(|e| anyhow::anyhow!("XPU event query failed: {:?}", e))
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
