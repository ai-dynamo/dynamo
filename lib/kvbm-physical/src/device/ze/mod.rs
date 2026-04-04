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
struct DeviceContextCache {
    _driver: ze::Driver,
    device: ze::Device,
    context: Arc<ze::Context>,
    event_pool: Arc<ze::EventPool>,
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

    let event_pool = Arc::new(
        context
            .create_event_pool(&[*device], 1024, ZE_EVENT_SCOPE_FLAG_HOST)
            .map_err(|e| anyhow::anyhow!("Failed to create event pool: {:?}", e))?,
    );

    let cache_entry = Arc::new(DeviceContextCache {
        _driver: driver.clone(),
        device: device.clone(),
        context,
        event_pool,
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

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        let cmd_list = self
            .cache
            .context
            .create_immediate_command_list(&self.cache.device)
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
        }))
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.device_id as u64)
    }
}

/// Memory pool wrapper for Level-Zero using `dynamo_memory::ZeMemPool`.
///
/// Bridges the `DeviceMemPoolOps` trait (which uses raw `u64` pointers) with
/// the `ZeMemPool` API (which uses RAII `DeviceBuffer` for ownership).
/// An internal `HashMap` tracks `DeviceBuffer` ownership by pointer so that
/// buffers stay alive until explicitly freed back to the pool.
struct ZeMemPoolWrapper {
    pool: dynamo_memory::ZeMemPool,
    /// Maps device pointer -> (DeviceBuffer, size) for ownership tracking.
    /// Entries are inserted on alloc and removed on free (returning the
    /// buffer to the pool's free-list).
    buffers: Mutex<HashMap<u64, (ze::DeviceBuffer, usize)>>,
}

impl std::fmt::Debug for ZeMemPoolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeMemPoolWrapper")
            .field("pool", &self.pool)
            .finish()
    }
}

impl DeviceMemPoolOps for ZeMemPoolWrapper {
    fn alloc_async(&self, size: usize, _stream: &dyn DeviceStreamOps) -> Result<u64> {
        let (buffer, ptr) = self.pool.alloc(size)?;
        self.buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .insert(ptr, (buffer, size));
        Ok(ptr)
    }

    fn free_async(&self, ptr: u64, _stream: &dyn DeviceStreamOps) -> Result<()> {
        let (buffer, size) = self
            .buffers
            .lock()
            .map_err(|e| anyhow::anyhow!("buffer map poisoned: {}", e))?
            .remove(&ptr)
            .ok_or_else(|| anyhow::anyhow!("ZeMemPoolWrapper: ptr {:#x} not found in buffer map", ptr))?;
        self.pool.free(buffer, ptr, size)?;
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
    event_pool: Arc<ze::EventPool>,
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

    // vectorized_copy: uses default fallback to batch_copy.
    // Future: override with SYCL JIT kernel for GPU-parallel copy.

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        let event = self
            .event_pool
            .create_event(Default::default(), Default::default())
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
