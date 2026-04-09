// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! XPU (Level-Zero) backend implementation
//!
//! Wraps Level-Zero API types with the device abstraction traits.

use crate::block_manager::v2::device::traits::*;
use anyhow::Result;
use level_zero::{self as ze, ZE_EVENT_SCOPE_FLAG_HOST};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// Global initialization state for Level-Zero runtime
fn ensure_ze_initialized() -> Result<()> {
    static INIT_RESULT: OnceLock<std::result::Result<(), String>> = OnceLock::new();

    // Use get_or_init to ensure ze::init() is called exactly once
    match INIT_RESULT.get_or_init(|| match ze::init() {
        Ok(()) => {
            eprintln!("[XPU] ✓ Level-Zero runtime initialized successfully");
            Ok(())
        }
        Err(e) => {
            eprintln!("[XPU] ✗ Level-Zero runtime initialization FAILED: {:?}", e);
            Err(format!("{:?}", e))
        }
    }) {
        Ok(()) => Ok(()),
        Err(err) => Err(anyhow::anyhow!(
            "Failed to initialize Level-Zero runtime: {}",
            err
        )),
    }
}

/// Device context cache entry
struct DeviceContextCache {
    /// Driver handle kept alive for RAII - ensures driver remains valid for context lifetime
    _driver: ze::Driver,
    device: ze::Device,
    context: Arc<ze::Context>,
    event_pool: Arc<ze::EventPool>,
}

// Level-Zero API is thread-safe by specification, so we can safely mark these as Send+Sync
// even though they contain raw pointers internally
unsafe impl Send for DeviceContextCache {}
unsafe impl Sync for DeviceContextCache {}

impl std::fmt::Debug for DeviceContextCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceContextCache")
            .field("_driver", &"<ze::Driver>")
            .field("device", &"<ze::Device>")
            .field("context", &"<Arc<ze::Context>>")
            .field("event_pool", &"<Arc<ze::EventPool>>")
            .finish()
    }
}

/// Global cache of Level-Zero contexts (one per device_id)
fn get_or_create_ze_context(device_id: u32) -> Result<Arc<DeviceContextCache>> {
    ensure_ze_initialized()?;

    static CONTEXT_CACHE: OnceLock<Mutex<HashMap<u32, Arc<DeviceContextCache>>>> = OnceLock::new();

    let mut cache = CONTEXT_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();

    if let Some(cached) = cache.get(&device_id) {
        eprintln!("[XPU] Device {} context found in cache", device_id);
        return Ok(Arc::clone(cached));
    }

    eprintln!("[XPU] Creating new context for device {}", device_id);

    // Get driver and devices
    let drivers =
        ze::drivers().map_err(|e| anyhow::anyhow!("Failed to get Level-Zero drivers: {:?}", e))?;

    let driver = drivers
        .get(0)
        .ok_or_else(|| anyhow::anyhow!("No Level-Zero drivers found"))?;

    let devices = driver
        .devices()
        .map_err(|e| anyhow::anyhow!("Failed to enumerate devices: {:?}", e))?;

    let device = devices
        .get(device_id as usize)
        .ok_or_else(|| anyhow::anyhow!("Device {} not found", device_id))?;

    // Create context
    let context = ze::Context::create(driver)
        .map_err(|e| anyhow::anyhow!("Failed to create Level-Zero context: {:?}", e))?;

    // Create event pool with 1024 events (similar to common CUDA patterns)
    let event_pool = context
        .create_event_pool(&[device.clone()], 1024, ZE_EVENT_SCOPE_FLAG_HOST)
        .map_err(|e| anyhow::anyhow!("Failed to create event pool: {:?}", e))?;

    let cache_entry = Arc::new(DeviceContextCache {
        _driver: driver.clone(),
        device: device.clone(),
        context: Arc::new(context),
        event_pool: Arc::new(event_pool),
    });

    cache.insert(device_id, Arc::clone(&cache_entry));
    eprintln!("[XPU] Device {} context created and cached", device_id);

    Ok(cache_entry)
}

/// XPU device context wrapping Level-Zero Context and Device
#[derive(Debug)]
pub struct ZeContext {
    device_id: u32,
    cache: Arc<DeviceContextCache>,
}

// ZeContext wraps thread-safe Arc<DeviceContextCache>
unsafe impl Send for ZeContext {}
unsafe impl Sync for ZeContext {}

impl ZeContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let cache = get_or_create_ze_context(device_id)?;

        Ok(Self { device_id, cache })
    }

    /// Get the underlying Level-Zero context
    pub fn inner(&self) -> &ze::Context {
        &self.cache.context
    }

    /// Get the device
    pub fn device(&self) -> &ze::Device {
        &self.cache.device
    }
}

impl DeviceContextOps for ZeContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        // Use immediate command list for stream-like behavior
        let cmd_list = self
            .cache
            .context
            .create_immediate_command_list(&self.cache.device.clone())
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
            .alloc_device(&self.cache.device.clone(), size, 1)
            .map_err(|e| anyhow::anyhow!("Level-Zero device allocation failed: {:?}", e))?;

        // Get pointer before moving buffer
        let ptr = buffer.as_mut_ptr() as u64;

        // Store buffer in a global map so it doesn't get dropped prematurely
        store_device_buffer(ptr, buffer, Arc::clone(&self.cache.context));

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

        // Get pointer before moving buffer
        let ptr = buffer.as_mut_ptr() as u64;

        // Store buffer in a global map so it doesn't get dropped prematurely
        store_host_buffer(ptr, buffer, Arc::clone(&self.cache.context));

        Ok(ptr)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        remove_host_buffer(ptr);
        Ok(())
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Level-Zero contexts are automatically thread-aware
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        Some(self.device_id as u64)
    }
}

/// Wrapper for ze::DeviceBuffer to add Send+Sync
/// Fields kept alive for RAII - prevents premature deallocation
struct SendSyncDeviceBuffer(
    #[allow(dead_code)] ze::DeviceBuffer,
    #[allow(dead_code)] Arc<ze::Context>,
);
unsafe impl Send for SendSyncDeviceBuffer {}
unsafe impl Sync for SendSyncDeviceBuffer {}

/// Global storage for device buffers (prevents premature Drop)
static DEVICE_BUFFERS: OnceLock<Mutex<HashMap<u64, SendSyncDeviceBuffer>>> = OnceLock::new();

fn store_device_buffer(ptr: u64, buffer: ze::DeviceBuffer, ctx: Arc<ze::Context>) {
    let mut map = DEVICE_BUFFERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    map.insert(ptr, SendSyncDeviceBuffer(buffer, ctx));
}

fn remove_device_buffer(ptr: u64) {
    if let Some(map_lock) = DEVICE_BUFFERS.get() {
        let mut map = map_lock.lock().unwrap();
        map.remove(&ptr);
    }
}

/// Wrapper for ze::HostBuffer to add Send+Sync
/// Fields kept alive for RAII - prevents premature deallocation
struct SendSyncHostBuffer(
    #[allow(dead_code)] ze::HostBuffer,
    #[allow(dead_code)] Arc<ze::Context>,
);
unsafe impl Send for SendSyncHostBuffer {}
unsafe impl Sync for SendSyncHostBuffer {}

/// Global storage for host buffers (prevents premature Drop)
static HOST_BUFFERS: OnceLock<Mutex<HashMap<u64, SendSyncHostBuffer>>> = OnceLock::new();

fn store_host_buffer(ptr: u64, buffer: ze::HostBuffer, ctx: Arc<ze::Context>) {
    let mut map = HOST_BUFFERS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();
    map.insert(ptr, SendSyncHostBuffer(buffer, ctx));
}

fn remove_host_buffer(ptr: u64) {
    if let Some(map_lock) = HOST_BUFFERS.get() {
        let mut map = map_lock.lock().unwrap();
        map.remove(&ptr);
    }
}

/// XPU stream wrapper using ImmediateCommandList
struct ZeStreamWrapper {
    cmd_list: Arc<ze::ImmediateCommandList>,
    event_pool: Arc<ze::EventPool>,
    device_id: u32,
}

// Level-Zero command lists are thread-safe by specification
unsafe impl Send for ZeStreamWrapper {}
unsafe impl Sync for ZeStreamWrapper {}

impl std::fmt::Debug for ZeStreamWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeStreamWrapper")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl DeviceStreamOps for ZeStreamWrapper {
    fn copy_h2d(&self, dst_device_ptr: u64, src_host_data: &[u8]) -> Result<()> {
        self.cmd_list
            .append_memcpy(
                dst_device_ptr as *mut std::ffi::c_void,
                src_host_data.as_ptr() as *const std::ffi::c_void,
                src_host_data.len(),
            )
            .map_err(|e| anyhow::anyhow!("XPU H2D copy failed: {:?}", e))?;

        Ok(())
    }

    fn copy_d2h(&self, dst_host_data: &mut [u8], src_device_ptr: u64) -> Result<()> {
        self.cmd_list
            .append_memcpy(
                dst_host_data.as_mut_ptr() as *mut std::ffi::c_void,
                src_device_ptr as *const std::ffi::c_void,
                dst_host_data.len(),
            )
            .map_err(|e| anyhow::anyhow!("XPU D2H copy failed: {:?}", e))?;

        Ok(())
    }

    fn copy_d2d(&self, dst_device_ptr: u64, src_device_ptr: u64, size: usize) -> Result<()> {
        self.cmd_list
            .append_memcpy(
                dst_device_ptr as *mut std::ffi::c_void,
                src_device_ptr as *const std::ffi::c_void,
                size,
            )
            .map_err(|e| anyhow::anyhow!("XPU D2D copy failed: {:?}", e))?;

        Ok(())
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        // Use a single global counter to allocate event indices across all streams
        // This prevents collisions when multiple streams share the same event pool
        static GLOBAL_EVENT_COUNTER: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(0);

        // Allocate a new event index from the pool (wraps at 1024)
        let event_idx =
            GLOBAL_EVENT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst) % 1024;

        let event = self
            .event_pool
            .create_event(
                event_idx,
                ZE_EVENT_SCOPE_FLAG_HOST,
                ZE_EVENT_SCOPE_FLAG_HOST,
            )
            .map_err(|e| anyhow::anyhow!("Failed to create event: {:?}", e))?;

        // Reset event first to ensure it's in a clean state
        event
            .host_reset()
            .map_err(|e| anyhow::anyhow!("Failed to reset event: {:?}", e))?;

        // Signal the event on the command list
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

    fn raw_handle(&self) -> Option<u64> {
        None // Command list handle is not directly exposable
    }
}

/// XPU event wrapper
struct ZeEventWrapper {
    event: ze::Event,
}

// SAFETY: Level-Zero event handles are thread-safe when accessed through the API.
unsafe impl Send for ZeEventWrapper {}
unsafe impl Sync for ZeEventWrapper {}

impl std::fmt::Debug for ZeEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeEventWrapper")
            .field("event", &"<Event>")
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

    fn raw_handle(&self) -> Option<u64> {
        None // Event handle is not exposed
    }
}

/// Check if XPU backend is available on this system
pub fn is_available() -> bool {
    // Try to initialize Level-Zero and enumerate devices
    if ze::init().is_err() {
        return false;
    }

    match ze::drivers() {
        Ok(drivers) => {
            if drivers.is_empty() {
                return false;
            }
            // Check if first driver has any devices
            if let Some(driver) = drivers.get(0) {
                if let Ok(devices) = driver.devices() {
                    return !devices.is_empty();
                }
            }
            false
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xpu_availability() {
        let available = is_available();
        println!("XPU available: {}", available);
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_xpu_context_creation() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");
        assert_eq!(ctx.device_id(), 0);
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_xpu_stream_creation() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        // Test synchronize
        stream.synchronize().expect("Failed to synchronize stream");
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_xpu_event_creation() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        let event = stream.record_event().expect("Failed to record event");

        // Event should complete quickly for an empty stream
        event.synchronize().expect("Failed to synchronize event");
        assert!(event.is_complete().expect("Failed to query event"));
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_xpu_h2d_d2h() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        // Allocate device memory
        let size = 1024;
        let dev_ptr = ctx
            .allocate_device(size)
            .expect("Failed to allocate device memory");

        // Prepare host data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut host_result = vec![0u8; size];

        // Copy H2D
        stream
            .copy_h2d(dev_ptr, &host_data)
            .expect("H2D copy failed");

        // Copy D2H
        stream
            .copy_d2h(&mut host_result, dev_ptr)
            .expect("D2H copy failed");

        // Synchronize
        stream.synchronize().expect("Failed to synchronize");

        // Verify data
        assert_eq!(
            host_data, host_result,
            "Data mismatch after H2D->D2H roundtrip"
        );

        // Cleanup
        ctx.free_device(dev_ptr)
            .expect("Failed to free device memory");
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_xpu_d2d() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        let size = 1024;
        let dev_ptr1 = ctx
            .allocate_device(size)
            .expect("Failed to allocate device memory 1");
        let dev_ptr2 = ctx
            .allocate_device(size)
            .expect("Failed to allocate device memory 2");

        // Prepare host data
        let host_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let mut host_result = vec![0u8; size];

        // Copy H2D to first buffer
        stream
            .copy_h2d(dev_ptr1, &host_data)
            .expect("H2D copy failed");

        // Copy D2D
        stream
            .copy_d2d(dev_ptr2, dev_ptr1, size)
            .expect("D2D copy failed");

        // Copy D2H from second buffer
        stream
            .copy_d2h(&mut host_result, dev_ptr2)
            .expect("D2H copy failed");

        // Synchronize
        stream.synchronize().expect("Failed to synchronize");

        // Verify data
        assert_eq!(host_data, host_result, "Data mismatch after H2D->D2D->D2H");

        // Cleanup
        ctx.free_device(dev_ptr1)
            .expect("Failed to free device memory 1");
        ctx.free_device(dev_ptr2)
            .expect("Failed to free device memory 2");
    }

    #[test]
    #[cfg(feature = "xpu")]
    fn test_pinned_memory() {
        if !is_available() {
            println!("Skipping test - no XPU devices available");
            return;
        }

        let ctx = ZeContext::new(0).expect("Failed to create XPU context");

        let size = 4096;
        let pinned_ptr = ctx
            .allocate_pinned(size)
            .expect("Failed to allocate pinned memory");

        // Verify we can write to it
        unsafe {
            let slice = std::slice::from_raw_parts_mut(pinned_ptr as *mut u8, size);
            slice.fill(42);
            assert_eq!(slice[0], 42);
        }

        ctx.free_pinned(pinned_ptr)
            .expect("Failed to free pinned memory");
    }
}
