// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HPU (Synapse) backend implementation
//!
//! Wraps Synapse API types with the device abstraction traits.

use crate::block_manager::v2::device::traits::*;
use anyhow::{Context as _, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use synapse::synapse_sys;
use synapse::{Device, DeviceBufferView, Event, HostBufferView, Stream};

/// Initialize Synapse runtime (called once per process)
/// CRITICAL: The Context must remain alive for the lifetime of the program,
/// as dropping it may invalidate the Synapse runtime state.
fn ensure_synapse_initialized() -> Result<()> {
    static SYNAPSE_CONTEXT: OnceLock<std::result::Result<synapse::Context, String>> =
        OnceLock::new();

    // Use get_or_init to ensure Context::new() is called exactly once with single-flight semantics
    match SYNAPSE_CONTEXT.get_or_init(|| match synapse::Context::new() {
        Ok(ctx) => {
            eprintln!("[HPU] ✓ Synapse runtime initialized successfully");
            Ok(ctx)
        }
        Err(e) => {
            eprintln!("[HPU] ✗ Synapse runtime initialization FAILED: {}", e);
            Err(e.to_string())
        }
    }) {
        Ok(_ctx) => Ok(()), // Context is stored in OnceLock and kept alive for program lifetime
        Err(err) => Err(anyhow::anyhow!("Synapse initialization failed: {}", err)),
    }
}

/// Global cache of acquired HPU devices (one per device_id)
/// The cached Device has owned=true, keeping the device acquired for the program lifetime.
/// We return unowned references to allow multiple HpuContext instances to share it.
fn get_or_acquire_device(device_id: u32) -> Result<Device> {
    // Ensure Synapse runtime is initialized before attempting device operations
    ensure_synapse_initialized()?;

    static DEVICE_CACHE: OnceLock<Mutex<HashMap<u32, Device>>> = OnceLock::new();

    let mut cache = DEVICE_CACHE
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap();

    // Check if device is already acquired and cached
    if cache.contains_key(&device_id) {
        eprintln!(
            "[HPU] Device {} already in cache, returning unowned ref",
            device_id
        );
        // Return unowned reference - the cache holds the owned device
        return Ok(Device::from_id_unowned(device_id));
    }

    eprintln!("[HPU] Device {} NOT in cache, acquiring...", device_id);

    // Try acquiring the device for standalone applications
    // Strategy: Try acquire_first() or acquire_by_module_id and validate the device ID matches
    let device = if device_id == 0 {
        // For device 0, use acquire_first() which is most reliable
        let dev = Device::acquire_first()
            .map_err(|e| anyhow::anyhow!("Failed to acquire HPU device {}: {}", device_id, e))?;

        let actual_id = dev.id();
        if actual_id != device_id {
            anyhow::bail!(
                "Requested HPU device {}, but acquire_first() returned device {}",
                device_id,
                actual_id
            );
        }

        eprintln!("[HPU] Device {} acquired via acquire_first()", actual_id);
        // Cache the owned device
        cache.insert(device_id, dev);
        eprintln!("[HPU] Device {} cached successfully", device_id);
        // Return unowned reference
        Device::from_id_unowned(device_id)
    } else {
        // For non-zero device IDs, try acquire_by_module_id
        let dev = Device::acquire_by_module_id(device_id).map_err(|e| {
            anyhow::anyhow!(
                "Failed to acquire HPU device {} by module_id: {}",
                device_id,
                e
            )
        })?;

        let actual_id = dev.id();
        if actual_id != device_id {
            anyhow::bail!(
                "Requested HPU device {}, but acquire_by_module_id() returned device {}",
                device_id,
                actual_id
            );
        }

        eprintln!("[HPU] Device {} acquired by module_id", device_id);
        cache.insert(device_id, dev);
        Device::from_id_unowned(device_id)
    };

    Ok(device)
}

/// HPU device context wrapping Synapse Device
#[derive(Debug)]
pub struct HpuContext {
    device: Device,
    device_id: u32,
}

impl HpuContext {
    pub fn new(device_id: u32) -> Result<Self> {
        let device = get_or_acquire_device(device_id)?;

        Ok(Self { device, device_id })
    }

    /// Create from an existing Device (for compatibility with existing code)
    pub fn from_device(device: Device, device_id: u32) -> Self {
        Self { device, device_id }
    }

    /// Get the underlying Synapse device
    pub fn inner(&self) -> &Device {
        &self.device
    }
}

impl DeviceContextOps for HpuContext {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn create_stream(&self) -> Result<Box<dyn DeviceStreamOps>> {
        let stream = Stream::new(&self.device).context("Failed to create HPU stream")?;

        Ok(Box::new(HpuStreamWrapper {
            stream: Arc::new(stream),
            device_id: self.device_id,
        }))
    }

    fn allocate_device(&self, size: usize) -> Result<u64> {
        let mut addr: u64 = 0;
        let status = unsafe {
            synapse_sys::runtime::synDeviceMalloc(
                self.device_id,
                size as u64,
                0, // reqAddr (0 = no preference)
                0, // flags
                &mut addr as *mut _,
            )
        };

        if status != synapse_sys::synStatus_synSuccess {
            anyhow::bail!("Synapse synDeviceMalloc failed with status: {:?}", status);
        }

        Ok(addr)
    }

    fn free_device(&self, ptr: u64) -> Result<()> {
        let status = unsafe {
            synapse_sys::runtime::synDeviceFree(
                self.device_id,
                ptr,
                0, // flags
            )
        };

        if status != synapse_sys::synStatus_synSuccess {
            anyhow::bail!("Synapse synDeviceFree failed with status: {:?}", status);
        }

        Ok(())
    }

    fn allocate_pinned(&self, size: usize) -> Result<u64> {
        let mut raw: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe {
            synapse_sys::runtime::synHostMalloc(
                self.device_id,
                size as u64,
                0, // flags (no write-combined support in current Synapse)
                &mut raw as *mut _,
            )
        };

        if status != synapse_sys::synStatus_synSuccess {
            anyhow::bail!("Synapse synHostMalloc failed with status: {:?}", status);
        }

        Ok(raw as u64)
    }

    fn free_pinned(&self, ptr: u64) -> Result<()> {
        let status = unsafe {
            synapse_sys::runtime::synHostFree(
                self.device_id,
                ptr as *const std::ffi::c_void,
                0, // flags
            )
        };

        if status != synapse_sys::synStatus_synSuccess {
            anyhow::bail!("Synapse synHostFree failed with status: {:?}", status);
        }

        Ok(())
    }

    fn bind_to_thread(&self) -> Result<()> {
        // Synapse contexts are automatically bound to threads
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        // Return the raw HPU device ID
        Some(self.device_id as u64)
    }
}

/// HPU stream wrapper
struct HpuStreamWrapper {
    stream: Arc<Stream>,
    device_id: u32, // Stored for creating events
}

impl std::fmt::Debug for HpuStreamWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HpuStreamWrapper")
            .field("stream", &"<Stream>")
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl DeviceStreamOps for HpuStreamWrapper {
    fn copy_h2d(&self, dst_device_ptr: u64, src_host_data: &[u8]) -> Result<()> {
        let src_view = HostBufferView::from_raw_parts(
            src_host_data.as_ptr() as usize as u64,
            src_host_data.len(),
        );
        let dst_view = DeviceBufferView::from_raw_parts(dst_device_ptr, src_host_data.len());

        synapse::copy_host_view_to_device(&self.stream, &src_view, &dst_view)
            .context("HPU H2D copy failed")?;

        Ok(())
    }

    fn copy_d2h(&self, dst_host_data: &mut [u8], src_device_ptr: u64) -> Result<()> {
        let src_view = DeviceBufferView::from_raw_parts(src_device_ptr, dst_host_data.len());
        let dst_view = HostBufferView::from_raw_parts(
            dst_host_data.as_mut_ptr() as usize as u64,
            dst_host_data.len(),
        );

        synapse::copy_device_to_host_view(&self.stream, &src_view, &dst_view)
            .context("HPU D2H copy failed")?;

        Ok(())
    }

    fn copy_d2d(&self, dst_device_ptr: u64, src_device_ptr: u64, size: usize) -> Result<()> {
        let src_view = DeviceBufferView::from_raw_parts(src_device_ptr, size);
        let dst_view = DeviceBufferView::from_raw_parts(dst_device_ptr, size);

        synapse::copy_device_to_device(&self.stream, &src_view, &dst_view)
            .context("HPU D2D copy failed")?;

        Ok(())
    }

    fn record_event(&self) -> Result<Box<dyn DeviceEventOps>> {
        // Create lightweight Device reference (doesn't acquire)
        let device = Device::from_id_unowned(self.device_id);

        // Create and record event (each record_event creates a new event, matching CUDA pattern)
        let event = Event::new(&device).context("Failed to create HPU event")?;
        event
            .record(&self.stream)
            .context("Failed to record HPU event")?;

        Ok(Box::new(HpuEventWrapper { event }))
    }

    fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .context("HPU stream synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        // Return the raw stream handle pointer
        None // Stream handle is opaque and not directly exposable
    }
}

/// HPU event wrapper
struct HpuEventWrapper {
    event: Event,
}

// SAFETY: Synapse event handles are thread-safe when accessed through the API.
// The raw pointer is managed by Synapse runtime and synchronized internally.
unsafe impl Send for HpuEventWrapper {}
unsafe impl Sync for HpuEventWrapper {}

impl std::fmt::Debug for HpuEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HpuEventWrapper")
            .field("event", &"<Event>")
            .finish()
    }
}

impl DeviceEventOps for HpuEventWrapper {
    fn is_complete(&self) -> Result<bool> {
        self.event.query().context("HPU event query failed")
    }

    fn synchronize(&self) -> Result<()> {
        self.event
            .synchronize()
            .context("HPU event synchronization failed")?;
        Ok(())
    }

    fn raw_handle(&self) -> Option<u64> {
        // Event handle is not exposed by synapse crate (Option 1: Honest API)
        None
    }
}

/// Check if HPU backend is available on this system
pub fn is_available() -> bool {
    // First ensure Synapse runtime is initialized
    if ensure_synapse_initialized().is_err() {
        return false;
    }

    // Try to acquire a device
    // This will fail if no HPU hardware is present or driver isn't loaded
    match Device::acquire_first() {
        Ok(_device) => {
            // Device acquired successfully - HPU is available
            // Device will be released when _device is dropped
            true
        }
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hpu_availability() {
        // Just check that the function doesn't panic
        let available = is_available();
        println!("HPU available: {}", available);
    }

    #[test]
    #[cfg(feature = "hpu")]
    fn test_hpu_context_creation() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");
        assert_eq!(ctx.device_id(), 0);
    }

    #[test]
    #[cfg(feature = "hpu")]
    fn test_hpu_stream_creation() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        // Test synchronize
        stream.synchronize().expect("Failed to synchronize stream");
    }

    #[test]
    #[cfg(feature = "hpu")]
    fn test_hpu_event_creation() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");
        let stream = ctx.create_stream().expect("Failed to create stream");

        let event = stream.record_event().expect("Failed to record event");

        // Event should complete quickly for an empty stream
        event.synchronize().expect("Failed to synchronize event");
        assert!(event.is_complete().expect("Failed to query event"));
    }

    #[test]
    #[cfg(feature = "hpu")]
    fn test_hpu_h2d_d2h() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");
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
    #[cfg(feature = "hpu")]
    fn test_hpu_d2d() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");
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
    #[cfg(feature = "hpu")]
    fn test_pinned_memory() {
        if !is_available() {
            println!("Skipping test - no HPU devices available");
            return;
        }

        let ctx = HpuContext::new(0).expect("Failed to create HPU context");

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
