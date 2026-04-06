// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Level Zero (Ze) device context and memory management.
//!
//! Provides [`ZeContext`] for Intel GPU device management, including:
//! - Device and driver initialization
//! - Command queue and event pool creation
//! - Pinned host and device memory allocation via [`StorageBackendOps`]
//! - Singleton [`Ze`] for managing per-device contexts

use crate::{StorageError, pinned::StorageBackendOps};

pub use level_zero::{Event as ZeEvent, EventPool as ZeEventPool, ZE_EVENT_SCOPE_FLAG_HOST};
use level_zero::{self, CommandList, CommandQueue, Context, Device, Driver, EventPool};
use std::{
    alloc::{Layout, alloc, dealloc},
    collections::HashMap,
    sync::{Arc, Mutex, OnceLock},
};

// ---------------------------------------------------------------------------
// ZeError
// ---------------------------------------------------------------------------

/// Errors specific to Level Zero operations.
#[derive(Debug, thiserror::Error)]
pub enum ZeError {
    /// Error from the Level Zero backend.
    #[error("Level Zero backend error: {0:?}")]
    Backend(level_zero::Error),
    /// No Level Zero driver found.
    #[error("No Level Zero driver found")]
    NoDriver,
    /// No Level Zero device found.
    #[error("No Level Zero device found")]
    NoDevice,
    /// Invalid device index.
    #[error("Invalid device index: {0}")]
    InvalidDeviceIndex(usize),
}

impl From<level_zero::Error> for ZeError {
    fn from(value: level_zero::Error) -> Self {
        Self::Backend(value)
    }
}

// ---------------------------------------------------------------------------
// ZeContext
// ---------------------------------------------------------------------------

/// Full Level Zero device context with driver, device, and command queue.
pub struct ZeContext {
    /// Ze device ordinal.
    pub device_id: usize,
    /// Level Zero driver handle.
    pub driver: Driver,
    /// Level Zero device handle.
    pub device: Device,
    /// Level Zero context handle.
    pub context: Arc<Context>,
    /// Default command queue for this device.
    queue: Arc<ZeCommandQueue>,
}

// SAFETY:
// Level Zero handles are opaque driver-managed resources intended to be passed
// across threads. Dynamo shares a single context/queue wrapper behind Arc for
// transfer workers; no Rust aliasing guarantees are violated by moving/sharing
// these handle containers between threads.
unsafe impl Send for ZeContext {}
// SAFETY: See Send rationale above.
unsafe impl Sync for ZeContext {}

impl std::fmt::Debug for ZeContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZeContext")
            .field("device_id", &self.device_id)
            .finish_non_exhaustive()
    }
}

impl ZeContext {
    /// Initialize a new Ze context for the given device index.
    pub fn new(device_index: usize) -> Result<Arc<Self>, ZeError> {
        level_zero::init().map_err(ZeError::from)?;

        let drivers = level_zero::drivers().map_err(ZeError::from)?;
        let Some(driver) = drivers.first().copied() else {
            return Err(ZeError::NoDriver);
        };

        let devices = driver.devices().map_err(ZeError::from)?;
        if devices.is_empty() {
            return Err(ZeError::NoDevice);
        }
        let Some(device) = devices.get(device_index).copied() else {
            return Err(ZeError::InvalidDeviceIndex(device_index));
        };

        let context = Arc::new(Context::create(&driver).map_err(ZeError::from)?);
        let queue = ZeCommandQueue::new(context.clone(), device)?;
        Ok(Arc::new(Self {
            device_id: device_index,
            driver,
            device,
            context,
            queue,
        }))
    }

    /// Get the default command queue for this context.
    pub fn command_queue(&self) -> Arc<ZeCommandQueue> {
        self.queue.clone()
    }

    /// Create a new command queue (analogous to CudaContext::new_stream).
    pub fn new_stream(&self) -> Result<Arc<ZeCommandQueue>, ZeError> {
        ZeCommandQueue::new(self.context.clone(), self.device)
    }
}

// ---------------------------------------------------------------------------
// ZeCommandQueue
// ---------------------------------------------------------------------------

/// Wrapper around a Level Zero command queue with associated context and events.
pub struct ZeCommandQueue {
    handle: CommandQueue,
    context: Arc<Context>,
    device: Device,
    event_pool: Arc<EventPool>,
}

// SAFETY:
// Command queue handles are externally synchronized by the Level Zero runtime;
// this wrapper only forwards API calls and stores opaque handles.
unsafe impl Send for ZeCommandQueue {}
// SAFETY: See Send rationale above.
unsafe impl Sync for ZeCommandQueue {}

impl ZeCommandQueue {
    fn new(context: Arc<Context>, device: Device) -> Result<Arc<Self>, ZeError> {
        let handle = context
            .create_command_queue(&device)
            .map_err(ZeError::from)?;
        let event_pool = context
            .create_event_pool(&[device], 1, 0)
            .map_err(ZeError::from)?;
        Ok(Arc::new(Self {
            handle,
            context,
            device,
            event_pool: Arc::new(event_pool),
        }))
    }

    /// Execute a command list without blocking.
    pub fn execute_nonblocking(&self, list: &mut CommandList) -> Result<(), ZeError> {
        self.handle.execute_nonblocking(list).map_err(ZeError::from)
    }

    /// Create a new command list for this device.
    pub fn create_command_list(&self) -> Result<CommandList, ZeError> {
        self.context
            .create_command_list(&self.device)
            .map_err(ZeError::from)
    }

    /// Get the Level Zero context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Get the Level Zero device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get the event pool.
    pub fn event_pool(&self) -> &Arc<EventPool> {
        &self.event_pool
    }
}

// ---------------------------------------------------------------------------
// Host memory helpers
// ---------------------------------------------------------------------------

fn ze_host_layout(size: usize) -> Result<Layout, StorageError> {
    Layout::from_size_align(size.max(1), 64).map_err(|e| {
        StorageError::AllocationFailed(format!("Invalid ZE host allocation layout: {}", e))
    })
}

pub(crate) unsafe fn malloc_host_prefer_writecombined_ze(
    size: usize,
) -> Result<*mut u8, StorageError> {
    let layout = ze_host_layout(size)?;
    // SAFETY: `layout` is validated by `ze_host_layout`.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        return Err(StorageError::AllocationFailed(format!(
            "ZE host allocation failed for {} bytes",
            size
        )));
    }
    tracing::debug!(
        "Allocated ZE host memory at 0x{:x} (size={})",
        ptr as usize,
        size
    );
    Ok(ptr)
}

pub(crate) unsafe fn free_host_ze(ptr: *mut u8, size: usize) -> Result<(), StorageError> {
    let layout = ze_host_layout(size)?;
    // SAFETY: `layout` is validated by `ze_host_layout`, and caller guarantees
    // that `ptr` was allocated with this layout.
    unsafe { dealloc(ptr, layout) };
    Ok(())
}

// ---------------------------------------------------------------------------
// Ze singleton
// ---------------------------------------------------------------------------

/// Singleton manager for Level Zero device contexts.
pub struct Ze {
    contexts: Mutex<HashMap<usize, Arc<ZeContext>>>,
}

impl Ze {
    fn new() -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
        }
    }

    /// Check if Level Zero is available on this system.
    /// Returns true only if the loader library exists.
    pub fn is_available() -> bool {
        std::fs::metadata("/usr/lib/x86_64-linux-gnu/libze_loader.so.1")
            .or_else(|_| std::fs::metadata("/usr/lib/x86_64-linux-gnu/libze_loader.so"))
            .or_else(|_| std::fs::metadata("/usr/local/lib/libze_loader.so.1"))
            .or_else(|_| std::fs::metadata("/usr/local/lib/libze_loader.so"))
            .is_ok()
    }

    /// Get an existing ZE context for a specific device.
    pub fn device(device_id: usize) -> Option<Arc<ZeContext>> {
        Ze::instance().get_existing_context(device_id)
    }

    /// Get or lazily create a ZE context for a specific device.
    pub fn device_or_create(device_id: usize) -> Result<Arc<ZeContext>, StorageError> {
        Ze::instance().get_context(device_id)
    }

    /// Check if a ZE context exists for a specific device.
    pub fn is_initialized(device_id: usize) -> bool {
        Ze::instance().has_context(device_id)
    }

    fn instance() -> &'static Ze {
        static INSTANCE: OnceLock<Ze> = OnceLock::new();
        INSTANCE.get_or_init(Ze::new)
    }

    fn get_context(&self, device_id: usize) -> Result<Arc<ZeContext>, StorageError> {
        if let Some(ctx) = self.contexts.lock().unwrap().get(&device_id) {
            return Ok(ctx.clone());
        }

        let ctx = ZeContext::new(device_id)
            .map_err(|e| StorageError::OperationFailed(format!("ZE context error: {:?}", e)))?;

        self.contexts.lock().unwrap().insert(device_id, ctx.clone());

        Ok(ctx)
    }

    /// Get a context if it exists, but don't create one.
    pub fn get_existing_context(&self, device_id: usize) -> Option<Arc<ZeContext>> {
        self.contexts.lock().unwrap().get(&device_id).cloned()
    }

    /// Check if a context exists for a device.
    pub fn has_context(&self, device_id: usize) -> bool {
        self.contexts.lock().unwrap().contains_key(&device_id)
    }
}

// ---------------------------------------------------------------------------
// StorageBackendOps for Arc<ZeContext>
// ---------------------------------------------------------------------------

impl StorageBackendOps for Arc<ZeContext> {
    unsafe fn alloc_pinned(&self, size: usize) -> Result<*mut u8, StorageError> {
        unsafe { malloc_host_prefer_writecombined_ze(size) }
    }

    unsafe fn free_pinned(&self, ptr: u64, size: usize) -> Result<(), StorageError> {
        unsafe { free_host_ze(ptr as *mut u8, size) }
    }

    unsafe fn alloc_device(
        &self,
        size: usize,
    ) -> Result<(u64, u32, Option<Box<dyn std::any::Any + Send + Sync>>), StorageError> {
        let ze_device_buffer = self
            .context
            .alloc_device(&self.device, size, 1)
            .map_err(|e| {
                StorageError::OperationFailed(format!(
                    "ZE alloc_device failed for {} bytes: {:?}",
                    size, e
                ))
            })?;

        let ptr = ze_device_buffer.as_mut_ptr() as u64;
        Ok((
            ptr,
            self.device_id as u32,
            Some(Box::new(ze_device_buffer)),
        ))
    }

    unsafe fn free_device(&self, _ptr: u64) -> Result<(), StorageError> {
        // Memory freed by DeviceBuffer's Drop implementation
        // which is stored as the type-erased metadata box.
        Ok(())
    }

    fn device_id(&self) -> u32 {
        self.device_id as u32
    }

    fn backend_type(&self) -> crate::pinned::BackendType {
        crate::pinned::BackendType::Ze
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn new_from_torch(
        &self,
        desc: &crate::pinned::TorchTensorDescriptor,
    ) -> crate::Result<(u64, usize)> {
        if !desc.is_xpu {
            return Err(crate::StorageError::OperationFailed(
                "Tensor is not an XPU/ZE tensor!".into(),
            ));
        }
        Ok((desc.data_ptr, desc.size_bytes))
    }
}
