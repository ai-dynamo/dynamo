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

pub use level_zero::{Event as ZeEvent, EventPool as ZeEventPool, ZE_EVENT_SCOPE_FLAG_HOST, sync_sycl_event, ze_context_handle_t, get_ze_context_from_sycl_queue};
use level_zero::{self, CommandList, CommandQueue, Context, Device, Driver, EventPool};
use std::{
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
        let Some(driver) = drivers.first().cloned() else {
            return Err(ZeError::NoDriver);
        };

        let devices = driver.devices().map_err(ZeError::from)?;
        if devices.is_empty() {
            return Err(ZeError::NoDevice);
        }
        let Some(device) = devices.get(device_index).cloned() else {
            return Err(ZeError::InvalidDeviceIndex(device_index));
        };

        let context = Arc::new(Context::create(&driver).map_err(ZeError::from)?);
        let queue = ZeCommandQueue::new(context.clone(), device.clone())?;
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
        ZeCommandQueue::new(self.context.clone(), self.device.clone())
    }

    /// Create a new command queue using a foreign context handle from SYCL.
    /// This allows sharing command queues between Level Zero and SYCL runtimes.
    pub fn new_stream_with_foreign_context(&self, foreign_ctx: ze_context_handle_t) -> Result<Arc<ZeCommandQueue>, ZeError> {
        ZeCommandQueue::new_with_foreign_context(self.context.clone(), self.device.clone(), foreign_ctx)
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
    foreign_src_context: Option<ze_context_handle_t>,
}

// SAFETY:
// Command queue handles are externally synchronized by the Level Zero runtime;
// this wrapper only forwards API calls and stores opaque handles.
unsafe impl Send for ZeCommandQueue {}
// SAFETY: See Send rationale above.
unsafe impl Sync for ZeCommandQueue {}

impl ZeCommandQueue {
    fn new(context: Arc<Context>, device: Device) -> Result<Arc<Self>, ZeError> {
        Self::new_inner(context, device, None)
    }

    fn new_with_foreign_context(context: Arc<Context>, device: Device, foreign_ctx: ze_context_handle_t) -> Result<Arc<Self>, ZeError> {
        // Create a Context wrapper around PyTorch's context handle.
        // Use PyTorch's context for the command queue and command list so that
        // device memory allocated by PyTorch is directly accessible.
        let foreign_context = unsafe { context.from_raw_handle(foreign_ctx) };
        let foreign_context = Arc::new(foreign_context);
        let handle = foreign_context
            .create_copy_command_queue(&device)
            .map_err(ZeError::from)?;
        let event_pool = foreign_context
            .create_event_pool(&[device.clone()], 1, 0)
            .map_err(ZeError::from)?;
        Ok(Arc::new(Self {
            handle,
            context: foreign_context,
            device,
            event_pool: Arc::new(event_pool),
            foreign_src_context: Some(foreign_ctx),
        }))
    }

    fn new_inner(context: Arc<Context>, device: Device, foreign_src_context: Option<ze_context_handle_t>) -> Result<Arc<Self>, ZeError> {
        let handle = context
            .create_copy_command_queue(&device)
            .map_err(ZeError::from)?;
        let event_pool = context
            .create_event_pool(&[device.clone()], 1, 0)
            .map_err(ZeError::from)?;
        Ok(Arc::new(Self {
            handle,
            context,
            device,
            event_pool: Arc::new(event_pool),
            foreign_src_context,
        }))
    }

    /// Execute a command list without blocking.
    pub fn execute_nonblocking(&self, list: &mut CommandList) -> Result<(), ZeError> {
        self.handle.execute_nonblocking(list).map_err(ZeError::from)
    }

    /// Create a new command list for this device.
    pub fn create_command_list(&self) -> Result<CommandList, ZeError> {
        self.context
            .create_copy_command_list(&self.device)
            .map_err(ZeError::from)
    }

    /// Get the Level Zero context.
    pub fn context(&self) -> &Arc<Context> {
        &self.context
    }

    /// Get the Level Zero device.
    pub fn device(&self) -> Device {
        self.device.clone()
    }

    /// Get the event pool.
    pub fn event_pool(&self) -> &Arc<EventPool> {
        &self.event_pool
    }

    /// Get the foreign source context handle for cross-context copies.
    pub fn foreign_src_context(&self) -> Option<ze_context_handle_t> {
        self.foreign_src_context
    }
}

// ---------------------------------------------------------------------------
// Host memory helpers
// ---------------------------------------------------------------------------

static ZE_USE_WRITE_COMBINED: OnceLock<bool> = OnceLock::new();

fn ze_use_write_combined(context: &Context) -> bool {
    *ZE_USE_WRITE_COMBINED.get_or_init(|| {
        if dynamo_config::env_is_truthy("DYN_KVBM_DISABLE_WRITE_COMBINED") {
            tracing::debug!("DYN_KVBM_DISABLE_WRITE_COMBINED set; Ze write-combined disabled");
            return false;
        }
        match context.alloc_host_wc(1, 64) {
            Ok(_buf) => {
                tracing::debug!("Ze write-combined USM host memory supported");
                true
            }
            Err(_) => {
                tracing::debug!(
                    "Ze write-combined memory not supported on this system; \
                     will use regular USM host memory"
                );
                false
            }
        }
    })
}

pub(crate) unsafe fn malloc_host_prefer_writecombined_ze(
    context: &Context,
    size: usize,
) -> Result<*mut u8, StorageError> {
    let host_buf = if ze_use_write_combined(context) {
        context.alloc_host_wc(size, 64)
    } else {
        context.alloc_host(size, 64)
    }
    .map_err(|e| {
        StorageError::AllocationFailed(format!(
            "ZE USM host allocation failed for {} bytes: {:?}",
            size, e
        ))
    })?;

    let ptr = host_buf.as_mut_ptr() as *mut u8;
    std::mem::forget(host_buf);

    tracing::debug!(
        "Allocated ZE USM host memory at 0x{:x} (size={})",
        ptr as usize,
        size
    );
    Ok(ptr)
}

pub(crate) unsafe fn free_host_ze(
    context: &Context,
    ptr: *mut u8,
) -> Result<(), StorageError> {
    unsafe {
        context
            .free_memory(ptr as *mut std::ffi::c_void)
            .map_err(|e| {
                StorageError::OperationFailed(format!(
                    "ZE host free failed for ptr 0x{:x}: {:?}",
                    ptr as usize, e
                ))
            })
    }
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
        unsafe { malloc_host_prefer_writecombined_ze(&self.context, size) }
    }

    unsafe fn free_pinned(&self, ptr: u64, _size: usize) -> Result<(), StorageError> {
        unsafe { free_host_ze(&self.context, ptr as *mut u8) }
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
