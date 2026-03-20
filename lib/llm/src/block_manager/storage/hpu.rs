// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HPU storage backend support for Synapse-based transfer paths.

use super::{
    DeviceStorage, StorageError,
    torch::TorchTensor,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use synapse::{
    Context, Device, DeviceBufferView, Event, HostBufferView, Stream,
    copy_device_to_device as synapse_copy_d2d,
    copy_device_to_host_view as synapse_copy_d2h,
    copy_host_view_to_device as synapse_copy_h2d,
};
use tokio::sync::oneshot;

pub fn device_storage_from_torch(_tensor: Arc<dyn TorchTensor>) -> Result<DeviceStorage, StorageError> {
    Err(StorageError::NotAccessible(
        "HPU device storage integration is not implemented yet".to_string(),
    ))
}

pub struct SynapseContext {
    _context: Context,
    device: Device,
    stream: Arc<Stream>,
}

impl SynapseContext {
    fn new(device_id: usize) -> Result<Self, StorageError> {
        let context = Context::new().map_err(|e| {
            StorageError::OperationFailed(format!("Failed to initialize Synapse context: {}", e))
        })?;

        let device = if device_id == 0 {
            Device::acquire_first().map_err(|e| {
                StorageError::OperationFailed(format!("Failed to acquire first Synapse device: {}", e))
            })?
        } else {
            Device::acquire_by_module_id(device_id as u32).map_err(|e| {
                StorageError::OperationFailed(format!(
                    "Failed to acquire Synapse device module {}: {}",
                    device_id, e
                ))
            })?
        };

        let stream = Arc::new(Stream::new(&device).map_err(|e| {
            StorageError::OperationFailed(format!("Failed to create Synapse stream: {}", e))
        })?);

        Ok(Self {
            _context: context,
            device,
            stream,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }
}

#[derive(Debug, Default, Clone)]
pub struct SynapseMemPool;

pub struct Synapse {
    contexts: Mutex<HashMap<usize, Arc<SynapseContext>>>,
}

impl Synapse {
    fn new() -> Self {
        Self {
            contexts: Mutex::new(HashMap::new()),
        }
    }

    pub fn device(device_id: usize) -> Option<Arc<SynapseContext>> {
        Self::instance()
            .contexts
            .lock()
            .unwrap()
            .get(&device_id)
            .cloned()
    }

    pub fn device_or_create(device_id: usize) -> Result<Arc<SynapseContext>, StorageError> {
        Self::instance().get_context(device_id)
    }

    pub fn stream_or_create(device_id: usize) -> Result<Arc<Stream>, StorageError> {
        Ok(Self::device_or_create(device_id)?.stream().clone())
    }

    pub fn is_initialized(device_id: usize) -> bool {
        Self::instance().contexts.lock().unwrap().contains_key(&device_id)
    }

    pub fn context_from_stream(stream: &Arc<Stream>) -> Option<Arc<SynapseContext>> {
        let contexts = Self::instance().contexts.lock().unwrap();
        contexts
            .values()
            .find(|ctx| Arc::ptr_eq(ctx.stream(), stream))
            .cloned()
    }

    fn instance() -> &'static Synapse {
        static INSTANCE: OnceLock<Synapse> = OnceLock::new();
        INSTANCE.get_or_init(Synapse::new)
    }

    fn get_context(&self, device_id: usize) -> Result<Arc<SynapseContext>, StorageError> {
        if let Some(ctx) = self.contexts.lock().unwrap().get(&device_id) {
            return Ok(ctx.clone());
        }

        let created = Arc::new(SynapseContext::new(device_id)?);
        self.contexts
            .lock()
            .unwrap()
            .insert(device_id, created.clone());

        Ok(created)
    }
}

pub fn signal_event(stream: Arc<Stream>, tx: oneshot::Sender<()>) -> Result<(), StorageError> {
    std::thread::spawn(move || {
        let result: Result<(), StorageError> = (|| {
            let ctx = Synapse::context_from_stream(&stream).ok_or_else(|| {
                StorageError::OperationFailed(
                    "Synapse stream is not registered in context cache".to_string(),
                )
            })?;

            let event = Event::new(ctx.device()).map_err(|e| {
                StorageError::OperationFailed(format!("Failed to create Synapse event: {}", e))
            })?;
            event.record(stream.as_ref()).map_err(|e| {
                StorageError::OperationFailed(format!("Failed to record Synapse event: {}", e))
            })?;
            event.synchronize().map_err(|e| {
                StorageError::OperationFailed(format!("Failed to synchronize Synapse event: {}", e))
            })?;

            Ok(())
        })();

        if let Err(e) = result {
            tracing::error!("Synapse event signaling failed: {}", e);
        }

        let _ = tx.send(());
    });

    Ok(())
}

fn validate_copy_ptrs(src_ptr: *const u8, dst_ptr: *mut u8, size: usize, label: &str) -> Result<(), StorageError> {
    if size == 0 {
        return Ok(());
    }
    if src_ptr.is_null() || dst_ptr.is_null() {
        return Err(StorageError::OperationFailed(format!(
            "{} copy encountered null source or destination pointer",
            label
        )));
    }
    Ok(())
}

pub fn copy_host_to_device_raw(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), StorageError> {
    validate_copy_ptrs(src_ptr, dst_ptr, size, "Synapse H2D")?;
    if size == 0 {
        return Ok(());
    }

    let src_view = HostBufferView::from_raw_parts(src_ptr as usize as u64, size);
    let dst_view = DeviceBufferView::from_raw_parts(dst_ptr as usize as u64, size);
    synapse_copy_h2d(stream, &src_view, &dst_view)
        .map_err(|e| StorageError::OperationFailed(format!("Synapse H2D copy failed: {}", e)))
}

pub fn copy_device_to_host_raw(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), StorageError> {
    validate_copy_ptrs(src_ptr, dst_ptr, size, "Synapse D2H")?;
    if size == 0 {
        return Ok(());
    }

    let src_view = DeviceBufferView::from_raw_parts(src_ptr as usize as u64, size);
    let dst_view = HostBufferView::from_raw_parts(dst_ptr as usize as u64, size);
    synapse_copy_d2h(stream, &src_view, &dst_view)
        .map_err(|e| StorageError::OperationFailed(format!("Synapse D2H copy failed: {}", e)))
}

pub fn copy_device_to_device_raw(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), StorageError> {
    validate_copy_ptrs(src_ptr, dst_ptr, size, "Synapse D2D")?;
    if size == 0 {
        return Ok(());
    }

    let src_view = DeviceBufferView::from_raw_parts(src_ptr as usize as u64, size);
    let dst_view = DeviceBufferView::from_raw_parts(dst_ptr as usize as u64, size);
    synapse_copy_d2d(stream, &src_view, &dst_view)
        .map_err(|e| StorageError::OperationFailed(format!("Synapse D2D copy failed: {}", e)))
}
