// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use super::context::{DeviceMemPool, DeviceStream, PinnedBuffer, TransferBackend};
use super::TransferError;

use dynamo_memory::{ZeCommandQueue, ZeEvent, ZeEventPool};
use dynamo_memory::ze::ZE_EVENT_SCOPE_FLAG_HOST;
use dynamo_runtime::utils::pool::SyncPoolItem;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;
use std::thread::JoinHandle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

// ============================================================================
// Ze transfer backend
// ============================================================================

#[derive(Debug, Default)]
pub struct ZeMemPool;

pub(super) struct TransferBackendZe {
    queue: Arc<ZeCommandQueue>,
    ze_mem_pool: Option<Arc<ZeMemPool>>,
    ze_event_tx: mpsc::UnboundedSender<(Arc<ZeEventPool>, ZeEvent, oneshot::Sender<()>)>,
    ze_event_worker: Option<JoinHandle<()>>,
    cancel_token: CancellationToken,
}

impl TransferBackendZe {
    pub fn new(queue: Arc<ZeCommandQueue>, _config: Option<&super::context::PoolConfig>) -> Self {
        let (ze_event_tx, ze_event_rx) =
            mpsc::unbounded_channel::<(Arc<ZeEventPool>, ZeEvent, oneshot::Sender<()>)>();

        let cancel_token = CancellationToken::new();
        let ze_event_worker = Self::setup_ze_event_worker(ze_event_rx, cancel_token.clone());

        Self {
            queue,
            ze_mem_pool: None,
            ze_event_tx,
            ze_event_worker: Some(ze_event_worker),
            cancel_token,
        }
    }

    fn setup_ze_event_worker(
        mut ze_event_rx: mpsc::UnboundedReceiver<(Arc<ZeEventPool>, ZeEvent, oneshot::Sender<()>)>,
        cancel_token: CancellationToken,
    ) -> JoinHandle<()> {
        std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to build Tokio runtime for Ze event worker.");

            runtime.block_on(async move {
                loop {
                    tokio::select! {
                        Some((_pool, event, tx)) = ze_event_rx.recv() => {
                            if let Err(e) = event.host_synchronize(u64::MAX) {
                                tracing::error!("Error synchronizing Ze event: {:?}", e);
                            }
                            let _ = tx.send(());
                        }
                        _ = cancel_token.cancelled() => {
                            break;
                        }
                    }
                }
            });
        })
    }
}

impl TransferBackend for TransferBackendZe {
    fn device_mem_pool(&self) -> Option<DeviceMemPool> {
        None
    }

    fn device_event(&self, tx: oneshot::Sender<()>) -> Result<(), TransferError> {
        let event_pool = self.queue.event_pool().clone();
        let event = event_pool
            .create_event(0, ZE_EVENT_SCOPE_FLAG_HOST, ZE_EVENT_SCOPE_FLAG_HOST)
            .map_err(|e| {
                TransferError::ExecutionError(format!("Ze event creation failed: {:?}", e))
            })?;

        // Create a command list that signals the event, then execute it.
        // Without this, the event is never signaled and host_synchronize hangs.
        let mut list = self.queue.create_command_list().map_err(|e| {
            TransferError::ExecutionError(format!("Ze command list creation failed: {:?}", e))
        })?;
        list.append_signal_event(&event).map_err(|e| {
            TransferError::ExecutionError(format!("Ze append_signal_event failed: {:?}", e))
        })?;
        list.close().map_err(|e| {
            TransferError::ExecutionError(format!("Ze command list close failed: {:?}", e))
        })?;
        self.queue.execute_nonblocking(&mut list).map_err(|e| {
            TransferError::ExecutionError(format!("Ze command list execute failed: {:?}", e))
        })?;

        self.ze_event_tx
            .send((event_pool, event, tx))
            .map_err(|_| TransferError::ExecutionError("Ze event worker exited.".into()))?;
        Ok(())
    }

    fn acquire_resources_for_transfer_sync(
        &self,
        _size: usize,
    ) -> Result<SyncPoolItem<PinnedBuffer>, TransferError> {
        Err(TransferError::ExecutionError(
            "Ze pinned buffer pool not yet implemented".into(),
        ))
    }

    fn device_stream(&self) -> DeviceStream {
        DeviceStream::Ze(self.queue.clone())
    }

    fn shutdown(&mut self) {
        self.cancel_token.cancel();
        if let Some(handle) = self.ze_event_worker.take()
            && let Err(e) = handle.join()
        {
            tracing::error!("Error joining Ze event worker: {:?}", e);
        }
    }
}

// ============================================================================
// Ze copy operations
// ============================================================================

type ZeMemcpyFnPtr = fn(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    queue: &ZeCommandQueue,
) -> Result<(), TransferError>;

fn ze_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<ZeMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::AsyncH2D | TransferStrategy::BlockingH2D => Ok(ze_memcpy_h2d),
        TransferStrategy::AsyncD2H | TransferStrategy::BlockingD2H => Ok(ze_memcpy_d2h),
        _ => Err(TransferError::ExecutionError(format!(
            "Unsupported ZE copy strategy: {:?}",
            strategy
        ))),
    }
}

/// Copy a block using ZE memcpy primitives.
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    queue: &ZeCommandQueue,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = ze_memcpy_fn_ptr(&strategy)?;

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            tracing::debug!(
                "ZE copy_block contiguous: strategy={:?}, src=0x{:x}, dst=0x{:x}, size={}",
                strategy,
                src_view.as_ptr() as usize,
                dst_view.as_mut_ptr() as usize,
                src_view.size()
            );
            memcpy_fn(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                queue,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            queue,
            strategy,
        )?;
    }

    Ok(())
}

/// Temporary ZE implementation for API parity with CUDA custom kernel path.
///
/// This currently falls back to batched ZE memcpys with a single command list.
pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    queue: &ZeCommandQueue,
    _ctx: &crate::block_manager::block::transfer::TransferContext,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if sources.len() != destinations.len() {
        return Err(TransferError::CountMismatch(
            sources.len(),
            destinations.len(),
        ));
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!(
            "ZE custom batch command list creation failed: {:?}",
            e
        ))
    })?;

    for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
        let src_data = src.block_data();
        let dst_data = dst.block_data_mut();

        if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
            let src_view = src_data.block_view()?;
            let mut dst_view = dst_data.block_view_mut()?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            if src_view.size() == 0 {
                continue;
            }

            unsafe {
                tracing::debug!(
                    "ZE copy_blocks_with_customized_kernel contiguous: src=0x{:x}, dst=0x{:x}, size={}",
                    src_view.as_ptr() as usize,
                    dst_view.as_mut_ptr() as usize,
                    src_view.size()
                );
                list.append_memcpy(
                    dst_view.as_mut_ptr() as *mut c_void,
                    src_view.as_ptr() as *const c_void,
                    src_view.size(),
                )
                .map_err(|e| {
                    TransferError::ExecutionError(format!(
                        "ZE custom batch append_memcpy (contiguous) failed: {:?}",
                        e
                    ))
                })?;
            }
        } else {
            assert_eq!(src_data.num_layers(), dst_data.num_layers());
            assert_eq!(src_data.num_outer_dims(), dst_data.num_outer_dims());

            for layer_idx in 0..src_data.num_layers() {
                for outer_idx in 0..src_data.num_outer_dims() {
                    let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                    let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                    debug_assert_eq!(src_view.size(), dst_view.size());

                    if src_view.size() == 0 {
                        continue;
                    }

                    unsafe {
                        tracing::debug!(
                            "ZE copy_blocks_with_customized_kernel layered: layer={}, outer={}, src=0x{:x}, dst=0x{:x}, size={}",
                            layer_idx,
                            outer_idx,
                            src_view.as_ptr() as usize,
                            dst_view.as_mut_ptr() as usize,
                            src_view.size()
                        );
                        list.append_memcpy(
                            dst_view.as_mut_ptr() as *mut c_void,
                            src_view.as_ptr() as *const c_void,
                            src_view.size(),
                        )
                        .map_err(|e| {
                            TransferError::ExecutionError(format!(
                                "ZE custom batch append_memcpy (layered) failed: {:?}",
                                e
                            ))
                        })?;
                    }
                }
            }
        }
    }

    list.close().map_err(|e| {
        TransferError::ExecutionError(format!("ZE custom batch list close failed: {:?}", e))
    })?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE custom batch queue submit failed: {:?}", e))
    })?;

    Ok(())
}

/// Copy a range of layers from source to destination using ZE memcpy.
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    queue: &ZeCommandQueue,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = ze_memcpy_fn_ptr(&strategy)?;

    assert_eq!(src_data.num_outer_dims(), dst_data.num_outer_dims());

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());
            unsafe {
                tracing::debug!(
                    "ZE copy_layers: strategy={:?}, layer={}, outer={}, src=0x{:x}, dst=0x{:x}, size={}",
                    strategy,
                    layer_idx,
                    outer_idx,
                    src_view.as_ptr() as usize,
                    dst_view.as_mut_ptr() as usize,
                    src_view.size()
                );
                memcpy_fn(
                    src_view.as_ptr(),
                    dst_view.as_mut_ptr(),
                    src_view.size(),
                    queue,
                )?;
            }
        }
    }

    Ok(())
}

#[inline(always)]
fn ze_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    queue: &ZeCommandQueue,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source ZE device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");

    if size == 0 {
        return Ok(());
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!("ZE D2H command list creation failed: {:?}", e))
    })?;
    list.append_memcpy(dst_ptr as *mut c_void, src_ptr as *const c_void, size)
        .map_err(|e| {
            TransferError::ExecutionError(format!("ZE D2H append_memcpy failed: {:?}", e))
        })?;
    list.close()
        .map_err(|e| TransferError::ExecutionError(format!("ZE D2H list close failed: {:?}", e)))?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE D2H queue submit failed: {:?}", e))
    })?;
    Ok(())
}

#[inline(always)]
fn ze_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    queue: &ZeCommandQueue,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination ZE device pointer is null");

    if size == 0 {
        return Ok(());
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!("ZE H2D command list creation failed: {:?}", e))
    })?;
    list.append_memcpy(dst_ptr as *mut c_void, src_ptr as *const c_void, size)
        .map_err(|e| {
            TransferError::ExecutionError(format!("ZE H2D append_memcpy failed: {:?}", e))
        })?;
    list.close()
        .map_err(|e| TransferError::ExecutionError(format!("ZE H2D list close failed: {:?}", e)))?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE H2D queue submit failed: {:?}", e))
    })?;
    Ok(())
}
