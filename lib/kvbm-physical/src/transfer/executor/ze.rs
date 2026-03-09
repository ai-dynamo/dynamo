// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Level Zero (XPU) executor for Intel GPU memory transfers.
//!
//! Provides the XPU counterpart to [`super::cuda`]: transfers between host (pinned)
//! and XPU device memory using Level Zero immediate command lists.
//!
//! Level Zero's `zeCommandListAppendMemoryCopy` auto-detects transfer direction
//! (H2D / D2H / D2D) from pointer types, similar to `cudaMemcpyDefault`.
//!
//! Async completion is tracked via [`ZeEvent`] signals and polled by the
//! notification subsystem, mirroring the CUDA event pattern.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::{can_use_whole_block_transfer, validate_layout_compatibility};
use anyhow::{Result, anyhow};
use syclrc::level_zero::ze::safe::{ZeDevice, ZeImmediateCmdList};
use syclrc::{ZeEvent, ZeEventPool, event_pool_flags};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

/// Cached per-device resources: immediate command list + event pool.
struct ZeDeviceResources {
    cmdlist: Arc<ZeImmediateCmdList>,
    event_pool: Arc<ZeEventPool>,
    /// Next event index within the pool (wraps around).
    next_event_idx: u32,
}

/// Event pool capacity per device.
/// Events are reused in ring-buffer fashion; the pool is HOST_VISIBLE so
/// [`ZeEvent::query_status`] works from the host.
const EVENT_POOL_SIZE: u32 = 64;

/// Get or create cached per-device resources (command list + event pool).
fn ze_resources(device_ordinal: u32) -> Result<Arc<Mutex<ZeDeviceResources>>> {
    static RESOURCES: OnceLock<Mutex<HashMap<u32, Arc<Mutex<ZeDeviceResources>>>>> = OnceLock::new();
    let mut map = RESOURCES.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&device_ordinal) {
        return Ok(existing.clone());
    }

    let dev = ZeDevice::new(device_ordinal as usize)
        .map_err(|e| anyhow!("Failed to create ZeDevice {}: {}", device_ordinal, e))?;
    // ZeDevice::new already returns Arc<ZeDevice>

    let cmdlist = ZeImmediateCmdList::new(Arc::clone(&dev))
        .map_err(|e| {
            anyhow!(
                "Failed to create ZeImmediateCmdList for device {}: {}",
                device_ordinal,
                e
            )
        })?;

    let event_pool = ZeEventPool::new(dev, EVENT_POOL_SIZE, event_pool_flags::HOST_VISIBLE)
        .map_err(|e| {
            anyhow!(
                "Failed to create ZeEventPool for device {}: {}",
                device_ordinal,
                e
            )
        })?;

    let res = Arc::new(Mutex::new(ZeDeviceResources {
        cmdlist: Arc::new(cmdlist),
        event_pool: Arc::new(event_pool),
        next_event_idx: 0,
    }));

    map.insert(device_ordinal, res.clone());
    Ok(res)
}

/// Allocate a fresh [`ZeEvent`] from the per-device pool.
///
/// Events rotate through pool slots; the caller is responsible for not
/// reusing an event that has not yet completed. With a pool of 64 slots
/// and the polling handler draining completions, this is unlikely.
fn allocate_event(resources: &mut ZeDeviceResources) -> Result<ZeEvent> {
    let idx = resources.next_event_idx;
    resources.next_event_idx = (idx + 1) % resources.event_pool.count();

    let event = ZeEvent::new(&resources.event_pool, idx)
        .map_err(|e| anyhow!("Failed to create ZeEvent at index {}: {}", idx, e))?;
    // Reset the event to unsignalled in case the slot was previously used.
    event
        .host_reset()
        .map_err(|e| anyhow!("Failed to reset ZeEvent at index {}: {}", idx, e))?;
    Ok(event)
}

/// Execute a Level Zero transfer between host and XPU device memory.
///
/// This executor handles transfers involving XPU memory using Level Zero APIs.
/// The last `zeCommandListAppendMemoryCopy` signals a [`ZeEvent`]; the event
/// is then registered with the [`TransferContext`] notification subsystem for
/// async polling, mirroring the CUDA event path.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
/// * `strategy` - XPU transfer strategy (ZeAsyncH2D, ZeAsyncD2H, ZeAsyncD2D)
/// * `ctx` - Transfer context
pub fn execute_ze_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // Validate layouts
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.num_layers() != dst_layout.num_layers() {
        return Err(anyhow!(
            "Layouts have incompatible layer counts: src={}, dst={}",
            src_layout.num_layers(),
            dst_layout.num_layers()
        ));
    }

    if src_layout.outer_dim() != dst_layout.outer_dim() {
        return Err(anyhow!(
            "Layouts have incompatible outer dimensions: src={}, dst={}",
            src_layout.outer_dim(),
            dst_layout.outer_dim()
        ));
    }

    // Validate layout compatibility (errors if transform would be needed)
    validate_layout_compatibility(src, dst)?;

    // Determine layer range
    let layers = layer_range.clone().unwrap_or(0..src_layout.num_layers());

    // Check if we can use optimized whole-block transfer
    let use_whole_block = can_use_whole_block_transfer(src, dst, layer_range.as_ref());

    // Determine XPU device ordinal from the layout locations
    let device_ordinal = match (src.location(), dst.location()) {
        (dynamo_memory::StorageKind::XpuDevice(id), _) => id,
        (_, dynamo_memory::StorageKind::XpuDevice(id)) => id,
        _ => {
            return Err(anyhow!(
                "ZeAsync strategy requires at least one XpuDevice endpoint, got src={:?}, dst={:?}",
                src.location(),
                dst.location()
            ));
        }
    };

    // Acquire per-device resources (command list + event pool)
    let resources_lock = ze_resources(device_ordinal)?;
    let mut resources = resources_lock.lock().unwrap();

    let cmdlist = resources.cmdlist.clone();
    let signal_event = allocate_event(&mut resources)?;

    // Release the lock before issuing memcpy operations
    drop(resources);

    // Determine direction name for logging
    let strategy_name = match strategy {
        TransferStrategy::ZeAsyncH2D => "H2D",
        TransferStrategy::ZeAsyncD2H => "D2H",
        TransferStrategy::ZeAsyncD2D => "D2D",
        _ => "Unknown",
    };

    match strategy {
        TransferStrategy::ZeAsyncH2D
        | TransferStrategy::ZeAsyncD2H
        | TransferStrategy::ZeAsyncD2D => {
            if use_whole_block {
                tracing::debug!(
                    strategy = strategy_name,
                    num_blocks = src_block_ids.len(),
                    bytes_per_block = src_layout.config().bytes_per_block(),
                    "Using XPU whole-block transfer (auto direction)"
                );
                execute_whole_block_ze(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    &cmdlist,
                    &signal_event,
                )?;
            } else {
                tracing::debug!(
                    strategy = strategy_name,
                    num_blocks = src_block_ids.len(),
                    num_layers = layers.len(),
                    "Using XPU per-chunk transfer"
                );
                execute_fc_lw_ze(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    &cmdlist,
                    &signal_event,
                )?;
            }
        }
        _ => {
            return Err(anyhow!("Invalid XPU transfer strategy: {:?}", strategy));
        }
    }

    // Register the signal event for async polling completion
    Ok(ctx.register_ze_event(signal_event))
}

/// Whole-block transfer using Level Zero memcpy.
///
/// For fully-contiguous layouts where all layers are transferred,
/// we issue one `zeCommandListAppendMemoryCopy` per block (covering the
/// entire block in a single copy). Direction is auto-detected by L0.
///
/// The **last** copy signals `signal_event` so the host can poll for completion.
fn execute_whole_block_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    cmdlist: &ZeImmediateCmdList,
    signal_event: &ZeEvent,
) -> Result<()> {
    let bytes_per_block = src.layout().config().bytes_per_block();
    let num_blocks = src_block_ids.len();

    if num_blocks == 0 {
        // Nothing to copy — signal the event immediately from the host.
        signal_event
            .host_signal()
            .map_err(|e| anyhow!("Failed to host_signal event for empty transfer: {}", e))?;
        return Ok(());
    }

    let last_idx = num_blocks - 1;

    for (i, (&src_block_id, &dst_block_id)) in
        src_block_ids.iter().zip(dst_block_ids.iter()).enumerate()
    {
        let src_region = src.memory_region(src_block_id, 0, 0)?;
        let dst_region = dst.memory_region(dst_block_id, 0, 0)?;

        // Signal the event only on the last copy
        let sig = if i == last_idx {
            signal_event.ze_handle()
        } else {
            std::ptr::null_mut()
        };

        unsafe {
            cmdlist
                .append_memcpy(
                    dst_region.addr() as *mut c_void,
                    src_region.addr() as *const c_void,
                    bytes_per_block,
                    sig,
                    &mut [], // no wait events
                )
                .map_err(|e| {
                    anyhow!(
                        "zeCommandListAppendMemoryCopy failed for whole-block: {}",
                        e
                    )
                })?;
        }
    }

    tracing::debug!(
        num_blocks,
        bytes_per_block,
        "XPU whole-block transfer appended"
    );

    Ok(())
}

/// Per-chunk transfer for mixed FC/LW layouts using Level Zero memcpy.
///
/// When layouts are not fully contiguous (e.g., one is layer-wise and the
/// other is fully-contiguous), we issue one L0 memcpy per (block, layer, outer)
/// chunk. This is the XPU equivalent of `execute_fc_lw_vectorized` in cuda.rs.
///
/// The **last** copy signals `signal_event`.
fn execute_fc_lw_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    cmdlist: &ZeImmediateCmdList,
    signal_event: &ZeEvent,
) -> Result<()> {
    let src_layout = src.layout();
    let nl = layers.len();
    let no = src_layout.outer_dim();
    let chunk_size =
        src_layout.page_size() * src_layout.inner_dim() * src_layout.dtype_width_bytes();
    let num_blocks = src_block_ids.len();

    if num_blocks == 0 || nl == 0 || no == 0 || chunk_size == 0 {
        // Nothing to copy — signal the event immediately from the host.
        signal_event
            .host_signal()
            .map_err(|e| anyhow!("Failed to host_signal event for empty transfer: {}", e))?;
        return Ok(());
    }

    // Pre-compute total number of copies to identify the last one
    let total_copies = num_blocks * nl * no;
    let mut copy_idx = 0usize;

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..no {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                copy_idx += 1;
                // Signal the event only on the last copy
                let sig = if copy_idx == total_copies {
                    signal_event.ze_handle()
                } else {
                    std::ptr::null_mut()
                };

                unsafe {
                    cmdlist
                        .append_memcpy(
                            dst_region.addr() as *mut c_void,
                            src_region.addr() as *const c_void,
                            src_region.size(),
                            sig,
                            &mut [], // no wait events
                        )
                        .map_err(|e| {
                            anyhow!(
                                "zeCommandListAppendMemoryCopy failed at block=({},{}), layer={}, outer={}: {}",
                                src_block_id, dst_block_id, layer_id, outer_id, e
                            )
                        })?;
                }
            }
        }
    }

    tracing::debug!(
        total_copies,
        chunk_size,
        "XPU per-chunk transfer appended"
    );

    Ok(())
}
