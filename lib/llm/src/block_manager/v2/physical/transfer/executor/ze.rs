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

use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use crate::block_manager::v2::physical::transfer::StorageKind;
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
fn allocate_event(resources: &mut ZeDeviceResources) -> Result<ZeEvent> {
    let idx = resources.next_event_idx;
    resources.next_event_idx = (idx + 1) % resources.event_pool.count();

    let event = ZeEvent::new(&resources.event_pool, idx)
        .map_err(|e| anyhow!("Failed to create ZeEvent at index {}: {}", idx, e))?;
    event
        .host_reset()
        .map_err(|e| anyhow!("Failed to reset ZeEvent at index {}: {}", idx, e))?;
    Ok(event)
}

/// Execute a Level Zero transfer between host and XPU device memory.
///
/// The last `zeCommandListAppendMemoryCopy` signals a [`ZeEvent`]; the event
/// is then registered with the [`TransferContext`] notification subsystem for
/// async polling, mirroring the CUDA event path.
///
/// If `ze_cmdlist` is provided, uses that command list instead of the
/// per-device cached one. The caller is responsible for synchronization.
pub fn execute_ze_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    ze_cmdlist: Option<Arc<ZeImmediateCmdList>>,
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

    // Determine layer range
    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    // Determine XPU device ordinal from the layout locations
    let device_ordinal = match (src.location(), dst.location()) {
        (StorageKind::XpuDevice(id), _) => id,
        (_, StorageKind::XpuDevice(id)) => id,
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

    // Use caller-provided command list if available, otherwise use cached one
    let caller_manages_sync = ze_cmdlist.is_some();
    let cmdlist = ze_cmdlist.unwrap_or_else(|| resources.cmdlist.clone());
    let signal_event = allocate_event(&mut resources)?;

    // Release the lock before issuing memcpy operations
    drop(resources);

    // Determine direction name for logging
    let _strategy_name = match strategy {
        TransferStrategy::ZeAsyncH2D => "H2D",
        TransferStrategy::ZeAsyncD2H => "D2H",
        TransferStrategy::ZeAsyncD2D => "D2D",
        _ => "Unknown",
    };

    match strategy {
        TransferStrategy::ZeAsyncH2D
        | TransferStrategy::ZeAsyncD2H
        | TransferStrategy::ZeAsyncD2D => {
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
        _ => {
            return Err(anyhow!("Invalid XPU transfer strategy: {:?}", strategy));
        }
    }

    // Register the signal event for async polling completion
    let _ = caller_manages_sync; // reserved for future: skip event if caller syncs
    Ok(ctx.register_ze_event(signal_event))
}

/// Per-chunk transfer for mixed FC/LW layouts using Level Zero memcpy.
///
/// We issue one L0 memcpy per (block, layer, outer) chunk.
/// The **last** copy signals `signal_event`.
fn execute_fc_lw_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
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
        total_copies = copy_idx,
        chunk_size,
        "XPU per-chunk transfer appended"
    );

    Ok(())
}
