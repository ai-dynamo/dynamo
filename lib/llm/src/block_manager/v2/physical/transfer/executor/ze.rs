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
use dynamo_memory::ZeMemPool;
use kvbm_kernels::ze_vectorized_copy as vc;
use syclrc::level_zero::ze::safe::{ZeDevice, ZeImmediateCmdList, ZeKernel, ZeModule};
use syclrc::level_zero::ze::sys;
use syclrc::{ZeEvent, ZeEventPool, event_pool_flags};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::{Arc, Mutex, OnceLock};

/// Cached per-device resources: command lists, event pool, kernel, memory pool.
struct ZeDeviceResources {
    /// BCS (copy engine) immediate command list.
    cmd_copy: Arc<ZeImmediateCmdList>,
    /// CCS (compute engine) immediate command list for kernel dispatch.
    cmd_compute: Arc<ZeImmediateCmdList>,
    event_pool: Arc<ZeEventPool>,
    /// Next event index within the pool (wraps around).
    next_event_idx: u32,
    /// SPIR-V vectorized_copy kernel (loaded lazily on first use).
    kernel: Arc<ZeKernel>,
    /// Scratch memory pool for pointer arrays.
    pool: Arc<ZeMemPool>,
    /// Module kept alive so kernel handle remains valid.
    _module: Arc<ZeModule>,
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

    let cmd_copy = ZeImmediateCmdList::new_copy(Arc::clone(&dev))
        .map_err(|e| anyhow!("Failed to create BCS cmd list for device {}: {}", device_ordinal, e))?;

    let cmd_compute = ZeImmediateCmdList::new_compute(Arc::clone(&dev))
        .map_err(|e| anyhow!("Failed to create CCS cmd list for device {}: {}", device_ordinal, e))?;

    let event_pool = ZeEventPool::new(Arc::clone(&dev), EVENT_POOL_SIZE, event_pool_flags::HOST_VISIBLE)
        .map_err(|e| anyhow!("Failed to create ZeEventPool for device {}: {}", device_ordinal, e))?;

    // Compile SPIR-V module and create vectorized_copy kernel.
    let module = Arc::new(
        ZeModule::from_spirv(&dev, vc::SPIRV, None)
            .map_err(|e| anyhow!("Failed to compile SPIR-V module for device {}: {}", device_ordinal, e))?,
    );
    let kernel = ZeKernel::new(&module, vc::KERNEL_NAME)
        .map_err(|e| anyhow!("Failed to create ZeKernel for device {}: {}", device_ordinal, e))?;
    kernel
        .set_group_size(vc::WORK_GROUP_SIZE, 1, 1)
        .map_err(|e| anyhow!("Failed to set kernel group size for device {}: {}", device_ordinal, e))?;

    // Scratch memory pool (16 MB reserve, 64 MB release threshold).
    let pool = ZeMemPool::builder(Arc::clone(&dev), 16 * 1024 * 1024)
        .release_threshold(64 * 1024 * 1024)
        .build()
        .map_err(|e| anyhow!("Failed to create ZeMemPool for device {}: {}", device_ordinal, e))?;

    let res = Arc::new(Mutex::new(ZeDeviceResources {
        cmd_copy: Arc::new(cmd_copy),
        cmd_compute: Arc::new(cmd_compute),
        event_pool: Arc::new(event_pool),
        next_event_idx: 0,
        kernel: Arc::new(kernel),
        pool: Arc::new(pool),
        _module: module,
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

    // Acquire per-device resources (command lists, kernel, pool, event pool)
    let resources_lock = ze_resources(device_ordinal)?;
    let mut resources = resources_lock.lock().unwrap();

    let caller_manages_sync = ze_cmdlist.is_some();
    let cmd_copy = ze_cmdlist.unwrap_or_else(|| resources.cmd_copy.clone());
    let cmd_compute = resources.cmd_compute.clone();
    let kernel = resources.kernel.clone();
    let pool = resources.pool.clone();
    let signal_event = allocate_event(&mut resources)?;

    // Release the lock before issuing GPU operations
    drop(resources);

    // Determine direction name for logging
    let _strategy_name = match strategy {
        TransferStrategy::ZeAsyncH2D => "H2D",
        TransferStrategy::ZeAsyncD2H => "D2H",
        TransferStrategy::ZeAsyncD2D => "D2D",
        _ => "Unknown",
    };

    match strategy {
        TransferStrategy::ZeAsyncD2D => {
            // Vectorized kernel: only for D2D (CCS kernel cannot
            // dereference host-pinned pointers on discrete GPUs).
            execute_fc_lw_vectorized_ze(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers,
                &cmd_copy,
                &cmd_compute,
                &kernel,
                &pool,
                &signal_event,
            )?;
        }
        TransferStrategy::ZeAsyncH2D | TransferStrategy::ZeAsyncD2H => {
            // H2D / D2H: fall back to per-chunk BCS memcpy.
            execute_fc_lw_ze(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers,
                &cmd_copy,
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


/// FC↔LW transfer using the SPIR-V vectorized_copy kernel (D2D only).
///
/// Mirrors [`kvbm_physical::transfer::executor::cuda::execute_fc_lw_vectorized`]:
/// 1. Builds flat (src, dst) pointer arrays on the host.
/// 2. Allocates scratch device memory from `ZeMemPool`.
/// 3. Uploads pointer arrays via BCS (`cmd_copy`).
/// 4. Dispatches `vectorized_copy` kernel via CCS (`cmd_compute`).
/// 5. Signals `signal_event` after kernel completion.
/// 6. Frees scratch allocations.
fn execute_fc_lw_vectorized_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: Range<usize>,
    cmd_copy: &ZeImmediateCmdList,
    cmd_compute: &ZeImmediateCmdList,
    kernel: &ZeKernel,
    pool: &ZeMemPool,
    signal_event: &ZeEvent,
) -> Result<()> {
    let src_layout = src.layout();
    let nl = layers.len();
    let no = src_layout.outer_dim();
    let chunk_size =
        src_layout.page_size() * src_layout.inner_dim() * src_layout.dtype_width_bytes();
    let num_blocks = src_block_ids.len();
    let total_chunks = num_blocks * nl * no;

    if total_chunks == 0 {
        signal_event
            .host_signal()
            .map_err(|e| anyhow!("Failed to host_signal event for empty transfer: {}", e))?;
        return Ok(());
    }

    // Build flat pointer arrays on host.
    let mut src_ptrs: Vec<u64> = Vec::with_capacity(total_chunks);
    let mut dst_ptrs: Vec<u64> = Vec::with_capacity(total_chunks);

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..no {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;
                src_ptrs.push(src_region.addr() as u64);
                dst_ptrs.push(dst_region.addr() as u64);
            }
        }
    }

    // Allocate scratch device memory for pointer arrays.
    let ptr_array_bytes = total_chunks * std::mem::size_of::<u64>();
    let src_ptrs_dev = pool
        .alloc(ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool alloc for src_ptrs failed: {}", e))?;
    let dst_ptrs_dev = pool
        .alloc(ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool alloc for dst_ptrs failed: {}", e))?;

    // Upload pointer arrays to device via BCS.
    unsafe {
        cmd_copy
            .append_memcpy(
                src_ptrs_dev as *mut c_void,
                src_ptrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .map_err(|e| anyhow!("Upload src_ptrs failed: {}", e))?;
        cmd_copy
            .append_memcpy(
                dst_ptrs_dev as *mut c_void,
                dst_ptrs.as_ptr() as *const c_void,
                ptr_array_bytes,
                std::ptr::null_mut(),
                &mut [],
            )
            .map_err(|e| anyhow!("Upload dst_ptrs failed: {}", e))?;
    }
    cmd_copy
        .host_synchronize(u64::MAX)
        .map_err(|e| anyhow!("BCS sync after pointer upload failed: {}", e))?;

    // Set kernel arguments and launch on CCS.
    let copy_sz = chunk_size as u64;
    let n_pairs = total_chunks as i32;
    unsafe {
        kernel.set_arg(0, &src_ptrs_dev).map_err(|e| anyhow!("set_arg(0) failed: {}", e))?;
        kernel.set_arg(1, &dst_ptrs_dev).map_err(|e| anyhow!("set_arg(1) failed: {}", e))?;
        kernel.set_arg(2, &copy_sz).map_err(|e| anyhow!("set_arg(2) failed: {}", e))?;
        kernel.set_arg(3, &n_pairs).map_err(|e| anyhow!("set_arg(3) failed: {}", e))?;
    }

    let num_groups = std::cmp::min(total_chunks as u32, vc::MAX_GROUPS);
    let group_count = sys::ze_group_count_t {
        groupCountX: num_groups,
        groupCountY: 1,
        groupCountZ: 1,
    };
    unsafe {
        cmd_compute
            .append_launch_kernel(kernel, &group_count, Some(signal_event), &mut [])
            .map_err(|e| anyhow!("append_launch_kernel failed: {}", e))?;
    }

    // Wait for kernel completion, then free scratch.
    cmd_compute
        .host_synchronize(u64::MAX)
        .map_err(|e| anyhow!("CCS sync after kernel launch failed: {}", e))?;

    pool.free(src_ptrs_dev, ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool free src_ptrs failed: {}", e))?;
    pool.free(dst_ptrs_dev, ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool free dst_ptrs failed: {}", e))?;

    tracing::debug!(
        total_chunks,
        chunk_size,
        "XPU vectorized_copy kernel transfer completed"
    );

    Ok(())
}

/// Per-chunk transfer for mixed FC/LW layouts using Level Zero memcpy.
///
/// We issue one L0 memcpy per (block, layer), coalescing all outer dims
/// into a single copy (outer dims are contiguous for FC and LW-BlockIsFirstDim).
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
    // Coalesce all outer dims into one copy per (block, layer).
    let layer_chunk_size = no * chunk_size;
    let num_blocks = src_block_ids.len();

    if num_blocks == 0 || nl == 0 || layer_chunk_size == 0 {
        // Nothing to copy — signal the event immediately from the host.
        signal_event
            .host_signal()
            .map_err(|e| anyhow!("Failed to host_signal event for empty transfer: {}", e))?;
        return Ok(());
    }

    let total_copies = num_blocks * nl;
    let mut copy_idx = 0usize;

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            // outer_id=0 gives the start of the contiguous outer-dim region.
            let src_region = src.memory_region(src_block_id, layer_id, 0)?;
            let dst_region = dst.memory_region(dst_block_id, layer_id, 0)?;

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
                        layer_chunk_size,
                        sig,
                        &mut [], // no wait events
                    )
                    .map_err(|e| {
                        anyhow!(
                            "zeCommandListAppendMemoryCopy failed at block=({},{}), layer={}: {}",
                            src_block_id, dst_block_id, layer_id, e
                        )
                    })?;
            }
        }
    }

    tracing::debug!(
        total_copies,
        layer_chunk_size,
        "XPU per-layer transfer appended"
    );

    Ok(())
}
