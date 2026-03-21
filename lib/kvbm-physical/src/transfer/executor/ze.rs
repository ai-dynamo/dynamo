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
//!
//! # SYCL kernel override
//!
//! By default, D2D FC↔LW transfers use the raw Level Zero SPIR-V kernel
//! (`vectorized_copy.spv` via `ZeModule::from_spirv`), which has the
//! lowest host-side dispatch latency.
//!
//! An alternative SYCL runtime path is available behind the `sycl-kernel`
//! feature. When enabled, set `KVBM_USE_SYCL_KERNEL=1` at runtime to
//! dispatch through `sycl::queue` instead of raw L0 APIs. This exercises
//! the full SYCL stack while sharing the same L0 context and device memory.
//!
//! ```sh
//! # 1. Build the SYCL shared library (requires icpx -fsycl):
//! make -C lib/kvbm-kernels/sycl
//!
//! # 2. Build Rust with the sycl-kernel feature:
//! cargo build -p kvbm-physical --features sycl-kernel
//!
//! # 3. At runtime, opt in and point to the .so:
//! export KVBM_USE_SYCL_KERNEL=1
//! export LD_LIBRARY_PATH=lib/kvbm-kernels/sycl:$LD_LIBRARY_PATH
//! # Or specify the exact path:
//! export SYCL_VC_LIB_PATH=lib/kvbm-kernels/sycl/libvectorized_copy_sycl.so
//! ```

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::{can_use_whole_block_transfer, validate_layout_compatibility};
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

/// Get or create a cached [`SyclVectorizedCopy`] for the given device.
///
/// The SYCL state shares the same Level Zero context as `ze_resources()`,
/// so device memory allocated by `ZeMemPool` is directly accessible.
#[cfg(feature = "sycl-kernel")]
fn sycl_vc_state(
    device_ordinal: u32,
) -> Result<Arc<kvbm_kernels::sycl_vectorized_copy::SyclVectorizedCopy>> {
    use kvbm_kernels::sycl_vectorized_copy::SyclVectorizedCopy;

    static SYCL_STATES: OnceLock<Mutex<HashMap<u32, Arc<SyclVectorizedCopy>>>> = OnceLock::new();
    let mut map = SYCL_STATES.get_or_init(Default::default).lock().unwrap();

    if let Some(existing) = map.get(&device_ordinal) {
        return Ok(existing.clone());
    }

    let dev = ZeDevice::new(device_ordinal as usize)
        .map_err(|e| anyhow!("Failed to create ZeDevice {}: {}", device_ordinal, e))?;

    let sycl_vc = SyclVectorizedCopy::new(
        dev.ze_context() as *mut c_void,
        dev.ze_device() as *mut c_void,
    )
    .map_err(|e| anyhow!("Failed to init SYCL vectorized_copy for device {}: {}", device_ordinal, e))?;

    let arc = Arc::new(sycl_vc);
    map.insert(device_ordinal, arc.clone());
    Ok(arc)
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
/// * `ze_cmdlist` - Optional caller-provided command list (caller manages sync)
/// * `ctx` - Transfer context
pub fn execute_ze_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
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
                    &cmd_copy,
                    &signal_event,
                )?;
            } else if matches!(strategy, TransferStrategy::ZeAsyncD2D) {
                // Vectorized kernel: only for D2D (CCS kernel cannot
                // dereference host-pinned pointers on discrete GPUs).
                //
                // Default: raw L0 SPIR-V kernel (lowest latency).
                // Override: set KVBM_USE_SYCL_KERNEL=1 to dispatch via SYCL runtime.
                #[cfg(feature = "sycl-kernel")]
                let use_sycl = std::env::var("KVBM_USE_SYCL_KERNEL").is_ok();
                #[cfg(not(feature = "sycl-kernel"))]
                let use_sycl = false;

                if use_sycl {
                    #[cfg(feature = "sycl-kernel")]
                    {
                        // Lazily init SYCL state using the same L0 context.
                        let sycl_vc = sycl_vc_state(device_ordinal)?;
                        tracing::debug!(
                            strategy = strategy_name,
                            num_blocks = src_block_ids.len(),
                            num_layers = layers.len(),
                            "Using SYCL vectorized_copy kernel (D2D)"
                        );
                        execute_fc_lw_vectorized_sycl_ze(
                            src,
                            dst,
                            src_block_ids,
                            dst_block_ids,
                            layers,
                            &cmd_copy,
                            &sycl_vc,
                            &pool,
                            &signal_event,
                        )?;
                    }
                } else {
                    tracing::debug!(
                        strategy = strategy_name,
                        num_blocks = src_block_ids.len(),
                        num_layers = layers.len(),
                        "Using XPU vectorized_copy kernel (D2D)"
                    );
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
            } else {
                // H2D / D2H: fall back to per-chunk BCS memcpy.
                tracing::debug!(
                    strategy = strategy_name,
                    num_blocks = src_block_ids.len(),
                    num_layers = layers.len(),
                    "Using XPU per-chunk memcpy (H2D/D2H)"
                );
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
        }
        _ => {
            return Err(anyhow!("Invalid XPU transfer strategy: {:?}", strategy));
        }
    }

    // Register the signal event for async polling completion.
    // When the caller provided their own command list, they manage synchronization,
    // but we still register the event so the notification system works uniformly.
    let _ = caller_manages_sync; // reserved for future: skip event if caller syncs
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


/// FC↔LW transfer using the SPIR-V vectorized_copy kernel (D2D only).
///
/// Mirrors [`super::cuda::execute_fc_lw_vectorized`]:
/// 1. Builds flat (src, dst) pointer arrays on the host.
/// 2. Allocates scratch device memory from `ZeMemPool`.
/// 3. Uploads pointer arrays via BCS (`cmd_copy`).
/// 4. Dispatches `vectorized_copy` kernel via CCS (`cmd_compute`).
/// 5. Signals `signal_event` after kernel completion.
/// 6. Frees scratch allocations.
fn execute_fc_lw_vectorized_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
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

/// Per-layer transfer for mixed FC/LW layouts using Level Zero memcpy.
///
/// When layouts are not fully contiguous (e.g., one is layer-wise and the
/// other is fully-contiguous), we issue one L0 memcpy per (block, layer)
/// covering all outer dimensions in a single copy.
///
/// This works because outer dims are contiguous within a (block, layer)
/// for both FC and LW-BlockIsFirstDim layouts (`outer_stride == region_size`).
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

/// FC↔LW transfer using the SYCL vectorized_copy kernel (D2D only).
///
/// Functionally identical to [`execute_fc_lw_vectorized_ze`] but dispatches
/// the kernel through the SYCL runtime instead of raw Level Zero APIs.
/// This exercises the full SYCL stack (queue, event, JIT) while sharing
/// the same L0 context, device memory, and BCS command list.
///
/// Steps 1-3 (build host arrays, alloc scratch, upload via BCS) are
/// identical to the L0 version. Step 4 calls `SyclVectorizedCopy::run()`
/// instead of `append_launch_kernel()`.
///
/// Requires the `sycl-kernel` feature and `libvectorized_copy_sycl.so`
/// to be built and loadable at runtime (`make -C lib/kvbm-kernels/sycl`).
#[cfg(feature = "sycl-kernel")]
fn execute_fc_lw_vectorized_sycl_ze(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    cmd_copy: &ZeImmediateCmdList,
    sycl_vc: &kvbm_kernels::sycl_vectorized_copy::SyclVectorizedCopy,
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

    // Dispatch via SYCL runtime (blocks until kernel completes).
    sycl_vc
        .run(src_ptrs_dev, dst_ptrs_dev, chunk_size as u64, total_chunks as i32)
        .map_err(|e| anyhow!("SYCL vectorized_copy kernel failed: {}", e))?;

    // Signal the event from the host (kernel already completed synchronously).
    signal_event
        .host_signal()
        .map_err(|e| anyhow!("Failed to host_signal event after SYCL kernel: {}", e))?;

    // Free scratch.
    pool.free(src_ptrs_dev, ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool free src_ptrs failed: {}", e))?;
    pool.free(dst_ptrs_dev, ptr_array_bytes)
        .map_err(|e| anyhow!("ZeMemPool free dst_ptrs failed: {}", e))?;

    tracing::debug!(
        total_chunks,
        chunk_size,
        "XPU SYCL vectorized_copy kernel transfer completed"
    );

    Ok(())
}
