// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer executors for different copy strategies.

pub(super) mod cuda;
mod memcpy;
mod nixl;
pub(crate) mod planner;

use super::strategy::select_strategy;
use super::strategy::{TransferPlan, TransferStrategy};
use super::validation::validate_block_transfer;
use super::{PhysicalLayout, TransferContext};
use crate::BlockId;
use crate::layout::KvBlockLayout;
use crate::manager::LayoutHandle;
use crate::transfer::BounceBufferInternal;
use crate::transfer::{StorageKind, context::TransferCompleteNotification};
use anyhow::Result;
use cudarc::driver::CudaStream;
use kvbm_common::KvbmTransferRoute;
use std::ops::Range;
use std::sync::Arc;
use tokio::sync::Mutex;

// Re-export the NIXL transfer builder for public use
pub use nixl::NixlTransferBuilder;

// PR-7.5.1: expose the transform-kernel dispatcher for benchmark.rs timing.
pub(crate) use planner::dispatch_transform_kernel;

/// Transformation kernel types for converting between different block layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransformKernel {
    /// No transformation needed - layouts are compatible, use copy
    None,
    /// Transform from operational (NHD/HND) to universal format
    BlockToUniversal { src_layout: KvBlockLayout },
    /// Transform from universal to operational (NHD/HND) format
    UniversalToBlock { dst_layout: KvBlockLayout },
    /// Transpose between operational formats (NHD <-> HND)
    OperationalTranspose,
    /// Layouts are incompatible and no kernel is available
    Unsupported,
}

/// Select the appropriate transformation kernel based on source and destination layouts.
///
/// Returns `TransformKernel::None` if the layouts are the same (copy is sufficient).
/// Returns `TransformKernel::Unsupported` if the layout combination is not supported.
#[allow(dead_code)]
pub(crate) fn select_transform_kernel(
    src_layout: KvBlockLayout,
    dst_layout: KvBlockLayout,
) -> TransformKernel {
    // Same layout - no transformation needed
    if !src_layout.requires_transform(&dst_layout) {
        return TransformKernel::None;
    }

    // Unknown layouts cannot be transformed
    if matches!(src_layout, KvBlockLayout::Unknown) || matches!(dst_layout, KvBlockLayout::Unknown)
    {
        return TransformKernel::Unsupported;
    }

    match (src_layout, dst_layout) {
        // Operational to Universal
        (
            KvBlockLayout::OperationalNHD | KvBlockLayout::OperationalHND,
            KvBlockLayout::Universal,
        ) => TransformKernel::BlockToUniversal { src_layout },

        // Universal to Operational
        (
            KvBlockLayout::Universal,
            KvBlockLayout::OperationalNHD | KvBlockLayout::OperationalHND,
        ) => TransformKernel::UniversalToBlock { dst_layout },

        // Operational NHD <-> HND transpose
        (KvBlockLayout::OperationalNHD, KvBlockLayout::OperationalHND)
        | (KvBlockLayout::OperationalHND, KvBlockLayout::OperationalNHD) => {
            TransformKernel::OperationalTranspose
        }

        // Custom layouts need explicit handling
        (KvBlockLayout::Custom(_), _) | (_, KvBlockLayout::Custom(_)) => {
            TransformKernel::Unsupported
        }

        // Fallback for any unhandled combinations
        _ => TransformKernel::Unsupported,
    }
}

/// Get the effective source layout, using override if provided.
#[expect(dead_code)]
pub(crate) fn effective_src_layout(
    src: &PhysicalLayout,
    override_layout: Option<KvBlockLayout>,
) -> KvBlockLayout {
    override_layout.unwrap_or_else(|| src.layout().block_layout())
}

/// Get the effective destination layout, using override if provided.
#[expect(dead_code)]
pub(crate) fn effective_dst_layout(
    dst: &PhysicalLayout,
    override_layout: Option<KvBlockLayout>,
) -> KvBlockLayout {
    override_layout.unwrap_or_else(|| dst.layout().block_layout())
}

#[derive(Default)]
#[expect(dead_code)]
pub(crate) struct TransferOptionsInternal {
    layer_range: Option<Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer: Option<BounceBufferInternal>,
    /// If provided, use this stream instead of acquiring from pool.
    /// Caller manages synchronization - no event is recorded by the executor.
    pub(crate) cuda_stream: Option<Arc<CudaStream>>,
    /// Override source block layout interpretation.
    /// If None, uses the layout's block_layout() method.
    pub(crate) src_kv_layout: Option<KvBlockLayout>,
    /// Override destination block layout interpretation.
    /// If None, uses the layout's block_layout() method.
    pub(crate) dst_kv_layout: Option<KvBlockLayout>,
    /// Logical route used for transfer metrics.
    pub(crate) metric_route: Option<KvbmTransferRoute>,
    /// Route through the stride-aware planner instead of legacy
    /// `select_strategy` + `execute_direct_transfer`. PR-5 wires this
    /// for the CudaAsync (H2D / D2H / D2D) strategies; other paths
    /// ignore the flag.
    pub(crate) use_planner: bool,
    /// AB-1d: per-axis coordinate-space restrictions for sliced
    /// cross-leader transfers. Empty = full extent (legacy behaviour).
    /// Non-empty requires `use_planner = true`; the planner threads
    /// them into [`crate::transfer::plan::TransferSelection`].
    pub(crate) axis_slices: Vec<kvbm_common::AxisIntersection>,
    /// Registry handles for the physical layouts. Present for normal
    /// manager-driven transfers and used as the prepared-plan cache key.
    pub(crate) plan_handles: Option<(LayoutHandle, LayoutHandle)>,
}

impl TransferOptionsInternal {
    pub(crate) fn builder() -> TransferOptionsInternalBuilder {
        TransferOptionsInternalBuilder::default()
    }
}

#[derive(Default)]
pub(crate) struct TransferOptionsInternalBuilder {
    layer_range: Option<Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer: Option<BounceBufferInternal>,
    cuda_stream: Option<Arc<CudaStream>>,
    src_kv_layout: Option<KvBlockLayout>,
    dst_kv_layout: Option<KvBlockLayout>,
    metric_route: Option<KvbmTransferRoute>,
    use_planner: bool,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
}

impl TransferOptionsInternalBuilder {
    pub(crate) fn layer_range(mut self, range: Range<usize>) -> Self {
        self.layer_range = Some(range);
        self
    }

    pub(crate) fn nixl_write_notification(mut self, notification: u64) -> Self {
        self.nixl_write_notification = Some(notification);
        self
    }

    pub(crate) fn bounce_buffer(mut self, bounce_buffer: BounceBufferInternal) -> Self {
        self.bounce_buffer = Some(bounce_buffer);
        self
    }

    /// Set a specific CUDA stream to use for this transfer.
    ///
    /// When provided, the executor will use this stream instead of acquiring
    /// one from the pool. The caller is responsible for synchronization -
    /// no event is recorded by the executor.
    ///
    /// This is useful for layer-wise transfers where all layers must execute
    /// on the same stream to allow proper event sequencing.
    pub(crate) fn cuda_stream(mut self, stream: Arc<CudaStream>) -> Self {
        self.cuda_stream = Some(stream);
        self
    }

    /// Override the source block layout interpretation.
    ///
    /// When set, the transfer executor will treat source blocks as having
    /// this layout instead of the layout's default block_layout().
    /// This enables transferring blocks that are stored in one format
    /// but should be interpreted as another.
    pub(crate) fn src_kv_layout(mut self, layout: KvBlockLayout) -> Self {
        self.src_kv_layout = Some(layout);
        self
    }

    /// Override the destination block layout interpretation.
    ///
    /// When set, the transfer executor will treat destination blocks as having
    /// this layout instead of the layout's default block_layout().
    /// This enables writing blocks in a different format than the destination
    /// layout's native format.
    pub(crate) fn dst_kv_layout(mut self, layout: KvBlockLayout) -> Self {
        self.dst_kv_layout = Some(layout);
        self
    }

    pub(crate) fn metric_route(mut self, route: KvbmTransferRoute) -> Self {
        self.metric_route = Some(route);
        self
    }

    pub(crate) fn use_planner(mut self, on: bool) -> Self {
        self.use_planner = on;
        self
    }

    /// Per-axis coordinate-space restrictions for sliced transfers.
    /// Forces `use_planner = true` when non-empty.
    pub(crate) fn axis_slices(mut self, slices: Vec<kvbm_common::AxisIntersection>) -> Self {
        if !slices.is_empty() {
            self.use_planner = true;
        }
        self.axis_slices = slices;
        self
    }

    pub(crate) fn handles(mut self, src: LayoutHandle, dst: LayoutHandle) -> Self {
        self.plan_handles = Some((src, dst));
        self
    }

    pub(crate) fn build(self) -> Result<TransferOptionsInternal> {
        if !self.axis_slices.is_empty() && !self.use_planner {
            anyhow::bail!("TransferOptionsInternal: axis_slices requires use_planner=true");
        }
        Ok(TransferOptionsInternal {
            layer_range: self.layer_range,
            nixl_write_notification: self.nixl_write_notification,
            bounce_buffer: self.bounce_buffer,
            cuda_stream: self.cuda_stream,
            src_kv_layout: self.src_kv_layout,
            dst_kv_layout: self.dst_kv_layout,
            metric_route: self.metric_route,
            use_planner: self.use_planner,
            axis_slices: self.axis_slices,
            plan_handles: self.plan_handles,
        })
    }
}

/// Execute a transfer between two physical layouts.
///
/// This is an internal entry point for all transfer operations called by TransportManager.
/// It selects the appropriate strategy and dispatches to the corresponding executor.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `options` - Transfer options
/// * `ctx` - Transfer context with CUDA stream and NIXL agent
pub(crate) fn execute_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    options: TransferOptionsInternal,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // Validate block IDs
    validate_block_transfer(src_block_ids, dst_block_ids, None, src, dst, None)?;

    // c6: auto-promote to the planner path when the layout pair requires
    // a kernel transform. The legacy `cuda::execute_cuda_transfer` path
    // calls `validate_layout_compatibility`, which rejects any pair where
    // `requires_transform = true`. The planner handles these via the
    // kernel catalog (see `executor::planner::plan_and_lower` →
    // `PlanOutcome::Transform` → `dispatch_transform_kernel`). c3 made
    // G2 = `KvBlockLayout::Universal` in `BlockLayoutMode::Universal`
    // while G1 stays operational, so the offload pipeline's default-
    // options call path (`TransferOptions::default()` → `use_planner =
    // false`) now needs this promotion.
    //
    // c6 phase 4b: `layer_range + requires_transform` is now supported.
    // The kernel takes `nl_full` + `nl_offset` so per-layer scatter
    // into a universal block writes the slice without head-interleave
    // corruption (see `kvbm_kernels_block_to_universal_kernel`).
    let mut options = options;
    let needs_transform = src
        .layout()
        .block_layout()
        .requires_transform(&dst.layout().block_layout());
    if needs_transform && !options.use_planner {
        tracing::debug!(
            src = ?src.layout().block_layout(),
            dst = ?dst.layout().block_layout(),
            "executor: auto-promoting to use_planner=true for cross-layout transform"
        );
        options.use_planner = true;
    }

    // Select transfer plan based on locations and capabilities
    let plan = select_strategy(src, dst, ctx)?;

    // Dispatch based on plan type
    match plan {
        TransferPlan::Direct(strategy) => execute_direct_transfer(
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            options.layer_range,
            strategy,
            options.cuda_stream,
            options.use_planner,
            options.bounce_buffer.as_ref(),
            options.axis_slices,
            options.plan_handles,
            ctx,
        ),
        TransferPlan::TwoHop {
            first,
            bounce_location,
            second,
        } => {
            if !options.axis_slices.is_empty() {
                anyhow::bail!(
                    "execute_transfer: axis_slices not supported on two-hop transfers \
                     (cross-leader sliced transfers must take a direct planner path)"
                );
            }
            execute_two_hop_transfer(TwoHopTransferParams {
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                first_strategy: first,
                bounce_location,
                second_strategy: second,
                options,
                ctx,
            })
        }
    }
}

/// Execute a direct single-hop transfer.
#[allow(clippy::too_many_arguments)]
fn execute_direct_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    use_planner: bool,
    bounce_buffer: Option<&BounceBufferInternal>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // axis_slices is only meaningful on planner paths; the legacy
    // memcpy / cuda / nixl-builder paths don't honour it and would
    // silently produce wrong data, so refuse early.
    if !axis_slices.is_empty() && !use_planner {
        anyhow::bail!(
            "execute_direct_transfer: axis_slices requires use_planner=true \
             (axis_slices contains {} entries)",
            axis_slices.len()
        );
    }
    match strategy {
        TransferStrategy::Memcpy => {
            if cuda_stream.is_some() {
                return Err(anyhow::anyhow!(
                    "cuda_stream option is not supported for Memcpy strategy"
                ));
            }
            memcpy::execute_memcpy_transfer(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layer_range,
                ctx,
            )
        }
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            if use_planner {
                // PR-5: planner-driven path for CudaAsync H2D / D2H / D2D.
                // Errors here are NOT silently fallen back to the
                // legacy executor — the caller flipped the flag, so a
                // failure is propagated as-is.
                //
                // c6 phase 4b: layer_range is honoured for the kernel-
                // transform path (the universal/transpose kernels take
                // `nl_full` + `nl_offset`). Same-layout direct copies
                // still reject layer_range because plan_copy's Direct
                // path doesn't slice on Layer — those stay on the
                // legacy executor until plan_copy grows the support.
                if layer_range.is_some()
                    && !src
                        .layout()
                        .block_layout()
                        .requires_transform(&dst.layout().block_layout())
                {
                    return Err(anyhow::anyhow!(
                        "use_planner=true currently incompatible with layer_range for \
                         same-layout direct copies; layer-restricted direct transfers stay \
                         on the legacy executor (transform pairs now supported via the \
                         kernel catalog)"
                    ));
                }
                return planner::execute_planner_cuda_transfer(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    strategy,
                    cuda_stream,
                    layer_range,
                    axis_slices,
                    plan_handles,
                    ctx,
                );
            }
            Ok(cuda::execute_cuda_transfer(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layer_range,
                strategy,
                cuda_stream,
                ctx,
            )?)
        }
        TransferStrategy::NixlRead
        | TransferStrategy::NixlWrite
        | TransferStrategy::NixlReadFlipped
        | TransferStrategy::NixlWriteFlipped => {
            if cuda_stream.is_some() {
                return Err(anyhow::anyhow!(
                    "cuda_stream option is not supported for NIXL strategies"
                ));
            }
            if use_planner {
                // PR-5.6: planner-driven NIXL path. Errors propagate;
                // no silent fallback to legacy (caller flipped the flag).
                if layer_range.is_some() {
                    return Err(anyhow::anyhow!(
                        "use_planner=true currently incompatible with layer_range; \
                         layer-restricted NIXL transfers stay on the legacy executor"
                    ));
                }
                return planner::execute_planner_nixl_transfer(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    strategy,
                    bounce_buffer,
                    axis_slices,
                    plan_handles,
                    ctx,
                );
            }
            let _ = bounce_buffer;
            let mut builder = NixlTransferBuilder::new()
                .src(src)
                .dst(dst)
                .src_blocks(src_block_ids)
                .dst_blocks(dst_block_ids)
                .strategy(strategy);

            if let Some(range) = layer_range {
                builder = builder.layer_range(range);
            }

            builder.execute(ctx)
        }
        TransferStrategy::Invalid => Err(anyhow::anyhow!(
            "Invalid transfer strategy for src={:?}, dst={:?}",
            src.location(),
            dst.location()
        )),
    }
}

/// Work-stealing bounce buffer transfer using two parallel tasks.
///
/// This function implements a work-stealing approach where two tasks each take
/// batches from a shared iterator and execute complete two-hop transfers.
/// This is simpler to maintain than double-buffering while still providing
/// good throughput through task parallelism.
///
/// # Algorithm
/// 1. Split bounce buffer into two groups (group 0 and group 1)
/// 2. Create a shared iterator over (src_block_id, dst_block_id) pairs
/// 3. Two parallel tasks each:
///    - Lock the iterator, take a batch of pairs
///    - Execute the complete two-hop transfer for that batch
///    - Repeat until iterator is exhausted
#[allow(clippy::too_many_arguments)]
async fn handle_buffered_transfer(
    src: &PhysicalLayout,
    bounce_layout: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    bounce_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    first_strategy: TransferStrategy,
    second_strategy: TransferStrategy,
    layer_range: &Option<Range<usize>>,
    ctx: &TransferContext,
) -> Result<()> {
    let bounce_groups =
        &bounce_block_ids[0..std::cmp::min(src_block_ids.len(), bounce_block_ids.len())];
    let (bounce_group_0, bounce_group_1) = bounce_groups.split_at(bounce_groups.len() / 2);
    let bounce_group_0 = bounce_group_0.to_vec();
    let bounce_group_1 = bounce_group_1.to_vec();

    let src_dst_iter = Arc::new(Mutex::new(src_block_ids.iter().zip(dst_block_ids.iter())));

    let transfer_task = async move |bounce_group: &[BlockId]| -> Result<()> {
        loop {
            let (src_ids, dst_ids): (Vec<BlockId>, Vec<BlockId>);
            {
                let mut x = src_dst_iter.lock().await;
                (src_ids, dst_ids) = x
                    .by_ref()
                    .take(bounce_group.len())
                    .map(|(&s, &d)| (s, d))
                    .unzip();
                if src_ids.is_empty() {
                    break;
                }
            }

            execute_two_hop_transfer_chunk(
                src,
                bounce_layout,
                dst,
                &src_ids,
                &bounce_group[0..src_ids.len()],
                &dst_ids,
                first_strategy,
                second_strategy,
                layer_range,
                ctx,
            )
            .await?;
        }

        Ok(())
    };

    let transfer_0 = transfer_task(&bounce_group_0);
    let transfer_1 = transfer_task(&bounce_group_1);

    futures::future::try_join(transfer_0, transfer_1).await?;

    Ok(())
}

/// Execute a single chunk of a two-hop transfer sequentially.
///
/// Used when bounce buffer has only a single block or as a fallback.
/// Performs src→bounce followed by bounce→dst sequentially.
#[allow(clippy::too_many_arguments)]
async fn execute_two_hop_transfer_chunk(
    src: &PhysicalLayout,
    bounce_layout: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    bounce_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    first_strategy: TransferStrategy,
    second_strategy: TransferStrategy,
    layer_range: &Option<Range<usize>>,
    ctx: &TransferContext,
) -> Result<()> {
    let bounce_ids_to_use = &bounce_block_ids[..src_block_ids.len()];

    execute_direct_transfer(
        src,
        bounce_layout,
        src_block_ids,
        bounce_ids_to_use,
        layer_range.clone(),
        first_strategy,
        None,       // Two-hop transfers don't support caller-provided streams
        false,      // Two-hop chunks stay on the legacy path for now
        None,       // bounce_buffer only used by use_planner=true NIXL transforms
        Vec::new(), // axis_slices: two-hop chunks never carry slices (rejected upstream)
        None,       // two-hop chunks do not map to the original handle pair
        ctx,
    )?
    .await?;

    execute_direct_transfer(
        bounce_layout,
        dst,
        bounce_ids_to_use,
        dst_block_ids,
        layer_range.clone(),
        second_strategy,
        None,  // Two-hop transfers don't support caller-provided streams
        false, // Two-hop chunks stay on the legacy path for now
        None,
        Vec::new(), // axis_slices: two-hop chunks never carry slices
        None,       // two-hop chunks do not map to the original handle pair
        ctx,
    )?
    .await?;

    Ok(())
}

/// Parameters for two-hop transfer execution
struct TwoHopTransferParams<'a> {
    src: &'a PhysicalLayout,
    dst: &'a PhysicalLayout,
    src_block_ids: &'a [BlockId],
    dst_block_ids: &'a [BlockId],
    first_strategy: TransferStrategy,
    bounce_location: StorageKind,
    second_strategy: TransferStrategy,
    options: TransferOptionsInternal,
    ctx: &'a TransferContext,
}

fn execute_two_hop_transfer(params: TwoHopTransferParams) -> Result<TransferCompleteNotification> {
    let TwoHopTransferParams {
        src,
        dst,
        src_block_ids,
        dst_block_ids,
        first_strategy,
        bounce_location,
        second_strategy,
        options,
        ctx,
    } = params;

    let event = ctx.event_system().new_event()?;
    let handle = event.into_handle();
    let awaiter = ctx.event_system().awaiter(handle)?;
    let system = ctx.event_system().clone();

    // TODO: Cloning all this stuff is not ideal.
    let src_clone = src.clone();
    let dst_clone = dst.clone();

    let src_block_ids = src_block_ids.to_vec();
    let dst_block_ids = dst_block_ids.to_vec();

    let ctx_clone = ctx.clone();
    // let options_clone = options.clone();

    ctx.tokio().spawn(async move {
        let Some(ref bounce_buffer_spec) = options.bounce_buffer else {
            let _ = system.poison(
                handle,
                "Two-hop transfers require a bounce buffer.".to_string(),
            );
            return;
        };

        if bounce_buffer_spec.layout.location() != bounce_location {
            let _ = system.poison(
                handle,
                "Bounce buffer layout does not match bounce location.".to_string(),
            );
            return;
        }

        let num_bounce_blocks = bounce_buffer_spec.block_ids.len();

        if num_bounce_blocks == 1 {
            // Single bounce block: use sequential processing for each block
            let bounce_block = bounce_buffer_spec.block_ids[0];
            for (src_block_id, dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
                if let Err(e) = execute_two_hop_transfer_chunk(
                    &src_clone,
                    &bounce_buffer_spec.layout,
                    &dst_clone,
                    &[*src_block_id],
                    &[bounce_block],
                    &[*dst_block_id],
                    first_strategy,
                    second_strategy,
                    &options.layer_range,
                    &ctx_clone,
                )
                .await
                {
                    let _ = system.poison(handle, e.to_string());
                    return;
                }
            }
            let _ = system.trigger(handle);
        } else {
            // Multiple bounce blocks: use work-stealing parallel transfer
            if let Err(e) = handle_buffered_transfer(
                &src_clone,
                &bounce_buffer_spec.layout,
                &dst_clone,
                &src_block_ids,
                &bounce_buffer_spec.block_ids,
                &dst_block_ids,
                first_strategy,
                second_strategy,
                &options.layer_range,
                &ctx_clone,
            )
            .await
            {
                let _ = system.poison(handle, e.to_string());
                return;
            }
            let _ = system.trigger(handle);
        }
    });

    Ok(TransferCompleteNotification::from_awaiter(awaiter))
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;

    #[test]
    fn test_select_transform_kernel_same_layout() {
        // Same layout - no transformation
        assert_eq!(
            select_transform_kernel(KvBlockLayout::OperationalNHD, KvBlockLayout::OperationalNHD),
            TransformKernel::None
        );
        assert_eq!(
            select_transform_kernel(KvBlockLayout::Universal, KvBlockLayout::Universal),
            TransformKernel::None
        );
    }

    #[test]
    fn test_select_transform_kernel_block_to_universal() {
        // Operational to Universal
        assert!(matches!(
            select_transform_kernel(KvBlockLayout::OperationalNHD, KvBlockLayout::Universal),
            TransformKernel::BlockToUniversal {
                src_layout: KvBlockLayout::OperationalNHD
            }
        ));
        assert!(matches!(
            select_transform_kernel(KvBlockLayout::OperationalHND, KvBlockLayout::Universal),
            TransformKernel::BlockToUniversal {
                src_layout: KvBlockLayout::OperationalHND
            }
        ));
    }

    #[test]
    fn test_select_transform_kernel_universal_to_block() {
        // Universal to Operational
        assert!(matches!(
            select_transform_kernel(KvBlockLayout::Universal, KvBlockLayout::OperationalNHD),
            TransformKernel::UniversalToBlock {
                dst_layout: KvBlockLayout::OperationalNHD
            }
        ));
        assert!(matches!(
            select_transform_kernel(KvBlockLayout::Universal, KvBlockLayout::OperationalHND),
            TransformKernel::UniversalToBlock {
                dst_layout: KvBlockLayout::OperationalHND
            }
        ));
    }

    #[test]
    fn test_select_transform_kernel_operational_transpose() {
        // NHD <-> HND
        assert_eq!(
            select_transform_kernel(KvBlockLayout::OperationalNHD, KvBlockLayout::OperationalHND),
            TransformKernel::OperationalTranspose
        );
        assert_eq!(
            select_transform_kernel(KvBlockLayout::OperationalHND, KvBlockLayout::OperationalNHD),
            TransformKernel::OperationalTranspose
        );
    }

    #[test]
    fn test_select_transform_kernel_unknown_unsupported() {
        // Unknown is always unsupported
        assert_eq!(
            select_transform_kernel(KvBlockLayout::Unknown, KvBlockLayout::OperationalNHD),
            TransformKernel::Unsupported
        );
        assert_eq!(
            select_transform_kernel(KvBlockLayout::OperationalNHD, KvBlockLayout::Unknown),
            TransformKernel::Unsupported
        );
    }

    #[test]
    fn test_select_transform_kernel_custom_unsupported() {
        // Custom layouts are unsupported (for now)
        let custom = KvBlockLayout::Custom([
            crate::layout::BlockDim::Head,
            crate::layout::BlockDim::Layer,
            crate::layout::BlockDim::Outer,
            crate::layout::BlockDim::Page,
        ]);
        assert_eq!(
            select_transform_kernel(custom, KvBlockLayout::OperationalNHD),
            TransformKernel::Unsupported
        );
    }
}
