// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer executors for different copy strategies.

pub(super) mod cuda;
mod memcpy;
mod nixl;
mod object_transfer;
mod descriptors;

use super::strategy::select_strategy;
use super::validation::validate_block_transfer;
use super::{PhysicalLayout, TransferContext, TransferPlan, TransferStrategy};
use crate::BlockId;
use crate::physical::transfer::BounceBufferInternal;
use crate::v2::physical::transfer::{StorageKind, context::TransferCompleteNotification};
use anyhow::Result;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// pub(crate) struct G4OnboardContext {}
// pub(crate) struct G4OffloadContext {}

// pub(crate) fn g4_read(
//     src: &G4OnboardContext,
//     src_seq_hash: &[SequenceHash],
//     dst: &PhysicalLayout,
//     dst_block_ids: &[BlockId],
//     ctx: &TransferContext,
// ) -> Result<()> {
//     unimplemented!()
// }

// pub(crate) fn g4_write(
//     src: &PhysicalLayout,
//     src_block_ids: &[BlockId],
//     dst: &G4OffloadContext,
//     dst_seq_hash: &[SequenceHash],
//     ctx: &TransferContext,
// ) -> Result<()> {
//     if !src.layout().is_fully_contiguous() {
//         anyhow::bail!("G4 write source layout must be fully contiguous");
//     }

//     // logical instance has created a multi-part upload
//     // this method simply dumps the full block into the offload context
//     // the offload context will consist of:
//     // - bucket name
//     // - object name
//     // - upload id
//     // - part number

//     unimplemented!()
// }

// Re-export the NIXL transfer builder for public use
pub use nixl::NixlTransferBuilder;

use super::options::BackendOptArgs;

#[derive(Default)]
#[expect(dead_code)]
pub(crate) struct TransferOptionsInternal {
    layer_range: Option<Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer: Option<BounceBufferInternal>,
    backend_opts: Option<Box<dyn BackendOptArgs>>,
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
    backend_opts: Option<Box<dyn BackendOptArgs>>,
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

    pub(crate) fn backend_opts(mut self, opts: Box<dyn BackendOptArgs>) -> Self {
        self.backend_opts = Some(opts);
        self
    }

    pub(crate) fn build(self) -> Result<TransferOptionsInternal> {
        Ok(TransferOptionsInternal {
            layer_range: self.layer_range,
            nixl_write_notification: self.nixl_write_notification,
            bounce_buffer: self.bounce_buffer,
            backend_opts: self.backend_opts,
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
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
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
            options.backend_opts,
            strategy,
            ctx,
        ),
        TransferPlan::TwoHop {
            first,
            bounce_location,
            second,
        } => execute_two_hop_transfer(TwoHopTransferParams {
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            first_strategy: first,
            bounce_location,
            second_strategy: second,
            options,
            ctx,
        }),
    }
}

/// Execute a direct single-hop transfer.
fn execute_direct_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    backend_opts: Option<Box<dyn BackendOptArgs>>,
    strategy: TransferStrategy,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    match strategy {
        TransferStrategy::Memcpy => memcpy::execute_memcpy_transfer(
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            layer_range,
            ctx,
        ),
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D
        | TransferStrategy::CudaBlockingH2D
        | TransferStrategy::CudaBlockingD2H => Ok(cuda::execute_cuda_transfer(
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            layer_range,
            strategy,
            ctx,
        )?),
        TransferStrategy::NixlRead
        | TransferStrategy::NixlWrite
        | TransferStrategy::NixlReadFlipped
        | TransferStrategy::NixlWriteFlipped => {
            let mut builder = NixlTransferBuilder::new()
                .src(src)
                .dst(dst)
                .src_blocks(src_block_ids)
                .dst_blocks(dst_block_ids)
                .strategy(strategy);

            if let Some(range) = layer_range {
                builder = builder.layer_range(range);
            }

            if let Some(opts) = backend_opts {
                // Re-box since backend_opts takes ownership via impl trait
                builder = builder.backend_opts(ReBoxedOptArgs(opts));
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

/// Wrapper to pass a boxed BackendOptArgs to the builder which expects impl BackendOptArgs.
#[derive(Debug)]
struct ReBoxedOptArgs(Box<dyn BackendOptArgs>);

impl BackendOptArgs for ReBoxedOptArgs {
    fn to_custom_param(&self) -> String {
        self.0.to_custom_param()
    }

    fn clone_box(&self) -> Box<dyn BackendOptArgs> {
        self.0.clone_box()
    }
}

/// Type alias for transfer group: (block_ids, bounce_buffer_group_index)
type TransferGroup = (Vec<BlockId>, bool);

/// Optimized bounce buffer transfer using double-buffering.
///
/// This function implements a pipelined approach that splits the bounce buffer into two groups
/// and overlaps transfers: while transferring src→bounce[0], it simultaneously transfers
/// bounce[1]→dst. This significantly improves throughput compared to sequential staging.
///
/// # Algorithm
/// 1. Split bounce buffer into two groups (group 0 and group 1)
/// 2. Loop alternating between groups:
///    - Stage src[i] → bounce_group[0] (parallel with bounce_group[1] → dst[i-1])
///    - Stage src[i+1] → bounce_group[1] (parallel with bounce_group[0] → dst[i])
/// 3. Continue until all blocks transferred
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
    // Split bounce buffer into two groups for double-buffering
    let bounce_groups =
        &bounce_block_ids[0..std::cmp::min(src_block_ids.len(), bounce_block_ids.len())];
    let bounce_groups = bounce_groups.split_at(bounce_groups.len() / 2);
    let bounce_groups = [bounce_groups.0, bounce_groups.1];

    let mut src_to_bounce: Option<TransferGroup> = None;
    let mut bounce_to_dst: Option<TransferGroup>;
    let mut src_iter = src_block_ids.iter();
    let mut dst_iter = dst_block_ids.iter();

    loop {
        // Prepare bounce→dst transfer from previous iteration's staging
        bounce_to_dst = src_to_bounce
            .as_ref()
            .map(|(src_ids, bounce_buffer_group)| {
                (
                    dst_iter
                        .by_ref()
                        .take(src_ids.len())
                        .copied()
                        .collect::<Vec<_>>(),
                    *bounce_buffer_group,
                )
            });

        // Determine which bounce group to use for next src→bounce transfer
        let bounce_group = src_to_bounce
            .map(|(_, bounce_buffer_group)| !bounce_buffer_group)
            .unwrap_or(false);

        // Prepare next src→bounce transfer
        let new_src_ids = src_iter
            .by_ref()
            .take(bounce_groups[bounce_group as usize].len())
            .copied()
            .collect::<Vec<_>>();

        src_to_bounce = if new_src_ids.is_empty() {
            None
        } else {
            Some((new_src_ids, bounce_group))
        };

        // Exit when no more transfers to do
        if src_to_bounce.is_none() && bounce_to_dst.is_none() {
            break;
        }

        // Execute transfers in parallel (src→bounce and bounce→dst overlap)
        let mut futures = Vec::new();
        if let Some(src_to_bounce) = src_to_bounce.as_ref() {
            let notification = execute_direct_transfer(
                src,
                bounce_layout,
                &src_to_bounce.0,
                &bounce_groups[src_to_bounce.1 as usize][0..src_to_bounce.0.len()],
                layer_range.clone(),
                None,
                first_strategy,
                ctx,
            )?;
            futures.push(notification.into_future());
        }

        if let Some(bounce_to_dst) = bounce_to_dst.as_ref() {
            let notification = execute_direct_transfer(
                bounce_layout,
                dst,
                &bounce_groups[bounce_to_dst.1 as usize][0..bounce_to_dst.0.len()],
                &bounce_to_dst.0,
                layer_range.clone(),
                None,
                second_strategy,
                ctx,
            )?;
            futures.push(notification.into_future());
        }

        futures::future::try_join_all(futures).await?;
    }

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
        None,
        first_strategy,
        ctx,
    )?
    .await?;

    execute_direct_transfer(
        bounce_layout,
        dst,
        bounce_ids_to_use,
        dst_block_ids,
        layer_range.clone(),
        None,
        second_strategy,
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
    let handle = event.handle();
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

        // Handle case where bounce buffer is smaller than transfer size
        if num_bounce_blocks < src_block_ids.len() {
            // Process in chunks that fit the bounce buffer
            for (src_block_ids, dst_block_ids) in src_block_ids
                .chunks(num_bounce_blocks)
                .zip(dst_block_ids.chunks(num_bounce_blocks))
            {
                let bounce_block_ids_to_use = &bounce_buffer_spec.block_ids[..src_block_ids.len()];
                if let Err(e) = execute_two_hop_transfer_chunk(
                    &src_clone,
                    &bounce_buffer_spec.layout,
                    &dst_clone,
                    src_block_ids,
                    bounce_block_ids_to_use,
                    dst_block_ids,
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
        } else if num_bounce_blocks == 1 {
            // Single bounce block: use sequential chunk processing
            let bounce_block_ids_to_use = &bounce_buffer_spec.block_ids[..src_block_ids.len()];
            let result = execute_two_hop_transfer_chunk(
                &src_clone,
                &bounce_buffer_spec.layout,
                &dst_clone,
                src_block_ids.as_slice(),
                bounce_block_ids_to_use,
                dst_block_ids.as_slice(),
                first_strategy,
                second_strategy,
                &options.layer_range,
                &ctx_clone,
            )
            .await;

            match result {
                Ok(_) => {
                    let _ = system.trigger(handle);
                }
                Err(e) => {
                    let _ = system.poison(handle, e.to_string());
                }
            }
        } else {
            // Multiple bounce blocks: use optimized double-buffering
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

pub struct TransferNotification {
    status: Arc<AtomicBool>,
}

impl Default for TransferNotification {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferNotification {
    pub fn new() -> Self {
        Self {
            status: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn done() -> Self {
        Self {
            status: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn is_complete(&self) -> bool {
        self.status.load(Ordering::Relaxed)
    }
}
