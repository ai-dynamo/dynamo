// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer executors for different copy strategies.

pub(super) mod cuda;
mod memcpy;
mod nixl;

use super::strategy::select_strategy;
use super::validation::validate_block_transfer;
use super::{PhysicalLayout, TransferContext, TransferOptions, TransferPlan, TransferStrategy};
use crate::block_manager::v2::physical::transfer::{
    StorageKind, context::TransferCompleteNotification,
};
use anyhow::Result;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Re-export the NIXL transfer builder for public use
pub use nixl::NixlTransferBuilder;

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
pub fn execute_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    options: TransferOptions,
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
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    match strategy {
        TransferStrategy::Memcpy => {
            memcpy::execute_memcpy_transfer(src, dst, src_block_ids, dst_block_ids, layer_range)
        }
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

            builder.execute(ctx)
        }
        TransferStrategy::Invalid => Err(anyhow::anyhow!(
            "Invalid transfer strategy for src={:?}, dst={:?}",
            src.location(),
            dst.location()
        )),
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_two_hop_transfer_chunk(
    src: &PhysicalLayout,
    bounce_layout: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    bounce_block_ids: &[usize],
    dst_block_ids: &[usize],
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
        ctx,
    )?
    .await?;

    Ok(())
}

/// Parameters for two-hop transfer execution
struct TwoHopTransferParams<'a> {
    src: &'a PhysicalLayout,
    dst: &'a PhysicalLayout,
    src_block_ids: &'a [usize],
    dst_block_ids: &'a [usize],
    first_strategy: TransferStrategy,
    bounce_location: StorageKind,
    second_strategy: TransferStrategy,
    options: TransferOptions,
    ctx: &'a TransferContext,
}

type TransferGroup = (Vec<usize>, bool);

#[allow(clippy::too_many_arguments)]
async fn handle_buffered_transfer(
    src: &PhysicalLayout,
    bounce_layout: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    bounce_block_ids: &[usize],
    dst_block_ids: &[usize],
    first_strategy: TransferStrategy,
    second_strategy: TransferStrategy,
    layer_range: &Option<Range<usize>>,
    ctx: &TransferContext,
) -> Result<()> {
    let bounce_groups =
        &bounce_block_ids[0..std::cmp::min(src_block_ids.len(), bounce_block_ids.len())];
    let bounce_groups = bounce_groups.split_at(bounce_groups.len() / 2);
    let bounce_groups = [bounce_groups.0, bounce_groups.1];

    let mut src_to_bounce: Option<TransferGroup> = None;
    let mut bounce_to_dst: Option<TransferGroup>;
    let mut src_iter = src_block_ids.iter();
    let mut dst_iter = dst_block_ids.iter();

    loop {
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

        let bounce_group = src_to_bounce
            .map(|(_, bounce_buffer_group)| !bounce_buffer_group)
            .unwrap_or(false);

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

        if src_to_bounce.is_none() && bounce_to_dst.is_none() {
            break;
        }

        let mut futures = Vec::new();
        if let Some(src_to_bounce) = src_to_bounce.as_ref() {
            futures.push(execute_direct_transfer(
                src,
                bounce_layout,
                &src_to_bounce.0,
                &bounce_groups[src_to_bounce.1 as usize][0..src_to_bounce.0.len()],
                layer_range.clone(),
                first_strategy,
                ctx,
            )?);
        }

        if let Some(bounce_to_dst) = bounce_to_dst.as_ref() {
            futures.push(execute_direct_transfer(
                bounce_layout,
                dst,
                &bounce_groups[bounce_to_dst.1 as usize][0..bounce_to_dst.0.len()],
                &bounce_to_dst.0,
                layer_range.clone(),
                second_strategy,
                ctx,
            )?);
        }

        futures::future::try_join_all(futures).await?;
    }

    Ok(())
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
    let (tx, rx) = tokio::sync::oneshot::channel();

    // TODO: Cloning all this stuff is not ideal.
    let src_clone = src.clone();
    let dst_clone = dst.clone();

    let src_block_ids = src_block_ids.to_vec();
    let dst_block_ids = dst_block_ids.to_vec();

    let options_clone = options.clone();

    let handle = ctx.tokio();
    let ctx_clone = ctx.clone();
    handle.spawn(async move {
        let Some(ref bounce_buffer_spec) = options_clone.bounce_buffer else {
            tx.send(Err(anyhow::anyhow!(
                "Two-hop transfers require a bounce buffer."
            )))
            .unwrap();
            return;
        };

        if bounce_buffer_spec.layout().location() != bounce_location {
            tx.send(Err(anyhow::anyhow!(
                "Bounce buffer layout does not match bounce location."
            )))
            .unwrap();
            return;
        }

        let num_bounce_blocks = bounce_buffer_spec.block_ids().len();

        if num_bounce_blocks == 1 {
            let bounce_block = bounce_buffer_spec.block_ids()[0];
            for (src_block_id, dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
                if let Err(e) = execute_two_hop_transfer_chunk(
                    &src_clone,
                    bounce_buffer_spec.layout(),
                    &dst_clone,
                    &[*src_block_id],
                    &[bounce_block],
                    &[*dst_block_id],
                    first_strategy,
                    second_strategy,
                    &options_clone.layer_range,
                    &ctx_clone,
                )
                .await
                {
                    tx.send(Err(e)).unwrap();
                    return;
                }
            }
            tx.send(Ok(())).unwrap();
        } else {
            if let Err(e) = handle_buffered_transfer(
                &src_clone,
                bounce_buffer_spec.layout(),
                &dst_clone,
                &src_block_ids,
                bounce_buffer_spec.block_ids(),
                &dst_block_ids,
                first_strategy,
                second_strategy,
                &options_clone.layer_range,
                &ctx_clone,
            )
            .await
            {
                tx.send(Err(e)).unwrap();
                return;
            }
            tx.send(Ok(())).unwrap();
        }
    });

    Ok(TransferCompleteNotification { status: rx })
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
