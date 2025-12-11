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

        if num_bounce_blocks < src_block_ids.len() {
            for (src_block_ids, dst_block_ids) in src_block_ids
                .chunks(num_bounce_blocks)
                .zip(dst_block_ids.chunks(num_bounce_blocks))
            {
                let bounce_block_ids_to_use =
                    &bounce_buffer_spec.block_ids()[..src_block_ids.len()];
                if let Err(e) = execute_two_hop_transfer_chunk(
                    &src_clone,
                    bounce_buffer_spec.layout(),
                    &dst_clone,
                    src_block_ids,
                    bounce_block_ids_to_use,
                    dst_block_ids,
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
            let bounce_block_ids_to_use = &bounce_buffer_spec.block_ids()[..src_block_ids.len()];
            let result = execute_two_hop_transfer_chunk(
                &src_clone,
                bounce_buffer_spec.layout(),
                &dst_clone,
                src_block_ids.as_slice(),
                bounce_block_ids_to_use,
                dst_block_ids.as_slice(),
                first_strategy,
                second_strategy,
                &options_clone.layer_range,
                &ctx_clone,
            )
            .await;

            tx.send(result).unwrap();
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
