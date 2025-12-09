// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer executors for different copy strategies.
//!
//! # Pipelined Two-Hop Transfers
//!
//! For Device ↔ Object Storage transfers, we use a pipelined approach:
//!
//! ```text
//! Sequential (old):
//! ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
//! │ GPU→Host C1  │ │ Host→S3 C1   │ │ GPU→Host C2  │ │ Host→S3 C2   │
//! └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
//!
//! Pipelined (new, with double-buffering):
//! ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
//! │ GPU→Host C1  │ │ GPU→Host C2  │ │ GPU→Host C3  │   (bounce slot A/B alternating)
//! └──────────────┘ └──────────────┘ └──────────────┘
//!       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
//!       │ Host→S3 C1   │ │ Host→S3 C2   │ │ Host→S3 C3   │  (overlapped)
//!       └──────────────┘ └──────────────┘ └──────────────┘
//! ```

mod contiguous_transfer;
pub(super) mod cuda;
mod memcpy;
mod nixl;
mod object_transfer;

use super::strategy::select_strategy;
use super::validation::validate_block_transfer;
use super::{
    DescriptorHint, PhysicalLayout, TransferContext, TransferOptions, TransferPlan,
    TransferStrategy,
};
use crate::block_manager::v2::physical::transfer::{
    StorageKind, context::TransferCompleteNotification,
};
use anyhow::Result;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// Re-export the NIXL transfer builder for public use
pub use nixl::NixlTransferBuilder;

// Re-export for testing (used by descriptor_tests)
#[cfg(feature = "testing-nixl")]
#[allow(unused_imports)]
pub(crate) use nixl::{DescriptorParams, build_descriptors};

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
            options.descriptor_hint,
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
    descriptor_hint: DescriptorHint,
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
                .strategy(strategy)
                .descriptor_hint(descriptor_hint);

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

/// Execute a single chunk of a two-hop transfer (sequential, for single-chunk transfers).
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
    descriptor_hint: DescriptorHint,
    ctx: &TransferContext,
) -> Result<()> {
    let bounce_ids_to_use = &bounce_block_ids[..src_block_ids.len()];
    let num_blocks = src_block_ids.len();
    let chunk_start = std::time::Instant::now();

    // First hop: src → bounce (use Auto, bounce is always host memory)
    let hop1_start = std::time::Instant::now();
    execute_direct_transfer(
        src,
        bounce_layout,
        src_block_ids,
        bounce_ids_to_use,
        layer_range.clone(),
        DescriptorHint::Auto,
        first_strategy,
        ctx,
    )?
    .await?;
    let hop1_elapsed = hop1_start.elapsed();

    // Second hop: bounce → dst (use the original hint for object storage)
    let hop2_start = std::time::Instant::now();
    execute_direct_transfer(
        bounce_layout,
        dst,
        bounce_ids_to_use,
        dst_block_ids,
        layer_range.clone(),
        descriptor_hint,
        second_strategy,
        ctx,
    )?
    .await?;
    let hop2_elapsed = hop2_start.elapsed();

    let chunk_elapsed = chunk_start.elapsed();

    tracing::debug!(
        target: "object_transfer_timing",
        blocks = num_blocks,
        hop1_ms = hop1_elapsed.as_secs_f64() * 1000.0,
        hop2_ms = hop2_elapsed.as_secs_f64() * 1000.0,
        total_ms = chunk_elapsed.as_secs_f64() * 1000.0,
        idle_pct = (hop1_elapsed.as_secs_f64() / chunk_elapsed.as_secs_f64()) * 100.0,
        "TWO_HOP_CHUNK: {} blocks | hop1(GPU→Host)={:.2}ms | hop2(Host→S3)={:.2}ms | total={:.2}ms | S3_idle_during_GPU={:.1}%",
        num_blocks,
        hop1_elapsed.as_secs_f64() * 1000.0,
        hop2_elapsed.as_secs_f64() * 1000.0,
        chunk_elapsed.as_secs_f64() * 1000.0,
        (hop1_elapsed.as_secs_f64() / chunk_elapsed.as_secs_f64()) * 100.0,
    );

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

/// Execute a two-hop transfer: src → bounce → dst.
///
/// The bounce buffer must be large enough to hold the entire transfer.
/// This is guaranteed by the ConnectorTransferBatcher which splits large
/// transfers into chunks that fit within the bounce buffer slot size.
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

    let src_clone = src.clone();
    let dst_clone = dst.clone();
    let src_block_ids = src_block_ids.to_vec();
    let dst_block_ids = dst_block_ids.to_vec();
    let options_clone = options.clone();

    let handle = ctx.tokio();
    let ctx_clone = ctx.clone();
    handle.spawn(async move {
        let Some(ref bounce_buffer_spec) = options_clone.bounce_buffer else {
            let _ = tx.send(Err(anyhow::anyhow!(
                "Two-hop transfers require a bounce buffer."
            )));
            return;
        };

        if bounce_buffer_spec.layout().location() != bounce_location {
            let _ = tx.send(Err(anyhow::anyhow!(
                "Bounce buffer layout does not match bounce location."
            )));
            return;
        }

        let num_bounce_blocks = bounce_buffer_spec.block_ids().len();
        let total_blocks = src_block_ids.len();

        // The batcher ensures transfers fit within the bounce buffer slot.
        // If this assertion fails, there's a configuration mismatch.
        if num_bounce_blocks < total_blocks {
            let _ = tx.send(Err(anyhow::anyhow!(
                "Transfer size ({}) exceeds bounce buffer capacity ({}). \
                 Ensure DYN_KVBM_TRANSFER_BATCH_SIZE <= bounce buffer slot size.",
                total_blocks,
                num_bounce_blocks
            )));
            return;
        }

        let transfer_start = std::time::Instant::now();

        tracing::debug!(
            target: "object_transfer_timing",
            total_blocks = total_blocks,
            bounce_size = num_bounce_blocks,
            "TWO_HOP_START: {} blocks, bounce_size={}",
            total_blocks,
            num_bounce_blocks,
        );

        let bounce_block_ids_to_use = &bounce_buffer_spec.block_ids()[..total_blocks];
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
            options_clone.descriptor_hint,
            &ctx_clone,
        )
        .await;

        let total_elapsed = transfer_start.elapsed();
        tracing::debug!(
            target: "object_transfer_timing",
            total_blocks = total_blocks,
            total_ms = total_elapsed.as_secs_f64() * 1000.0,
            "TWO_HOP_COMPLETE: {} blocks, total={:.2}ms",
            total_blocks,
            total_elapsed.as_secs_f64() * 1000.0,
        );

        // Use let _ to gracefully handle receiver being dropped (e.g., due to timeout)
        let _ = tx.send(result);
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
