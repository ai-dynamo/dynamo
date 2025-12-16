// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer executors for different copy strategies.

pub(super) mod cuda;
mod memcpy;
mod nixl;

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
use tokio::sync::Mutex;

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

#[derive(Default)]
#[expect(dead_code)]
pub(crate) struct TransferOptionsInternal {
    layer_range: Option<Range<usize>>,
    nixl_write_notification: Option<u64>,
    bounce_buffer: Option<BounceBufferInternal>,
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

    pub(crate) fn build(self) -> Result<TransferOptionsInternal> {
        Ok(TransferOptionsInternal {
            layer_range: self.layer_range,
            nixl_write_notification: self.nixl_write_notification,
            bounce_buffer: self.bounce_buffer,
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
