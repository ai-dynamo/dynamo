// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stub collective operations implementation.
//!
//! This module provides a no-op implementation of [`CollectiveOps`] for testing
//! and single-worker scenarios where no actual collective communication is needed.

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use dynamo_nova::events::LocalEventSystem;

use crate::physical::transfer::TransferCompleteNotification;
use crate::v2::logical::LogicalLayoutHandle;
use crate::v2::BlockId;

use super::CollectiveOps;

/// Stub collective operations implementation.
///
/// This implementation completes immediately without actually performing any
/// collective communication. Use for testing or when collective operations
/// are not yet implemented (e.g., before NCCL integration).
///
/// # Safety
///
/// This stub does NOT perform actual data transfer. Using it in production
/// with `ReplicatedDataWorker` will result in incorrect behavior where
/// non-rank-0 workers have uninitialized data.
///
/// # Example
///
/// ```rust,ignore
/// use kvbm::v2::distributed::collectives::StubCollectiveOps;
///
/// let collective = StubCollectiveOps::new(events, 0, 1);
///
/// // Operations complete immediately without data transfer
/// let notification = collective.broadcast(
///     LogicalLayoutHandle::G1,
///     LogicalLayoutHandle::G1,
///     &src_block_ids,
///     &dst_block_ids,
///     None,
/// )?;
/// ```
pub struct StubCollectiveOps {
    events: Arc<LocalEventSystem>,
    rank: usize,
    world_size: usize,
}

impl StubCollectiveOps {
    /// Create a new stub collective ops.
    ///
    /// # Arguments
    /// * `events` - The event system for creating completion notifications
    /// * `rank` - The rank of this worker in the collective group
    /// * `world_size` - The total number of workers in the collective group
    pub fn new(events: Arc<LocalEventSystem>, rank: usize, world_size: usize) -> Self {
        Self {
            events,
            rank,
            world_size,
        }
    }

    /// Create a stub for single-worker scenarios (rank 0, world_size 1).
    pub fn single_worker(events: Arc<LocalEventSystem>) -> Self {
        Self::new(events, 0, 1)
    }
}

impl CollectiveOps for StubCollectiveOps {
    fn broadcast(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification> {
        tracing::warn!(
            rank = self.rank,
            world_size = self.world_size,
            ?src,
            ?dst,
            num_src_blocks = src_block_ids.len(),
            num_dst_blocks = dst_block_ids.len(),
            ?layer_range,
            "StubCollectiveOps::broadcast called - completing immediately without actual transfer"
        );

        // Create an event that's already triggered (immediate completion)
        let event = self.events.new_event()?;
        let handle = event.handle();
        event.trigger()?;

        let awaiter = self.events.awaiter(handle)?;
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}
