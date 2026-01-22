// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Collective communication operations for distributed workers.
//!
//! This module provides the infrastructure for collective operations needed by
//! replicated data workers. Initially provides a stub implementation; NCCL
//! integration will be added later.
//!
//! # Collective Operations
//!
//! - **Broadcast**: Send data from rank 0 to all other ranks (used for G1 loading)
//!
//! # Example
//!
//! ```rust,ignore
//! use kvbm::v2::distributed::parallelism::collective::{CollectiveOps, StubCollectiveOps};
//!
//! let collective = StubCollectiveOps::new(events);
//!
//! // Broadcast G1 blocks from rank 0 to all ranks
//! let notification = collective.g1_broadcast(&block_ids, Some(0..32))?;
//! notification.await_completion()?;
//! ```

use std::ops::Range;
use std::sync::Arc;

use anyhow::Result;
use dynamo_nova::events::LocalEventSystem;

use crate::physical::transfer::context::TransferCompleteNotification;
use crate::v2::BlockId;

/// Collective communication operations for distributed workers.
///
/// This trait defines the collective operations needed by replicated data workers
/// to broadcast data across ranks. Implementations may use NCCL, NIXL, or other
/// collective communication libraries.
pub trait CollectiveOps: Send + Sync {
    /// Broadcast G1 blocks from rank 0 to all other ranks.
    ///
    /// This operation transfers the specified blocks in the G1 (GPU) tier from
    /// rank 0 to all other ranks. Optionally, a layer range can be specified
    /// to transfer only a subset of layers (for pipelined loading).
    ///
    /// # Arguments
    /// * `block_ids` - The block IDs to broadcast
    /// * `layer_range` - Optional range of layers to transfer. If None, all layers are transferred.
    ///
    /// # Returns
    /// A notification that completes when the broadcast is done on all ranks.
    fn g1_broadcast(
        &self,
        block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification>;
}

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
pub struct StubCollectiveOps {
    events: Arc<LocalEventSystem>,
}

impl StubCollectiveOps {
    /// Create a new stub collective ops.
    ///
    /// # Arguments
    /// * `events` - The event system for creating completion notifications
    pub fn new(events: Arc<LocalEventSystem>) -> Self {
        Self { events }
    }
}

impl CollectiveOps for StubCollectiveOps {
    fn g1_broadcast(
        &self,
        block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification> {
        tracing::warn!(
            num_blocks = block_ids.len(),
            ?layer_range,
            "StubCollectiveOps::g1_broadcast called - completing immediately without actual transfer"
        );

        // Create an event that's already triggered (immediate completion)
        let event = self.events.new_event()?;
        let handle = event.handle();
        event.trigger()?;

        let awaiter = self.events.awaiter(handle)?;
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }
}

// Tests for StubCollectiveOps require integration with KvbmRuntime to access
// the proper event system. See integration tests for collective operation testing.
