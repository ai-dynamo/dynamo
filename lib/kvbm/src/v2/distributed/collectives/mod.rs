// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Collective communication operations for distributed workers.
//!
//! This module provides infrastructure for collective operations needed by
//! replicated data workers. It defines the [`CollectiveOps`] trait and provides
//! multiple implementations:
//!
//! - [`StubCollectiveOps`]: No-op implementation for testing and single-worker scenarios
//! - [`NcclCollectives`]: NCCL-based implementation for GPU collective operations (requires `nccl` feature)
//!
//! # Architecture
//!
//! In MLA (Multi-head Latent Attention) scenarios, KV blocks are replicated across
//! all workers rather than sharded. This means only rank 0 needs G2/G3 storage -
//! other ranks receive data via broadcast from rank 0 after it loads from G2/G3.
//!
//! ```text
//! Rank 0:   G3 (disk) ←→ G2 (host) ←→ G1 (GPU) ───broadcast──→ Other ranks G1
//! Rank 1-N: [no G2/G3]                G1 (GPU) ←──────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use kvbm::v2::distributed::collectives::{CollectiveOps, StubCollectiveOps};
//!
//! let collective = StubCollectiveOps::new(events);
//!
//! // Broadcast G1 blocks from rank 0 to all ranks
//! let notification = collective.broadcast(
//!     LogicalLayoutHandle::G1,
//!     LogicalLayoutHandle::G1,
//!     &src_block_ids,
//!     &dst_block_ids,
//!     Some(0..32),
//! )?;
//! notification.await_completion()?;
//! ```

mod stub;

#[cfg(feature = "nccl")]
mod bootstrap;
#[cfg(feature = "nccl")]
mod nccl;

pub use stub::StubCollectiveOps;

#[cfg(feature = "nccl")]
pub use bootstrap::NcclBootstrap;
#[cfg(feature = "nccl")]
pub use nccl::{CudaEventRegistrar, LayoutResolver, NcclCollectives};

use std::ops::Range;

use anyhow::Result;

use crate::physical::transfer::TransferCompleteNotification;
use crate::v2::BlockId;
use crate::v2::logical::LogicalLayoutHandle;

/// Collective communication operations for distributed workers.
///
/// This trait defines the collective operations needed by replicated data workers
/// to broadcast data across ranks. Implementations may use NCCL, NIXL, or other
/// collective communication libraries.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow sharing across threads.
/// NCCL operations are inherently thread-safe when used correctly (one stream
/// per communicator per thread).
pub trait CollectiveOps: Send + Sync {
    /// Broadcast blocks from rank 0 to all other ranks.
    ///
    /// This operation transfers the specified blocks from the source layout on
    /// rank 0 to the destination layout on all other ranks. Optionally, a layer
    /// range can be specified to transfer only a subset of layers (for pipelined
    /// loading).
    ///
    /// # Arguments
    /// * `src` - The source logical layout (typically G1 on rank 0)
    /// * `dst` - The destination logical layout (typically G1 on all ranks)
    /// * `src_block_ids` - The block IDs to read from on the source
    /// * `dst_block_ids` - The block IDs to write to on the destination
    /// * `layer_range` - Optional range of layers to transfer. If None, all layers are transferred.
    ///
    /// # Returns
    /// A notification that completes when the broadcast is done on all ranks.
    ///
    /// # Synchronization
    ///
    /// This is a collective operation - all ranks must call this method with
    /// the same arguments for the broadcast to complete correctly. The returned
    /// notification signals local completion; global completion is guaranteed
    /// by the collective semantics of the underlying implementation.
    fn broadcast(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification>;

    /// Get the rank of this worker in the collective group.
    fn rank(&self) -> usize;

    /// Get the total number of workers in the collective group.
    fn world_size(&self) -> usize;
}
