// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replicated data worker for MLA (Multi-head Latent Attention) scenarios.
//!
//! In MLA architectures, KV blocks are replicated across all workers rather than
//! sharded. This means only rank 0 needs G2/G3 storage - other ranks receive
//! data via broadcast from rank 0 after it loads from G2/G3.
//!
//! # Architecture
//!
//! ```text
//! Rank 0:   G3 (disk) ←→ G2 (host) ←→ G1 (GPU) ───broadcast──→ Other ranks G1
//! Rank 1-N: [no G2/G3]                G1 (GPU) ←──────────────────────┘
//! ```
//!
//! # Transfer Semantics
//!
//! | Operation | Behavior |
//! |-----------|----------|
//! | G2/G3 → G1 (onboard) | Rank 0 transfers, then broadcasts to all ranks |
//! | G1 → G2/G3 (offload) | Rank 0 only (other ranks don't have G2/G3) |
//! | G2 ↔ G3 | Rank 0 only |
//! | G1 → G1 (local) | All ranks execute (data is replicated) |

use super::*;

use crate::KvbmRuntime;
use anyhow::Result;

use std::{ops::Range, sync::Arc};

// todo: re-house this in distributed::collectives
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
    fn broadcast(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        layer_range: Option<Range<usize>>,
    ) -> Result<TransferCompleteNotification>;
}

/// Replicated data worker for MLA scenarios.
///
/// Only rank 0 has G2/G3 storage. When loading data to G1, rank 0 transfers
/// from G2/G3 and then broadcasts to all other ranks via collective operations.
///
/// # Requirements
///
/// - Workers must be initialized such that only rank 0 has G2/G3 handles
/// - A [`CollectiveOps`] implementation must be provided for broadcasting
///
/// # Trait Implementations
///
/// - [`WorkerTransfers`]: Specialized routing based on source/destination tiers
/// - [`ParallelWorker`]: Delegates to inner SpmdWorker
/// - [`ObjectBlockOps`]: Routes to rank 0 only (it has the G2 layout for resolution)
pub struct ReplicatedDataWorker {
    inner: Arc<PhysicalWorker>,
    runtime: Arc<KvbmRuntime>,
    collective: Arc<dyn CollectiveOps>,
}

impl ReplicatedDataWorker {
    /// Create a new ReplicatedDataWorker.
    ///
    /// # Arguments
    /// * `workers` - The underlying workers (one per rank). Only workers[0] should have G2/G3.
    /// * `events` - The event system for aggregating completion notifications
    /// * `runtime` - The tokio runtime handle for spawning aggregation tasks
    /// * `collective` - The collective ops implementation for broadcasting
    ///
    /// # Panics
    ///
    /// Debug builds will panic if workers.len() < 1.
    pub fn new(
        worker: Arc<PhysicalWorker>, // perhaps use a trait to abstract this
        runtime: Arc<KvbmRuntime>,
        collective: Arc<dyn CollectiveOps>,
    ) -> Self {
        // todo: ensure worker has a rank

        Self {
            inner: worker,
            runtime,
            collective,
        }
    }

    /// Get access to the underlying SpmdWorker.
    pub fn inner(&self) -> &PhysicalWorker {
        &self.inner
    }

    /// Get the rank of the underlying worker.
    pub fn rank(&self) -> usize {
        self.inner.rank().expect("Worker must have a rank")
    }

    fn broadcast(
        &self,
        xfer_completion: TransferCompleteNotification,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        unimplemented!()
    }
}

impl WorkerTransfers for ReplicatedDataWorker {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let is_rank0 = self.rank() == 0;
        let use_bcast = dst == LogicalLayoutHandle::G1;

        if src == LogicalLayoutHandle::G1 && dst == LogicalLayoutHandle::G1 {
            return self.inner.execute_local_transfer(
                src,
                dst,
                src_block_ids,
                dst_block_ids.clone(),
                options,
            );
        }

        if !is_rank0 && !use_bcast {
            return Ok(TransferCompleteNotification::completed());
        } else if is_rank0 {
            let xfer_completion = self.inner.execute_local_transfer(
                src,
                dst,
                src_block_ids,
                dst_block_ids.clone(),
                options.clone(),
            )?;

            if use_bcast {
                self.broadcast(xfer_completion, dst, dst_block_ids, options)
            } else {
                Ok(xfer_completion)
            }
        } else {
            let xfer_completion = TransferCompleteNotification::completed();
            self.broadcast(xfer_completion, dst, dst_block_ids, options)
        }
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        unimplemented!()
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        unimplemented!()
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        // Use the shared implementation
        self.inner.connect_remote(instance_id, metadata)
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        self.inner.has_remote_metadata(instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: crate::physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        unimplemented!()
    }
}
