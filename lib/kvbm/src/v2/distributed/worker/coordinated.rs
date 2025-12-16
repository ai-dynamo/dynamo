// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CoordinatedWorker - Leader's view of a worker with coordination state.
//!
//! This module provides a wrapper around the Worker trait that adds coordination
//! state needed by the leader, including local layout handles and remote handle
//! mappings for cross-leader transfers.

use std::collections::HashMap;
use std::sync::{OnceLock, RwLock};

use anyhow::Result;
use futures::future::BoxFuture;

use crate::physical::manager::LayoutHandle;
use crate::physical::transfer::{PhysicalLayout, TransferOptions};
use crate::v2::distributed::object::ObjectBlockOps;
use crate::v2::{BlockId, InstanceId, SequenceHash};

use super::{
    LogicalLayoutHandle, RemoteDescriptor, SerializedLayout, TransferCompleteNotification, Worker,
    WorkerLayoutResponse,
};

/// Leader's view of a worker with coordination state.
///
/// # Coordination State vs Execution State
///
/// CoordinatedWorker maintains **coordination state** - the leader's view of what
/// handles a worker has and how to route transfers. This is distinct from
/// **execution state** which [`DirectWorker`] maintains for actual transfer execution.
///
/// | State Type | Owner | Purpose |
/// |------------|-------|---------|
/// | Execution | DirectWorker | Handles needed by TransferManager to execute |
/// | Coordination | CoordinatedWorker | Leader's tracking for routing decisions |
///
/// When the inner worker is a DirectWorker, handles exist in both places. This
/// duplication is intentional:
/// - DirectWorker needs handles to call TransferManager
/// - CoordinatedWorker provides uniform API for local AND remote workers
/// - NovaWorkerClient is stateless, so leader must track handles somewhere
///
/// # Usage
///
/// ```ignore
/// // Leader creates CoordinatedWorker wrapping actual worker
/// let worker = CoordinatedWorker::new(
///     Box::new(direct_worker),
///     rank,
///     host_instance,
/// );
///
/// // After configure_layouts RPC, populate coordination state
/// worker.apply_layout_response(&response)?;
///
/// // Leader can now query handles for routing
/// if let Some(g2) = worker.local_g2() {
///     // Route G2 transfers through this worker
/// }
/// ```
///
/// # Remote Handle Mappings
///
/// For cross-leader transfers (e.g., Prefill pulling from Decode), the leader
/// imports remote worker metadata and stores rank-aware mappings:
///
/// ```ignore
/// // Prefill leader imports Decode workers' metadata
/// worker.import_remote_metadata(decode_leader_id, decode_rank, metadata).await?;
///
/// // Later, execute transfer using stored mapping
/// worker.transfer_from_remote(
///     decode_leader_id,
///     decode_rank,
///     LogicalLayoutHandle::G2,  // source
///     src_block_ids,
///     LogicalLayoutHandle::G2,  // destination
///     dst_block_ids,
///     options,
/// )?;
/// ```
///
/// [`DirectWorker`]: super::DirectWorker
pub struct CoordinatedWorker {
    /// The actual worker (local DirectWorker or remote NovaWorkerClient).
    /// CoordinatedWorker delegates execution to this inner worker.
    inner: Box<dyn Worker>,

    /// This worker's rank under its leader (0-indexed).
    /// Used for asymmetric TP routing between leaders with different worker counts.
    rank: usize,

    /// Instance ID of the process hosting this worker.
    /// For DirectWorker: same as leader's instance.
    /// For NovaWorkerClient: the remote worker's instance.
    host_instance: InstanceId,

    // =========================================================================
    // Coordination State - leader's view of this worker's handles
    // =========================================================================
    /// G1 (GPU KV cache) layout handle.
    /// Populated from WorkerLayoutResponse after configure_layouts RPC.
    local_g1: OnceLock<LayoutHandle>,

    /// G2 (Host/pinned cache) layout handle.
    /// Populated from WorkerLayoutResponse after configure_layouts RPC.
    local_g2: OnceLock<LayoutHandle>,

    /// G3 (Disk cache) layout handle.
    /// Populated from WorkerLayoutResponse after configure_layouts RPC.
    local_g3: OnceLock<LayoutHandle>,

    /// Remote handle mappings for cross-leader transfers.
    /// Key: (remote_leader_id, remote_rank, logical_type) → physical_handle
    ///
    /// Unlike DirectWorker's remote_handles (keyed by instance only), this
    /// includes rank for asymmetric TP routing. When Prefill (TP=4) pulls from
    /// Decode (TP=2), each Prefill worker needs to know which Decode worker(s)
    /// to pull from.
    remote_handles: RwLock<HashMap<(InstanceId, usize, LogicalLayoutHandle), LayoutHandle>>,
}

impl CoordinatedWorker {
    /// Create a new CoordinatedWorker wrapping an existing Worker.
    pub fn new(inner: Box<dyn Worker>, rank: usize, host_instance: InstanceId) -> Self {
        Self {
            inner,
            rank,
            host_instance,
            local_g1: OnceLock::new(),
            local_g2: OnceLock::new(),
            local_g3: OnceLock::new(),
            remote_handles: RwLock::new(HashMap::new()),
        }
    }

    /// Get this worker's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the instance ID of the process hosting this worker.
    pub fn host_instance(&self) -> InstanceId {
        self.host_instance
    }

    /// Get a reference to the underlying Worker.
    pub fn inner(&self) -> &dyn Worker {
        &*self.inner
    }

    /// Set the local G1 (GPU KV) handle.
    ///
    /// # Arguments
    /// * `handle` - G1 layout handle
    ///
    /// # Errors
    /// Returns error if G1 handle was already set.
    pub fn set_local_g1(&self, handle: LayoutHandle) -> Result<()> {
        self.local_g1
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G1 handle already set"))
    }

    /// Set the local G2 (Host) handle.
    ///
    /// # Arguments
    /// * `handle` - G2 layout handle
    ///
    /// # Errors
    /// Returns error if G2 handle was already set.
    pub fn set_local_g2(&self, handle: LayoutHandle) -> Result<()> {
        self.local_g2
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G2 handle already set"))
    }

    /// Set the local G3 (Disk) handle.
    ///
    /// # Arguments
    /// * `handle` - G3 layout handle
    ///
    /// # Errors
    /// Returns error if G3 handle was already set.
    pub fn set_local_g3(&self, handle: LayoutHandle) -> Result<()> {
        self.local_g3
            .set(handle)
            .map_err(|_| anyhow::anyhow!("G3 handle already set"))
    }

    /// Apply layout response from configure_layouts RPC.
    ///
    /// This is the primary way to populate coordination state. After the leader
    /// sends a configure_layouts RPC to the worker, the response contains the
    /// handles that were created. This method extracts those handles from the
    /// serialized metadata.
    ///
    /// # Arguments
    /// * `response` - The WorkerLayoutResponse from configure_layouts RPC
    ///
    /// # Example
    /// ```ignore
    /// // Leader calls configure_layouts on worker
    /// let response = worker_client.configure_layouts(config).await?;
    ///
    /// // Populate coordination state from response
    /// coordinated_worker.apply_layout_response(&response)?;
    /// ```
    pub fn apply_layout_response(&self, response: &WorkerLayoutResponse) -> Result<()> {
        // Extract handles from the metadata
        let unpacked = response.metadata.unpack()?;

        for descriptor in &unpacked.layouts {
            match descriptor.logical_type {
                LogicalLayoutHandle::G1 => {
                    let _ = self.local_g1.set(descriptor.handle);
                }
                LogicalLayoutHandle::G2 => {
                    let _ = self.local_g2.set(descriptor.handle);
                }
                LogicalLayoutHandle::G3 => {
                    let _ = self.local_g3.set(descriptor.handle);
                }
                LogicalLayoutHandle::G4 => {
                    // G4 (object store) not tracked locally
                }
            }
        }

        Ok(())
    }

    /// Get the local G1 handle if set.
    pub fn local_g1(&self) -> Option<LayoutHandle> {
        self.local_g1.get().copied()
    }

    /// Get the local G2 handle if set.
    pub fn local_g2(&self) -> Option<LayoutHandle> {
        self.local_g2.get().copied()
    }

    /// Get the local G3 handle if set.
    pub fn local_g3(&self) -> Option<LayoutHandle> {
        self.local_g3.get().copied()
    }

    /// Import metadata from a remote worker and store handle mappings.
    ///
    /// This is called when the leader receives metadata from another leader's
    /// workers during cross-leader coordination (e.g., prefill→decode).
    ///
    /// # Arguments
    /// * `remote_leader_id` - Instance ID of the remote leader
    /// * `remote_rank` - Rank of the remote worker under its leader
    /// * `metadata` - Serialized layout metadata from the remote worker
    pub async fn import_remote_metadata(
        &self,
        remote_leader_id: InstanceId,
        remote_rank: usize,
        metadata: SerializedLayout,
    ) -> Result<()> {
        // Unpack metadata to get logical type info
        let unpacked = metadata.unpack()?;

        // Import into the underlying worker so NIXL knows about the remote
        let repacked = SerializedLayout::pack(
            unpacked.worker_address.clone(),
            unpacked.nixl_metadata.clone(),
            unpacked.layouts.clone(),
        )?;
        let response = self.inner.import_metadata(repacked)?;
        let _handles = response.await?;

        // Store mappings for later lookups
        let mut mapping = self.remote_handles.write().unwrap();
        for descriptor in &unpacked.layouts {
            mapping.insert(
                (remote_leader_id, remote_rank, descriptor.logical_type),
                descriptor.handle,
            );
        }

        Ok(())
    }

    /// Look up physical handle for a remote transfer.
    ///
    /// # Arguments
    /// * `remote_leader_id` - Instance ID of the remote leader
    /// * `remote_rank` - Rank of the remote worker
    /// * `logical_type` - Logical layout type (G1/G2/G3)
    pub fn resolve_remote_handle(
        &self,
        remote_leader_id: InstanceId,
        remote_rank: usize,
        logical_type: LogicalLayoutHandle,
    ) -> Option<LayoutHandle> {
        self.remote_handles
            .read()
            .unwrap()
            .get(&(remote_leader_id, remote_rank, logical_type))
            .copied()
    }

    /// Check if remote metadata has been imported for a specific remote worker.
    pub fn has_remote_metadata(&self, remote_leader_id: InstanceId, remote_rank: usize) -> bool {
        let handles = self.remote_handles.read().unwrap();
        handles
            .keys()
            .any(|(leader, rank, _)| *leader == remote_leader_id && *rank == remote_rank)
    }

    /// Execute transfer from a remote worker.
    ///
    /// This method looks up the remote handle from stored mappings and
    /// executes an RDMA transfer to pull data from the remote worker.
    ///
    /// # Arguments
    /// * `remote_leader_id` - Instance ID of the remote leader
    /// * `remote_rank` - Rank of the source worker under its leader
    /// * `src_logical` - Source logical layout type (e.g., G2)
    /// * `src_block_ids` - Block IDs on the remote to pull
    /// * `dst_logical` - Destination logical layout type on this worker
    /// * `dst_block_ids` - Destination block IDs
    /// * `options` - Transfer options
    #[allow(clippy::too_many_arguments)]
    pub fn transfer_from_remote(
        &self,
        remote_leader_id: InstanceId,
        remote_rank: usize,
        src_logical: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst_logical: LogicalLayoutHandle,
        dst_block_ids: std::sync::Arc<[BlockId]>,
        options: TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let src_handle = self
            .resolve_remote_handle(remote_leader_id, remote_rank, src_logical)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No mapping for remote ({}, rank {}, {:?})",
                    remote_leader_id,
                    remote_rank,
                    src_logical
                )
            })?;

        let src = RemoteDescriptor::Layout {
            handle: src_handle,
            block_ids: src_block_ids,
        };
        self.inner
            .execute_remote_onboard(src, dst_logical, dst_block_ids, options)
    }
}

impl ObjectBlockOps for CoordinatedWorker {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        // Delegate to inner worker - Worker trait now extends ObjectBlockOps
        self.inner.has_blocks(keys)
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Delegate to inner worker - Worker trait now extends ObjectBlockOps
        self.inner.put_blocks(keys, layout, block_ids)
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        layout: PhysicalLayout,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // Delegate to inner worker - Worker trait now extends ObjectBlockOps
        self.inner.get_blocks(keys, layout, block_ids)
    }
}

#[cfg(test)]
mod tests {
    // TODO: Add tests with mock Worker implementation
}
