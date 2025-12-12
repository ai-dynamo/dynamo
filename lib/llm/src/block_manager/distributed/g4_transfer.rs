// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G4 (Object/Disk Storage) transfer handler
//!
//! This module provides [`G4TransferHandler`] - a handler for remote storage transfers.
//!
//! ## Architecture
//!
//! ```text
//! Offload (Host -> Remote):
//!   Host Block (PinnedStorage) -> NIXL Write -> Object/Disk
//!
//! Onboard (Remote -> Host -> Device):
//!   Object/Disk -> NIXL Read -> Host Bounce -> Device Block
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Result, bail};
use nixl_sys::{Agent as NixlAgent, MemType, XferDescList, XferOp};

use crate::block_manager::block::data::local::LocalBlockData;
use crate::block_manager::block::transfer::{TransferContext, WriteTo};
use crate::block_manager::config::ObjectStorageConfig;
use crate::block_manager::metrics_kvbm::KvbmMetrics;
use crate::block_manager::storage::{DeviceStorage, ObjectStorage, PinnedStorage};

use super::registry::{DistributedRegistry, SequenceHashRegistry};


/// Direction of G4 (remote storage) transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum G4TransferDirection {
    /// Local → Remote Storage (offload/persist)
    Offload,
    /// Remote Storage → Local (onboard/restore)
    Onboard,
}

/// Kind of remote storage for G4 transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemoteStorageKind {
    Object,
    Disk,
}

/// Result of a G4 transfer operation.
#[derive(Debug)]
pub struct G4TransferResult {
    /// Number of blocks successfully transferred
    pub transferred: usize,
}

/// Descriptor for a remote block in G4 transfers.
///
/// Encapsulates all information needed to identify and transfer
/// a single block to/from remote storage.
#[derive(Debug, Clone)]
pub enum RemoteDescriptor {
    /// Object storage (S3/GCS) - bucket + key
    Object {
        key: u64,
        bucket: String,
    },
    /// Disk/file storage - key only
    Disk {
        key: u64,
    },
}

impl RemoteDescriptor {
    /// Create a new remote descriptor for object storage.
    pub fn object(key: u64, bucket: String) -> Self {
        Self::Object { key, bucket }
    }

    /// Create a new remote descriptor for disk storage.
    pub fn disk(key: u64) -> Self {
        Self::Disk { key }
    }

    /// Get the key regardless of storage type.
    pub fn key(&self) -> u64 {
        match self {
            Self::Object { key, .. } => *key,
            Self::Disk { key, .. } => *key,
        }
    }

    /// Get the storage kind.
    pub fn storage_kind(&self) -> RemoteStorageKind {
        match self {
            Self::Object { .. } => RemoteStorageKind::Object,
            Self::Disk { .. } => RemoteStorageKind::Disk,
        }
    }
}

#[derive(Clone)]
pub struct G4TransferHandler {
    /// NIXL agent (with OBJ/FILE backends loaded)
    agent: Arc<Option<NixlAgent>>,

    local_registry: SequenceHashRegistry,

    /// Distributed registry for cross-worker deduplication and lookup
    distributed_registry: Option<Arc<dyn DistributedRegistry>>,

    /// Object storage configuration
    config: ObjectStorageConfig,

    /// Host blocks for bounce buffers (shared from host pool)
    host_blocks: Arc<Vec<LocalBlockData<PinnedStorage>>>,

    /// Device blocks for source/destination
    device_blocks: Arc<Vec<LocalBlockData<DeviceStorage>>>,

    /// Transfer context (stream, NIXL agent, buffer pool)
    context: Arc<TransferContext>,

    /// Worker ID for bucket resolution
    worker_id: u64,

    /// Optional KVBM metrics for tracking offload/onboard stats
    kvbm_metrics: Option<KvbmMetrics>,
}

impl G4TransferHandler {
    /// # Arguments
    /// * `agent` - VNIXL agent with OBJ/FILE backends loaded (wrapped in Option)
    /// * `local_registry` - Local registry for tracking offloaded hashes
    /// * `distributed_registry` - Optional distributed registry for cross-worker dedup
    /// * `config` - Object storage configuration (bucket, endpoint, region)
    /// * `host_blocks` - Host blocks for bounce buffers
    /// * `device_blocks` - Device blocks
    /// * `context` - transfer context
    /// * `worker_id` - Worker ID
    /// * `kvbm_metrics` - Optional KVBM metrics for tracking offload/onboard stats
    pub fn new(
        agent: Arc<Option<NixlAgent>>,
        local_registry: SequenceHashRegistry,
        distributed_registry: Option<Arc<dyn DistributedRegistry>>,
        config: ObjectStorageConfig,
        host_blocks: Vec<LocalBlockData<PinnedStorage>>,
        device_blocks: Vec<LocalBlockData<DeviceStorage>>,
        context: Arc<TransferContext>,
        worker_id: u64,
        kvbm_metrics: Option<KvbmMetrics>,
    ) -> Result<Self> {
        // Verify agent is available
        if agent.is_none() {
            bail!("NIXL agent required for G4TransferHandler but was None");
        }

        tracing::info!(
            "Creating G4TransferHandler for worker {}",
            worker_id
        );
        tracing::debug!(
            "G4TransferHandler: {} host blocks, {} device blocks",
            host_blocks.len(),
            device_blocks.len()
        );

        Ok(Self {
            agent,
            local_registry,
            distributed_registry,
            config,
            host_blocks: Arc::new(host_blocks),
            device_blocks: Arc::new(device_blocks),
            context,
            worker_id,
            kvbm_metrics,
        })
    }

    /// Execute a G4 (remote storage) transfer.
    ///
    /// This is the main entry point for remote storage transfers. It delegates
    /// to either offload or onboard based on the direction, and handles both
    /// Object and Disk storage types.
    ///
    /// # Arguments
    /// * `direction` - Whether to offload (local→remote) or onboard (remote→local)
    /// * `storage_kind` - Type of remote storage (Object or Disk)
    /// * `local_block_ids` - Block indices in the local layout
    /// * `remote_descriptors` - Remote descriptors containing object keys and bucket info
    ///
    /// # Returns
    /// Result containing the number of blocks successfully transferred
    pub fn execute_g4_transfer(
        &self,
        direction: G4TransferDirection,
        storage_kind: RemoteStorageKind,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        match (direction, storage_kind) {
            (G4TransferDirection::Offload, RemoteStorageKind::Object) => {
                self.execute_g4_offload_object(local_block_ids, remote_descriptors)
            }
            (G4TransferDirection::Offload, RemoteStorageKind::Disk) => {
                self.execute_g4_offload_disk(local_block_ids, remote_descriptors)
            }
            (G4TransferDirection::Onboard, RemoteStorageKind::Object) => {
                self.execute_g4_onboard_object(local_block_ids, remote_descriptors)
            }
            (G4TransferDirection::Onboard, RemoteStorageKind::Disk) => {
                self.execute_g4_onboard_disk(local_block_ids, remote_descriptors)
            }
        }
    }

    /// Async version of [`Self::execute_g4_transfer`].
    pub async fn execute_g4_transfer_async(
        &self,
        direction: G4TransferDirection,
        storage_kind: RemoteStorageKind,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        match (direction, storage_kind) {
            (G4TransferDirection::Offload, RemoteStorageKind::Object) => {
                self.execute_g4_offload_object_async(local_block_ids, remote_descriptors).await
            }
            (G4TransferDirection::Offload, RemoteStorageKind::Disk) => {
                self.execute_g4_offload_disk_async(local_block_ids, remote_descriptors).await
            }
            (G4TransferDirection::Onboard, RemoteStorageKind::Object) => {
                self.execute_g4_onboard_object_async(local_block_ids, remote_descriptors).await
            }
            (G4TransferDirection::Onboard, RemoteStorageKind::Disk) => {
                self.execute_g4_onboard_disk_async(local_block_ids, remote_descriptors).await
            }
        }
    }

    /// Execute G4 offload to object storage
    fn execute_g4_offload_object(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        // Use blocking runtime for sync version
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available"))?;

        rt.block_on(self.execute_g4_offload_object_async(local_block_ids, remote_descriptors))
    }

    /// Execute G4 offload to object storage.
    async fn execute_g4_offload_object_async(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        use crate::block_manager::block::data::BlockDataViews;

        if local_block_ids.len() != remote_descriptors.len() {
            bail!(
                "local_block_ids.len() ({}) != remote_descriptors.len() ({})",
                local_block_ids.len(),
                remote_descriptors.len()
            );
        }

        if local_block_ids.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        let agent = self.agent.as_ref().as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not available for G4 offload"))?;

        let num_blocks = local_block_ids.len();

        tracing::debug!(
            "Batched G4 offload: {} blocks to object storage",
            num_blocks
        );

        // Use a scope block to ensure all non-Send types are dropped before await
        let (xfer_req, still_pending) = {
            // Register ALL object storage regions with NIXL
            let mut obj_storages = Vec::with_capacity(num_blocks);
            let mut _registration_handles = Vec::with_capacity(num_blocks);

            // Get block size from first block
            let first_block = &self.host_blocks[local_block_ids[0]];
            let first_view = first_block.local_block_view()?;
            let block_size = first_view.size();

            for desc in remote_descriptors.iter() {
                let RemoteDescriptor::Object { key, bucket } = desc else {
                    anyhow::bail!("Expected Object descriptor for object storage offload");
                };
                let obj_storage = ObjectStorage::new(bucket, *key, block_size)
                    .map_err(|e| anyhow::anyhow!("Failed to create object storage: {:?}", e))?;

                let handle = agent
                    .register_memory(&obj_storage, None)
                    .map_err(|e| anyhow::anyhow!("Failed to register object storage: {:?}", e))?;

                obj_storages.push(obj_storage);
                _registration_handles.push(handle);
            }

            // Build transfer descriptor lists with ALL blocks
            let mut src_dl = XferDescList::new(MemType::Dram)?;
            let mut dst_dl = XferDescList::new(MemType::Object)?;

            for (&block_id, desc) in local_block_ids.iter().zip(remote_descriptors.iter()) {
                let host_block = &self.host_blocks[block_id];
                let block_view = host_block.local_block_view()?;
                let addr = unsafe { block_view.as_ptr() as usize };

                // src_dl: source data in host DRAM
                src_dl.add_desc(addr, block_size, 0); // device_id=0 for host

                // dst_dl: destination in object storage (key used as device_id)
                dst_dl.add_desc(0, block_size, desc.key());
            }

            // Create ONE transfer request for ALL blocks
            let agent_name = agent.name();
            let xfer_req = agent.create_xfer_req(
                XferOp::Write,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = agent.post_xfer_req(&xfer_req, None)?;

            (xfer_req, still_pending)
        };

        // Wait for completion if transfer is pending
        if still_pending {
            loop {
                let status = agent.get_xfer_status(&xfer_req)?;
                if status.is_success() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        }

        tracing::debug!(
            "Batched G4 offload complete: {} blocks transferred",
            num_blocks
        );

        Ok(G4TransferResult { transferred: num_blocks })
    }

    /// Execute G4 onboard from object storage (sync version).
    fn execute_g4_onboard_object(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available"))?;

        rt.block_on(self.execute_g4_onboard_object_async(local_block_ids, remote_descriptors))
    }

    /// Execute G4 onboard from object storage (async version).
    ///
    /// Transfers blocks from object storage to host memory using NIXL OBJ backend.
    /// All blocks are batched into a single NIXL transfer request for efficiency.
    async fn execute_g4_onboard_object_async(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        use crate::block_manager::block::data::BlockDataViews;

        if local_block_ids.len() != remote_descriptors.len() {
            bail!(
                "local_block_ids.len() ({}) != remote_descriptors.len() ({})",
                local_block_ids.len(),
                remote_descriptors.len()
            );
        }

        if local_block_ids.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        let agent = self.agent.as_ref().as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not available for G4 onboard"))?;

        let num_blocks = local_block_ids.len();

        tracing::debug!(
            "Batched G4 onboard: {} blocks to host memory",
            num_blocks
        );

        // Use a scope block to ensure all non-Send types are dropped before await
        let (xfer_req, still_pending) = {
            // Register ALL object storage regions with NIXL
            let mut obj_storages = Vec::with_capacity(num_blocks);
            let mut _registration_handles = Vec::with_capacity(num_blocks);

            // Get block size from first block
            let first_block = &self.host_blocks[local_block_ids[0]];
            let first_view = first_block.local_block_view()?;
            let block_size = first_view.size();

            for desc in remote_descriptors.iter() {
                let RemoteDescriptor::Object { key, bucket } = desc else {
                    anyhow::bail!("Expected Object descriptor for object storage onboard");
                };
                let obj_storage = ObjectStorage::new(bucket, *key, block_size)
                    .map_err(|e| anyhow::anyhow!("Failed to create object storage: {:?}", e))?;

                let handle = agent
                    .register_memory(&obj_storage, None)
                    .map_err(|e| anyhow::anyhow!("Failed to register object storage: {:?}", e))?;

                obj_storages.push(obj_storage);
                _registration_handles.push(handle);
            }

            // Build transfer descriptor lists with ALL blocks
            // IMPORTANT: For OBJ backend READ, src_dl must be local DRAM (destination)
            let mut src_dl = XferDescList::new(MemType::Dram)?;  // Local DRAM (destination)
            let mut dst_dl = XferDescList::new(MemType::Object)?;  // Object storage (source)

            for (&block_id, desc) in local_block_ids.iter().zip(remote_descriptors.iter()) {
                let host_block = &self.host_blocks[block_id];
                let block_view = host_block.local_block_view()?;
                let addr = unsafe { block_view.as_ptr() as usize };

                // src_dl = local DRAM buffer where data will be read INTO
                src_dl.add_desc(addr, block_size, 0); // device_id=0 for host

                // dst_dl = object storage (the key is used as device_id)
                dst_dl.add_desc(0, block_size, desc.key());
            }

            // Create ONE transfer request for ALL blocks
            let agent_name = agent.name();
            let xfer_req = agent.create_xfer_req(
                XferOp::Read,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = agent.post_xfer_req(&xfer_req, None)?;

            (xfer_req, still_pending)
        };

        // Wait for completion if transfer is pending
        if still_pending {
            loop {
                let status = agent.get_xfer_status(&xfer_req)?;
                if status.is_success() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        }

        tracing::debug!(
            "Batched G4 onboard complete: {} blocks transferred",
            num_blocks
        );

        Ok(G4TransferResult { transferred: num_blocks })
    }


    /// Execute G4 offload to disk storage.
    fn execute_g4_offload_disk(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available"))?;

        rt.block_on(self.execute_g4_offload_disk_async(local_block_ids, remote_descriptors))
    }

    /// Execute G4 offload to disk storage (async version).
    ///
    /// Transfers blocks from host memory to disk using NIXL FILE backend.
    /// All blocks are batched into a single NIXL transfer request for efficiency.
    async fn execute_g4_offload_disk_async(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        use crate::block_manager::block::data::BlockDataViews;

        if local_block_ids.len() != remote_descriptors.len() {
            bail!(
                "local_block_ids.len() ({}) != remote_descriptors.len() ({})",
                local_block_ids.len(),
                remote_descriptors.len()
            );
        }

        if local_block_ids.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        let agent = self.agent.as_ref().as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not available for disk offload"))?;

        let num_blocks = local_block_ids.len();

        tracing::debug!(
            "Batched disk offload: {} blocks",
            num_blocks
        );

        // Use a scope block to ensure all non-Send types are dropped before await
        let (xfer_req, still_pending) = {
            // Get block size from first block
            let first_block = &self.host_blocks[local_block_ids[0]];
            let first_view = first_block.local_block_view()?;
            let block_size = first_view.size();

            // Build transfer descriptor lists with ALL blocks
            let mut src_dl = XferDescList::new(MemType::Dram)?;
            let mut dst_dl = XferDescList::new(MemType::File)?;

            for (&block_id, desc) in local_block_ids.iter().zip(remote_descriptors.iter()) {
                let host_block = &self.host_blocks[block_id];
                let block_view = host_block.local_block_view()?;
                let addr = unsafe { block_view.as_ptr() as usize };

                // src_dl: source data in host DRAM
                src_dl.add_desc(addr, block_size, 0);

                // dst_dl: destination in file (object_key used as file device_id)
                dst_dl.add_desc(0, block_size, desc.key());
            }

            // Create ONE transfer request for ALL blocks
            let agent_name = agent.name();
            let xfer_req = agent.create_xfer_req(
                XferOp::Write,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = agent.post_xfer_req(&xfer_req, None)?;

            (xfer_req, still_pending)
        };

        // Wait for completion if transfer is pending
        if still_pending {
            loop {
                let status = agent.get_xfer_status(&xfer_req)?;
                if status.is_success() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        }

        tracing::debug!(
            "Batched disk offload complete: {} blocks transferred",
            num_blocks
        );

        Ok(G4TransferResult { transferred: num_blocks })
    }

    /// Execute G4 onboard from disk storage (sync version).
    fn execute_g4_onboard_disk(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        let rt = tokio::runtime::Handle::try_current()
            .map_err(|_| anyhow::anyhow!("No tokio runtime available"))?;

        rt.block_on(self.execute_g4_onboard_disk_async(local_block_ids, remote_descriptors))
    }

    /// Execute G4 onboard from disk storage (async version).
    ///
    /// Transfers blocks from disk to host memory using NIXL FILE backend.
    /// All blocks are batched into a single NIXL transfer request for efficiency.
    async fn execute_g4_onboard_disk_async(
        &self,
        local_block_ids: &[usize],
        remote_descriptors: &[RemoteDescriptor],
    ) -> Result<G4TransferResult> {
        use crate::block_manager::block::data::BlockDataViews;

        if local_block_ids.len() != remote_descriptors.len() {
            bail!(
                "local_block_ids.len() ({}) != remote_descriptors.len() ({})",
                local_block_ids.len(),
                remote_descriptors.len()
            );
        }

        if local_block_ids.is_empty() {
            return Ok(G4TransferResult { transferred: 0 });
        }

        let agent = self.agent.as_ref().as_ref()
            .ok_or_else(|| anyhow::anyhow!("NIXL agent not available for disk onboard"))?;

        let num_blocks = local_block_ids.len();

        tracing::debug!(
            "Batched disk onboard: {} blocks",
            num_blocks
        );

        // Use a scope block to ensure all non-Send types are dropped before await
        let (xfer_req, still_pending) = {
            // Get block size from first block
            let first_block = &self.host_blocks[local_block_ids[0]];
            let first_view = first_block.local_block_view()?;
            let block_size = first_view.size();

            // Build transfer descriptor lists with ALL blocks
            let mut src_dl = XferDescList::new(MemType::Dram)?;  // Local DRAM (destination)
            let mut dst_dl = XferDescList::new(MemType::File)?;  // File (source)

            for (&block_id, desc) in local_block_ids.iter().zip(remote_descriptors.iter()) {
                let host_block = &self.host_blocks[block_id];
                let block_view = host_block.local_block_view()?;
                let addr = unsafe { block_view.as_ptr() as usize };

                src_dl.add_desc(addr, block_size, 0);
                dst_dl.add_desc(0, block_size, desc.key());
            }

            // Create ONE transfer request for ALL blocks
            let agent_name = agent.name();
            let xfer_req = agent.create_xfer_req(
                XferOp::Read,
                &src_dl,
                &dst_dl,
                &agent_name,
                None,
            )?;

            let still_pending = agent.post_xfer_req(&xfer_req, None)?;

            (xfer_req, still_pending)
        };

        // Wait for completion if transfer is pending
        if still_pending {
            loop {
                let status = agent.get_xfer_status(&xfer_req)?;
                if status.is_success() {
                    break;
                }
                tokio::task::yield_now().await;
            }
        }

        tracing::debug!(
            "Batched disk onboard complete: {} blocks transferred",
            num_blocks
        );

        Ok(G4TransferResult { transferred: num_blocks })
    }

    // =========================================================================
    // High-Level Operations with Deduplication
    // =========================================================================

    /// Offload blocks to object storage with deduplication (write-through).
    ///
    /// Uses local and distributed registries for deduplication. Data stays
    /// in host memory after offload (write-through semantics).
    ///
    /// # Flow
    /// 1. Check local registry for existing hashes (fast local dedup)
    /// 2. Check distributed registry for cross-worker dedup
    /// 3. For each new hash: execute NIXL OBJ write
    /// 4. Register hashes in local + distributed registries
    ///
    /// # Arguments
    /// * `host_block_ids` - Source block indices in host pool
    /// * `sequence_hashes` - Sequence hashes for each block (object keys)
    ///
    /// # Returns
    /// Number of blocks successfully offloaded (excludes already-existing)
    pub async fn offload(
        &self,
        host_block_ids: &[usize],
        sequence_hashes: &[u64],
    ) -> Result<usize> {
        if host_block_ids.len() != sequence_hashes.len() {
            bail!(
                "host_block_ids.len() ({}) != sequence_hashes.len() ({})",
                host_block_ids.len(),
                sequence_hashes.len()
            );
        }

        if sequence_hashes.is_empty() {
            return Ok(0);
        }

        // Phase 1: Local dedup
        let local_existing = self.local_registry.match_keys(sequence_hashes);
        let local_existing_set: HashSet<_> = local_existing.into_iter().collect();

        let after_local_filter: Vec<_> = host_block_ids
            .iter()
            .zip(sequence_hashes)
            .filter(|(_, hash)| !local_existing_set.contains(hash))
            .collect();

        if after_local_filter.is_empty() {
            tracing::debug!(
                "All {} hashes already in local registry",
                sequence_hashes.len()
            );
            return Ok(0);
        }

        // Phase 2: Distributed dedup (if configured)
        let bucket = self.config.resolve_bucket(self.worker_id as u32);
        let to_offload: Vec<_> = if let Some(distributed) = &self.distributed_registry {
            let hashes_to_check: Vec<u64> = after_local_filter.iter().map(|(_, h)| **h).collect();

            let offload_result = distributed.can_offload(&bucket, &hashes_to_check).await?;

            tracing::debug!(
                "Distributed registry: can_offload={}, already_stored={}, leased={}",
                offload_result.can_offload.len(),
                offload_result.already_stored.len(),
                offload_result.leased.len()
            );

            let granted_set: HashSet<_> = offload_result.can_offload.into_iter().collect();

            after_local_filter
                .into_iter()
                .filter(|(_, hash)| granted_set.contains(hash))
                .collect()
        } else {
            after_local_filter
        };

        if to_offload.is_empty() {
            tracing::debug!(
                "All {} hashes already exist after distributed check",
                sequence_hashes.len()
            );
            return Ok(0);
        }

        // Check registry capacity
        if !self.local_registry.can_register() {
            tracing::warn!(
                "Object registry at capacity, cannot offload {} blocks)",
                to_offload.len()
            );
            return Ok(0);
        }

        // Phase 3: Build descriptors and execute transfer
        let local_ids: Vec<usize> = to_offload.iter().map(|(id, _)| **id).collect();
        let descriptors: Vec<RemoteDescriptor> = to_offload
            .iter()
            .map(|(_, hash)| RemoteDescriptor::object(**hash, bucket.clone()))
            .collect();

        let result = self.execute_g4_transfer_async(
            G4TransferDirection::Offload,
            RemoteStorageKind::Object,
            &local_ids,
            &descriptors,
        ).await?;

        // Phase 4: Register successfully offloaded hashes
        let offloaded_hashes: Vec<u64> = to_offload
            .iter()
            .take(result.transferred)
            .map(|(_, h)| **h)
            .collect();

        if !offloaded_hashes.is_empty() {
            self.local_registry.register(&offloaded_hashes);

            if let Some(distributed) = &self.distributed_registry {
                if let Err(e) = distributed.register(&bucket, &offloaded_hashes).await {
                    tracing::warn!(
                        "Failed to register {} hashes in distributed registry: {}",
                        offloaded_hashes.len(),
                        e
                    );
                }
            }
        }

        // Track offload metric
        if result.transferred > 0 {
            if let Some(metrics) = &self.kvbm_metrics {
                metrics.offload_blocks_d2o.inc_by(result.transferred as u64);
            }
        }

        tracing::debug!(
            offloaded = result.transferred,
            total = sequence_hashes.len(),
            "Object storage offload complete"
        );

        Ok(result.transferred)
    }

    /// Onboard blocks from object storage to device.
    ///
    /// Uses host blocks as bounce buffers, then transfers to device.
    ///
    /// # Flow
    /// 1. Read from object storage -> host bounce buffers
    /// 2. Transfer host bounce buffers -> device blocks
    ///
    /// # Arguments
    /// * `sequence_hashes` - Sequence hashes to onboard (object keys)
    /// * `host_block_ids` - Host block IDs for bounce buffers
    /// * `device_block_ids` - Destination device block IDs
    ///
    /// # Returns
    /// Vector of hashes that were successfully onboarded
    pub async fn onboard(
        &self,
        sequence_hashes: &[u64],
        host_block_ids: &[usize],
        device_block_ids: &[usize],
    ) -> Result<Vec<u64>> {
        if sequence_hashes.is_empty() {
            return Ok(vec![]);
        }

        let num_blocks = sequence_hashes.len();

        if num_blocks > host_block_ids.len() {
            bail!(
                "Not enough host bounce blocks: need {}, have {}",
                num_blocks,
                host_block_ids.len()
            );
        }

        if num_blocks > device_block_ids.len() {
            bail!(
                "Not enough device blocks: need {}, have {}",
                num_blocks,
                device_block_ids.len()
            );
        }

        tracing::debug!(
            "Onboarding {} blocks from G4",
            num_blocks
        );

        let bucket = self.config.resolve_bucket(self.worker_id as u32);

        // Phase 1: Object Storage -> Host (bounce buffers)
        let descriptors: Vec<RemoteDescriptor> = sequence_hashes
            .iter()
            .map(|hash| RemoteDescriptor::object(*hash, bucket.clone()))
            .collect();

        let host_ids: Vec<usize> = host_block_ids.iter().take(num_blocks).copied().collect();

        let result = self.execute_g4_transfer_async(
            G4TransferDirection::Onboard,
            RemoteStorageKind::Object,
            &host_ids,
            &descriptors,
        ).await?;

        if result.transferred != num_blocks {
            tracing::warn!(
                "Only onboarded {}/{} blocks from object storage",
                result.transferred,
                num_blocks
            );
        }

        // Phase 2: Host -> Device (batched to respect buffer pool limits)
        // The transfer buffer pool is sized for MAX_TRANSFER_BATCH_SIZE blocks,
        // so we need to batch the H->D transfers to avoid buffer overflow.
        const H2D_BATCH_SIZE: usize = 16; // Match MAX_TRANSFER_BATCH_SIZE

        let transferred_count = result.transferred;
        for batch_start in (0..transferred_count).step_by(H2D_BATCH_SIZE) {
            let batch_end = std::cmp::min(batch_start + H2D_BATCH_SIZE, transferred_count);

            let host_blocks_batch: Vec<_> = host_block_ids[batch_start..batch_end]
                .iter()
                .map(|id| self.host_blocks[*id].clone())
                .collect();

            let mut device_blocks_batch: Vec<_> = device_block_ids[batch_start..batch_end]
                .iter()
                .map(|id| self.device_blocks[*id].clone())
                .collect();

            host_blocks_batch
                .write_to(&mut device_blocks_batch, self.context.clone())?
                .await?;
        }

        // Register in local cache
        let onboarded_hashes: Vec<u64> = sequence_hashes
            .iter()
            .take(result.transferred)
            .copied()
            .collect();
        self.local_registry.register(&onboarded_hashes);

        tracing::debug!(
            onboarded = result.transferred,
            "Object storage onboard complete"
        );

        Ok(onboarded_hashes)
    }


    /// Check which sequence hashes exist in object storage.
    ///
    /// Returns the contiguous prefix of hashes that exist (checks local registry).
    pub fn lookup(&self, sequence_hashes: &[u64]) -> Vec<u64> {
        self.local_registry.match_keys(sequence_hashes)
    }

    /// Check if a single hash exists in object storage.
    pub fn contains(&self, hash: u64) -> bool {
        !self.local_registry.match_keys(&[hash]).is_empty()
    }

    /// Get the local registry.
    pub fn local_registry(&self) -> &SequenceHashRegistry {
        &self.local_registry
    }

    /// Get the distributed registry (if configured).
    pub fn distributed_registry(&self) -> Option<&Arc<dyn DistributedRegistry>> {
        self.distributed_registry.as_ref()
    }

    /// Get the object storage config.
    pub fn config(&self) -> &ObjectStorageConfig {
        &self.config
    }

    /// Get the worker ID.
    pub fn worker_id(&self) -> u64 {
        self.worker_id
    }

    /// Get the host blocks (for external access if needed).
    pub fn host_blocks(&self) -> &[LocalBlockData<PinnedStorage>] {
        &self.host_blocks
    }

    /// Get the device blocks (for external access if needed).
    pub fn device_blocks(&self) -> &[LocalBlockData<DeviceStorage>] {
        &self.device_blocks
    }

}

impl std::fmt::Debug for G4TransferHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("G4TransferHandler")
            .field("worker_id", &self.worker_id)
            .field("bucket", &self.config.bucket_template)
            .field("num_host_blocks", &self.host_blocks.len())
            .field("num_device_blocks", &self.device_blocks.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_g4_handler_debug() {
        // Just verify Debug trait works
        let handler_repr = format!("{:?}", "G4TransferHandler { placeholder }");
        assert!(handler_repr.contains("G4TransferHandler"));
    }

    #[test]
    fn test_remote_descriptor_constructors() {
        let obj_desc = RemoteDescriptor::object(0x1234, "my-bucket".to_string());
        assert_eq!(obj_desc.key(), 0x1234);
        assert_eq!(obj_desc.storage_kind(), RemoteStorageKind::Object);
        match obj_desc {
            RemoteDescriptor::Object { key, bucket } => {
                assert_eq!(key, 0x1234);
                assert_eq!(bucket, "my-bucket");
            }
            _ => panic!("Expected Object variant"),
        }

        let disk_desc = RemoteDescriptor::disk(0x5678);
        assert_eq!(disk_desc.key(), 0x5678);
        assert_eq!(disk_desc.storage_kind(), RemoteStorageKind::Disk);
        match disk_desc {
            RemoteDescriptor::Disk { key } => {
                assert_eq!(key, 0x5678);
            }
            _ => panic!("Expected Disk variant"),
        }
    }

    #[test]
    fn test_direction_and_storage_kind_equality() {
        assert_eq!(G4TransferDirection::Offload, G4TransferDirection::Offload);
        assert_ne!(G4TransferDirection::Offload, G4TransferDirection::Onboard);

        assert_eq!(RemoteStorageKind::Object, RemoteStorageKind::Object);
        assert_ne!(RemoteStorageKind::Object, RemoteStorageKind::Disk);
    }
}
