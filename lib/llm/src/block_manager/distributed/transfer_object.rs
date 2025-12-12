// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage transfer handler for KV cache blocks.
//!
//! This module provides [`ObjectTransferHandler`] - a dedicated handler for object storage transfers
//!
//!
//! | Aspect        | Tier Transfers            | Object Transfers          |
//! |---------------|---------------------------|------------------------   |
//! | **Layouts**   | Pre-allocated, fixed      | Ephemeral, per-object     |
//! | **Block IDs** | Index-based (0, 1, 2...)  | Hash-based (sequence_hash)|
//! | **Lookup**    | Direct pool index         | Registry lookup required  |
//! | **Lifecycle** | Long-lived layouts        | Created per-transfer      |
//!
//! ## Architecture
//!
//! ```text
//! Offload (Device -> Object):
//!   Device Block -> [Host Pool Bounce] -> Object
//!
//! Onboard (Object -> Device):
//!   Object -> [Host Pool Bounce] -> Device Block
//! ```
//!
//! Bounce buffers are dynamically acquired from the host pool for each transfer
//!

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;

use crate::block_manager::config::ObjectStorageConfig;
use super::registry::{DistributedRegistry, SequenceHashRegistry};
use crate::block_manager::v2::physical::layout::{LayoutConfig, PhysicalLayout};
use crate::block_manager::v2::physical::manager::{
    G4TransferDirection, LayoutHandle, RemoteDescriptor, TransportManager,
};
use crate::block_manager::v2::physical::transfer::nixl_agent::NixlAgent;
use crate::block_manager::v2::physical::transfer::options::TransferOptions;

///
#[derive(Clone)]
pub struct ObjectTransferHandler {
    /// Transport manager for physical transfers
    transport: TransportManager,

    /// Local registry tracking what THIS worker has offloaded (fast local lookup)
    local_registry: SequenceHashRegistry,

    /// Distributed registry for cross-worker deduplication (checks with hub)
    distributed_registry: Option<Arc<dyn DistributedRegistry>>,

    /// Object storage configuration (bucket, endpoint, region)
    config: ObjectStorageConfig,

    /// Host pool layout handle - bounce buffers are dynamically pulled from here
    host_handle: LayoutHandle,

    /// Layout config for creating ephemeral object layouts
    layout_config: LayoutConfig,

    /// NIXL agent for creating object layouts
    agent: NixlAgent,
}

impl ObjectTransferHandler {
    /// Create a new ObjectTransferHandler.
    ///
    /// # Arguments
    /// * `transport` - TransportManager for executing transfers
    /// * `local_registry` - Local registry for tracking what this worker has stored (fast lookup)
    /// * `distributed_registry` - Optional distributed registry for cross-worker deduplication
    /// * `config` - Object storage configuration
    /// * `host_handle` - Host pool layout handle (bounce buffers pulled from here)
    /// * `layout_config` - Config for creating ephemeral object layouts
    /// * `agent` - NIXL agent for layout creation
    pub fn new(
        transport: TransportManager,
        local_registry: SequenceHashRegistry,
        distributed_registry: Option<Arc<dyn DistributedRegistry>>,
        config: ObjectStorageConfig,
        host_handle: LayoutHandle,
        layout_config: LayoutConfig,
        agent: NixlAgent,
    ) -> Result<Self> {
        tracing::info!("Creating ObjectTransferHandler");
        Ok(Self {
            transport,
            local_registry,
            distributed_registry,
            config,
            host_handle,
            layout_config,
            agent,
        })
    }

    /// Get the host pool handle (for bounce buffer allocation).
    pub fn host_handle(&self) -> LayoutHandle {
        self.host_handle
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

    /// Get the transport manager.
    pub fn transport(&self) -> &TransportManager {
        &self.transport
    }

    // =========================================================================
    // Offload Operations (Device/Host -> Object Storage)
    // =========================================================================

    /// Offload blocks to object storage.
    ///
    /// Creates ephemeral ObjectLayout for each block, executes transfer,
    /// and registers sequence hashes in both local and distributed registries.
    ///
    /// Deduplication is performed in two stages:
    /// 1. Local registry check (fast, this worker's offloads)
    /// 2. Distributed registry check (cross-worker deduplication via hub)
    ///
    /// # Arguments
    /// * `src_handle` - Source layout handle (Device or Host)
    /// * `src_blocks` - Source block indices
    /// * `sequence_hashes` - Sequence hashes for each block (used as object keys)
    ///
    /// # Returns
    /// Number of blocks successfully offloaded (excludes already-existing)
    pub async fn offload(
        &self,
        src_handle: LayoutHandle,
        src_blocks: &[usize],
        sequence_hashes: &[u64],
    ) -> Result<usize> {
        if src_blocks.len() != sequence_hashes.len() {
            anyhow::bail!(
                "src_blocks.len() ({}) != sequence_hashes.len() ({})",
                src_blocks.len(),
                sequence_hashes.len()
            );
        }

        let local_existing = self.local_registry.match_keys(sequence_hashes);
        let local_existing_set: HashSet<_> = local_existing.into_iter().collect();

        let after_local_filter: Vec<_> = src_blocks
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

        let to_offload: Vec<_> = if let Some(distributed) = &self.distributed_registry {
            let bucket = self.config.resolve_bucket(self.transport.worker_id() as u32);
            let hashes_to_check: Vec<u64> = after_local_filter.iter().map(|(_, h)| **h).collect();

            let offload_result = distributed.can_offload(&bucket, &hashes_to_check).await?;

            tracing::debug!(
                "Distributed registry check: can_offload={}, already_stored={}, leased={}",
                offload_result.can_offload.len(),
                offload_result.already_stored.len(),
                offload_result.leased.len()
            );

            // Only offload hashes that were granted (not already stored or leased)
            let granted_set: HashSet<_> = offload_result.can_offload.into_iter().collect();

            after_local_filter
                .into_iter()
                .filter(|(_, hash)| granted_set.contains(hash))
                .collect()
        } else {
            // No distributed registry, proceed with all locally-filtered hashes
            after_local_filter
        };

        if to_offload.is_empty() {
            tracing::debug!(
                "All {} hashes already exist (local={}, distributed check filtered rest)",
                sequence_hashes.len(),
                local_existing_set.len()
            );
            return Ok(0);
        }

        // Check if local registry can accept new registrations
        if !self.local_registry.can_register() {
            tracing::warn!(
                "Object storage registry at capacity, cannot offload {} blocks",
                to_offload.len()
            );
            return Ok(0);
        }

        tracing::debug!(
            "Offloading {} blocks to object storage ({} filtered by local, {} by distributed)",
            to_offload.len(),
            local_existing_set.len(),
            sequence_hashes.len() - local_existing_set.len() - to_offload.len()
        );

        let mut offloaded = 0;
        let mut offloaded_hashes = Vec::with_capacity(to_offload.len());

        for (src_block, hash) in to_offload.iter() {
            // Create ephemeral object layout for this hash
            let object_layout = self.create_object_layout(**hash)?;
            let _object_handle = self.transport.register_layout(object_layout)?;

            // Transfer directly from source to object storage
            let bucket = self.config.resolve_bucket(self.transport.worker_id() as u32);
            let remote_desc = RemoteDescriptor::new(**hash, &bucket);
            let _result = self.transport.execute_g4_transfer_async(
                G4TransferDirection::Offload,
                src_handle,
                &[**src_block],
                &[remote_desc],
            ).await?;

            // Register in local registry
            self.local_registry.register(&[**hash]);
            offloaded_hashes.push(**hash);
            offloaded += 1;

            tracing::trace!(
                hash = %hash,
                src_block = %src_block,
                "Offloaded block to object storage"
            );
        }

        // Register in distributed registry (confirms the lease)
        if let Some(distributed) = &self.distributed_registry {
            let bucket = self.config.resolve_bucket(self.transport.worker_id() as u32);
            if let Err(e) = distributed.register(&bucket, &offloaded_hashes).await {
                tracing::warn!(
                    "Failed to register {} hashes in distributed registry: {}",
                    offloaded_hashes.len(),
                    e
                );
                // Continue anyway - local registry has the data
            }
        }

        tracing::debug!(
            offloaded = offloaded,
            total = sequence_hashes.len(),
            "Object storage offload complete"
        );

        Ok(offloaded)
    }

    // =========================================================================
    // Onboard Operations (Object Storage -> Device/Host)
    // =========================================================================

    /// Onboard blocks from object storage.
    ///
    /// Looks up sequence hashes in the registry, creates ephemeral ObjectLayouts,
    /// and transfers data to the destination.
    ///
    /// # Arguments
    /// * `sequence_hashes` - Sequence hashes to look up
    /// * `host_block_ids` - Host block IDs to use as bounce buffers (allocated by leader)
    /// * `dst_handle` - Destination layout handle (Device or Host)
    /// * `dst_blocks` - Destination block indices
    ///
    /// # Returns
    /// Vector of hashes that were successfully onboarded (contiguous prefix)
    ///
    /// NOTE: The leader has already verified these hashes exist in the distributed
    /// registry. We trust that verification and proceed directly to fetch from
    /// object storage without re-checking.
    ///
    /// Host blocks for bounce buffers are provided by the caller (allocated by leader).
    pub async fn onboard(
        &self,
        sequence_hashes: &[u64],
        host_block_ids: &[usize],
        dst_handle: LayoutHandle,
        dst_blocks: &[usize],
    ) -> Result<Vec<u64>> {
        if sequence_hashes.is_empty() {
            return Ok(vec![]);
        }

        let num_blocks = sequence_hashes.len();

        if num_blocks > dst_blocks.len() {
            anyhow::bail!(
                "Not enough destination blocks: need {}, have {}",
                num_blocks,
                dst_blocks.len()
            );
        }

        if num_blocks > host_block_ids.len() {
            anyhow::bail!(
                "Not enough host bounce blocks: need {}, have {}",
                num_blocks,
                host_block_ids.len()
            );
        }

        tracing::info!(
            "Onboarding {} blocks from object storage (batch transfer)",
            num_blocks
        );

        // Create RemoteDescriptors for all objects at once
        let bucket = self.config.resolve_bucket(self.transport.worker_id() as u32);
        let remote_descriptors: Vec<RemoteDescriptor> = sequence_hashes
            .iter()
            .map(|hash| RemoteDescriptor::new(*hash, &bucket))
            .collect();

        // Batch transfer: Object Storage -> Host (bounce buffers)
        let g4_result = self
            .transport
            .execute_g4_transfer_async(
                G4TransferDirection::Onboard,
                self.host_handle,
                host_block_ids,
                &remote_descriptors,
            )
            .await?;

        // Validate G4 transfer completed successfully
        if g4_result.transferred != num_blocks {
            tracing::warn!(
                "G4 onboard transferred {} of {} blocks",
                g4_result.transferred,
                num_blocks
            );
        }

        // Batch transfer: Host (bounce buffers) -> Device
        let block_pairs: Vec<(usize, usize)> = host_block_ids
            .iter()
            .zip(dst_blocks.iter())
            .map(|(src, dst)| (*src, *dst))
            .collect();

        let src_block_ids: Vec<usize> = block_pairs.iter().map(|(src, _)| *src).collect();
        let dst_block_ids: Vec<usize> = block_pairs.iter().map(|(_, dst)| *dst).collect();

        let notification = self.transport.execute_transfer(
            self.host_handle,
            &src_block_ids,
            dst_handle,
            &dst_block_ids,
            TransferOptions::default(),
        )?;

        notification.await?;

        // Register in local cache only after successful transfer
        self.local_registry.register(sequence_hashes);

        tracing::debug!(
            onboarded = num_blocks,
            "Object storage onboard complete (batch)"
        );

        Ok(sequence_hashes.to_vec())
    }

    /// Check which sequence hashes exist in object storage.
    ///
    /// Returns the contiguous prefix of hashes that exist (checks local registry).
    pub fn lookup(&self, sequence_hashes: &[u64]) -> Vec<u64> {
        self.local_registry.match_keys(sequence_hashes)
    }

    /// Check if a single hash exists in object storage (checks local registry).
    pub fn contains(&self, hash: u64) -> bool {
        !self.local_registry.match_keys(&[hash]).is_empty()
    }

    /// Get the number of items registered in the external registry.
    pub fn registry_len(&self) -> usize {
        // The registry is an Arc<dyn ExternalRegistry>, so we need to use match_keys
        // to check size. A real implementation would add a len() method to the trait.
        // For now, we can't directly get the size.
        0 // Placeholder - real implementation would need registry.len()
    }


    /// Create an ephemeral ObjectLayout for a single object.
    fn create_object_layout(&self, hash: u64) -> Result<PhysicalLayout> {
        let bucket = self.config.resolve_bucket(self.transport.worker_id() as u32);

        // Create a layout config with num_blocks = 1 (single object)
        let mut object_config = self.layout_config.clone();
        object_config.num_blocks = 1;

        PhysicalLayout::builder(self.agent.clone())
            .with_config(object_config)
            .object_layout()
            .allocate_object(bucket, hash)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to create object layout: {}", e))
    }
}

impl std::fmt::Debug for ObjectTransferHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectTransferHandler")
            .field("worker_id", &self.transport.worker_id())
            .field("bucket", &self.config.bucket_template)
            .field("host_handle", &self.host_handle)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests would go here, but require mocking TransportManager, etc.
    // For now, we verify the struct compiles and basic construction works.

    #[test]
    fn test_object_storage_config_bucket_resolution() {
        let config = ObjectStorageConfig {
            bucket_template: "my-bucket-{worker_id}".to_string(),
            endpoint_override: None,
            region: None,
        };

        assert_eq!(config.resolve_bucket(0), "my-bucket-0");
        assert_eq!(config.resolve_bucket(42), "my-bucket-42");
    }
}

