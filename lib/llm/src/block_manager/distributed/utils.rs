// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::block_manager::connector::protocol::LeaderTransferRequest;

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_WORKER_METADATA_MESSAGE: &str = "worker_metadata";
pub const ZMQ_LEADER_METADATA_MESSAGE: &str = "leader_metadata";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetadata {
    pub num_device_blocks: usize,
    pub bytes_per_block: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderMetadata {
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
    pub num_object_blocks: usize,
    pub object_storage_config: Option<ObjectStorageConfig>,
}

/// Configuration for object storage (S3-compatible) backend.
/// All values are read from environment variables.
///
/// Objects are allocated dynamically - keys are derived from sequence hashes
/// when blocks are stored to object storage.
///
/// The bucket name can be a template with `{worker_id}` placeholder:
/// - `kvcache` → all workers use the same bucket
/// - `kvcache-worker-{worker_id}` → each worker gets its own bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectStorageConfig {
    /// S3 bucket name template (can contain `{worker_id}` placeholder)
    pub bucket_template: String,
    /// S3 endpoint override (for MinIO or custom S3-compatible endpoints)
    pub endpoint_override: Option<String>,
    /// AWS region (optional)
    pub region: Option<String>,
}

impl ObjectStorageConfig {
    /// Create ObjectStorageConfig from environment variables.
    /// Returns None if required variables are not set.
    pub fn from_env() -> Option<Self> {
        use dynamo_runtime::config::environment_names::kvbm::object_storage;

        let bucket_template = std::env::var(object_storage::DYN_KVBM_OBJECT_BUCKET).ok()?;
        let endpoint_override = std::env::var(object_storage::DYN_KVBM_OBJECT_ENDPOINT).ok();
        let region = std::env::var(object_storage::DYN_KVBM_OBJECT_REGION).ok();

        Some(Self {
            bucket_template,
            endpoint_override,
            region,
        })
    }

    /// Resolve the bucket name for a specific worker.
    ///
    /// Substitutes `{worker_id}` in the template with the actual worker ID.
    pub fn resolve_bucket(&self, worker_id: u32) -> String {
        self.bucket_template.replace("{worker_id}", &worker_id.to_string())
    }

    /// Get the number of object blocks from environment variable.
    /// This represents the maximum number of blocks that can be stored in object storage.
    pub fn num_blocks_from_env() -> usize {
        use dynamo_runtime::config::environment_names::kvbm::object_storage;

        std::env::var(object_storage::DYN_KVBM_OBJECT_NUM_BLOCKS)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Check if object storage offloading is enabled.
    /// Returns true if DYN_KVBM_USE_OBJECT_OFFLOAD=1
    pub fn is_offload_enabled() -> bool {
        use dynamo_runtime::config::environment_names::kvbm::object_storage;

        std::env::var(object_storage::DYN_KVBM_USE_OBJECT_OFFLOAD)
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

    /// Check if write-through caching is enabled for object storage.
    /// When enabled, blocks offloaded to S3 are ALSO kept in host cache.
    /// Returns true if DYN_KVBM_OBJECT_WRITE_THROUGH=1
    pub fn is_write_through_enabled() -> bool {
        use dynamo_runtime::config::environment_names::kvbm::object_storage;

        std::env::var(object_storage::DYN_KVBM_OBJECT_WRITE_THROUGH)
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false)
    }

}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy)]
pub enum BlockTransferPool {
    Device,
    Host,
    Disk,
    Object,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ConnectorTransferType {
    Store,
    Load,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConnectorRequestLeader {
    pub req_id: String,
    pub txn_id: u64,
    pub transfer_type: ConnectorTransferType,
}

#[derive(Serialize, Deserialize, Debug, Getters, Clone)]
pub struct BlockTransferRequest {
    pub from_pool: BlockTransferPool,
    pub to_pool: BlockTransferPool,
    pub blocks: Vec<(usize, usize)>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_req: Option<LeaderTransferRequest>,

    /// Optional bounce block IDs for Device↔Object transfers.
    /// When provided, these host block IDs are used for staging instead of
    /// the internal bounce buffer allocator. This allows the caller to
    /// dynamically allocate bounce blocks from the host pool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bounce_block_ids: Option<Vec<usize>>,
}

impl BlockTransferRequest {
    #[allow(dead_code)]
    pub fn new(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: None,
            bounce_block_ids: None,
        }
    }

    pub fn new_with_trigger_id(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: Some(connector_req),
            bounce_block_ids: None,
        }
    }

    /// Create a transfer request with externally-provided bounce block IDs.
    /// Use this for Device↔Object transfers when bounce blocks are allocated
    /// from the host pool by the caller.
    pub fn new_with_bounce_blocks(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        bounce_block_ids: Vec<usize>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: None,
            bounce_block_ids: Some(bounce_block_ids),
        }
    }
}
