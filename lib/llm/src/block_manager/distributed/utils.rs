// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::block_manager::connector::protocol::LeaderTransferRequest;
use crate::block_manager::config::ObjectStorageConfig;

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_WORKER_METADATA_MESSAGE: &str = "worker_metadata";
pub const ZMQ_LEADER_METADATA_MESSAGE: &str = "leader_metadata";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";
pub const ZMQ_G4_ONBOARD_MESSAGE: &str = "g4_onboard";

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

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Copy)]
pub enum BlockTransferPool {
    Device,
    Host,
    Disk,
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

    /// Sequence hashes for G4 write-through (only used for Device -> Host transfers).
    /// When present, worker will also offload these blocks to object storage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_hashes: Option<Vec<u64>>,
}

/// Request to onboard blocks from G4 object storage directly to device.
/// Worker handles the G4→Host→Device transfer atomically.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct G4OnboardRequest {
    /// Request ID for correlation
    pub request_id: String,

    /// Operation ID for tracking
    pub operation_id: uuid::Uuid,

    /// Sequence hashes to onboard (lookup keys in object storage)
    pub sequence_hashes: Vec<u64>,

    /// Destination device block IDs
    pub device_block_ids: Vec<usize>,

    /// Host block IDs to use as bounce buffers (allocated by leader)
    pub host_block_ids: Vec<usize>,

    /// Optional connector request for scheduling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_req: Option<LeaderTransferRequest>,
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
            sequence_hashes: None,
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
            sequence_hashes: None,
        }
    }

    /// Create a new request with sequence hashes for G4 write-through.
    pub fn new_with_g4_hashes(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        connector_req: LeaderTransferRequest,
        sequence_hashes: Vec<u64>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: Some(connector_req),
            sequence_hashes: Some(sequence_hashes),
        }
    }
}

impl G4OnboardRequest {
    pub fn new(
        request_id: String,
        operation_id: uuid::Uuid,
        sequence_hashes: Vec<u64>,
        device_block_ids: Vec<usize>,
        host_block_ids: Vec<usize>,
    ) -> Self {
        Self {
            request_id,
            operation_id,
            sequence_hashes,
            device_block_ids,
            host_block_ids,
            connector_req: None,
        }
    }

    pub fn new_with_connector_req(
        request_id: String,
        operation_id: uuid::Uuid,
        sequence_hashes: Vec<u64>,
        device_block_ids: Vec<usize>,
        host_block_ids: Vec<usize>,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        Self {
            request_id,
            operation_id,
            sequence_hashes,
            device_block_ids,
            host_block_ids,
            connector_req: Some(connector_req),
        }
    }
}
