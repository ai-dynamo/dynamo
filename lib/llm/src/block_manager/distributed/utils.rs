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

    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_hashes: Option<Vec<u64>>,

    #[serde(default)]
    pub write_through: bool,
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
            write_through: false,
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
            write_through: false,
        }
    }

    pub fn new_with_write_through(
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
            write_through: true,
        }
    }

    #[allow(dead_code)]
    pub fn new_with_hashes(
        from_pool: BlockTransferPool,
        to_pool: BlockTransferPool,
        blocks: Vec<(usize, usize)>,
        sequence_hashes: Vec<u64>,
    ) -> Self {
        Self {
            from_pool,
            to_pool,
            blocks,
            connector_req: None,
            sequence_hashes: Some(sequence_hashes),
            write_through: false,
        }
    }

    pub fn new_g4_onboard(
        sequence_hashes: Vec<u64>,
        host_block_ids: Vec<usize>,
        device_block_ids: Vec<usize>,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        // blocks: (host_bounce_idx, device_dest_idx) pairs
        let blocks: Vec<(usize, usize)> = host_block_ids
            .iter()
            .zip(device_block_ids.iter())
            .map(|(&h, &d)| (h, d))
            .collect();

        Self {
            from_pool: BlockTransferPool::Object,
            to_pool: BlockTransferPool::Device,
            blocks,
            connector_req: Some(connector_req),
            sequence_hashes: Some(sequence_hashes),
            write_through: false,
        }
    }

    pub fn new_g4_offload(
        host_block_ids: Vec<usize>,
        sequence_hashes: Vec<u64>,
    ) -> Self {
        let blocks: Vec<(usize, usize)> = host_block_ids
            .iter()
            .map(|&h| (h, 0))
            .collect();

        Self {
            from_pool: BlockTransferPool::Host,
            to_pool: BlockTransferPool::Object,
            blocks,
            connector_req: None,
            sequence_hashes: Some(sequence_hashes),
            write_through: false,
        }
    }
}
