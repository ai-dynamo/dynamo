// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_getters::Getters;
use serde::{Deserialize, Serialize};

use crate::block_manager::block::transfer::remote::{
    RemoteBlockDescriptor, RemoteTransferDirection, RemoteTransferPipeline,
};
use crate::block_manager::connector::protocol::LeaderTransferRequest;

pub const ZMQ_PING_MESSAGE: &str = "ping";
pub const ZMQ_WORKER_METADATA_MESSAGE: &str = "worker_metadata";
pub const ZMQ_LEADER_METADATA_MESSAGE: &str = "leader_metadata";
pub const ZMQ_TRANSFER_BLOCKS_MESSAGE: &str = "transfer_blocks";
pub const ZMQ_REMOTE_TRANSFER_MESSAGE: &str = "remote_transfer";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerMetadata {
    pub num_device_blocks: usize,
    pub bytes_per_block: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderMetadata {
    pub num_host_blocks: usize,
    pub num_disk_blocks: usize,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_hashes: Option<Vec<u64>>,
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
}

/// Request for remote storage transfers (G4 object storage or remote disk).
///
/// This request wraps a `RemoteTransferPipeline` with tracking metadata
/// for the connector system. Used for both onboard (remote -> device)
/// and offload (device -> remote) operations.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct RemoteTransferRequest {
    /// Request ID for tracking
    pub request_id: String,
    /// Unique operation ID
    pub operation_id: uuid::Uuid,
    /// The transfer pipeline (direction, descriptors, block IDs)
    pub pipeline: SerializableRemoteTransferPipeline,
    /// Optional connector request for completion tracking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_req: Option<LeaderTransferRequest>,
}

/// Serializable version of RemoteTransferPipeline for ZMQ transport.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableRemoteTransferPipeline {
    pub direction: SerializableTransferDirection,
    pub descriptors: Vec<SerializableRemoteBlockDescriptor>,
    pub bounce_block_ids: Option<Vec<usize>>,
    pub device_block_ids: Option<Vec<usize>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializableTransferDirection {
    Onboard,
    Offload,
}

impl From<RemoteTransferDirection> for SerializableTransferDirection {
    fn from(dir: RemoteTransferDirection) -> Self {
        match dir {
            RemoteTransferDirection::Onboard => Self::Onboard,
            RemoteTransferDirection::Offload => Self::Offload,
        }
    }
}

impl From<SerializableTransferDirection> for RemoteTransferDirection {
    fn from(dir: SerializableTransferDirection) -> Self {
        match dir {
            SerializableTransferDirection::Onboard => Self::Onboard,
            SerializableTransferDirection::Offload => Self::Offload,
        }
    }
}

/// Serializable version of RemoteBlockDescriptor.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SerializableRemoteBlockDescriptor {
    /// Object: (bucket, key), Disk: (path, key)
    pub storage_type: SerializableStorageType,
    pub location: String,
    pub key: String,
    pub size: usize,
    pub sequence_hash: Option<u64>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializableStorageType {
    Object,
    Disk,
}

impl From<&RemoteBlockDescriptor> for SerializableRemoteBlockDescriptor {
    fn from(desc: &RemoteBlockDescriptor) -> Self {
        use crate::block_manager::block::transfer::remote::RemoteKey;
        match desc.key() {
            RemoteKey::Object(obj) => Self {
                storage_type: SerializableStorageType::Object,
                location: obj.bucket.clone(),
                key: obj.key.clone(),
                size: desc.size(),
                sequence_hash: desc.sequence_hash(),
            },
            RemoteKey::Disk(disk) => Self {
                storage_type: SerializableStorageType::Disk,
                location: disk.path.clone(),
                key: disk.key.clone(),
                size: desc.size(),
                sequence_hash: desc.sequence_hash(),
            },
        }
    }
}

impl From<&RemoteTransferPipeline> for SerializableRemoteTransferPipeline {
    fn from(pipeline: &RemoteTransferPipeline) -> Self {
        match pipeline {
            RemoteTransferPipeline::Direct {
                direction,
                remote_descriptors,
            } => Self {
                direction: (*direction).into(),
                descriptors: remote_descriptors.iter().map(|d| d.into()).collect(),
                bounce_block_ids: None,
                device_block_ids: None,
            },
            RemoteTransferPipeline::WithBounce {
                direction,
                remote_descriptors,
                bounce_block_ids,
                device_block_ids,
            } => Self {
                direction: (*direction).into(),
                descriptors: remote_descriptors.iter().map(|d| d.into()).collect(),
                bounce_block_ids: Some(bounce_block_ids.clone()),
                device_block_ids: Some(device_block_ids.clone()),
            },
        }
    }
}

impl SerializableRemoteBlockDescriptor {
    /// Convert back to RemoteBlockDescriptor.
    pub fn to_descriptor(&self) -> RemoteBlockDescriptor {
        use crate::block_manager::block::transfer::remote::{
            DiskKey, ObjectKey, RemoteBlockMetadata, RemoteKey,
        };
        let key = match self.storage_type {
            SerializableStorageType::Object => {
                RemoteKey::Object(ObjectKey::new(&self.location, &self.key))
            }
            SerializableStorageType::Disk => {
                RemoteKey::Disk(DiskKey::new(&self.location, &self.key))
            }
        };
        if let Some(hash) = self.sequence_hash {
            RemoteBlockDescriptor::with_metadata(key, self.size, RemoteBlockMetadata::new(hash))
        } else {
            RemoteBlockDescriptor::new(key, self.size)
        }
    }
}

impl SerializableRemoteTransferPipeline {
    /// Convert back to RemoteTransferPipeline.
    pub fn to_pipeline(&self) -> RemoteTransferPipeline {
        let descriptors: Vec<RemoteBlockDescriptor> =
            self.descriptors.iter().map(|d| d.to_descriptor()).collect();
        let direction: RemoteTransferDirection = self.direction.into();

        match (&self.bounce_block_ids, &self.device_block_ids) {
            (Some(bounce), Some(device)) => {
                if direction.is_onboard() {
                    RemoteTransferPipeline::onboard_with_bounce(
                        descriptors,
                        bounce.clone(),
                        device.clone(),
                    )
                } else {
                    RemoteTransferPipeline::offload_with_bounce(
                        descriptors,
                        bounce.clone(),
                        device.clone(),
                    )
                }
            }
            _ => {
                if direction.is_onboard() {
                    RemoteTransferPipeline::onboard_direct(descriptors)
                } else {
                    RemoteTransferPipeline::offload_direct(descriptors)
                }
            }
        }
    }
}

impl RemoteTransferRequest {
    /// Create a new remote transfer request.
    pub fn new(
        request_id: String,
        operation_id: uuid::Uuid,
        pipeline: &RemoteTransferPipeline,
    ) -> Self {
        Self {
            request_id,
            operation_id,
            pipeline: pipeline.into(),
            connector_req: None,
        }
    }

    /// Create a new remote transfer request with connector tracking.
    pub fn new_with_connector_req(
        request_id: String,
        operation_id: uuid::Uuid,
        pipeline: &RemoteTransferPipeline,
        connector_req: LeaderTransferRequest,
    ) -> Self {
        Self {
            request_id,
            operation_id,
            pipeline: pipeline.into(),
            connector_req: Some(connector_req),
        }
    }

    /// Check if this is an onboard (remote -> device) request.
    pub fn is_onboard(&self) -> bool {
        self.pipeline.direction == SerializableTransferDirection::Onboard
    }

    /// Check if this is an offload (device -> remote) request.
    pub fn is_offload(&self) -> bool {
        self.pipeline.direction == SerializableTransferDirection::Offload
    }

    /// Get the number of blocks in this transfer.
    pub fn num_blocks(&self) -> usize {
        self.pipeline.descriptors.len()
    }

    /// Convert the serializable pipeline back to the real type.
    pub fn to_pipeline(&self) -> RemoteTransferPipeline {
        self.pipeline.to_pipeline()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializable_remote_block_descriptor_roundtrip() {
        let desc = RemoteBlockDescriptor::object_from_hash("my-bucket", 0x1234abcd, 4096);

        // Convert to serializable
        let serializable: SerializableRemoteBlockDescriptor = (&desc).into();
        assert_eq!(serializable.storage_type, SerializableStorageType::Object);
        assert_eq!(serializable.location, "my-bucket");
        assert_eq!(serializable.size, 4096);
        assert_eq!(serializable.sequence_hash, Some(0x1234abcd));

        // Convert back
        let restored = serializable.to_descriptor();
        assert_eq!(restored.size(), 4096);
        assert_eq!(restored.sequence_hash(), Some(0x1234abcd));
    }

    #[test]
    fn test_serializable_remote_transfer_pipeline_onboard_with_bounce() {
        let descs = vec![
            RemoteBlockDescriptor::object_from_hash("bucket", 0x1111, 4096),
            RemoteBlockDescriptor::object_from_hash("bucket", 0x2222, 4096),
        ];
        let pipeline = RemoteTransferPipeline::onboard_with_bounce(descs, vec![0, 1], vec![10, 11]);

        // Convert to serializable
        let serializable: SerializableRemoteTransferPipeline = (&pipeline).into();
        assert_eq!(
            serializable.direction,
            SerializableTransferDirection::Onboard
        );
        assert_eq!(serializable.descriptors.len(), 2);
        assert_eq!(serializable.bounce_block_ids, Some(vec![0, 1]));
        assert_eq!(serializable.device_block_ids, Some(vec![10, 11]));

        // Convert back
        let restored = serializable.to_pipeline();
        assert_eq!(restored.direction(), RemoteTransferDirection::Onboard);
        assert!(restored.has_bounce());
        assert_eq!(restored.num_blocks(), 2);
        assert_eq!(restored.bounce_block_ids(), Some([0, 1].as_slice()));
        assert_eq!(restored.device_block_ids(), Some([10, 11].as_slice()));
    }

    #[test]
    fn test_serializable_remote_transfer_pipeline_offload_direct() {
        let descs = vec![RemoteBlockDescriptor::object_from_hash(
            "bucket", 0x3333, 8192,
        )];
        let pipeline = RemoteTransferPipeline::offload_direct(descs);

        // Convert to serializable
        let serializable: SerializableRemoteTransferPipeline = (&pipeline).into();
        assert_eq!(
            serializable.direction,
            SerializableTransferDirection::Offload
        );
        assert_eq!(serializable.bounce_block_ids, None);
        assert_eq!(serializable.device_block_ids, None);

        // Convert back
        let restored = serializable.to_pipeline();
        assert_eq!(restored.direction(), RemoteTransferDirection::Offload);
        assert!(!restored.has_bounce());
    }

    #[test]
    fn test_remote_transfer_request_json_roundtrip() {
        let descs = vec![RemoteBlockDescriptor::object_from_hash(
            "bucket", 0xabc, 4096,
        )];
        let pipeline = RemoteTransferPipeline::onboard_with_bounce(descs, vec![5], vec![15]);
        let request =
            RemoteTransferRequest::new("req-123".to_string(), uuid::Uuid::new_v4(), &pipeline);

        // Serialize to JSON (as we do for ZMQ)
        let json = serde_json::to_vec(&request).unwrap();

        // Deserialize back
        let restored: RemoteTransferRequest = serde_json::from_slice(&json).unwrap();
        assert_eq!(restored.request_id, "req-123");
        assert!(restored.is_onboard());
        assert_eq!(restored.num_blocks(), 1);

        // Verify pipeline conversion
        let restored_pipeline = restored.to_pipeline();
        assert!(restored_pipeline.has_bounce());
    }

    #[test]
    fn test_disk_descriptor_roundtrip() {
        use crate::block_manager::block::transfer::remote::{DiskKey, RemoteKey};

        let key = RemoteKey::Disk(DiskKey::new("/mnt/nvme", "block_001"));
        let desc = RemoteBlockDescriptor::new(key, 8192);

        let serializable: SerializableRemoteBlockDescriptor = (&desc).into();
        assert_eq!(serializable.storage_type, SerializableStorageType::Disk);
        assert_eq!(serializable.location, "/mnt/nvme");
        assert_eq!(serializable.key, "block_001");

        let restored = serializable.to_descriptor();
        assert_eq!(
            restored.kind(),
            crate::block_manager::block::transfer::remote::RemoteStorageKind::Disk
        );
    }
}
