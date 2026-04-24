// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared protocol types for KVBM conditional disaggregation.
//!
//! This crate intentionally contains only serializable control-plane data. It
//! is shared by the connector, hub, and future admission code without making
//! any one of those crates depend on another.

use kvbm_common::BlockId;
use serde::{Deserialize, Serialize};
use velo_common::InstanceId;

/// Current conditional-disaggregation protocol version.
pub const DISAGG_PROTOCOL_VERSION: u16 = 1;

/// Unique identifier for a conditional-disaggregation session.
pub type SessionId = uuid::Uuid;

/// JSON-safe representation of a KVBM sequence hash.
///
/// Native KVBM hashes are currently backed by `u128`, which `serde_json` does
/// not support directly. The wire protocol carries the decimal representation.
pub type DisaggSequenceHash = String;

mod serde_uuid_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;

    pub fn serialize<S>(id: &Uuid, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Uuid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value).map_err(D::Error::custom)
    }
}

mod serde_instance_id_string {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};
    use uuid::Uuid;
    use velo_common::InstanceId;

    pub fn serialize<S>(id: &InstanceId, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&id.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<InstanceId, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Uuid::parse_str(&value)
            .map(InstanceId::from)
            .map_err(D::Error::custom)
    }
}

/// Role advertised by a worker or instance to the hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerRole {
    Decode,
    Prefill,
    Hybrid,
}

/// Opaque endpoint descriptor for a session/control channel.
///
/// Later PRs can standardize concrete endpoint kinds such as velo-streaming
/// anchors without changing where protocol ownership lives.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEndpoint {
    pub kind: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub payload: serde_json::Value,
}

/// Serializable descriptor for a block made available through a disagg
/// session. `layout_handle_raw` is intentionally encoded as a decimal string
/// so this protocol crate does not depend on kvbm-physical and remains
/// JSON-compatible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DisaggBlockRef {
    pub block_id: BlockId,
    pub sequence_hash: DisaggSequenceHash,
    pub layout_handle_raw: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<BlockDescriptor>,
}

/// Opaque RDMA descriptor payload. Real NIXL descriptors are carried here as
/// bytes; mocks can use small sentinel byte vectors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockDescriptor {
    pub bytes: Vec<u8>,
}

impl BlockDescriptor {
    pub fn new(bytes: impl Into<Vec<u8>>) -> Self {
        Self {
            bytes: bytes.into(),
        }
    }
}

/// Select all session blocks, or a specific hash subset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "hashes", rename_all = "snake_case")]
pub enum HashSelection {
    All,
    Hashes(Vec<DisaggSequenceHash>),
}

/// Ask the holder for descriptors for all or a subset of blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorRequest {
    pub hashes: HashSelection,
}

/// Holder response to a descriptor request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescriptorResponse {
    pub ready_blocks: Vec<DisaggBlockRef>,
    pub pending_hashes: Vec<DisaggSequenceHash>,
}

/// Request that the peer release session-held pins for all or a subset of
/// blocks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnpinRequest {
    pub request_id: String,
    pub hashes: HashSelection,
}

/// Mandatory acknowledgement after a session unpin request has been applied.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnpinAck {
    pub request_id: String,
    pub hashes: HashSelection,
}

/// The puller completed an RDMA pull. The corresponding `PullAck` proves the
/// holder remained live long enough to observe completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullComplete {
    pub pull_id: u64,
    pub hashes: Vec<DisaggSequenceHash>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullAck {
    pub pull_id: u64,
}

/// Typed transfer parameters carried in request metadata.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferParams {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_prefill: Option<RemotePrefillParams>,
}

impl TransferParams {
    pub fn remote_prefill(params: RemotePrefillParams) -> Self {
        Self {
            remote_prefill: Some(params),
        }
    }
}

/// Parameters identifying a remote prefill session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillParams {
    pub protocol_version: u16,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    #[serde(default)]
    pub sequence_hashes: Vec<DisaggSequenceHash>,
}

impl RemotePrefillParams {
    pub fn new(session_id: SessionId, initiator_instance_id: InstanceId) -> Self {
        Self {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id,
            initiator_instance_id,
            decode_endpoint: None,
            sequence_hashes: Vec::new(),
        }
    }
}

/// Payload enqueued by a decode worker and consumed by a prefill worker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemotePrefillRequest {
    pub protocol_version: u16,
    pub request_id: String,
    #[serde(with = "serde_uuid_string")]
    pub session_id: SessionId,
    #[serde(with = "serde_instance_id_string")]
    pub initiator_instance_id: InstanceId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_endpoint: Option<SessionEndpoint>,
    #[serde(default)]
    pub sequence_hashes: Vec<DisaggSequenceHash>,
    pub token_ids: Vec<u32>,
    pub num_computed_tokens: usize,
}

impl RemotePrefillRequest {
    pub fn remote_prefill_params(&self) -> RemotePrefillParams {
        RemotePrefillParams {
            protocol_version: self.protocol_version,
            session_id: self.session_id,
            initiator_instance_id: self.initiator_instance_id,
            decode_endpoint: self.decode_endpoint.clone(),
            sequence_hashes: self.sequence_hashes.clone(),
        }
    }

    pub fn transfer_params(&self) -> TransferParams {
        TransferParams::remote_prefill(self.remote_prefill_params())
    }
}

/// Frames sent from a decode worker to a prefill worker over a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecodeToPrefillFrame {
    DescriptorResponse(DescriptorResponse),
    UnpinRequest(UnpinRequest),
    UnpinAck(UnpinAck),
    PullComplete(PullComplete),
    PullAck(PullAck),
    BlocksReady { blocks: Vec<DisaggBlockRef> },
    OutputBlocksPulled { hashes: Vec<DisaggSequenceHash> },
    Detach,
    Error { message: String },
}

/// Frames sent from a prefill worker to a decode worker over a session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PrefillToDecodeFrame {
    Attach {
        #[serde(with = "serde_instance_id_string")]
        prefill_instance_id: InstanceId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefill_endpoint: Option<SessionEndpoint>,
    },
    DescriptorRequest(DescriptorRequest),
    UnpinRequest(UnpinRequest),
    UnpinAck(UnpinAck),
    PullComplete(PullComplete),
    PullAck(PullAck),
    InitialBlocksPulled {
        hashes: Vec<DisaggSequenceHash>,
    },
    OutputBlocksReady {
        blocks: Vec<DisaggBlockRef>,
    },
    Detach,
    Error {
        message: String,
    },
}

/// Hub-issued lifecycle/control signal for decode or prefill workers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ControlSignal {
    Pause {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
    Resume,
    Drain,
    ShutdownGracefully,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(position: u64) -> DisaggSequenceHash {
        position.to_string()
    }

    fn instance_id() -> InstanceId {
        uuid::Uuid::new_v4().into()
    }

    #[test]
    fn remote_prefill_request_builds_transfer_params() {
        let request = RemotePrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: "req-1".to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: Some(SessionEndpoint {
                kind: "test".to_string(),
                payload: serde_json::json!({"anchor": "a"}),
            }),
            sequence_hashes: vec![hash(0), hash(1)],
            token_ids: vec![1, 2, 3],
            num_computed_tokens: 16,
        };

        let params = request
            .transfer_params()
            .remote_prefill
            .expect("remote params populated");
        assert_eq!(params.protocol_version, request.protocol_version);
        assert_eq!(params.session_id, request.session_id);
        assert_eq!(params.initiator_instance_id, request.initiator_instance_id);
        assert_eq!(params.decode_endpoint, request.decode_endpoint);
        assert_eq!(params.sequence_hashes, request.sequence_hashes);
    }

    #[test]
    fn transfer_params_round_trips_json() {
        let params = TransferParams::remote_prefill(RemotePrefillParams {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: instance_id(),
            decode_endpoint: None,
            sequence_hashes: vec![hash(0)],
        });

        let encoded = serde_json::to_vec(&params).unwrap();
        let decoded: TransferParams = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, params);
    }

    #[test]
    fn session_frames_round_trip_json() {
        let block = DisaggBlockRef {
            block_id: 7,
            sequence_hash: hash(0),
            layout_handle_raw: "42".to_string(),
            descriptor: Some(BlockDescriptor::new([1, 2, 3])),
        };
        let frame = PrefillToDecodeFrame::OutputBlocksReady {
            blocks: vec![block],
        };

        let encoded = serde_json::to_vec(&frame).unwrap();
        let decoded: PrefillToDecodeFrame = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, frame);
    }

    #[test]
    fn descriptor_and_unpin_frames_round_trip_json() {
        let descriptor = DescriptorResponse {
            ready_blocks: vec![DisaggBlockRef {
                block_id: 9,
                sequence_hash: hash(9),
                layout_handle_raw: "100".to_string(),
                descriptor: Some(BlockDescriptor::new([0xaa])),
            }],
            pending_hashes: vec![hash(10)],
        };
        let frame = DecodeToPrefillFrame::DescriptorResponse(descriptor);
        let encoded = serde_json::to_vec(&frame).unwrap();
        let decoded: DecodeToPrefillFrame = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, frame);

        let ack = PrefillToDecodeFrame::UnpinAck(UnpinAck {
            request_id: "req".to_string(),
            hashes: HashSelection::All,
        });
        let encoded = serde_json::to_vec(&ack).unwrap();
        let decoded: PrefillToDecodeFrame = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded, ack);
    }

    #[test]
    fn control_signal_round_trips_json() {
        let signal = ControlSignal::Pause {
            reason: Some("operator".to_string()),
        };

        let encoded = serde_json::to_string(&signal).unwrap();
        let decoded: ControlSignal = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, signal);
    }
}
