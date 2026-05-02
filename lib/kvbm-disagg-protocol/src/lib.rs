// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared protocol types for KVBM conditional disaggregation.
//!
//! This crate intentionally contains only serializable control-plane data. It
//! is shared by the connector, hub, and future admission code without making
//! any one of those crates depend on another.

use kvbm_common::SequenceHash;
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
    pub sequence_hashes: Vec<SequenceHash>,
    /// Decode-side `num_computed_tokens`. Prefill needs this to
    /// translate its 0-indexed position in `sequence_hashes`
    /// (which carries decode's local-match slice) back to the
    /// absolute token-block index in the original sequence —
    /// `expected_hashes[i]` is at absolute position
    /// `(num_computed_tokens / block_size) + i`.
    #[serde(default)]
    pub num_computed_tokens: usize,
}

impl RemotePrefillParams {
    pub fn new(session_id: SessionId, initiator_instance_id: InstanceId) -> Self {
        Self {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id,
            initiator_instance_id,
            decode_endpoint: None,
            sequence_hashes: Vec::new(),
            num_computed_tokens: 0,
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
    pub sequence_hashes: Vec<SequenceHash>,
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
            num_computed_tokens: self.num_computed_tokens,
        }
    }

    pub fn transfer_params(&self) -> TransferParams {
        TransferParams::remote_prefill(self.remote_prefill_params())
    }
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
            sequence_hashes: vec![
                SequenceHash::new(0, None, 0),
                SequenceHash::new(1, Some(0), 1),
            ],
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
            sequence_hashes: vec![SequenceHash::new(0, None, 0)],
            num_computed_tokens: 0,
        });

        let encoded = serde_json::to_vec(&params).unwrap();
        let decoded: TransferParams = serde_json::from_slice(&encoded).unwrap();

        assert_eq!(decoded, params);
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
