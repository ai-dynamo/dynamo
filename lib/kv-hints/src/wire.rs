// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV transfer hints attached to backend requests, the runtime-config keys that advertise
//! them, and, re-exported from the indexer that produces it, who can be the source of a fetch.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use dynamo_kv_hashing::ExternalSequenceBlockHash;

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data key: the hint worker type; a target fetches only from a source of the
/// same type.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
/// Worker runtime-data key: the control endpoint of each global DP rank that can be a source,
/// as a JSON map from rank to endpoint.
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.fetch@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    /// The source's control endpoint the target fetches from.
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

/// One versioned action in a [`KvHint`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHintAction {
    /// Caller-chosen id, unique within the hint.
    pub action_id: String,
    /// The action's type, such as `kv.fetch`.
    pub action_type: String,
    /// The version of that type the payload follows.
    pub action_version: String,
    /// The payload, kept opaque so an engine forwards actions it does not know.
    pub payload: BTreeMap<String, serde_json::Value>,
}

impl KvHintAction {
    /// An action of any type and version over an already-built payload.
    pub fn new(
        action_id: impl Into<String>,
        action_type: impl Into<String>,
        action_version: impl Into<String>,
        payload: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            action_id: action_id.into(),
            action_type: action_type.into(),
            action_version: action_version.into(),
            payload,
        }
    }

    /// The `kv.fetch@1.0` action with its typed payload.
    pub fn fetch(action_id: impl Into<String>, payload: KvSourceLocationsPayload) -> Self {
        let KvSourceLocationsPayload {
            source_control_endpoint,
            block_hashes,
        } = payload;
        Self::new(
            action_id,
            KV_FETCH_ACTION_TYPE,
            KV_FETCH_ACTION_VERSION,
            BTreeMap::from([
                (
                    "source_control_endpoint".to_string(),
                    serde_json::json!(source_control_endpoint),
                ),
                ("block_hashes".to_string(), serde_json::json!(block_hashes)),
            ]),
        )
    }
}

/// One versioned KV hint message for the selected backend request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHint {
    /// The hint protocol version the message follows.
    pub protocol_version: String,
    /// Caller-chosen id of this message.
    pub message_id: String,
    /// The actions, applied in order.
    pub actions: Vec<KvHintAction>,
}

impl KvHint {
    /// A hint at the current protocol version.
    pub fn new(message_id: impl Into<String>, actions: Vec<KvHintAction>) -> Self {
        Self {
            protocol_version: KV_HINT_PROTOCOL_VERSION.to_string(),
            message_id: message_id.into(),
            actions,
        }
    }
}

/// Who holds a KV prefix a target could fetch: a worker rank, or a cache owner by its compact
/// residency key. The indexer produces it with a match and owns it; this crate keeps the
/// public path because [`crate::transfer`] names it in every source choice.
pub use dynamo_kv_indexer::kv_hints::KvTransferCandidateSource;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_versioned_fetch_action() {
        let hint = KvHint::new(
            "msg-123",
            vec![KvHintAction::fetch(
                "a1",
                KvSourceLocationsPayload {
                    source_control_endpoint: "tcp://127.0.0.1:23280".to_string(),
                    block_hashes: vec![
                        ExternalSequenceBlockHash(11),
                        ExternalSequenceBlockHash(22),
                    ],
                },
            )],
        );

        assert_eq!(
            serde_json::to_value(hint).unwrap(),
            serde_json::json!({
                "protocol_version": "0.1",
                "message_id": "msg-123",
                "actions": [{
                    "action_id": "a1",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": {
                        "source_control_endpoint": "tcp://127.0.0.1:23280",
                        "block_hashes": [11, 22],
                    },
                }],
            })
        );
    }

    #[test]
    fn deserializes_unknown_action_as_opaque_transport_data() {
        let value = serde_json::json!({
            "protocol_version": "0.1",
            "message_id": "msg-future",
            "actions": [{
                "action_id": "a-future",
                "action_type": "kv.future_action",
                "action_version": "7.3",
                "payload": {
                    "nested": {"enabled": true},
                    "items": [1, 2, 3],
                },
            }],
        });

        let hint: KvHint = serde_json::from_value(value.clone()).unwrap();

        assert_eq!(hint.actions[0].action_type, "kv.future_action");
        assert_eq!(serde_json::to_value(hint).unwrap(), value);
    }
}
