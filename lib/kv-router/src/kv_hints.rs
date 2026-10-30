// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

mod sources;

pub use dynamo_kv_indexer::kv_hints::CacheOwnerKvTransferKeys;
pub use sources::{
    KvTransferScope, KvTransferSourceChoice, KvTransferSourceSummary, KvTransferTarget,
    WorkerKvTransferKeys, WorkerKvTransferKeysCache,
};

pub use dynamo_router_protocol::kv_hints::{
    KV_HINT_TRANSFER_CAPABILITY_KEY, KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, KvHint, KvHintAction, KvSourceLocationsPayload,
};

pub use dynamo_kv_indexer::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::ExternalSequenceBlockHash;

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
