// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

mod sources;

use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwnerKey, ResidencyRoutingSnapshot, WorkerWithDpRank,
};

#[cfg_attr(not(feature = "standalone-selection"), allow(unused_imports))]
pub(crate) use sources::kv_copy_locations;
pub use sources::{
    CacheOwnerKvTransferKeys, KvCopyLocations, KvCopySource, KvTransferScope,
    KvTransferSourceChoice, KvTransferSourceSummary, KvTransferTarget, WorkerKvTransferKeys,
    WorkerKvTransferKeysCache,
};

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
/// Upper bound on the actions one hint carries.
pub const MAX_KV_HINT_ACTIONS: usize = 32;
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.fetch@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Root-aligned request hashes through the fetchable end. `block_hashes[i]`
    /// corresponds to request block `i`; the target fetches its missing suffix.
    /// With fetch continuation, the source need not hold the target's local prefix.
    /// If that prefix has been evicted, the receiver must use actual fetch
    /// results and recompute from any remaining gap; the hint is not a pin.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

/// One versioned action in a [`KvHint`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHintAction {
    pub action_id: String,
    pub action_type: String,
    pub action_version: String,
    pub payload: BTreeMap<String, serde_json::Value>,
}

impl KvHintAction {
    /// An action with an explicit ID. When the router composes a hint, it numbers every action
    /// again (see [`KvHint::append_numbered`]), so a hint policy can use [`Self::unnumbered`].
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

    /// An action without an ID. The router numbers it when it composes the hint.
    pub fn unnumbered(
        action_type: impl Into<String>,
        action_version: impl Into<String>,
        payload: BTreeMap<String, serde_json::Value>,
    ) -> Self {
        Self::new(String::new(), action_type, action_version, payload)
    }

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
    pub protocol_version: String,
    pub message_id: String,
    pub actions: Vec<KvHintAction>,
}

impl KvHint {
    pub fn new(message_id: impl Into<String>, actions: Vec<KvHintAction>) -> Self {
        Self {
            protocol_version: KV_HINT_PROTOCOL_VERSION.to_string(),
            message_id: message_id.into(),
            actions,
        }
    }

    /// An empty hint for one routed request. Its message ID is the request ID.
    pub fn for_request(request_id: impl Into<String>) -> Self {
        Self::new(request_id, Vec::new())
    }

    /// Append one producer's actions whole, numbering each by its position in the hint: `a1`,
    /// `a2`, and so on. Numbering replaces any ID the producer set, so action IDs are unique
    /// without producers coordinating them.
    ///
    /// Rejects all of `actions`, leaving the hint unchanged, when they would take it past
    /// [`MAX_KV_HINT_ACTIONS`].
    pub fn append_numbered(
        &mut self,
        actions: Vec<KvHintAction>,
    ) -> Result<(), KvHintActionLimitError> {
        if self.actions.len() + actions.len() > MAX_KV_HINT_ACTIONS {
            return Err(KvHintActionLimitError {
                present: self.actions.len(),
                added: actions.len(),
            });
        }
        for mut action in actions {
            action.action_id = format!("a{}", self.actions.len() + 1);
            self.actions.push(action);
        }
        Ok(())
    }

    /// Whether the hint carries no action.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

/// Actions that would take a hint past [`MAX_KV_HINT_ACTIONS`].
#[derive(Debug, thiserror::Error)]
#[error(
    "a hint carries at most {max} actions; it has {present} and {added} more were offered",
    max = MAX_KV_HINT_ACTIONS
)]
pub struct KvHintActionLimitError {
    present: usize,
    added: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KvTransferCandidateSource {
    Worker(WorkerWithDpRank),
    CacheOwner(ResidencyOwnerKey),
}

impl From<WorkerWithDpRank> for KvTransferCandidateSource {
    fn from(worker: WorkerWithDpRank) -> Self {
        Self::Worker(worker)
    }
}

/// One source's contiguous coverage of request blocks `[start, end)`.
/// A destination can use it only when its actual local prefix reaches `start`.
/// `start` may be clamped to the first destination frontier that needs the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvTransferSourceInterval {
    pub source: KvTransferCandidateSource,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KvTransferCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(KvTransferCandidateSource, usize)>,
    /// Suffix-only coverage that can extend an original device+host prefix.
    pub source_intervals: Vec<KvTransferSourceInterval>,
    pub routing_snapshot: Option<Arc<ResidencyRoutingSnapshot>>,
}

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
    fn appended_actions_are_numbered_and_bounded() {
        let action = |id: &str| KvHintAction::new(id, "test.action", "1", BTreeMap::new());
        let mut hint = KvHint::for_request("request-1");
        assert!(hint.is_empty());
        hint.append_numbered(vec![
            action("x"),
            KvHintAction::unnumbered("test.action", "1", BTreeMap::new()),
        ])
        .unwrap();
        hint.append_numbered(vec![action("x")]).unwrap();
        let ids: Vec<_> = hint
            .actions
            .iter()
            .map(|action| action.action_id.as_str())
            .collect();
        assert_eq!(ids, ["a1", "a2", "a3"]);
        assert_eq!(hint.message_id, "request-1");

        // A batch that would overflow the hint is rejected whole.
        let error = hint
            .append_numbered(vec![action("y"); MAX_KV_HINT_ACTIONS - 2])
            .unwrap_err();
        assert!(error.to_string().contains("at most 32 actions"), "{error}");
        assert_eq!(hint.actions.len(), 3);
        hint.append_numbered(vec![action("y"); MAX_KV_HINT_ACTIONS - 3])
            .unwrap();
        assert_eq!(hint.actions.len(), MAX_KV_HINT_ACTIONS);
        assert_eq!(hint.actions[MAX_KV_HINT_ACTIONS - 1].action_id, "a32");
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
