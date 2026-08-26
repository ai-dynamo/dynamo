// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

use std::{collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::protocols::{
    ExternalSequenceBlockHash, ResidencyOwnerKey, ResidencyRoutingSnapshot, WorkerWithDpRank,
};

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
const KV_SOURCE_LOCATIONS_ACTION_TYPE: &str = "kv.source_locations";
const KV_SOURCE_LOCATIONS_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.source_locations@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
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

    pub fn source_locations(
        action_id: impl Into<String>,
        payload: KvSourceLocationsPayload,
    ) -> Self {
        let KvSourceLocationsPayload {
            source_control_endpoint,
            block_hashes,
        } = payload;
        Self::new(
            action_id,
            KV_SOURCE_LOCATIONS_ACTION_TYPE,
            KV_SOURCE_LOCATIONS_ACTION_VERSION,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTransferCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(KvTransferCandidateSource, usize)>,
    pub routing_snapshot: Option<Arc<ResidencyRoutingSnapshot>>,
}

impl KvTransferCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(KvTransferCandidateSource, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(KvTransferCandidateSource) -> bool,
    {
        let (source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(source, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*source)
            })
            .max_by(|(left_source, left_blocks), (right_source, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_source.cmp(left_source))
            })?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_versioned_source_locations_action() {
        let hint = KvHint::new(
            "msg-123",
            vec![KvHintAction::source_locations(
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
                    "action_type": "kv.source_locations",
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

    #[test]
    fn best_source_selects_longest_eligible_prefix() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let excluded = WorkerWithDpRank::new(9, 0);
        let candidates = KvTransferCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![
                (worker_b.into(), 2),
                (excluded.into(), 3),
                (worker_a.into(), 3),
            ],
            routing_snapshot: None,
        };

        let selected = candidates.best_source(0, |source| source != excluded.into());

        assert_eq!(
            selected,
            Some((
                worker_a.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            ))
        );
    }

    #[test]
    fn best_source_fails_closed_on_invalid_prefix_length() {
        let candidates = KvTransferCandidates {
            block_hashes: vec![ExternalSequenceBlockHash(101)],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0).into(), 2)],
            routing_snapshot: None,
        };

        assert!(candidates.best_source(0, |_| true).is_none());
    }

    #[test]
    fn best_source_requires_prefix_longer_than_threshold() {
        let worker_a = WorkerWithDpRank::new(7, 0);
        let worker_b = WorkerWithDpRank::new(8, 0);
        let candidates = KvTransferCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
                ExternalSequenceBlockHash(104),
            ],
            owner_prefix_blocks: vec![(worker_a.into(), 3), (worker_b.into(), 4)],
            routing_snapshot: None,
        };

        assert!(
            candidates
                .best_source(3, |source| source == worker_a.into())
                .is_none()
        );
        assert_eq!(
            candidates.best_source(3, |_| true),
            Some((
                worker_b.into(),
                vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                    ExternalSequenceBlockHash(104),
                ],
            ))
        );
    }
}
