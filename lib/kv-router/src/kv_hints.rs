// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "kv_hint.transfer.v1";

/// Current version of the Dynamo-to-engine KV hint envelope.
pub const KV_HINT_PROTOCOL_VERSION: &str = "0.1";

/// Action discriminator for a point-to-point KV source location.
pub const KV_SOURCE_LOCATIONS_ACTION_TYPE: &str = "kv.source_locations";

/// Current payload version for [`KvSourceLocationsPayload`].
pub const KV_SOURCE_LOCATIONS_ACTION_VERSION: &str = "1.0";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "kv_hint_transfer_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "kv_hint_transfer_source_control_endpoints";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvHintProtocolVersion {
    #[default]
    #[serde(rename = "0.1")]
    V0_1,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum KvSourceLocationsActionVersion {
    #[serde(rename = "1.0")]
    V1_0,
}

/// Typed payload for the `kv.source_locations@1.0` point-to-point action.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Root-aligned source-side KV block hashes. `block_hashes[i]`
    /// corresponds to request block `i`; the target decides which suffix to fetch.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
}

/// One typed action in a [`KvHints`] envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "action_type")]
pub enum KvHintAction {
    #[serde(rename = "kv.source_locations")]
    SourceLocations {
        action_id: String,
        action_version: KvSourceLocationsActionVersion,
        payload: KvSourceLocationsPayload,
    },
}

impl KvHintAction {
    pub fn source_locations(payload: KvSourceLocationsPayload) -> Self {
        Self::source_locations_with_id(format!("a-{}", Uuid::new_v4().simple()), payload)
    }

    pub fn source_locations_with_id(
        action_id: impl Into<String>,
        payload: KvSourceLocationsPayload,
    ) -> Self {
        Self::SourceLocations {
            action_id: action_id.into(),
            action_version: KvSourceLocationsActionVersion::V1_0,
            payload,
        }
    }
}

/// Versioned actions for the selected backend request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvHints {
    pub protocol_version: KvHintProtocolVersion,
    pub message_id: String,
    pub actions: Vec<KvHintAction>,
}

impl KvHints {
    pub fn new(actions: Vec<KvHintAction>) -> Self {
        Self::with_message_id(format!("msg-{}", Uuid::new_v4().simple()), actions)
    }

    pub fn with_message_id(message_id: impl Into<String>, actions: Vec<KvHintAction>) -> Self {
        Self {
            protocol_version: KvHintProtocolVersion::V0_1,
            message_id: message_id.into(),
            actions,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTransferCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(WorkerWithDpRank, usize)>,
}

impl KvTransferCandidates {
    pub fn best_source<F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut is_eligible_source: F,
    ) -> Option<(WorkerWithDpRank, Vec<ExternalSequenceBlockHash>)>
    where
        F: FnMut(WorkerWithDpRank) -> bool,
    {
        let (source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(worker, blocks)| {
                *blocks > prefix_blocks_to_beat && is_eligible_source(*worker)
            })
            .max_by(|(left_worker, left_blocks), (right_worker, right_blocks)| {
                left_blocks
                    .cmp(right_blocks)
                    .then_with(|| right_worker.cmp(left_worker))
            })?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_versioned_source_locations_action() {
        let hints = KvHints::with_message_id(
            "msg-123",
            vec![KvHintAction::source_locations_with_id(
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
            serde_json::to_value(hints).unwrap(),
            serde_json::json!({
                "protocol_version": KV_HINT_PROTOCOL_VERSION,
                "message_id": "msg-123",
                "actions": [{
                    "action_id": "a1",
                    "action_type": KV_SOURCE_LOCATIONS_ACTION_TYPE,
                    "action_version": KV_SOURCE_LOCATIONS_ACTION_VERSION,
                    "payload": {
                        "source_control_endpoint": "tcp://127.0.0.1:23280",
                        "block_hashes": [11, 22],
                    },
                }],
            })
        );
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
            owner_prefix_blocks: vec![(worker_b, 2), (excluded, 3), (worker_a, 3)],
        };

        let selected = candidates.best_source(0, |worker| worker != excluded);

        assert_eq!(
            selected,
            Some((
                worker_a,
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
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 0), 2)],
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
            owner_prefix_blocks: vec![(worker_a, 3), (worker_b, 4)],
        };

        assert!(
            candidates
                .best_source(3, |worker| worker == worker_a)
                .is_none()
        );
        assert_eq!(
            candidates.best_source(3, |_| true),
            Some((
                worker_b,
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
