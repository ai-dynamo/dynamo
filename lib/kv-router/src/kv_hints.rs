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

/// Separate, opt-in direct GPU endpoint capability. It never authorizes the
/// legacy CPU `kv.fetch` action. Sources must hold eviction-safe GPU leases
/// until transport completion; event inventory alone is not a lease.
pub const KV_HINT_GPU_DIRECT_CAPABILITY_KEY: &str = "router_hint_gpu_direct";
pub const KV_HINT_GPU_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_gpu_source_control_endpoints";

/// A GPU-only source/target protocol. The single variants prevent a typed GPU
/// action from accidentally requesting CPU landing slots or a staged transport.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvGpuMemoryKind {
    Gpu,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvGpuTransport {
    NixlGpuDirect,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvGpuSourceLocationsPayload {
    pub source_control_endpoint: String,
    /// Root-aligned complete GPU blocks; the target fetches only its missing suffix.
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub source_memory_kind: KvGpuMemoryKind,
    pub target_memory_kind: KvGpuMemoryKind,
    pub transport: KvGpuTransport,
    pub compatibility_domain: String,
}

const KV_HINT_PROTOCOL_VERSION: &str = "0.1";
const KV_FETCH_ACTION_TYPE: &str = "kv.fetch";
const KV_FETCH_ACTION_VERSION: &str = "1.0";

/// Typed payload for the `kv.fetch@1.0` point-to-point action.
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

    /// The separate action is emitted only to an explicitly GPU-capable target.
    /// Legacy readers continue receiving exactly `kv.fetch@1.0` CPU payloads.
    pub fn fetch_gpu(
        action_id: impl Into<String>,
        payload: KvSourceLocationsPayload,
        compatibility_domain: String,
    ) -> Self {
        Self::new(
            action_id,
            "kv.fetch_gpu",
            "1.0",
            BTreeMap::from([
                (
                    "source_control_endpoint".into(),
                    serde_json::json!(payload.source_control_endpoint),
                ),
                (
                    "block_hashes".into(),
                    serde_json::json!(payload.block_hashes),
                ),
                (
                    "source_memory_kind".into(),
                    serde_json::json!(KvGpuMemoryKind::Gpu),
                ),
                (
                    "target_memory_kind".into(),
                    serde_json::json!(KvGpuMemoryKind::Gpu),
                ),
                (
                    "transport".into(),
                    serde_json::json!(KvGpuTransport::NixlGpuDirect),
                ),
                (
                    "compatibility_domain".into(),
                    serde_json::json!(compatibility_domain),
                ),
            ]),
        )
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
        self.best_ranked_source(prefix_blocks_to_beat, |source| {
            is_eligible_source(source).then_some(())
        })
    }

    /// Like [`Self::best_source`], but a higher `rank` wins before a longer
    /// prefix. A `None` rank makes the source ineligible.
    pub fn best_ranked_source<R, F>(
        &self,
        prefix_blocks_to_beat: usize,
        mut rank: F,
    ) -> Option<(KvTransferCandidateSource, Vec<ExternalSequenceBlockHash>)>
    where
        R: Ord,
        F: FnMut(KvTransferCandidateSource) -> Option<R>,
    {
        let (_, source, prefix_blocks) = self
            .owner_prefix_blocks
            .iter()
            .copied()
            .filter(|(_, blocks)| *blocks > prefix_blocks_to_beat)
            .filter_map(|(source, blocks)| Some((rank(source)?, source, blocks)))
            .max_by(
                |(left_rank, left_source, left_blocks),
                 (right_rank, right_source, right_blocks)| {
                    left_rank
                        .cmp(right_rank)
                        .then_with(|| left_blocks.cmp(right_blocks))
                        .then_with(|| right_source.cmp(left_source))
                },
            )?;

        Some((source, self.block_hashes.get(..prefix_blocks)?.to_vec()))
    }
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
    fn best_ranked_source_prefers_rank_over_prefix_length() {
        let near = WorkerWithDpRank::new(7, 0);
        let far = WorkerWithDpRank::new(8, 0);
        let farther = WorkerWithDpRank::new(9, 0);
        // Would win the tie with `farther` if it were eligible.
        let excluded = WorkerWithDpRank::new(6, 0);
        let candidates = KvTransferCandidates {
            block_hashes: (101..=104).map(ExternalSequenceBlockHash).collect(),
            owner_prefix_blocks: vec![
                (near.into(), 2),
                (far.into(), 3),
                (farther.into(), 4),
                (excluded.into(), 4),
            ],
            routing_snapshot: None,
        };
        let rank = |preferred: WorkerWithDpRank| {
            move |source: KvTransferCandidateSource| {
                (source != excluded.into()).then_some(source == preferred.into())
            }
        };

        // A higher rank wins over a longer prefix.
        let (source, block_hashes) = candidates.best_ranked_source(0, rank(near)).unwrap();
        assert_eq!((source, block_hashes.len()), (near.into(), 2));
        // Within one rank the longest prefix wins, and a `None` rank is skipped.
        let (source, _) = candidates.best_ranked_source(0, rank(excluded)).unwrap();
        assert_eq!(source, farther.into());
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
    #[test]
    fn gpu_fetch_is_explicit_and_does_not_change_legacy_cpu_wire() {
        let action = KvHintAction::fetch_gpu(
            "gpu",
            KvSourceLocationsPayload {
                source_control_endpoint: "tcp://gpu:23290".into(),
                block_hashes: vec![ExternalSequenceBlockHash(11), ExternalSequenceBlockHash(22)],
            },
            "weights-layout-tp2-v1".into(),
        );
        assert_eq!(action.action_type, "kv.fetch_gpu");
        assert_eq!(action.action_version, "1.0");
        assert_eq!(
            serde_json::to_value(&action.payload).unwrap(),
            serde_json::json!({
                "source_control_endpoint": "tcp://gpu:23290", "block_hashes": [11, 22],
                "source_memory_kind": "gpu", "target_memory_kind": "gpu",
                "transport": "nixl_gpu_direct", "compatibility_domain": "weights-layout-tp2-v1"
            })
        );
        let mut payload = serde_json::to_value(action.payload).unwrap();
        payload["target_memory_kind"] = serde_json::json!("cpu");
        assert!(serde_json::from_value::<KvGpuSourceLocationsPayload>(payload).is_err());
    }
}
