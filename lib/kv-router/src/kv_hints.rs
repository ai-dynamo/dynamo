// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed KV-cache hints attached to selected backend requests.

mod sources;

pub use dynamo_kv_indexer::kv_hints::CacheOwnerKvTransferKeys;
pub use sources::{
    KvTransferScope, KvTransferSourceChoice, KvTransferSourceSummary, KvTransferTarget,
    ResolvedTransferSource, WorkerKvTransferKeys,
};

pub use dynamo_router_protocol::kv_hints::{
    KV_HINT_TRANSFER_CAPABILITY_KEY, KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, KvHint, KvHintAction, KvSourceLocationsPayload,
};

pub use dynamo_kv_indexer::kv_hints::{KvTransferCandidateSource, KvTransferCandidates};

use std::collections::HashMap;
use std::sync::OnceLock;

use dynamo_kv_indexer::kv_hints::{label_key, topology_keys};

use crate::protocols::{KvTransferEnforcement, WorkerConfigLike, WorkerId};

/// Derives a worker's hashed hint identity from its runtime config. Lives beside
/// [`WorkerConfigLike`] because the selection stage does not know the config trait.
pub trait WorkerKvTransferKeysExt: Sized {
    /// Hash the worker's hint labels and collect its source ranks.
    fn of<C: WorkerConfigLike + ?Sized>(config: &C) -> Self;
}

impl WorkerKvTransferKeysExt for WorkerKvTransferKeys {
    fn of<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
        let start = config.data_parallel_start_rank();
        let mut worker_type = None;
        let mut source_ranks = Vec::new();
        for rank in start..start.saturating_add(config.data_parallel_size()) {
            let Some(metadata) = config.kv_hint_transfer_metadata_for_dp_rank(rank) else {
                continue;
            };
            worker_type.get_or_insert_with(|| label_key(metadata.worker_type));
            if metadata
                .source_control_endpoint
                .is_some_and(|endpoint| !endpoint.is_empty())
            {
                source_ranks.push(rank);
            }
        }
        let topology = config
            .topology_domains()
            .map(topology_keys)
            .unwrap_or_default();
        let transfer_domain = config.kv_transfer_domain().map(label_key);
        WorkerKvTransferKeys::from_parts(
            worker_type,
            source_ranks,
            topology,
            transfer_domain,
            config.kv_transfer_enforcement() == Some(KvTransferEnforcement::Preferred),
        )
    }
}

/// [`WorkerKvTransferKeys`] derived from a worker config on first use. The
/// config's other fields decide equality, and it must not change afterwards.
#[derive(Clone, Debug, Default)]
pub struct WorkerKvTransferKeysCache(OnceLock<WorkerKvTransferKeys>);

impl WorkerKvTransferKeysCache {
    pub fn get_or_init<C: WorkerConfigLike + ?Sized>(&self, config: &C) -> &WorkerKvTransferKeys {
        self.0.get_or_init(|| WorkerKvTransferKeys::of(config))
    }
}

impl PartialEq for WorkerKvTransferKeysCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// Resolve a request's transfer candidates into the sources a selection can rank: skip any
/// no longer than every target's local prefix or longer than the retained chain, drop worker
/// ranks that publish no source endpoint, and look cache owners up in the routing snapshot
/// (an owner without topology sits in its attached worker's island).
pub fn resolve_transfer_sources<C: WorkerConfigLike>(
    candidates: &KvTransferCandidates,
    workers: &HashMap<WorkerId, C>,
    scope: &KvTransferScope,
) -> Vec<ResolvedTransferSource> {
    let min_local_prefix_blocks = scope.min_local_prefix_blocks();
    let needs_islands = scope.needs_islands();
    let snapshot = candidates.routing_snapshot.as_deref();
    let mut resolved = Vec::new();
    for &(source, prefix_blocks) in &candidates.owner_prefix_blocks {
        // A source cannot share more of the prompt than the retained chain.
        if prefix_blocks <= min_local_prefix_blocks || prefix_blocks > candidates.block_hashes.len()
        {
            continue;
        }
        let (worker_type, attachment, topology) = match source {
            KvTransferCandidateSource::Worker(worker) => {
                let Some(config) = workers.get(&worker.worker_id) else {
                    continue;
                };
                let keys = config.kv_transfer_keys();
                let Some(worker_type) = keys.worker_type() else {
                    continue;
                };
                if !keys.is_source(worker.dp_rank) {
                    continue;
                }
                (worker_type, Some(worker), keys.topology().to_vec())
            }
            KvTransferCandidateSource::CacheOwner(owner) => {
                let Some(owner) = snapshot.and_then(|snapshot| snapshot.router_hint_source(owner))
                else {
                    continue;
                };
                let keys = &owner.kv_transfer_keys;
                if !keys.is_source {
                    continue;
                }
                // An owner that publishes no topology is in its attached worker's island.
                let topology = if keys.topology.is_empty() && needs_islands {
                    owner
                        .attached_worker
                        .and_then(|worker| workers.get(&worker.worker_id))
                        .map(|config| config.kv_transfer_keys().topology().to_vec())
                        .unwrap_or_default()
                } else {
                    keys.topology.clone()
                };
                (keys.worker_type, owner.attached_worker, topology)
            }
        };
        resolved.push(ResolvedTransferSource {
            source,
            prefix_blocks,
            worker_type,
            attachment,
            topology,
        });
    }
    resolved
}

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
