// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How workers and cache owners advertise that they take part in the KV hint protocol: the
//! `router_hint` runtime-data keys, each worker's hashed copy metadata, and cache-owner source
//! metadata. The router reads them to decide which workers get copy offers and hints.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use super::sources::{KvTransferTarget, island};
use crate::protocols::{KvTransferEnforcement, WorkerConfigLike, WorkerId, WorkerWithDpRank};

/// The selected worker can consume a `TRANSFER` hint with the v1 payload.
// TODO: Rename these constants and wire values with the matching KVCC names.
pub const KV_HINT_TRANSFER_CAPABILITY_KEY: &str = "router_hint";

/// Worker runtime-data keys used to build transfer hints.
pub const KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY: &str = "router_hint_worker_type";
pub const KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY: &str =
    "router_hint_source_control_endpoints";

/// Hash a label once, when its config or snapshot is published, so matching
/// sources to targets per request compares integers instead of strings.
pub(super) fn label_key(label: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    hasher.finish()
}

/// Sorted, so keys built from equal labels compare equal.
fn topology_keys(topology: &HashMap<String, String>) -> Vec<(u64, u64)> {
    let mut keys: Vec<_> = topology
        .iter()
        .map(|(domain, value)| (label_key(domain), label_key(value)))
        .collect();
    keys.sort_unstable();
    keys
}

/// A worker's router-hint identity, with its labels hashed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WorkerKvTransferKeys {
    /// Hint worker type; `None` when the worker cannot consume hints.
    pub(super) worker_type: Option<u64>,
    /// DP ranks with a source control endpoint, ascending.
    source_ranks: Vec<u32>,
    /// `(domain, value)` topology labels.
    pub(super) topology: Vec<(u64, u64)>,
    /// The `kv_transfer_domain` this worker fetches within, and its value for it.
    transfer_domain: Option<u64>,
    transfer_island: Option<u64>,
    preferred: bool,
}

impl WorkerKvTransferKeys {
    pub fn new<C: WorkerConfigLike + ?Sized>(config: &C) -> Self {
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
        let transfer_island = transfer_domain.and_then(|domain| island(&topology, domain));
        Self {
            worker_type,
            source_ranks,
            topology,
            transfer_domain,
            transfer_island,
            preferred: config.kv_transfer_enforcement() == Some(KvTransferEnforcement::Preferred),
        }
    }

    pub(super) fn is_source(&self, dp_rank: u32) -> bool {
        self.source_ranks.binary_search(&dp_rank).is_ok()
    }

    /// What a selection needs to find this worker's source; `None` when the
    /// worker cannot consume hints.
    pub fn target(&self) -> Option<KvTransferTarget> {
        Some(KvTransferTarget {
            worker_type: self.worker_type?,
            transfer_domain: self.transfer_domain,
            transfer_island: self.transfer_island,
            preferred: self.preferred,
        })
    }
}

/// [`WorkerKvTransferKeys`] derived from a worker config on first use. The
/// config's other fields decide equality, and it must not change afterwards.
#[derive(Clone, Debug, Default)]
pub struct WorkerKvTransferKeysCache(OnceLock<WorkerKvTransferKeys>);

impl WorkerKvTransferKeysCache {
    pub fn get_or_init<C: WorkerConfigLike + ?Sized>(&self, config: &C) -> &WorkerKvTransferKeys {
        self.0.get_or_init(|| WorkerKvTransferKeys::new(config))
    }
}

impl PartialEq for WorkerKvTransferKeysCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

/// A cache owner's router-hint identity, hashed when its routing snapshot is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheOwnerKvTransferKeys {
    pub(super) worker_type: u64,
    pub(super) is_source: bool,
    pub(super) topology: Vec<(u64, u64)>,
}

impl CacheOwnerKvTransferKeys {
    pub fn new(metadata: &RouterHintSourceMetadata) -> Self {
        Self {
            worker_type: label_key(&metadata.worker_type),
            is_source: !metadata.source_control_endpoint.is_empty(),
            topology: topology_keys(&metadata.topology_domains),
        }
    }
}

/// Persistent source metadata used to resolve a cache owner into a router hint.
///
/// This is advisory discovery metadata. Endpoint health and ownership takeover
/// are guaranteed by the persistent cache implementation, not probed by Dynamo.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouterHintSourceMetadata {
    pub source_control_endpoint: String,
    pub worker_type: String,
    /// Topology labels of the cache, like a worker's `topology_domains`. Hint
    /// targets with a `kv_transfer_domain` only fetch from the same island.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub topology_domains: HashMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedRouterHintSource {
    pub metadata: RouterHintSourceMetadata,
    pub attached_worker: Option<WorkerWithDpRank>,
    pub kv_transfer_keys: CacheOwnerKvTransferKeys,
}

/// Whether any worker in a partition snapshot advertises `router_hint` (worker-level metadata, so
/// one rank suffices).
#[cfg_attr(not(feature = "standalone-selection"), allow(dead_code))]
pub(crate) fn hint_capable_partition<C: WorkerConfigLike>(configs: &HashMap<WorkerId, C>) -> bool {
    configs
        .values()
        .any(|config| config.kv_transfer_keys().target().is_some())
}
