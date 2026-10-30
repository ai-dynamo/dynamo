// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV transfer candidates the indexer emits with a match, and the hashed hint identity of a
//! cache owner that `ResidencyRoutingSnapshot` carries for them.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use dynamo_router_protocol::block_hash::ExternalSequenceBlockHash;
use dynamo_router_protocol::metadata::RouterHintSourceMetadata;

use crate::residency::ResidencyRoutingSnapshot;

pub use dynamo_router_protocol::kv_hints::KvTransferCandidateSource;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvTransferCandidates {
    pub block_hashes: Vec<ExternalSequenceBlockHash>,
    pub owner_prefix_blocks: Vec<(KvTransferCandidateSource, usize)>,
    pub routing_snapshot: Option<Arc<ResidencyRoutingSnapshot>>,
}

/// Hash a label once, when its config or snapshot is published, so matching
/// sources to targets per request compares integers instead of strings.
pub fn label_key(label: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    label.hash(&mut hasher);
    hasher.finish()
}

/// Sorted, so keys built from equal labels compare equal.
pub fn topology_keys(topology: &HashMap<String, String>) -> Vec<(u64, u64)> {
    let mut keys: Vec<_> = topology
        .iter()
        .map(|(domain, value)| (label_key(domain), label_key(value)))
        .collect();
    keys.sort_unstable();
    keys
}

/// A cache owner's router-hint identity, hashed when its routing snapshot is built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheOwnerKvTransferKeys {
    /// Hashed hint worker type.
    pub worker_type: u64,
    /// Whether the owner exposes a source control endpoint.
    pub is_source: bool,
    /// `(domain, value)` topology labels, hashed and sorted.
    pub topology: Vec<(u64, u64)>,
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
