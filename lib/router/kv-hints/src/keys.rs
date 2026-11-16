// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A worker's hashed hint identity, built once from its runtime config: the key builder over
//! the config's fields, the once-per-config cache, and the two plain data types a worker's
//! runtime config speaks about hints. The label hashing ([`label_key`], [`topology_keys`]) is
//! the indexer's, re-exported here so worker keys and the indexer's cache-owner keys hash the
//! same way.

use std::collections::HashMap;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::transfer::WorkerKvTransferKeys;

pub use dynamo_kv_indexer::{label_key, topology_keys};

/// TRANSFER hint metadata a worker config exposes for one global DP rank.
///
/// Borrowed from the config so candidate filtering can check capability, role compatibility,
/// and source-endpoint presence without allocating. `source_control_endpoint` is optional
/// because a target only consumes hints, while a source must publish an endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KvHintTransferWorkerMetadata<'a> {
    /// The hint worker type; a target fetches only from a source of the same type.
    pub worker_type: &'a str,
    /// The control endpoint a target fetches this rank's blocks from; `None` when the rank is
    /// not a source.
    pub source_control_endpoint: Option<&'a str>,
}

/// How a worker's `kv_transfer_domain` constrains the KV transfer: whether a same-island
/// source is required or only preferred.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvTransferEnforcement {
    /// Put the generated topology taint in `RoutingConstraints.required_taints`.
    Required,
    /// Put the generated topology taint in `RoutingConstraints.preferred_taints`.
    Preferred,
}

impl WorkerKvTransferKeys {
    /// Hash a worker's hint labels and collect its source ranks from the fields of its runtime
    /// config: the global DP ranks `start_rank..start_rank + dp_size`, each rank's hint
    /// metadata, the topology labels, the transfer domain, and the enforcement mode. The
    /// scheduler kernel's `WorkerKvTransferKeysExt::of` reads those fields off a
    /// `WorkerConfigLike` and calls this.
    pub fn from_worker<'a>(
        start_rank: u32,
        dp_size: u32,
        per_rank_metadata: impl Fn(u32) -> Option<KvHintTransferWorkerMetadata<'a>>,
        topology: Option<&HashMap<String, String>>,
        transfer_domain: Option<&str>,
        enforcement: Option<KvTransferEnforcement>,
    ) -> Self {
        let mut worker_type = None;
        let mut source_ranks = Vec::new();
        for rank in start_rank..start_rank.saturating_add(dp_size) {
            let Some(metadata) = per_rank_metadata(rank) else {
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
        let topology = topology.map(topology_keys).unwrap_or_default();
        let transfer_domain = transfer_domain.map(label_key);
        WorkerKvTransferKeys::from_parts(
            worker_type,
            source_ranks,
            topology,
            transfer_domain,
            enforcement == Some(KvTransferEnforcement::Preferred),
        )
    }
}

/// [`WorkerKvTransferKeys`] derived from a worker config on first use. The
/// config's other fields decide equality, and it must not change afterwards.
#[derive(Clone, Debug, Default)]
pub struct WorkerKvTransferKeysCache(OnceLock<WorkerKvTransferKeys>);

impl WorkerKvTransferKeysCache {
    /// The cached keys, built with `init` on the first call.
    pub fn get_or_init(
        &self,
        init: impl FnOnce() -> WorkerKvTransferKeys,
    ) -> &WorkerKvTransferKeys {
        self.0.get_or_init(init)
    }
}

impl PartialEq for WorkerKvTransferKeysCache {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_transfer_enforcement_serde() {
        assert_eq!(
            serde_json::to_string(&KvTransferEnforcement::Required).unwrap(),
            r#""required""#
        );
        assert_eq!(
            serde_json::from_str::<KvTransferEnforcement>(r#""preferred""#).unwrap(),
            KvTransferEnforcement::Preferred
        );
        assert!(serde_json::from_str::<KvTransferEnforcement>(r#""fallback""#).is_err());
    }
}
