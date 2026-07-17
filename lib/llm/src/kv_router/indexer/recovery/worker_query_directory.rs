// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;
use dynamo_kv_router::protocols::{DpRank, WorkerId};
use dynamo_runtime::component::Instance;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryInstanceId, EndpointInstanceId};
use dynamo_runtime::protocols::EndpointId;

use super::worker_query_state::RecoveryKey;

/// Prefix for worker KV indexer query endpoint names.
const QUERY_ENDPOINT_PREFIX: &str = "worker_kv_indexer_query_dp";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredQueryEndpoint {
    pub(super) source_endpoint: EndpointId,
    pub(super) worker_id: WorkerId,
    pub(super) dp_rank: DpRank,
    pub(super) target: Instance,
}

#[derive(Debug, Clone)]
struct RecoveryTargetEntry {
    target: Instance,
    generation: u64,
}

#[derive(Debug, Default)]
pub(super) struct WorkerQueryEndpointDirectory {
    targets: DashMap<RecoveryKey, RecoveryTargetEntry>,
    next_generation: AtomicU64,
}

impl WorkerQueryEndpointDirectory {
    pub(super) fn parse_endpoint_name(
        endpoint_name: &str,
        route_worker_id: WorkerId,
    ) -> Option<(WorkerId, DpRank)> {
        let suffix = endpoint_name.strip_prefix(QUERY_ENDPOINT_PREFIX)?;
        let (dp_rank, worker_id) = match suffix.split_once("_worker") {
            Some((dp_rank, worker_id)) => (dp_rank.parse().ok()?, worker_id.parse().ok()?),
            None => (suffix.parse().ok()?, route_worker_id),
        };
        Some((worker_id, dp_rank))
    }

    pub(super) fn parse_added(instance: DiscoveryInstance) -> Option<DiscoveredQueryEndpoint> {
        let DiscoveryInstance::Endpoint(inst) = instance else {
            return None;
        };
        let source_endpoint = inst.source_endpoint.clone()?;
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&inst.endpoint, inst.instance_id)?;
        Some(DiscoveredQueryEndpoint {
            source_endpoint,
            worker_id,
            dp_rank,
            target: inst,
        })
    }

    pub(super) fn parse_removed(
        id: DiscoveryInstanceId,
    ) -> Option<(EndpointId, WorkerId, DpRank, EndpointInstanceId)> {
        let DiscoveryInstanceId::Endpoint(eid) = id else {
            return None;
        };
        let source_endpoint = eid.source_endpoint.clone()?;
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&eid.endpoint, eid.instance_id)?;
        Some((source_endpoint, worker_id, dp_rank, eid))
    }

    pub(super) fn insert(&self, endpoint: &DiscoveredQueryEndpoint) -> (Option<Instance>, u64) {
        let key = (endpoint.worker_id, endpoint.dp_rank);
        match self.targets.entry(key) {
            dashmap::mapref::entry::Entry::Occupied(mut entry) => {
                if entry.get().target == endpoint.target {
                    return (None, entry.get().generation);
                }
                let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
                let previous = entry.insert(RecoveryTargetEntry {
                    target: endpoint.target.clone(),
                    generation,
                });
                (Some(previous.target), generation)
            }
            dashmap::mapref::entry::Entry::Vacant(entry) => {
                let generation = self.next_generation.fetch_add(1, Ordering::Relaxed);
                entry.insert(RecoveryTargetEntry {
                    target: endpoint.target.clone(),
                    generation,
                });
                (None, generation)
            }
        }
    }

    #[cfg(test)]
    pub(super) fn target_for(&self, worker_id: WorkerId, dp_rank: DpRank) -> Option<Instance> {
        self.targets
            .get(&(worker_id, dp_rank))
            .map(|entry| entry.target.clone())
    }

    pub(super) fn target_with_generation(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
    ) -> Option<(Instance, u64)> {
        self.targets
            .get(&(worker_id, dp_rank))
            .map(|entry| (entry.target.clone(), entry.generation))
    }

    pub(super) fn keys(&self) -> HashSet<RecoveryKey> {
        self.targets.iter().map(|entry| *entry.key()).collect()
    }

    #[cfg(test)]
    pub(super) fn remove_dp(&self, worker_id: WorkerId, dp_rank: DpRank) {
        self.targets.remove(&(worker_id, dp_rank));
    }

    pub(super) fn remove_if_matches(
        &self,
        worker_id: WorkerId,
        dp_rank: DpRank,
        endpoint_id: &EndpointInstanceId,
    ) -> bool {
        let key = (worker_id, dp_rank);
        let dashmap::mapref::entry::Entry::Occupied(entry) = self.targets.entry(key) else {
            return false;
        };
        if entry.get().target.endpoint_instance_id() != *endpoint_id {
            return false;
        }
        entry.remove();
        true
    }
}
