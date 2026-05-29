// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dashmap::DashMap;
use dynamo_kv_router::protocols::{DpRank, WorkerId};
use dynamo_runtime::component::Instance;
use dynamo_runtime::discovery::{DiscoveryInstance, DiscoveryInstanceId, EndpointInstanceId};

use super::worker_query_state::RecoveryKey;

/// Prefix for worker KV indexer query endpoint names.
const QUERY_ENDPOINT_PREFIX: &str = "worker_kv_indexer_query_dp";

/// Prefix for worker Velo peer-info endpoint names (velo-recovery feature only).
#[cfg(feature = "velo-recovery")]
const VELO_PEER_ENDPOINT_PREFIX: &str = "worker_kv_velo_peer_dp";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DiscoveredQueryEndpoint {
    pub(super) worker_id: WorkerId,
    pub(super) dp_rank: DpRank,
    pub(super) target: Instance,
}

#[derive(Debug, Default)]
pub(super) struct WorkerQueryEndpointDirectory {
    targets: DashMap<RecoveryKey, Instance>,
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
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&inst.endpoint, inst.instance_id)?;
        Some(DiscoveredQueryEndpoint {
            worker_id,
            dp_rank,
            target: inst,
        })
    }

    pub(super) fn parse_removed(
        id: DiscoveryInstanceId,
    ) -> Option<(WorkerId, DpRank, EndpointInstanceId)> {
        let DiscoveryInstanceId::Endpoint(eid) = id else {
            return None;
        };
        let (worker_id, dp_rank) = Self::parse_endpoint_name(&eid.endpoint, eid.instance_id)?;
        Some((worker_id, dp_rank, eid))
    }

    pub(super) fn insert(&self, endpoint: &DiscoveredQueryEndpoint) -> Option<Instance> {
        self.targets.insert(
            (endpoint.worker_id, endpoint.dp_rank),
            endpoint.target.clone(),
        )
    }

    pub(super) fn target_for(&self, worker_id: WorkerId, dp_rank: DpRank) -> Option<Instance> {
        self.targets
            .get(&(worker_id, dp_rank))
            .map(|target| target.value().clone())
    }

    /// Parse a velo peer-info endpoint name.
    ///
    /// Accepts both `worker_kv_velo_peer_dp{N}` (route instance = logical worker)
    /// and `worker_kv_velo_peer_dp{N}_worker{W}` (multi-worker-per-pod), mirroring
    /// the `_worker{id}` convention used by the legacy query endpoint.
    #[cfg(feature = "velo-recovery")]
    pub(super) fn parse_velo_peer_endpoint_name(
        endpoint_name: &str,
        route_instance_id: WorkerId,
    ) -> Option<(WorkerId, DpRank)> {
        let suffix = endpoint_name.strip_prefix(VELO_PEER_ENDPOINT_PREFIX)?;
        let (dp_rank, worker_id) = match suffix.split_once("_worker") {
            Some((dp_str, worker_str)) => (dp_str.parse().ok()?, worker_str.parse().ok()?),
            None => (suffix.parse().ok()?, route_instance_id),
        };
        Some((worker_id, dp_rank))
    }

    /// Parse a removal id, accepting either endpoint-name format.
    ///
    /// Tries `worker_kv_indexer_query_dp{N}[_worker{W}]` first, then
    /// `worker_kv_velo_peer_dp{N}[_worker{W}]`.  In practice only one of the two
    /// is active at a time: velo-recovery mode uses the Velo peer endpoint as the
    /// lifecycle anchor and skips the legacy query endpoint.
    pub(super) fn parse_removed_any(
        id: DiscoveryInstanceId,
    ) -> Option<(WorkerId, DpRank, EndpointInstanceId)> {
        let DiscoveryInstanceId::Endpoint(eid) = id else {
            return None;
        };
        // Standard query endpoint.
        if let Some((worker_id, dp_rank)) =
            Self::parse_endpoint_name(&eid.endpoint, eid.instance_id)
        {
            return Some((worker_id, dp_rank, eid));
        }
        // Velo peer endpoint.
        #[cfg(feature = "velo-recovery")]
        if let Some((worker_id, dp_rank)) =
            Self::parse_velo_peer_endpoint_name(&eid.endpoint, eid.instance_id)
        {
            return Some((worker_id, dp_rank, eid));
        }
        None
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
        if entry.get().endpoint_instance_id() != *endpoint_id {
            return false;
        }
        entry.remove();
        true
    }
}
