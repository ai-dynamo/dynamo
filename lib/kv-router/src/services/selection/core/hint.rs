// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV transfer hint attached to a booked selection.

use super::*;
use crate::kv_hints::KvTransferSourceChoice;
use crate::protocols::KvTransferEnforcement;

/// Whether any worker in the published partition snapshot advertises a router
/// hint worker type (worker-level metadata, so one rank suffices).
pub(super) fn hint_capable_partition(configs: &HashMap<WorkerId, SelectionWorkerConfig>) -> bool {
    configs.values().any(|config| {
        config
            .router_hint_worker_type
            .as_deref()
            .is_some_and(|worker_type| !worker_type.is_empty())
    })
}

/// Shared island policy for same-role hints and cross-role donor lookups.
/// Compatibility and source capability must be checked separately before ranking.
pub(super) fn transfer_source_topology_rank(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target_domains: Option<&HashMap<String, String>>,
    transfer_domain: Option<&str>,
    enforcement: Option<KvTransferEnforcement>,
    candidates: &KvTransferCandidates,
    source: KvTransferCandidateSource,
) -> Option<bool> {
    let Some(domain) = transfer_domain else {
        return Some(false);
    };
    let target_island = target_domains.and_then(|domains| domains.get(domain));
    let source_island = match source {
        KvTransferCandidateSource::Worker(worker) => configs
            .get(&worker.worker_id)
            .and_then(|config| config.topology_domains.get(domain)),
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.router_hint_source(owner))
            .and_then(|source| {
                let topology = if source.metadata.topology_domains.is_empty() {
                    &configs
                        .get(&source.attached_worker?.worker_id)?
                        .topology_domains
                } else {
                    &source.metadata.topology_domains
                };
                topology.get(domain)
            }),
    };
    let same_island = target_island.is_some() && source_island == target_island;
    (same_island || enforcement == Some(KvTransferEnforcement::Preferred)).then_some(same_island)
}

/// The fetch payload for the source worker selection chose (see
/// [`crate::kv_hints::KvTransferSourceSummary`]). The endpoint comes from the
/// current snapshot, so a source that left since selection gets no hint.
pub(super) fn transfer_hint_payload(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    source: KvTransferSourceChoice,
    candidates: &KvTransferCandidates,
) -> Option<KvSourceLocationsPayload> {
    let source_control_endpoint = match source.source {
        KvTransferCandidateSource::Worker(worker) => configs
            .get(&worker.worker_id)?
            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)?
            .source_control_endpoint?
            .to_string(),
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()?
            .router_hint_source(owner)?
            .metadata
            .source_control_endpoint
            .clone(),
    };
    let block_hashes = candidates.block_hashes.get(..source.prefix_blocks)?;
    if source_control_endpoint.is_empty() || block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes: block_hashes.to_vec(),
    })
}
