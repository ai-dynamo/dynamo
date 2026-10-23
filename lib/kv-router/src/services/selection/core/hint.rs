// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV transfer hint attached to a booked selection.

use super::*;
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

/// Pick the best router-hint source for `target`: a same-role worker (or
/// cache owner) holding a longer root-aligned prefix than the target's own
/// `target_cached_prefix_blocks`, with a non-empty control endpoint.
///
/// When the target sets a `kv_transfer_domain`, the source must also be in the
/// target's island: its `topology_domains` value for that domain equals the
/// target's. A source without that value is in no island, and a cache owner
/// that publishes no topology uses its attached worker's. With `preferred`
/// enforcement, a cross-island source is used only when no same-island source
/// qualifies.
pub(super) fn transfer_hint_for_selection(
    configs: &HashMap<WorkerId, SelectionWorkerConfig>,
    target: WorkerWithDpRank,
    target_cached_prefix_blocks: u32,
    candidates: Option<&KvTransferCandidates>,
) -> Option<KvSourceLocationsPayload> {
    let candidates = candidates?;
    let target_config = configs.get(&target.worker_id)?;
    let target_metadata = target_config.kv_hint_transfer_metadata_for_dp_rank(target.dp_rank)?;

    let is_eligible = |source| match source {
        KvTransferCandidateSource::Worker(worker) => {
            worker != target
                && configs.get(&worker.worker_id).is_some_and(|config| {
                    config.kv_event_source_mode.as_deref() != Some("state_agent_v2")
                        && config
                            .kv_hint_transfer_metadata_for_dp_rank(worker.dp_rank)
                            .is_some_and(|source_metadata| {
                                source_metadata.worker_type == target_metadata.worker_type
                                    && source_metadata
                                        .source_control_endpoint
                                        .is_some_and(|endpoint| !endpoint.is_empty())
                            })
                })
        }
        KvTransferCandidateSource::CacheOwner(owner) => candidates
            .routing_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.router_hint_source(owner))
            .is_some_and(|source| {
                source.attached_worker != Some(target)
                    && source.metadata.worker_type == target_metadata.worker_type
                    && !source.metadata.source_control_endpoint.is_empty()
            }),
    };

    let prefix_blocks_to_beat = usize::try_from(target_cached_prefix_blocks).unwrap_or(usize::MAX);
    let selected = candidates.best_ranked_source(prefix_blocks_to_beat, |source| {
        if !is_eligible(source) {
            return None;
        }
        transfer_source_topology_rank(
            configs,
            Some(&target_config.topology_domains),
            target_config.kv_transfer_domain.as_deref(),
            target_config.kv_transfer_enforcement,
            candidates,
            source,
        )
    });
    let (source, block_hashes) = selected?;
    let source_control_endpoint = match source {
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
    if block_hashes.is_empty() {
        return None;
    }
    Some(KvSourceLocationsPayload {
        source_control_endpoint,
        block_hashes,
    })
}
